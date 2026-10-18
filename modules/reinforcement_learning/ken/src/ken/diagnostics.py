"""Fixed-puzzle diagnostics for operand usage, target sensitivity, and exploration."""
import random

import torch

from .countdown import (
    EOS, OPS, _uses_allowed_numbers, decode, encode, exact_reward,
    generate_puzzle, prompt_for, safe_eval,
)


@torch.no_grad()
def _greedy(net, prompt, max_new):
    ids = list(prompt)
    device = next(net.parameters()).device
    if len(ids) + max_new > net.pos.num_embeddings:
        raise ValueError("prompt plus max_new exceeds model context")
    for _ in range(max_new):
        token = int(net(torch.tensor([ids], device=device))[0, -1].argmax())
        ids.append(token)
        if token == EOS:
            break
    return ids


def _score(ids, plen, numbers, target):
    expression = decode(ids[plen:])
    value = safe_eval(expression)
    return dict(expression=expression, value=value,
                allowed_numbers=_uses_allowed_numbers(expression, numbers),
                valid_expression=value is not None,
                exact=bool(exact_reward(expression, numbers, target)))


def _target_variants(numbers, rng, count=4):
    """Construct distinct solvable targets from the same operands."""
    variants = {}
    for _ in range(512):
        operands = rng.sample(numbers, len(numbers))
        value, expression = operands[0], str(operands[0])
        for operand in operands[1:]:
            op = rng.choice(OPS)
            value = {"+": value + operand, "-": value - operand,
                     "*": value * operand}[op]
            expression = f"({expression}{op}{operand})"
        variants.setdefault(value, expression)
        if len(variants) == count:
            break
    return variants


def evaluate_model(net, count=100, n_numbers=3, samples=12, seed=12345,
                   max_new=48, target_sets=5):
    """Use fixed puzzles, isolate sampling RNG, and restore the model's mode.

    at_least_one_exact is the observed success rate in sets of `samples`
    stochastic completions, not a greedy or best-answer selection score.
    allowed_numbers checks operands independently of arithmetic syntax.
    """
    from .grpo_toy import sample

    if count < 1 or samples < 1 or n_numbers < 2 or max_new < 1 or target_sets < 0:
        raise ValueError("positive count/samples/max_new, numbers >= 2, target_sets >= 0 required")
    rng = random.Random(seed)
    device = next(net.parameters()).device
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    was_training = net.training
    rows, probes = [], []
    try:
        net.eval()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            for _ in range(count):
                numbers, target = generate_puzzle(rng, n_numbers)
                prompt = encode(prompt_for(numbers, target))
                greedy = _score(_greedy(net, prompt, max_new), len(prompt), numbers, target)
                candidates = [_score(ids, len(prompt), numbers, target)
                              for ids in sample(net, prompt, samples, max_new=max_new)]
                rows.append(dict(numbers=numbers, target=target, greedy=greedy,
                                 samples=candidates))
            # Separate RNG so probe settings do not change evaluation puzzles.
            probe_rng = random.Random(seed + 1)
            for _ in range(target_sets):
                numbers, _ = generate_puzzle(probe_rng, n_numbers)
                answers = []
                for target, witness in _target_variants(numbers, probe_rng).items():
                    prompt = encode(prompt_for(numbers, target))
                    answer = _score(_greedy(net, prompt, max_new), len(prompt), numbers, target)
                    answers.append(dict(target=target, witness=witness, **answer))
                probes.append(dict(numbers=numbers, answers=answers,
                                   distinct_expressions=len({a['expression'] for a in answers}),
                                   distinct_values=len({a['value'] for a in answers})))
    finally:
        net.train(was_training)
    greedy_rows = [row['greedy'] for row in rows]
    sampled = [answer for row in rows for answer in row['samples']]
    def rate(items, key):
        return sum(item[key] for item in items) / len(items)
    summary = dict(
        greedy_exact=rate(greedy_rows, 'exact'),
        sampled_exact=rate(sampled, 'exact'),
        at_least_one_exact=sum(any(a['exact'] for a in r['samples']) for r in rows) / count,
        greedy_allowed_numbers=rate(greedy_rows, 'allowed_numbers'),
        sampled_allowed_numbers=rate(sampled, 'allowed_numbers'),
        greedy_valid_expression=rate(greedy_rows, 'valid_expression'),
        sampled_valid_expression=rate(sampled, 'valid_expression'),
        mean_distinct_sampled_expressions=sum(len({a['expression'] for a in r['samples']}) for r in rows) / count,
    )
    return dict(seed=seed, count=count, n_numbers=n_numbers, samples=samples,
                max_new=max_new, summary=summary, results=rows, target_probes=probes)


def print_report(report):
    print(f"\nDiagnostics: {report['count']} puzzles, {report['samples']} samples/puzzle, seed={report['seed']}")
    for name, value in report['summary'].items():
        print(f"  {name}: {value:.2f}" if name.startswith('mean_') else f"  {name}: {value:.1%}")
    for probe in report['target_probes']:
        print(f"\nSame numbers {probe['numbers']}; distinct outputs: {probe['distinct_expressions']}")
        for answer in probe['answers']:
            print(f"  target {answer['target']:>6}: {answer['expression']!r} = {answer['value']} | exact={answer['exact']}")
    if report['target_probes']:
        print("Changed outputs alone do not prove target understanding; check their values and correctness.")


def diagnose(model_path='grpo_toy.pt', device=None, **kwargs):
    from .grpo_toy import TinyGPT
    net = TinyGPT().to(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    net.load_state_dict(torch.load(model_path, map_location=next(net.parameters()).device,
                                   weights_only=True))
    report = evaluate_model(net, **kwargs)
    report['checkpoint'] = str(model_path)
    print_report(report)
    return report
