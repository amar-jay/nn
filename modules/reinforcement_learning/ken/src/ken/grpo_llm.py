"""Stage 2: GRPO on Qwen2.5-0.5B-Instruct with TRL, countdown reward.

Fits a 4GB GPU: 0.5B bf16 weights (~1GB), LoRA-only grads, gradient
checkpointing, and beta=0 (no frozen reference model in VRAM).
"""

import argparse
import random
import re

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from .countdown import generate_puzzle, reward

SYSTEM = (
    "Using each of the given numbers at most once, write one arithmetic "
    "expression with + - * / and parentheses that equals the target. "
    "Reply with only the expression, nothing else.\n"
    "Example:\nNumbers: [3, 7, 11, 4]\nTarget: 24\nExpression:\n(11-3)*(7-4)"
)

FRAG_RE = re.compile(r"[0-9+\-*/() ]+")


def make_dataset(n: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        numbers, target = generate_puzzle(rng, 4)
        rows.append(
            {
                "prompt": f"{SYSTEM}\nNumbers: {numbers}\nTarget: {target}\nExpression:",
                "numbers": numbers,
                "target": target,
            }
        )
    return Dataset.from_list(rows)


def countdown_reward(completions, **kwargs) -> list[float]:
    numbers = kwargs["numbers"]
    target = kwargs["target"]
    scores = []
    for text, nums, tgt in zip(completions, numbers, target):
        # Score the best arithmetic-looking fragment: instruct models ramble,
        # so grade the strongest candidate instead of only the last line.
        frags = [f.strip() for f in FRAG_RE.findall(text)]
        candidates = [f for f in frags if any(ch.isdigit() for ch in f) and any(op in f for op in "+-*/")]
        last_lines = text.strip().splitlines()
        candidates.append(last_lines[-1].strip() if last_lines else "")
        scores.append(max([reward(c, list(nums), int(tgt)) for c in candidates] or [0.0]))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default="outputs/grpo-countdown")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    config = GRPOConfig(
        output_dir=args.out,
        num_generations=args.generations,
        max_completion_length=64,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=4,
        max_steps=args.steps,
        learning_rate=5e-6,
        temperature=0.9,
        beta=0.0,  # no reference model: fits 4GB VRAM
        logging_steps=1,
        save_steps=25,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=countdown_reward,
        args=config,
        train_dataset=make_dataset(200, seed=0),
        processing_class=tok,
        peft_config=LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM"),
    )
    trainer.train()


if __name__ == "__main__":
    main()
