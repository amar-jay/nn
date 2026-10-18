"""Countdown game: verifiable reward task for GRPO practice.

A puzzle gives N numbers and a target. The policy must emit an arithmetic
expression that evaluates to the target, using each number at most once.
"""

import math
import random
import re
import warnings

OPS = ("+", "-", "*")
TOKEN_RE = re.compile(r"\d+|[+\-*/()]")

# ----------------- ENCODING -----------------
CHARS = "0123456789+-*/() :="
BOS, EOS, PAD = 0, 1, 2
stoi = {"<bos>": BOS, "<eos>": EOS, "<pad>": PAD, **{c: i + 3 for i, c in enumerate(CHARS)}}
VOCAB = len(stoi)
itos = {i: c for c, i in stoi.items()}


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids if i >= 3)


def prompt_for(numbers: list[int], target: int) -> str:
    return f"{' '.join(map(str, numbers))}:{target}="


# ----------------- PUZZLE GENERATION & REWARD -----------------

def generate_puzzle(rng: random.Random, n_numbers: int = 4) -> tuple[list[int], int]:
    """Sample numbers plus a guaranteed-solvable target (built with +,-,*)."""
    numbers = [rng.randint(1, 20) for _ in range(n_numbers)]
    pool = numbers[:]
    rng.shuffle(pool)
    # Apply ops left-to-right with plain Python arithmetic:
    total = pool[0]
    for x in pool[1:]:
        op = rng.choice(OPS)
        total = {"+": total + x, "-": total - x, "*": total * x}[op]
    return numbers, total


def _uses_allowed_numbers(expr: str, numbers: list[int]) -> bool:
    used = [int(t) for t in TOKEN_RE.findall(expr) if t.isdigit()]
    remaining = list(numbers)
    for u in used:
        if u not in remaining:
            return False
        remaining.remove(u)
    return len(used) >= 2


def safe_eval(expr: str) -> float | None:
    """Evaluate arithmetic expr; None if invalid."""
    if not expr or "**" in expr or "//" in expr:
        return None
    if not re.fullmatch(r"[0-9+\-*/() ]+", expr):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            val = eval(expr, {"__builtins__": {}}, {})  # locked-down globals
        except (SyntaxError, NameError, TypeError, ZeroDivisionError, ValueError, OverflowError):
            return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    if isinstance(val, int) and val.bit_length() > 1024:
        return None
    try:
        val = float(val)
    except (OverflowError, ValueError):
        return None
    return val if math.isfinite(val) else None

def random_valid_expr(numbers: list[int], rng: random.Random) -> str:
    """Random well-formed expression using all supplied numbers, not a solution."""
    operands = rng.sample(numbers, len(numbers))
    e = str(operands[0])
    for operand in operands[1:]:
        op = rng.choice(OPS)
        e = f"({e}{op}{operand})"
    return e



def reward(expr: str, numbers: list[int], target: int) -> float:
    """1.0 exact hit, 0.3 valid-but-wrong, 0.1 parseable-only, else 0.0."""
    val = safe_eval(expr)
    if val is None:
        return 0.0
    if not _uses_allowed_numbers(expr, numbers):
        return 0.1
    return 1.0 if abs(val - target) < 1e-6 else 0.3
