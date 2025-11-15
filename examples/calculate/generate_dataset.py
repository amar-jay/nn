# ...existing code...
"""
CSV dataset generator

Produces rows with columns:
 - a (float)
 - b (float or empty for unary ops)
 - op (string)
 - result (float or empty if invalid)
 - valid (bool)
 - error (text reason for invalid result)

Usage (examples):
  python generate_dataset.py --out dataset.csv --n 1000000 --min -10 --max 10
  python generate_dataset.py --out /tmp/big.csv --n 5000000 --min 0.001 --max 100 --chunk 200000

Notes:
 - Binary ops: add - x / (division by zero marked invalid)
 - Unary ops: ln, exp, sin, cos, tan, sinh, cosh, tanh,
   arcsin, arccos, arctan, arcsinh, arccosh, arctanh, radtodeg, degtorad
 - For domain-restricted unary ops (e.g. arcsin, arccosh, ln), by default the script
   samples values inside the valid domain when possible to maximize valid examples.
 - The script writes in chunks (default 100k rows) to avoid excessive memory use.
"""

import argparse
import sys
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Operation definitions
_BINARY_OPS = ["add", "sub", "mul", "div"]
_UNARY_OPS = [
    "ln", "exp",
    "sin", "cos", "tan",
    "sinh", "cosh", "tanh",
    "arcsin", "arccos", "arctan",
    "arcsinh", "arccosh", "arctanh",
    "radtodeg", "degtorad",
]
ALL_OPS = _BINARY_OPS + _UNARY_OPS

# Mapping op name -> function that accepts numpy arrays a, b (b may be None)
# Returns tuple (result_array, valid_mask, error_reason_array)
def _compute_op(op: str, a: np.ndarray, b: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = a.shape[0]
    res = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    error = np.full(n, "", dtype=object)

    try:
        if op == "add":
            res = a + b
            valid = np.isfinite(res)
        elif op == "sub":
            res = a - b
            valid = np.isfinite(res)
        elif op == "mul":
            res = a * b
            valid = np.isfinite(res)
        elif op == "div":
            mask_zero = (b == 0)
            valid_mask = ~mask_zero
            res[valid_mask] = a[valid_mask] / b[valid_mask]
            valid = np.logical_and(np.isfinite(res), valid_mask)
            error[mask_zero] = "division_by_zero"
        elif op == "ln":
            mask_pos = (a > 0)
            res[mask_pos] = np.log(a[mask_pos])
            # valid = np.logical_and(np.isfinite(res), ~mask_pos)
            valid = np.isfinite(res)
            error[~mask_pos] = "ln_domain"
        elif op == "exp":
            # exp can overflow; will produce inf if too large
            res = np.exp(a)
            valid = np.isfinite(res)
            error[~valid] = "overflow_or_invalid"
        elif op == "sin":
            res = np.sin(a)
            valid = np.isfinite(res)
        elif op == "cos":
            res = np.cos(a)
            valid = np.isfinite(res)
        elif op == "tan":
            res = np.tan(a)
            valid = np.isfinite(res)
        elif op == "sinh":
            res = np.sinh(a)
            valid = np.isfinite(res)
            error[~valid] = "overflow_or_invalid"
        elif op == "cosh":
            res = np.cosh(a)
            valid = np.isfinite(res)
            error[~valid] = "overflow_or_invalid"
        elif op == "tanh":
            res = np.tanh(a)
            valid = np.isfinite(res)
        elif op == "arcsin":
            mask = (a >= -1) & (a <= 1)
            res[mask] = np.arcsin(a[mask])
            valid = np.isfinite(res) & mask
            error[~mask] = "arcsin_domain"
        elif op == "arccos":
            mask = (a >= -1) & (a <= 1)
            res[mask] = np.arccos(a[mask])
            valid = np.isfinite(res) & mask
            error[~mask] = "arccos_domain"
        elif op == "arctan":
            res = np.arctan(a)
            valid = np.isfinite(res)
        elif op == "arcsinh":
            res = np.arcsinh(a)
            valid = np.isfinite(res)
        elif op == "arccosh":
            mask = (a >= 1)
            res[mask] = np.arccosh(a[mask])
            valid = np.isfinite(res) & mask
            error[~mask] = "arccosh_domain"
        elif op == "arctanh":
            mask = (a > -1) & (a < 1)
            res[mask] = np.arctanh(a[mask])
            valid = np.isfinite(res) & mask
            error[~mask] = "arctanh_domain"
        elif op == "radtodeg":
            res = np.degrees(a)
            valid = np.isfinite(res)
        elif op == "degtorad":
            res = np.radians(a)
            valid = np.isfinite(res)
        else:
            error[:] = "unsupported_op"
            # valid[:] = False
    except Exception as e:
        # Fallback: mark all rows in this chunk invalid with the exception message
        valid[:] = False
        error[:] = str(e)
    return res, valid, error

def _op_is_unary(op: str) -> bool:
    return op in _UNARY_OPS

def _domain_sample_range_for_op(op: str, global_min: float, global_max: float) -> Tuple[float, float]:
    # For domain-restricted ops, return a preferred sampling range intersected with user range.
    if op in ("arcsin", "arccos"):
        # domain [-1,1]
        lo, hi = max(global_min, -1.0), min(global_max, 1.0)
        return lo, hi
    if op == "arctanh":
        lo, hi = max(global_min, -0.999999999), min(global_max, 0.999999999)
        return lo, hi
    if op == "arccosh":
        lo, hi = max(global_min, 1.0), global_max
        return lo, hi
    if op == "ln":
        lo, hi = max(global_min, 1e-12), global_max
        return lo, hi
    # others: use full user provided range
    return global_min, global_max

def _sample_uniform(n: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    if lo == hi:
        return np.full(n, lo, dtype=np.float64)
    return rng.uniform(lo, hi, size=n)

def generate_dataset(outfile: str,
                     n: int,
                     lo: float,
                     hi: float,
                     ops: Optional[List[str]] = None,
                     chunk: int = 100000,
                     domain_sensitive: bool = True,
                     seed: Optional[int] = None,
                     balanced: bool = True) -> None:
    """
    Generate dataset and write CSV to outfile.

    - ops: list of operation names (subset of ALL_OPS). Defaults to ALL_OPS.
    - n: total number of rows (roughly). If balanced=True, roughly n/len(ops) per op.
    - chunk: number of rows written per iteration.
    - domain_sensitive: attempt to sample from valid domains for domain-restricted ops.
    - seed: RNG seed.
    """
    if ops is None:
        ops = ALL_OPS
    for o in ops:
        if o not in ALL_OPS:
            raise ValueError(f"Unknown op: {o}")

    rng = np.random.default_rng(seed)

    # Prepare counts per operation
    if balanced:
        base = n // len(ops)
        counts = [base] * len(ops)
        rem = n - base * len(ops)
        for i in range(rem):
            counts[i % len(ops)] += 1
    else:
        # sample ops randomly
        counts = None

    header_written = False
    total_written = 0

    pbar = tqdm(total=n, unit="rows", desc="Generating")

    try:
        with open(outfile, "w", encoding="utf-8") as f:
            # Write header now using pandas for consistency (no `error` column)
            df0 = pd.DataFrame(columns=["a", "b", "op", "result", "valid"])
            df0.to_csv(f, index=False)
            header_written = True

        # Now append chunks
        for op_idx, op in enumerate(ops):
            want = counts[op_idx] if counts is not None else None
            if want == 0:
                continue
            produced = 0
            while (want is None and total_written < n) or (want is not None and produced < want):
                to_make = chunk
                if want is not None:
                    to_make = min(to_make, want - produced)
                else:
                    to_make = min(to_make, n - total_written)
                if to_make <= 0:
                    break

                # Decide sampling ranges for a and b
                if _op_is_unary(op):
                    a_lo, a_hi = (lo, hi)
                    if domain_sensitive:
                        a_lo, a_hi = _domain_sample_range_for_op(op, lo, hi)
                    a = _sample_uniform(to_make, a_lo, a_hi, rng)
                    b = np.full(to_make, np.nan, dtype=np.float64)
                else:
                    # binary
                    a_lo, a_hi = lo, hi
                    b_lo, b_hi = lo, hi
                    if domain_sensitive:
                        # For binary operations no special domain except division; sample b avoiding zero for division
                        if op == "div":
                            # Ensure b is not zero by sampling from two ranges if zero is inside.
                            if lo < 0 < hi:
                                # sample from [lo, -eps] U [eps, hi]
                                eps = max(1e-12, (hi - lo) * 1e-12)
                                left_size = rng.uniform(0.0, 1.0)
                                # simple approach: sample uniformly but redraw zeros
                                b = _sample_uniform(to_make, lo, hi, rng)
                                zero_mask = (b == 0)
                                tries = 0
                                while zero_mask.any() and tries < 10:
                                    b[zero_mask] = _sample_uniform(zero_mask.sum(), lo, hi, rng)
                                    zero_mask = (b == 0)
                                    tries += 1
                                # if still zero, set tiny eps
                                b[zero_mask] = eps
                                a = _sample_uniform(to_make, a_lo, a_hi, rng)
                                # compute below
                                res, valid, error = _compute_op(op, a, b)
                                # assemble df etc.
                                df = pd.DataFrame({
                                    "a": a,
                                    "b": b,
                                    "op": [op] * to_make,
                                    "result": res,
                                    "valid": valid,
                                    # "error": error,
                                })
                                df.to_csv(outfile, mode="a", header=False, index=False)
                                produced += to_make
                                total_written += to_make
                                pbar.update(to_make)
                                continue
                            else:
                                # zero not in range -> fine
                                pass
                        # otherwise use straightforward uniform sampling
                    a = _sample_uniform(to_make, a_lo, a_hi, rng)
                    # If b was not already created (division special-case above), create normally:
                    if 'b' not in locals() or (isinstance(b, np.ndarray) and b.shape[0] != to_make):
                        b = _sample_uniform(to_make, b_lo, b_hi, rng)

                a = np.floor(a).astype(np.float64)
                b = np.floor(b).astype(np.float64)
                # Compute results (ignore internal `error` array; we don't write it)
                res, valid, _ = _compute_op(op, a, b if not _op_is_unary(op) else None)

                df = pd.DataFrame({
                    "a": a,
                    "b": (b if not _op_is_unary(op) else ["" for _ in range(to_make)]),
                    "op": [op] * to_make,
                    "result": res,
                    "valid": valid,
                })

                # Append to CSV
                df.to_csv(outfile, mode="a", header=False, index=False)

                produced += to_make
                total_written += to_make
                pbar.update(to_make)

                # cleanup local b if set in division handling
                if 'b' in locals() and isinstance(b, np.ndarray):
                    del b

                # Stop if we reached global n in non-balanced mode
                if counts is None and total_written >= n:
                    break

            # break early if reached global n
            if counts is None and total_written >= n:
                break

    finally:
        pbar.close()

    print(f"Written approx {total_written} rows to {outfile}")

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Massive arithmetic dataset generator")
    p.add_argument("--out", "-o", required=True, help="Output CSV file path")
    p.add_argument("--n", type=int, default=1000000, help="Total number of rows to generate")
    p.add_argument("--min", type=float, default=-10.0, dest="lo", help="Minimum value for sampling")
    p.add_argument("--max", type=float, default=10.0, dest="hi", help="Maximum value for sampling")
    p.add_argument("--ops", type=str, default=",".join(ALL_OPS),
                   help=f"Comma-separated ops to include. Available: {','.join(ALL_OPS)}")
    p.add_argument("--chunk", type=int, default=100000, help="Rows per write chunk")
    p.add_argument("--no-domain-sensitive", action="store_true", help="Disable domain-sensitive sampling")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    # Balanced generation is the default. Provide a flag to disable it.
    p.add_argument("--no-balanced", action="store_true", help="Disable balancing across operations (default: balanced)")
    args = p.parse_args(argv)
    args.ops = [op.strip() for op in args.ops.split(",") if op.strip()]
    # If the user passed --no-balanced, set balanced False; otherwise default to True
    args.balanced = not getattr(args, "no_balanced", False)
    return args

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    generate_dataset(outfile=args.out,
                     n=args.n,
                     lo=args.lo,
                     hi=args.hi,
                     ops=args.ops,
                     chunk=args.chunk,
                     domain_sensitive=not args.no_domain_sensitive,
                     seed=args.seed,
                     balanced=args.balanced)