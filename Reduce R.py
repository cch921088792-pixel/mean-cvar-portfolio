import argparse, os, time
import numpy as np
import pandas as pd


def load_R(path: str) -> pd.DataFrame:
    """
    Load a scenario–asset return matrix R from CSV or Excel.
    The first column is treated as index; non-numeric values are coerced to NaN.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"):
        R = pd.read_excel(path, index_col=0)
    else:
        R = pd.read_csv(path, index_col=0)
    R = R.apply(pd.to_numeric, errors="coerce")
    return R


def reduce_uniform(R: pd.DataFrame, n_out: int, seed: int) -> pd.DataFrame:
    """
    Uniform subsampling without replacement.
    """
    if n_out >= len(R):
        return R.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(R), size=n_out, replace=False)
    idx.sort()
    return R.iloc[idx].copy()


def reduce_stratified(R: pd.DataFrame, n_out: int, seed: int, bins: int = 200) -> pd.DataFrame:
    """
    Stratified sampling by the equal-weight portfolio return.
    Steps:
      1) Compute r_bar = mean across assets for each scenario (row).
      2) Cut r_bar into `bins` quantile buckets (unique boundaries).
      3) Allocate samples proportionally across buckets (≥1 per bucket),
         then sample within each bucket without replacement.
    This keeps tail coverage while preserving distributional structure.
    """
    N = len(R)
    if n_out >= N:
        return R.copy()

    rng = np.random.default_rng(seed)

    r_bar = R.mean(axis=1).values

    qs = np.unique(np.quantile(r_bar, np.linspace(0, 1, bins + 1)))

    layer_ids = []
    for i in range(len(qs) - 1):
        lo, hi = qs[i], qs[i + 1]
        if i == len(qs) - 2:
            ids = np.where((r_bar >= lo) & (r_bar <= hi))[0]
        else:
            ids = np.where((r_bar >= lo) & (r_bar < hi))[0]
        if ids.size > 0:
            layer_ids.append(ids)

    sizes = np.array([len(a) for a in layer_ids], dtype=float)
    prop = sizes / sizes.sum()
    alloc = np.floor(prop * n_out).astype(int)

    alloc = np.maximum(alloc, 1)

    diff = n_out - alloc.sum()
    if diff > 0:
        order = np.argsort(-sizes)
        i = 0
        while diff > 0:
            alloc[order[i % len(order)]] += 1
            i += 1
            diff -= 1
    elif diff < 0:
        order = np.argsort(-alloc)
        i = 0
        while diff < 0 and np.any(alloc > 1):
            j = order[i % len(order)]
            if alloc[j] > 1:
                alloc[j] -= 1
                diff += 1
            i += 1

    picked = []
    for ids, k in zip(layer_ids, alloc):
        if k >= len(ids):
            picked.append(ids)
        else:
            sel = rng.choice(ids, size=k, replace=False)
            picked.append(sel)

    idx = np.concatenate(picked)
    idx.sort()
    out = R.iloc[idx].copy()
    return out


def reduce_block(R: pd.DataFrame, n_out: int, seed: int, block: int = 20) -> pd.DataFrame:
    """
    Block bootstrap along the time axis.
    Randomly choose starting indices and take contiguous segments of length `block`,
    concatenating segments until `n_out` rows are collected.
    Use only when rows of R are time-ordered.
    """
    N = len(R)
    if n_out >= N:
        return R.copy()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max(1, N - block + 1), size=int(np.ceil(n_out / block)))
    rows = []
    for s in starts:
        e = min(N, s + block)
        rows.extend(range(s, e))
        if len(rows) >= n_out:
            break
    rows = rows[:n_out]
    rows.sort()
    return R.iloc[rows].copy()


def main():
    ap = argparse.ArgumentParser(description="Reduce R scenarios to a target size while preserving structure.")
    ap.add_argument("--infile", type=str, default="R.csv", help="input R file (csv/xlsx, index_col=0)")
    ap.add_argument("--outfile", type=str, default="", help="output csv; auto-named if empty")
    ap.add_argument("--method", type=str, default="stratified",
                    choices=["uniform", "stratified", "block"],
                    help="reduction method")
    ap.add_argument("--n_out", type=int, default=10000, help="target number of scenarios")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--bins", type=int, default=200, help="quantile bins for stratified")
    ap.add_argument("--block", type=int, default=20, help="block size for block bootstrap")
    args = ap.parse_args()

    t0 = time.time()
    R = load_R(args.infile)
    N, M = R.shape
    print(f"[info] Loaded R: {N} scenarios × {M} assets from '{args.infile}'")
    print(f"[info] method={args.method}, n_out={args.n_out}, seed={args.seed}")

    if args.n_out <= 0:
        raise ValueError("n_out must be positive.")
    if args.n_out > N:
        print("[warn] n_out > N; nothing to reduce. Will just copy input.")

    if args.method == "uniform":
        R_out = reduce_uniform(R, args.n_out, args.seed)
    elif args.method == "stratified":
        R_out = reduce_stratified(R, args.n_out, args.seed, bins=args.bins)
    else:
        R_out = reduce_block(R, args.n_out, args.seed, block=args.block)

    if args.outfile:
        out_path = args.outfile
    else:
        stem = os.path.splitext(os.path.basename(args.infile))[0]
        suffix = f"{args.method}_{args.n_out}"
        out_path = f"{stem}_{suffix}.csv"

    R_out.to_csv(out_path)

    mu_full = R.mean(axis=0)
    mu_out  = R_out.mean(axis=0)

    stat_path = os.path.splitext(out_path)[0] + "_summary.csv"
    pd.DataFrame({
        "mu_full": mu_full,
        "mu_reduced": mu_out,
        "mu_diff": (mu_out - mu_full)
    }).to_csv(stat_path)

    elapsed = time.time() - t0
    print(f"[ok] Reduced -> {len(R_out)}×{M}, saved to '{out_path}'")
    print(f"[ok] Summary saved to '{stat_path}'")
    print(f"[time] {elapsed:.2f}s")


if __name__ == "__main__":
    main()
