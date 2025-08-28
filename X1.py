import os, time, math, argparse, random
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pulp as pl

ALPHA = 0.95
W_MIN, W_MAX = 0.005, 0.15
K_MIN, K_MAX = 10, 20

# GA population size
GA_POP   = 80
# Number of elites
GA_ELITE = 8
# Mutation probability (swap/perturb)
GA_MUT_P = 0.20
# Noise strength for perturbation
GA_MUT_W = 0.15
# Logging interval in seconds
REFRESH_EVERY = 5.0

SCENARIO_LIKE = {"scenario","scenarios","scene","scen","id","index","日期","时间","date","time"}

def set_seed(sd: int):
    """Set NumPy and Python random seeds for reproducibility."""
    np.random.seed(sd)
    random.seed(sd)

def load_returns(path: str, n_scenes: int) -> np.ndarray:
    """Robustly load CSV/Excel; convert prices to returns if detected; fill NaN by column mean; take the first n_scenes rows."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".xls", ".xlsx"):
            df = pd.read_excel(path)
        else:
            try:
                df = pd.read_csv(path)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(path, encoding="gbk")
                except Exception as e:
                    raise RuntimeError(f"Unable to read file：{path}（Tried utf-8 / gbk）") from e
    except Exception as e:
        raise e

    def looks_like_scenario(col: str) -> bool:
        c = (col or "").strip().lower()
        return any(x in c for x in SCENARIO_LIKE)

    num_cols = [c for c in df.columns if (pd.api.types.is_numeric_dtype(df[c]) and not looks_like_scenario(c))]
    if not num_cols:
        raise ValueError("文件中未找到数值列")
    df = df[num_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    med_med = df.abs().median(numeric_only=True).median()
    if pd.notna(med_med) and float(med_med) > 10:
        df = df.pct_change().iloc[1:].copy()

    df = df.fillna(df.mean(numeric_only=True))

    if n_scenes and n_scenes > 0:
        df = df.iloc[:n_scenes].copy()

    R = df.to_numpy(dtype=float)
    return R

def choose_assets(R: np.ndarray, n_assets: int) -> np.ndarray:
    """Select the top-variance assets (n_assets columns) to stabilize reproducibility."""
    if n_assets and n_assets < R.shape[1]:
        vars_ = np.nanvar(R, axis=0)
        idx = np.argsort(vars_)[::-1][:n_assets]
        R = R[:, idx]
    return R

def cvar_discrete_from_losses(losses: np.ndarray, alpha: float) -> float:
    """Discrete CVaR as the mean of the worst (1-α) tail; `losses` is the loss array."""
    S = len(losses)
    m = int(math.ceil((1.0 - alpha) * S))
    m = max(1, min(m, S))
    part = np.partition(losses, -m)[-m:]
    return float(np.mean(part))

def cvar_of_w(R: np.ndarray, w: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Compute discrete CVaR and mean for a given weight vector using the global ALPHA."""
    p = R @ w
    loss = -p
    c = cvar_discrete_from_losses(loss, alpha)
    mu = float(np.mean(p))
    return c, mu

def project_box_simplex(v: np.ndarray, L: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Project a vector onto the box-constrained simplex: sum(x)=1, W_MIN<=x_i<=W_MAX on active support."""
    lo = float(np.min(v - U))
    hi = float(np.max(v - L))
    for _ in range(80):
        tau = 0.5 * (lo + hi)
        x = np.clip(v - tau, L, U)
        s = float(np.sum(x))
        if s > 1.0:
            lo = tau
        else:
            hi = tau
        if abs(hi - lo) < 1e-12:
            break
    tau = 0.5 * (lo + hi)
    x = np.clip(v - tau, L, U)
    return x

def build_cbc_solver(milp_time: int, gap: float, msg: bool=False):
    """Return a configured CBC solver; supports multiple PuLP argument names across versions."""
    try:
        return pl.PULP_CBC_CMD(timeLimit=milp_time, fracGap=gap, msg=msg)
    except TypeError:
        pass
    try:
        return pl.PULP_CBC_CMD(timeLimit=milp_time, gapRel=gap, msg=msg)
    except TypeError:
        pass
    try:
        return pl.PULP_CBC_CMD(maxSeconds=milp_time, gapRel=gap, msg=msg)
    except TypeError:
        pass
    try:
        return pl.PULP_CBC_CMD(msg=msg, options=[f"-sec {milp_time}", f"-ratio {gap}"])
    except Exception as e:
        raise RuntimeError("need PuLP/CBC：conda install -c conda-forge pulp coincbc") from e

def milp_min_cvar(R: np.ndarray, alpha: float,
                  w_min: float, w_max: float,
                  k_min: int, k_max: int,
                  milp_time: int = 900, gap: float = 1e-4) -> Tuple[np.ndarray, Dict]:
    """Build and solve the Rockafellar–Uryasev min-CVaR MILP with sum-to-one, box, and cardinality constraints."""
    S, N = R.shape
    prob = pl.LpProblem("min_CVaR", pl.LpMinimize)

    w = [pl.LpVariable(f"w_{i}", lowBound=0) for i in range(N)]
    y = [pl.LpVariable(f"y_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(N)]
    t = pl.LpVariable("t", lowBound=None)
    u = [pl.LpVariable(f"u_{s}", lowBound=0) for s in range(S)]

    coef = 1.0 / ((1.0 - alpha) * S)
    prob += t + coef * pl.lpSum(u)

    prob += pl.lpSum(w) == 1.0
    prob += pl.lpSum(y) >= int(k_min)
    prob += pl.lpSum(y) <= int(k_max)
    for i in range(N):
        prob += w[i] >= float(w_min) * y[i]
        prob += w[i] <= float(w_max) * y[i]
    for s in range(S):
        rsw = pl.lpDot(list(R[s, :]), w)
        prob += u[s] >= -rsw - t

    solver = build_cbc_solver(milp_time, gap, msg=False)
    t0 = time.time()
    prob.solve(solver)
    elapsed = time.time() - t0

    status = pl.LpStatus[prob.status]
    w_star = np.array([float(v.value()) for v in w])
    c_star, mu_star = cvar_of_w(R, w_star, alpha)

    info = {
        "status": status,
        "solve_time": elapsed,
        "MILP_CVaR": c_star,
        "MILP_Mean": mu_star,
    }
    return w_star, info

# GA: min-CVaR (same feasible region as MILP)
def repair(w: np.ndarray, k_min: int=K_MIN, k_max: int=K_MAX,
           w_min: float=W_MIN, w_max: float=W_MAX) -> np.ndarray:
    """Force feasibility: K in [K_MIN, K_MAX], W within [W_MIN, W_MAX] on the active set, and sum-to-one."""
    N = len(w)
    w = np.maximum(w, 0.0)
    idx = np.where(w > 1e-12)[0]

    if len(idx) == 0:
        k = np.random.randint(k_min, k_max + 1)
        idx = np.random.choice(N, k, replace=False)
    elif len(idx) < k_min:
        need = k_min - len(idx)
        cand = np.setdiff1d(np.arange(N), idx, assume_unique=False)
        add = cand[:need] if len(cand) >= need else np.random.choice(cand, need, replace=True)
        idx = np.concatenate([idx, add])
    elif len(idx) > k_max:
        idx = np.random.choice(idx, k_max, replace=False)

    v = np.zeros(N)
    v[idx] = w[idx] if np.sum(w[idx]) > 0 else np.random.rand(len(idx))
    y = np.zeros(N)
    y[idx] = 1.0
    L = w_min * y
    U = w_max * y
    return project_box_simplex(v, L, U)

def crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two-parent linear crossover with Beta(2,2) mixing."""
    lam = np.random.beta(2, 2)
    c = lam * a + (1 - lam) * b
    return c

def mutate(x: np.ndarray, p_swap: float=GA_MUT_P, noise: float=GA_MUT_W) -> np.ndarray:
    """Sparse swap and in-support Gaussian perturbation mutation."""
    z = x.copy()
    if np.random.rand() < p_swap:
        nz = np.where(z > 1e-12)[0]
        if len(nz) >= 2:
            i, j = np.random.choice(nz, 2, replace=False)
            z[i], z[j] = z[j], z[i]
    nz = np.where(z > 1e-12)[0]
    if len(nz) > 0:
        z[nz] = z[nz] * (1.0 + np.random.normal(0.0, noise, size=len(nz)))
        z = np.maximum(z, 0.0)
    return z

def fit(R: np.ndarray, w: np.ndarray, alpha: float) -> float:
    """Fitness function returning CVaR for a weight vector."""
    c, _ = cvar_of_w(R, w, alpha)
    return c

def ga_min_cvar(R: np.ndarray, alpha: float,
                time_limit: int, seeds: int=7) -> Dict:
    """Genetic algorithm to minimize CVaR under the same feasible region as the MILP."""
    S, N = R.shape
    bests = []
    logs = []

    for sd in range(seeds):
        set_seed(20240000 + 97 * sd)
        pop = [repair(np.random.rand(N)) for _ in range(GA_POP)]
        f = np.array([fit(R, x, alpha) for x in pop])
        best = float(f.min()); bw = pop[int(f.argmin())].copy()

        t0 = time.time(); last = t0
        while time.time() - t0 < time_limit:
            ords = np.argsort(f)
            elites = [pop[i].copy() for i in ords[:GA_ELITE]]

            newp = elites.copy()
            while len(newp) < GA_POP:
                i, j = np.random.choice(GA_ELITE, 2, replace=True)
                c = repair(mutate(crossover(elites[i], elites[j])))
                newp.append(c)
            pop = newp

            f = np.array([fit(R, x, alpha) for x in pop])
            if f.min() < best:
                best = float(f.min())
                bw = pop[int(f.argmin())].copy()

            now = time.time()
            if now - last >= REFRESH_EVERY:
                logs.append((sd, now - t0, best))
                last = now

        bests.append((sd, best, bw))

    info = {
        "bests": bests,
        "logs": logs
    }
    return info

# Plotting (optional)
def plot_convergence(logs: List[Tuple[int,float,float]], milp_cvar: float, out: str):
    """Plot GA convergence versus time relative to the MILP CVaR baseline."""
    if not logs:
        return
    df = pd.DataFrame(logs, columns=["seed","t","best"])
    df["delta_pct"] = (df["best"] - milp_cvar) / abs(milp_cvar) * 100.0

    ts = sorted(df["t"].unique())
    meds, q1s, q3s = [], [], []
    for t in ts:
        x = df.loc[df["t"] == t, "delta_pct"].to_numpy()
        meds.append(np.median(x))
        q1s.append(np.quantile(x, 0.25))
        q3s.append(np.quantile(x, 0.75))

    plt.figure(figsize=(6.0,4.0), dpi=140)
    plt.plot(ts, meds, "-o", ms=3, color="#2563eb", label="GA median ΔCVaR(t)")
    plt.fill_between(ts, q1s, q3s, color="#60a5fa", alpha=0.25, label="IQR")
    plt.axhline(0, ls="--", color="#9ca3af", label="MILP level")
    plt.legend(frameon=False)
    plt.xlabel("Seconds"); plt.ylabel("ΔCVaR / MILP (%)")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()

def plot_scatter(row: Dict, out: str, title: str="X1: min-CVaR baseline"):
    """Scatter plot of (CVaR, mean) for MILP and GA results."""
    plt.figure(figsize=(5.2,4.2), dpi=140)
    plt.scatter([row["MILP_CVaR"]], [row["MILP_Mean"]], c="#111827", s=70, label="MILP")
    if "GA_CVaR_mean" in row and "GA_Mean_mean" in row:
        plt.errorbar(row["GA_CVaR_mean"], row["GA_Mean_mean"],
                     xerr=[[row.get("GA_CVaR_mean") - row.get("GA_CVaR_q1", row["GA_CVaR_mean"])],
                           [row.get("GA_CVaR_q3", row["GA_CVaR_mean"]) - row.get("GA_CVaR_mean")]],
                     yerr=[[row.get("GA_Mean_mean") - row.get("GA_Mean_q1", row["GA_Mean_mean"])],
                           [row.get("GA_Mean_q3", row["GA_Mean_mean"]) - row.get("GA_Mean_mean")]],
                     fmt="s", ms=7, color="#2563eb", ecolor="#2563eb", capsize=3, label="GA")
    plt.xlabel("CVaR"); plt.ylabel("Mean")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()

# Main pipeline
def run_x1(file: str, scenes: int, assets: int,
           milp_time: int, ga_time: int, seeds: int,
           outdir: str):
    """Run the X1 baseline experiment: load data, select assets, solve MILP, run GA seeds, and save outputs."""
    t0 = time.time()

    R = load_returns(file, scenes)
    R = choose_assets(R, assets)

    w_star, info = milp_min_cvar(R, ALPHA, W_MIN, W_MAX, K_MIN, K_MAX, milp_time=milp_time)
    row = dict(info)

    ga = ga_min_cvar(R, ALPHA, time_limit=ga_time, seeds=seeds)
    bests = ga["bests"]
    logs = ga["logs"]

    c_list = [b[1] for b in bests]
    w_list = [b[2] for b in bests]
    mus = [np.mean(R @ w) for w in w_list]

    row["GA_CVaR_mean"] = float(np.mean(c_list))
    row["GA_CVaR_q1"]   = float(np.quantile(c_list, 0.25))
    row["GA_CVaR_q3"]   = float(np.quantile(c_list, 0.75))

    row["GA_Mean_mean"] = float(np.mean(mus))
    row["GA_Mean_q1"]   = float(np.quantile(mus, 0.25))
    row["GA_Mean_q3"]   = float(np.quantile(mus, 0.75))

    row["Delta_CVaR_pct"] = (row["GA_CVaR_mean"] - row["MILP_CVaR"]) / abs(row["MILP_CVaR"]) * 100.0
    row["total_time"] = time.time() - t0

    os.makedirs(outdir, exist_ok=True)

    df_row = pd.DataFrame([row])
    csv_path = os.path.join(outdir, "x1_summary.csv")
    df_row.to_csv(csv_path, index=False)

    if logs:
        log_df = pd.DataFrame(logs, columns=["seed","t","best"])
        log_df.to_csv(os.path.join(outdir, "x1_ga_logs.csv"), index=False)
        plot_convergence(logs, row["MILP_CVaR"], os.path.join(outdir, "x1_convergence.png"))

    plot_scatter(row, os.path.join(outdir, "x1_scatter.png"))

def parse_args():
    """Parse command-line arguments for X1."""
    ap = argparse.ArgumentParser(description="X1 baseline: min-CVaR (MILP) and GA convergence.")
    ap.add_argument("--file", required=True, help="收益/价格 CSV 或 Excel；按 --scenes 截断")
    ap.add_argument("--scenes", type=int, default=5000)
    ap.add_argument("--assets", type=int, default=24)
    ap.add_argument("--ga_time", type=int, default=60)
    ap.add_argument("--milp_time", type=int, default=900)
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--outdir", type=str, default="results_x1")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    run_x1(args.file, args.scenes, args.assets, args.milp_time, args.ga_time, args.seeds, args.outdir)
