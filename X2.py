import os, time, math, argparse, random
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.95
K_MIN, K_MAX = 10, 20
W_MIN, W_MAX = 0.005, 0.15

GA_POP, GA_ELITE, GA_MUT_P, GA_MUT_W = 80, 8, 0.20, 0.15
REFRESH_EVERY = 5.0
PEN_CVAR = 3e5

SCENARIO_LIKE = {"scenario","scenarios","scene","scen","id","index","日期","时间","date","time"}

# Utilities
def set_seed(sd:int):
    """Set both NumPy and Python random seeds."""
    np.random.seed(sd)
    random.seed(sd)

def load_returns(path:str, n_scenes:int)->np.ndarray:
    """
    Load returns or price-like data, attempt robust CSV/Excel reading, convert prices to returns if detected,
    fill NaNs by column means, and truncate to the first n_scenes rows.
    """
    ext = os.path.splitext(path)[1].lower()
    df = None
    if ext in (".xlsx",".xls"):
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception:
            df = pd.read_excel(path)
    else:
        for enc in ("utf-8-sig","utf-8","gbk","cp936"):
            try:
                df = pd.read_csv(path, header=0, low_memory=False, encoding=enc)
                break
            except Exception:
                pass
        if df is None:
            raise RuntimeError(f"无法读取文件：{path}")
    keep = [c for c in df.columns if str(c).strip().lower() not in SCENARIO_LIKE]
    df = df[keep]
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        raise ValueError("未找到数值列")
    df = df[num].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    med_med = df.abs().median(numeric_only=True).median()
    if pd.notna(med_med) and float(med_med) > 0.05:
        df = df.pct_change().dropna()
    if df.isna().any().any():
        df = df.fillna(df.mean(numeric_only=True))
    return df.to_numpy(dtype=float)[:n_scenes,:]

def choose_assets(R:np.ndarray, n_assets:int)->Tuple[np.ndarray,np.ndarray]:
    """
    Select the top-variance assets to stabilize reproducibility.
    Returns the reduced matrix and selected column indices.
    """
    idx = np.argsort(R.std(axis=0))[::-1][:n_assets]
    return R[:,idx], idx

def cvar_discrete_from_losses(losses: np.ndarray, alpha: float = ALPHA) -> float:
    """
    Discrete CVaR defined as the mean of the worst (1 − alpha) tail of losses.
    """
    losses = np.asarray(losses, float)
    S = losses.size
    m = max(1, int(np.ceil((1.0 - alpha) * S)))
    tail = np.partition(losses, S - m)[S - m:]
    return float(np.mean(tail))

def cvar_of_w(R:np.ndarray, w:np.ndarray, alpha:float=ALPHA)->float:
    """
    Compute discrete CVaR for a given weight vector under returns matrix R.
    """
    return cvar_discrete_from_losses(-R @ w, alpha)

def project_box_simplex(v, L, U, iters=60, tol=1e-12):
    """
    Projection onto the box-constrained simplex:
    solve min ||x − v||^2 subject to sum(x)=1 and L ≤ x ≤ U via bisection on the Lagrange multiplier.
    """
    v = np.asarray(v,float); L = np.asarray(L,float); U = np.asarray(U,float)
    L = np.minimum(L,U)
    lo = np.min(v - U); hi = np.max(v - L)
    for _ in range(iters):
        tau = 0.5*(lo+hi)
        x = np.minimum(np.maximum(v - tau, L), U)
        if x.sum() > 1.0:
            lo = tau
        else:
            hi = tau
        if hi - lo < tol:
            break
    tau = 0.5*(lo+hi)
    x = np.minimum(np.maximum(v - tau, L), U)
    return x

def build_cbc_solver(milp_time: int, gap: float = 1e-4, msg: bool = False):
    """
    Create a CBC solver with compatible argument names across PuLP versions.
    """
    import pulp as pl
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
    return pl.PULP_CBC_CMD(msg=msg, options=[f"-sec {milp_time}", f"-ratio {gap}"])

# MILP: maximize mean s.t. CVaR ≤ τ
def milp_max_mean_cvar_leq_tau(R:np.ndarray, tau:float, timelimit:int=600, msg:bool=False)->Dict:
    """
    Rockafellar–Uryasev linearized MILP for maximizing mean subject to CVaR ≤ τ
    under sum-to-one, box bounds, and cardinality constraints.
    """
    try:
        import pulp as pl
    except Exception as e:
        raise RuntimeError("Need PuLP/CBC：conda install -c conda-forge pulp coincbc") from e

    S,N = R.shape
    mu = R.mean(axis=0)

    prob = pl.LpProblem("max_mean_given_CVaR", pl.LpMaximize)
    w = pl.LpVariable.dicts("w", range(N), lowBound=0, upBound=W_MAX, cat=pl.LpContinuous)
    y = pl.LpVariable.dicts("y", range(N), lowBound=0, upBound=1,   cat=pl.LpBinary)
    v = pl.LpVariable("VaR", lowBound=None, upBound=None, cat=pl.LpContinuous)
    u = pl.LpVariable.dicts("u", range(S), lowBound=0, upBound=None, cat=pl.LpContinuous)

    prob += pl.lpSum(mu[i]*w[i] for i in range(N))

    prob += pl.lpSum(w[i] for i in range(N)) == 1.0
    prob += pl.lpSum(y[i] for i in range(N)) >= K_MIN
    prob += pl.lpSum(y[i] for i in range(N)) <= K_MAX
    for i in range(N):
        prob += w[i] <= W_MAX*y[i]
        prob += w[i] >= W_MIN*y[i]

    coef = 1.0/((1.0-ALPHA)*S)
    for s in range(S):
        prob += u[s] >= -pl.lpSum(R[s,i]*w[i] for i in range(N)) - v
    prob += v + coef*pl.lpSum(u[s] for s in range(S)) <= tau

    t0=time.time()
    solver = build_cbc_solver(timelimit, gap=1e-4, msg=msg)
    status = prob.solve(solver)
    t1=time.time()
    st = pl.LpStatus.get(prob.status,"unknown")

    w_opt = np.array([w[i].value() for i in range(N)], float)
    return dict(status=st, time=t1-t0, w=w_opt,
                CVaR=cvar_of_w(R,w_opt), Mean=float(mu @ w_opt))

# GA: maximize mean s.t. CVaR ≤ τ (penalized)
def repair(w:np.ndarray)->np.ndarray:
    """
    Force feasibility: K in [K_MIN, K_MAX], weights within [W_MIN, W_MAX] on active support, and sum-to-one.
    """
    N=len(w); w=np.maximum(w,0.0)
    idx=np.where(w>1e-12)[0]
    if idx.size==0:
        k=np.random.randint(K_MIN,K_MAX+1); idx=np.random.choice(N,k,replace=False)
    elif idx.size<K_MIN:
        add=np.setdiff1d(np.arange(N),idx)
        idx=np.concatenate([idx, np.random.choice(add,K_MIN-idx.size,replace=False)])
    elif idx.size>K_MAX:
        idx=np.random.choice(idx,K_MAX,replace=False)
    v=np.zeros(N); v[idx]=w[idx] if w[idx].sum()>0 else np.random.rand(idx.size)
    y=np.zeros(N); y[idx]=1.0
    L=W_MIN*y; U=W_MAX*y
    return project_box_simplex(v,L,U)

def crossover(a,b):
    """
    Two-parent linear crossover with Beta(2,2) mixing.
    """
    lam=np.random.beta(2,2); return lam*a+(1-lam)*b

def mutate(w):
    """
    Sparse swap followed by in-support Gaussian perturbation.
    """
    w=w.copy()
    if np.random.rand()<GA_MUT_P:
        idx=np.where(w>1e-12)[0]; non=np.where(w<=1e-12)[0]
        if idx.size>0 and non.size>0:
            out=np.random.choice(idx); inn=np.random.choice(non); w[inn]=w[out]; w[out]=0.0
    idx=np.where(w>1e-12)[0]
    if idx.size>0:
        noise=np.random.normal(0,GA_MUT_W,size=idx.size); w[idx]=np.maximum(w[idx]*(1+noise),0.0)
    return w

def ga_max_mean_under_cvar(R:np.ndarray, tau:float, time_limit:int=45, seeds:int=7)->Tuple[List[Dict],List[Dict]]:
    """
    GA with penalized objective: minimize Φ(w) = PEN_CVAR·max(0, CVaR(w) − τ) − mean(w),
    while always repairing individuals to the hard feasible region.
    Returns per-seed finals and time logs.
    """
    logs, finals = [], []
    mu = R.mean(axis=0)

    def fitness(x):
        c = cvar_of_w(R,x)
        m = float(mu @ x)
        return PEN_CVAR*max(0.0, c - tau) - m, c, m

    for sd in range(seeds):
        set_seed(20240000 + 113*sd)
        pop = [repair(np.random.rand(R.shape[1])) for _ in range(GA_POP)]
        vals = [fitness(x) for x in pop]
        f = np.array([v[0] for v in vals])
        best_idx = int(f.argmin()); best = vals[best_idx]; bw = pop[best_idx].copy()

        t0=time.time(); last=t0
        while time.time()-t0 < time_limit:
            ords=np.argsort(f); elites=[pop[i].copy() for i in ords[:GA_ELITE]]
            newp = elites.copy()
            while len(newp) < GA_POP:
                i,j = np.random.choice(GA_ELITE,2,replace=True)
                c = repair(mutate(crossover(elites[i], elites[j]))); newp.append(c)
            pop = newp
            vals = [fitness(x) for x in pop]; f = np.array([v[0] for v in vals])
            if f.min() < best[0]:
                best = vals[int(f.argmin())]; bw = pop[int(f.argmin())].copy()
            now=time.time()
            if now-last >= REFRESH_EVERY or now-t0 >= time_limit-1e-3:
                logs.append(dict(seed=sd, t=now-t0, best_cvar=best[1]))
                last=now

        fc, cvar, mean = fitness(bw)
        finals.append(dict(seed=sd, CVaR=cvar, Mean=mean, Time=time.time()-t0,
                           feasible=1 if cvar <= tau*(1+1e-8) else 0))
    return finals, logs

# τ selection
def sample_tau_from_cvar_quantiles(R:np.ndarray, qs:List[float], n_samples:int=2000)->List[float]:
    """
    Sample random feasible weights under the same hard constraints as MILP/GA,
    compute CVaR for each, and return quantiles at levels `qs`.
    """
    N = R.shape[1]
    cv = []
    for _ in range(n_samples):
        k = np.random.randint(K_MIN, K_MAX+1)
        idx = np.random.choice(N, k, replace=False)
        v = np.zeros(N); v[idx] = np.random.rand(k)
        y = np.zeros(N); y[idx] = 1.0
        L = W_MIN*y; U=W_MAX*y
        w = project_box_simplex(v, L, U)
        cv.append(cvar_of_w(R, w))
    cv = np.array(cv, float)
    return [float(np.quantile(cv, q)) for q in qs]

# Plotting
def plot_convergence(log_df:pd.DataFrame, milp_cvar:float, tau:float, out_png:str):
    """
    Plot GA convergence in ΔCVaR(t) relative to MILP baseline level.
    """
    if log_df.empty:
        return
    ts = np.linspace(1, max(1.0, float(log_df["t"].max())), 40)
    meds, q1s, q3s = [], [], []
    denom = max(1e-12, abs(milp_cvar))
    for t in ts:
        cur = log_df[log_df["t"]<=t].groupby("seed")["best_cvar"].min().values
        if cur.size==0:
            meds.append(np.nan); q1s.append(np.nan); q3s.append(np.nan)
        else:
            rel = (cur - milp_cvar) / denom * 100.0
            meds.append(float(np.nanmedian(rel)))
            q1s.append(float(np.nanquantile(rel, 0.25)))
            q3s.append(float(np.nanquantile(rel, 0.75)))
    plt.figure(figsize=(8.6,5.2))
    plt.plot(ts, meds, "-o", ms=3, color="#2563eb", label="GA median ΔCVaR(t)")
    plt.fill_between(ts, q1s, q3s, color="#60a5fa", alpha=0.25, label="IQR")
    plt.axhline(0, ls="--", color="#9ca3af", label="MILP level")
    plt.xlabel("Seconds"); plt.ylabel("ΔCVaR / MILP (%)")
    plt.title(f"Convergence @ τ={tau:.6g}")
    plt.grid(ls="--", alpha=0.25); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out_png, dpi=240); plt.close()

def plot_frontier(summary_df:pd.DataFrame, out_png:str, title="Mini ε-frontier (MILP vs GA)"):
    """
    Plot (CVaR, Mean) points for MILP and GA with GA error bars and τ annotations.
    """
    if summary_df.empty:
        return
    plt.figure(figsize=(8.6,5.2))
    plt.scatter(summary_df["MILP_CVaR"], summary_df["MILP_Mean"], c="#111827", s=60, label="MILP")
    plt.errorbar(summary_df["GA_CVaR_mean"], summary_df["GA_Mean_mean"],
                 xerr=summary_df["GA_CVaR_ci"], yerr=summary_df["GA_Mean_ci"],
                 fmt="s", ms=7, color="#2563eb", ecolor="#2563eb", capsize=3, label="GA")
    for _,r in summary_df.iterrows():
        plt.text(r["MILP_CVaR"], r["MILP_Mean"], f"τ={r['tau']:.6g}", fontsize=9, ha="left", va="bottom")
    plt.xlabel("CVaR (lower is better)"); plt.ylabel("Mean return"); plt.title(title)
    plt.grid(ls="--", alpha=0.25); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out_png, dpi=240); plt.close()

def run_x2(file:str, n_assets:int, n_scenes:int, tau_list:List[float], tau_qs:List[float],
           seeds:int, ga_time:int, milp_time:int, outdir:str, n_samples:int):
    """
    End-to-end X2 pipeline: load data, select assets, obtain τ list, solve MILP, run GA seeds, and save outputs.
    """
    os.makedirs(outdir, exist_ok=True)
    Rfull = load_returns(file, n_scenes)
    R, idx = choose_assets(Rfull, n_assets)

    if (tau_list is None or len(tau_list)==0):
        qs = tau_qs if tau_qs else [0.60,0.75,0.90]
        taus = sample_tau_from_cvar_quantiles(R, qs, n_samples=n_samples)
    else:
        taus = list(tau_list)

    rows = []
    raw_rows = []

    for k, tau in enumerate(taus):
        milp = milp_max_mean_cvar_leq_tau(R, tau, timelimit=milp_time, msg=False)
        print(f"[MILP τ={tau:.6g}] CVaR={milp['CVaR']:.6f}, Mean={milp['Mean']:.6f}, "
              f"time={milp['time']:.1f}s, status={milp['status']}")

        ga_finals, ga_logs = ga_max_mean_under_cvar(R, tau, time_limit=ga_time, seeds=seeds)
        logs_df = pd.DataFrame(ga_logs)
        plot_convergence(logs_df, milp["CVaR"], tau, os.path.join(outdir, f"x2_convergence_{k+1}.png"))

        ga_df = pd.DataFrame(ga_finals)
        n = len(ga_df)
        g_cvar_mean = float(ga_df["CVaR"].mean())
        g_mean_mean = float(ga_df["Mean"].mean"])
        g_cvar_ci = float(1.96*ga_df["CVaR"].std(ddof=1)/math.sqrt(n)) if n>1 else 0.0
        g_mean_ci = float(1.96*ga_df["Mean"].std(ddof=1)/math.sqrt(n)) if n>1 else 0.0

        delta_cvar_pct = float((g_cvar_mean - milp["CVaR"]) / max(1e-12,abs(milp["CVaR"])) * 100.0)
        delta_mean_pct = float((g_mean_mean - milp["Mean"]) / (abs(milp["Mean"])+1e-12) * 100.0)

        row = dict(
            tau=float(tau),
            MILP_CVaR=float(milp["CVaR"]), MILP_Mean=float(milp["Mean"]),
            MILP_status=str(milp["status"]), MILP_time=float(milp["time"]),
            GA_CVaR_mean=g_cvar_mean, GA_Mean_mean=g_mean_mean,
            GA_CVaR_ci=g_cvar_ci, GA_Mean_ci=g_mean_ci,
            GA_time_mean=float(ga_df["Time"].mean()),
            GA_feasible_rate=float((ga_df["feasible"]==1).mean()),
            Delta_CVaR_pct=delta_cvar_pct, Delta_Mean_pct=delta_mean_pct
        )
        rows.append(row)

        ga_df2 = ga_df.copy()
        ga_df2["tau"] = tau
        ga_df2["method"] = "GA"
        raw_rows.append(ga_df2)

    summary = pd.DataFrame(rows)
    raw = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
    summary.to_csv(os.path.join(outdir,"x2_summary.csv"), index=False)
    raw.to_csv(os.path.join(outdir,"x2_raw.csv"), index=False)
    plot_frontier(summary, os.path.join(outdir,"x2_frontier.png"))
    print(f"[SAVE] {os.path.join(outdir,'x2_summary.csv')}")
    print(f"[SAVE] {os.path.join(outdir,'x2_raw.csv')}")
    print("[OK] X2 done.")

def parse_args():
    """
    CLI for running the X2 pipeline.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Yield/Price CSV or Excel")
    ap.add_argument("--assets", type=int, default=48)
    ap.add_argument("--scenes", type=int, default=10000)
    ap.add_argument("--tau", type=str, default="", help='Comma-separated constants τ, e.g. "0.02,0.03,0.04"')
    ap.add_argument("--tau_q", type=str, default="0.60,0.75,0.90", help='Comma separated quantiles (taken from the feasible region CVaR distribution), e.g. "0.60,0.75,0.90"')
    ap.add_argument("--n_samples", type=int, default=2000, help="The number of random feasible weight samples used to estimate τ")
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--ga_time", type=int, default=45)
    ap.add_argument("--milp_time", type=int, default=600)
    ap.add_argument("--out", type=str, default="results_x2_48x10k")
    return ap.parse_args()

if __name__ == "__main__":
    a = parse_args()
    tau_list = [float(x) for x in a.tau.split(",") if x.strip()!=""]
    tau_qs   = [float(x) for x in a.tau_q.split(",") if x.strip()!=""]
    run_x2(a.file, a.assets, a.scenes, tau_list, tau_qs, a.seeds, a.ga_time, a.milp_time, a.out, a.n_samples)
