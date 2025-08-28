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
PEN_CVAR  = 3e5
PEN_COST  = 3e5
EPS_TRADE = 1e-9

SCENARIO_LIKE = {"scenario","scenarios","scene","scen","id","index","date","time","date","time"}

def set_seed(sd:int):
    np.random.seed(sd); random.seed(sd)

# Data loading
def _read_table(path:str)->pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    df = None
    if ext in (".xlsx",".xls"):
        try: df = pd.read_excel(path, engine="openpyxl")
        except Exception: df = pd.read_excel(path)
    else:
        for enc in ("utf-8-sig","utf-8","gbk","cp936"):
            try: df = pd.read_csv(path, header=0, low_memory=False, encoding=enc); break
            except Exception: pass
    if df is None: raise RuntimeError(f"无法读取文件：{path}")
    return df

def load_returns_df(path:str, n_scenes:int)->pd.DataFrame:
    df = _read_table(path)
    keep = [c for c in df.columns if str(c).strip().lower() not in SCENARIO_LIKE]
    df = df[keep]
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num: raise ValueError("未找到数值列")
    df = df[num].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    med_med = df.abs().median(numeric_only=True).median()
    if pd.notna(med_med) and float(med_med) > 0.05:
        df = df.pct_change().dropna()
    if df.isna().any().any():
        df = df.fillna(df.mean(numeric_only=True))
    return df.iloc[:n_scenes,:].copy()

def choose_assets(R:np.ndarray, cols:List[str], n_assets:int)->Tuple[np.ndarray,List[int],List[str]]:
    idx = np.argsort(R.std(axis=0))[::-1][:n_assets]
    return R[:,idx], list(idx), [cols[i] for i in idx]

def try_load_w0(w0_file:str, full_df:pd.DataFrame, sel_idx:List[int])->np.ndarray:
    N_sel = len(sel_idx)
    if not w0_file or not os.path.exists(w0_file):
        return np.zeros(N_sel, dtype=float)
    df = _read_table(w0_file)
    cols = list(full_df.columns)
    sel_cols = [cols[i] for i in sel_idx]
    inter = [c for c in sel_cols if c in df.columns]
    if len(inter) == len(sel_cols):
        row = df.iloc[0][sel_cols].astype(float).to_numpy()
        return row
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        vec = num.iloc[0].to_numpy(dtype=float)
        if vec.size == full_df.shape[1]:
            return vec[sel_idx]
        if vec.size == N_sel:
            return vec
    return np.zeros(N_sel, dtype=float)

# Risk and cost functions
def cvar_tail_mean(losses: np.ndarray, alpha: float = ALPHA) -> float:
    losses = np.asarray(losses, float); S = losses.size
    m = max(1, int(np.ceil((1.0 - alpha) * S)))
    tail = np.partition(losses, S - m)[S - m:]
    return float(np.mean(tail))

def cvar_of_w(R:np.ndarray, w:np.ndarray, alpha:float=ALPHA)->float:
    return cvar_tail_mean(-R @ w, alpha)

def cost_of_w(w:np.ndarray, w0:np.ndarray, c_lin:float, c_fix:float)->Tuple[float,float,int]:
    diff = np.abs(w - w0)
    turnover = float(np.sum(diff))
    trades = int(np.sum(diff > EPS_TRADE))
    cost = c_lin*turnover + c_fix*trades
    return cost, turnover, trades

# Projection: box + simplex
def project_box_simplex(v, L, U, iters=60, tol=1e-12):
    v = np.asarray(v,float); L = np.asarray(L,float); U = np.asarray(U,float)
    L = np.minimum(L,U)
    lo = np.min(v - U); hi = np.max(v - L)
    for _ in range(iters):
        tau = 0.5*(lo+hi)
        x = np.minimum(np.maximum(v - tau, L), U)
        if x.sum() > 1.0: lo = tau
        else: hi = tau
        if hi - lo < tol: break
    tau = 0.5*(lo+hi)
    x = np.minimum(np.maximum(v - tau, L), U)
    return x

# CBC solver construction (compatible)
def build_cbc_solver(milp_time: int, gap: float = 1e-4, msg: bool = False):
    import pulp as pl
    try: return pl.PULP_CBC_CMD(timeLimit=milp_time, fracGap=gap, msg=msg)
    except TypeError: pass
    try: return pl.PULP_CBC_CMD(timeLimit=milp_time, gapRel=gap, msg=msg)
    except TypeError: pass
    try: return pl.PULP_CBC_CMD(maxSeconds=milp_time, gapRel=gap, msg=msg)
    except TypeError: pass
    return pl.PULP_CBC_CMD(msg=msg, options=[f"-sec {milp_time}", f"-ratio {gap}"])

# MILP: max mean s.t. CVaR ≤ τ and cost ≤ B
def milp_max_mean_with_cost(R:np.ndarray, tau:float, w0:np.ndarray,
                            c_lin:float, c_fix:float, B:float,
                            timelimit:int=600, msg:bool=False)->Dict:
    try:
        import pulp as pl
    except Exception as e:
        raise RuntimeError("Need PuLP/CBC：conda install -c conda-forge pulp coincbc") from e

    S,N = R.shape
    mu = R.mean(axis=0)

    prob = pl.LpProblem("max_mean_cvar_cost", pl.LpMaximize)
    w = pl.LpVariable.dicts("w", range(N), lowBound=0, upBound=W_MAX, cat=pl.LpContinuous)
    y = pl.LpVariable.dicts("y", range(N), lowBound=0, upBound=1,   cat=pl.LpBinary)
    v = pl.LpVariable("VaR", lowBound=None, upBound=None, cat=pl.LpContinuous)
    u = pl.LpVariable.dicts("u", range(S), lowBound=0, upBound=None, cat=pl.LpContinuous)
    d = pl.LpVariable.dicts("d", range(N), lowBound=0, upBound=1.0,  cat=pl.LpContinuous)
    s = pl.LpVariable.dicts("s", range(N), lowBound=0, upBound=1.0,  cat=pl.LpBinary)

    prob += pl.lpSum(mu[i]*w[i] for i in range(N))

    prob += pl.lpSum(w[i] for i in range(N)) == 1.0
    prob += pl.lpSum(y[i] for i in range(N)) >= K_MIN
    prob += pl.lpSum(y[i] for i in range(N)) <= K_MAX
    for i in range(N):
        prob += w[i] <= W_MAX*y[i]
        prob += w[i] >= W_MIN*y[i]

    coef = 1.0/((1.0-ALPHA)*S)
    for s_idx in range(S):
        prob += u[s_idx] >= -pl.lpSum(R[s_idx,i]*w[i] for i in range(N)) - v
    prob += v + coef*pl.lpSum(u[s_idx] for s_idx in range(S)) <= tau

    for i in range(N):
        prob += d[i] >=  w[i] - float(w0[i])
        prob += d[i] >= -w[i] + float(w0[i])
        prob += d[i] <= s[i]
    prob += c_lin*pl.lpSum(d[i] for i in range(N)) + c_fix*pl.lpSum(s[i] for i in range(N)) <= B

    t0=time.time()
    solver = build_cbc_solver(timelimit, gap=1e-4, msg=msg)
    _ = prob.solve(solver)
    t1=time.time()
    st = pl.LpStatus.get(prob.status,"unknown")

    w_opt = np.array([w[i].value() for i in range(N)], float)
    cvar = cvar_of_w(R,w_opt); mean=float(mu @ w_opt)
    cost, turnover, trades = cost_of_w(w_opt, w0, c_lin, c_fix)
    return dict(status=st, time=t1-t0, w=w_opt, CVaR=cvar, Mean=mean,
                cost=cost, turnover=turnover, trades=trades)

# GA: same feasible region + cost budget (penalized)
def repair(w:np.ndarray)->np.ndarray:
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

def crossover(a,b): lam=np.random.beta(2,2); return lam*a+(1-lam)*b

def mutate(w):
    w=w.copy()
    if np.random.rand()<GA_MUT_P:
        idx=np.where(w>1e-12)[0]; non=np.where(w<=1e-12)[0]
        if idx.size>0 and non.size>0:
            out=np.random.choice(idx); inn=np.random.choice(non); w[inn]=w[out]; w[out]=0.0
    idx=np.where(w>1e-12)[0]
    if idx.size>0:
        noise=np.random.normal(0,GA_MUT_W,size=idx.size); w[idx]=np.maximum(w[idx]*(1+noise),0.0)
    return w

def ga_max_mean_with_cost(R:np.ndarray, tau:float, w0:np.ndarray,
                          c_lin:float, c_fix:float, B:float,
                          time_limit:int=45, seeds:int=7)->Tuple[List[Dict],List[Dict]]:
    logs, finals = [], []
    mu = R.mean(axis=0)
    def fitness(x):
        cvar = cvar_of_w(R,x)
        mean = float(mu @ x)
        cost, turnover, trades = cost_of_w(x, w0, c_lin, c_fix)
        pen = PEN_CVAR*max(0.0, cvar - tau) + PEN_COST*max(0.0, cost - B)
        return pen - mean, cvar, mean, cost, turnover, trades
    for sd in range(seeds):
        set_seed(20240000 + 211*sd)
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
                k = int(f.argmin()); best = vals[k]; bw = pop[k].copy()
            now=time.time()
            if now-last >= REFRESH_EVERY or now-t0 >= time_limit-1e-3:
                logs.append(dict(seed=sd, t=now-t0, best_cvar=best[1]))
                last=now
        _, cvar, mean, cost, turnover, trades = fitness(bw)
        finals.append(dict(seed=sd, CVaR=cvar, Mean=mean, cost=cost,
                           turnover=turnover, trades=trades, Time=time.time()-t0,
                           feasible=1 if (cvar<=tau*(1+1e-8) and cost<=B*(1+1e-8)) else 0))
    return finals, logs

# τ estimation (same as X2)
def sample_tau_from_cvar_quantiles(R:np.ndarray, qs:List[float], n_samples:int=2000)->List[float]:
    N = R.shape[1]
    cv = []
    for _ in range(n_samples):
        k = np.random.randint(K_MIN, K_MAX+1)
        idx = np.random.choice(N, k, replace=False)
        v = np.zeros(N); v[idx] = np.random.rand(k)
        y = np.zeros(N); y[idx] = 1.0
        L = W_MIN*y; U = W_MAX*y
        w = project_box_simplex(v, L, U)
        cv.append(cvar_of_w(R, w))
    cv = np.array(cv, float)
    return [float(np.quantile(cv, q)) for q in qs]

# Plotting
def plot_convergence(log_df:pd.DataFrame, milp_cvar:float, tau:float, out_png:str):
    if log_df.empty: return
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

def plot_frontier(summary_df:pd.DataFrame, out_png:str):
    if summary_df.empty: return
    plt.figure(figsize=(8.6,5.2))
    plt.scatter(summary_df["MILP_CVaR"], summary_df["MILP_Mean"], c="#111827", s=60, label="MILP")
    plt.errorbar(summary_df["GA_CVaR_mean"], summary_df["GA_Mean_mean"],
                 xerr=summary_df["GA_CVaR_ci"], yerr=summary_df["GA_Mean_ci"],
                 fmt="s", ms=7, color="#2563eb", ecolor="#2563eb", capsize=3, label="GA")
    for _,r in summary_df.iterrows():
        plt.text(r["MILP_CVaR"], r["MILP_Mean"], f"τ={r['tau']:.6g}", fontsize=9, ha="left", va="bottom")
    plt.xlabel("CVaR (lower is better)"); plt.ylabel("Mean return")
    plt.title("Mini ε-frontier (MILP vs GA, with costs)")
    plt.grid(ls="--", alpha=0.25); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out_png, dpi=240); plt.close()

def run_x3(file:str, n_assets:int, n_scenes:int,
           tau_list:List[float], tau_qs:List[float], n_samples:int,
           seeds:int, ga_time:int, milp_time:int,
           c_lin:float, c_fix:float, budget:float, w0_file:str, outdir:str):
    os.makedirs(outdir, exist_ok=True)
    dfR = load_returns_df(file, n_scenes)
    R_full = dfR.to_numpy(dtype=float)
    R, sel_idx, sel_cols = choose_assets(R_full, list(dfR.columns), n_assets)
    w0 = try_load_w0(w0_file, dfR, sel_idx)
    if (tau_list is None) or (len(tau_list)==0):
        qs = tau_qs if tau_qs else [0.60,0.75,0.90]
        taus = sample_tau_from_cvar_quantiles(R, qs, n_samples=n_samples)
    else:
        taus = list(tau_list)
    rows, raw_rows = [], []
    for k, tau in enumerate(taus):
        milp = milp_max_mean_with_cost(R, tau, w0, c_lin, c_fix, budget,
                                       timelimit=milp_time, msg=False)
        print(f"[MILP τ={tau:.6g}] CVaR={milp['CVaR']:.6f}, Mean={milp['Mean']:.6f}, "
              f"cost={milp['cost']:.6f}, time={milp['time']:.1f}s, status={milp['status']}")
        ga_finals, ga_logs = ga_max_mean_with_cost(R, tau, w0, c_lin, c_fix, budget,
                                                   time_limit=ga_time, seeds=seeds)
        plot_convergence(pd.DataFrame(ga_logs), milp["CVaR"], tau,
                         os.path.join(outdir, f"x3_convergence_{k+1}.png"))
        ga_df = pd.DataFrame(ga_finals)
        n = len(ga_df)
        g_cvar_mean = float(ga_df["CVaR"].mean())
        g_mean_mean = float(ga_df["Mean"].mean())
        g_cvar_ci = float(1.96*ga_df["CVaR"].std(ddof=1)/math.sqrt(n)) if n>1 else 0.0
        g_mean_ci = float(1.96*ga_df["Mean"].std(ddof=1)/math.sqrt(n)) if n>1 else 0.0
        g_cost_mean = float(ga_df["cost"].mean())
        g_turn_mean = float(ga_df["turnover"].mean())
        g_trd_mean  = float(ga_df["trades"].mean())
        delta_cvar_pct = float((g_cvar_mean - milp["CVaR"]) / max(1e-12,abs(milp["CVaR"])) * 100.0)
        delta_mean_pct = float((g_mean_mean - milp["Mean"]) / (abs(milp["Mean"])+1e-12) * 100.0)
        row = dict(
            tau=float(tau),
            MILP_CVaR=float(milp["CVaR"]), MILP_Mean=float(milp["Mean"]),
            MILP_cost=float(milp["cost"]), MILP_trades=int(milp["trades"]),
            MILP_turnover=float(milp["turnover"]), MILP_status=str(milp["status"]), MILP_time=float(milp["time"]),
            GA_CVaR_mean=g_cvar_mean, GA_Mean_mean=g_mean_mean,
            GA_CVaR_ci=g_cvar_ci, GA_Mean_ci=g_mean_ci,
            GA_cost_mean=g_cost_mean, GA_turnover_mean=g_turn_mean, GA_trades_mean=g_trd_mean,
            GA_time_mean=float(ga_df["Time"].mean()),
            GA_feasible_rate=float((ga_df["feasible"]==1).mean()),
            Delta_CVaR_pct=delta_cvar_pct, Delta_Mean_pct=delta_mean_pct,
            cost_budget=float(budget), cost_linear=float(c_lin), cost_fix=float(c_fix)
        )
        rows.append(row)
        ga_df2 = ga_df.copy(); ga_df2["tau"]=tau; ga_df2["method"]="GA"
        raw_rows.append(ga_df2)
    summary = pd.DataFrame(rows).sort_values("tau").reset_index(drop=True)
    raw = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
    summary.to_csv(os.path.join(outdir,"x3_summary.csv"), index=False)
    raw.to_csv(os.path.join(outdir,"x3_raw.csv"), index=False)
    plot_frontier(summary, os.path.join(outdir,"x3_frontier.png"))
    print(f"[SAVE] {os.path.join(outdir,'x3_summary.csv')}")
    print(f"[SAVE] {os.path.join(outdir,'x3_raw.csv')}")
    print("[OK] X3 done.")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Yield/Price CSV or Excel")
    ap.add_argument("--assets", type=int, default=48)
    ap.add_argument("--scenes", type=int, default=10000)
    ap.add_argument("--tau", type=str, default="", help='Comma separated constants τ, for example "0.026,0.028,0.030"')
    ap.add_argument("--tau_q", type=str, default="0.60,0.75,0.90", help='Comma separated quantiles (taken from the feasible region CVaR distribution)')
    ap.add_argument("--n_samples", type=int, default=2000, help="The number of random feasible weight samples for estimating τ")
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--ga_time", type=int, default=45)
    ap.add_argument("--milp_time", type=int, default=600)
    ap.add_argument("--cost_linear", type=float, default=0.001)
    ap.add_argument("--cost_fix",    type=float, default=0.0005)
    ap.add_argument("--cost_budget", type=float, default=0.04)
    ap.add_argument("--w0_file", type=str, default="", help="Initial weight w0 (CSV/Excel, optional; default is all 0)")
    ap.add_argument("--out", type=str, default="results_x3_48x10k_cost")
    return ap.parse_args()

if __name__ == "__main__":
    a = parse_args()
    tau_list = [float(x) for x in a.tau.split(",") if x.strip()!=""]
    tau_qs   = [float(x) for x in a.tau_q.split(",") if x.strip()!=""]
    run_x3(
        file=a.file, n_assets=a.assets, n_scenes=a.scenes,
        tau_list=tau_list, tau_qs=tau_qs, n_samples=a.n_samples,
        seeds=a.seeds, ga_time=a.ga_time, milp_time=a.milp_time,
        c_lin=a.cost_linear, c_fix=a.cost_fix, budget=a.cost_budget,
        w0_file=a.w0_file, outdir=a.out
    )