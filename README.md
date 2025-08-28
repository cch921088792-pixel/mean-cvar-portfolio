# mean-cvar-portfolio

Code and reproducible experiments for a thesis on **mean–CVaR portfolio optimisation**:
- **X1**: Minimise CVaR (Rockafellar–Uryasev MILP) + GA convergence
- **X2**: Maximise mean subject to **CVaR ≤ τ** (ε-frontier)
- **X3**: X2 + **transaction-cost budget** (linear + fixed)

All experiments use the same feasible region: sum-to-one, box bounds `W ∈ [W_MIN, W_MAX]`, and cardinality `K ∈ [K_MIN, K_MAX]`.

---

## Environment

```bash
conda create -n mcvar python=3.10 -y
conda activate mcvar
conda install -c conda-forge numpy pandas matplotlib pulp coincbc pandas-datareader -y
