# DAY 54 — Sample Average Approximation

Day 53 used all 103 historical weeks as scenarios. Real problems rarely offer a neat, complete list: the distribution is a simulator or a fitted model, and there are endless possible weeks. **Sample average approximation (SAA)** handles that:

- Draw $$N$$ scenarios.
- Solve the two-stage model on them.
- Judge the plan on scenarios it has never seen.

Three questions follow: how big must $$N$$ be, how optimistic is the in-sample answer, and how close is the plan to the true optimum when you can't compute the true optimum? Today the 103 weeks play the "true" distribution, so for once you can check the answers.

| Time | Block |
|---|---|
| 0:00 – 0:30 | 54.1 The true problem |
| 0:30 – 1:45 | 54.2 SAA with N sampled weeks |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 54.3 Bounds without knowing the truth |
| 2:45 – 3:45 | A planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: would SAA still lease the ovens? |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Day 57:

- `bakery_oven_log`: 26 weeks of oven hours, actual against standard, for every product.
- `bakery_robust`: what an overrun oven hour costs.

## 54.1 The true problem

Create `day54/saa.py`:

```python
"""Day 54: sample average approximation. Solve the two-stage model on a sample of N scenarios, then judge the plan
on scenarios it has never seen.

Import it:  from saa import sample_demand, solve_plan, evaluate_plan
"""
import numpy as np
import pyomo.environ as pyo

from bakery import solve
from two_stage import build_two_stage


def sample_demand(demand, products, n_weeks, idx):
    """Scenarios for the sampled week indices idx (drawn with replacement): scenario k is historical week idx[k]."""
    return {(p, k): demand[p, int(w)] for p in products for k, w in enumerate(idx)}, [str(int(w)) for w in idx]


def solve_plan(d, econ, st, demand, weeks):
    """Solve the SAA problem; return the plan and its in-sample expected profit (the SAA optimum)."""
    m = build_two_stage(d, econ, st, demand, weeks)
    solve(m)
    return {p: max(0.0, pyo.value(m.x[p])) for p in m.P}, pyo.value(m.expected_profit)


def evaluate_plan(d, econ, st, demand, weeks, plan):
    """The plan's expected profit on other scenarios: first stage fixed, second stage re-optimized in each."""
    m = build_two_stage(d, econ, st, demand, weeks, plan=plan)
    solve(m)
    return pyo.value(m.expected_profit)
```

Then `day54/run_saa.py`:

```python
# ---------- 54.1 The true problem ----------
print("--- truth")
best_plan, v_star = solve_plan(d, econ, st, demand, weeks)
print(f"all {W} weeks, equally likely: the true optimum is ${v_star:,.2f} a week")
```

```text
--- truth
all 103 weeks, equally likely: the true optimum is $112,600.11 a week
```

## 54.2 SAA with N sampled weeks

For each $$N$$, draw 20 independent samples of $$N$$ weeks (with replacement), solve each, and record three numbers:

- The **in-sample** optimum.
- The plan's **true** value on all 103 weeks.
- Its **optimism**: how much the SAA plan beats the true best plan *on the SAA plan's own sample*. This is never negative.

```python
# ---------- 54.2 SAA with N sampled weeks ----------
print("--- saa")
rng = np.random.default_rng(54)
rows, curve = [], []
for n in (5, 10, 20, 40, 80):
    inside, true, optimism = [], [], []
    for _ in range(20):                                          # 20 independent samples of n weeks
        dm, wk = sample_demand(demand, P, n, rng.integers(W, size=n))
        plan, v = solve_plan(d, econ, st, dm, wk)
        inside.append(v)
        true.append(evaluate_plan(d, econ, st, demand, weeks, plan))
        optimism.append(v - evaluate_plan(d, econ, st, dm, wk, best_plan))   # >= 0: it beats the true best plan on its own sample
    inside, true = np.array(inside), np.array(true)
    rows.append((n, round(inside.mean(), 2), round(inside.std(ddof=1), 2), round(np.mean(optimism), 2),
                 round(true.mean(), 2), round(true.min(), 2), round(v_star - true.mean(), 2)))
    curve.append((n, inside.mean(), true.mean(), 1.96 * inside.std(ddof=1) / np.sqrt(len(inside))))
print(pd.DataFrame(rows, columns=["N", "in_sample_mean", "in_sample_sd", "optimism", "true_value_mean", "true_value_worst",
                                  "loss_vs_optimum"]).to_string(index=False))
c = np.array(curve)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.errorbar(c[:, 0], c[:, 1], yerr=c[:, 3], fmt="o-", color="#e5484d", capsize=3,
            label="SAA optimum, in sample (95% interval)")
ax.plot(c[:, 0], c[:, 2], "o-", color="#3b82f6", label="the SAA plan's true value")
ax.axhline(v_star, ls="--", color="#6b7280", label="true optimum")
ax.set(xscale="log", xlabel="scenarios N (log scale)", ylabel="expected profit a week ($)",
       title="SAA: noisy in sample, converging out of sample")
ax.set_xticks(c[:, 0], [str(int(n)) for n in c[:, 0]])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/day54_saa.png", dpi=150)
print("saved figures/day54_saa.png")
```

```text
--- saa
 N  in_sample_mean  in_sample_sd  optimism  true_value_mean  true_value_worst  loss_vs_optimum
 5       113070.82       1347.79    202.79        112375.33         112108.44           224.78
10       112937.89        961.48    117.24        112491.96         112407.15           108.15
20       112335.24       1238.08    125.42        112478.92         112240.87           121.19
40       112544.50        526.91     33.05        112562.09         112506.88            38.02
80       112505.64        605.51     29.14        112566.21         112498.02            33.90
saved figures/day54_saa.png
```

![SAA as N grows](figures/day54_saa.png)

- **The SAA plan's true value rises toward the optimum**: it loses $225 a week at $$N = 5$$ and $34 at $$N = 80$$. Even 5 sampled weeks give a decent plan. The worst of 20 plans at $$N = 40$$ is within $94 of the optimum.
- **The in-sample optimum is optimistic**: on its own sample, the SAA plan always looks better than the truly best plan, by $203 at $$N = 5$$ and $29 at $$N = 80$$. On average, then, an SAA optimum **overstates** what its plan will earn (Day 48's optimizer's curse again).
- The in-sample numbers are also very **noisy**. A sample that happens to include a Christmas week looks much better than one without, so with these heavy-tailed weeks the noise can hide the optimism. Never report an in-sample optimum as the expected profit.

## 54.3 Bounds without knowing the truth

In practice there is no "all 103 weeks" to check against. Two estimates bracket the true optimum (for a maximization):

- An **upper bound**: the average of several independent SAA optima. They are optimistic on average, so their mean is at least the true optimum.
- A **lower bound**: one plan's value, estimated on a **fresh** sample. Any plan's true value is at most the optimum.

Choose the plan on one sample and judge it on another, so the choice doesn't flatter its own estimate.

```python
# ---------- 54.3 Bounds without knowing the truth ----------
print("--- bounds")
n, reps = 20, 20
rng = np.random.default_rng(540)
optima, plans = [], []
for _ in range(reps):
    dm, wk = sample_demand(demand, P, n, rng.integers(W, size=n))
    plan, v = solve_plan(d, econ, st, dm, wk)
    optima.append(v)
    plans.append(plan)
upper, half_u = mean_ci(optima)
pick, pwk = sample_demand(demand, P, 500, rng.integers(W, size=500))         # sample A: to choose among the plans
values = [evaluate_plan(d, econ, st, pick, pwk, plan) for plan in plans]
k = int(np.argmax(values))
check, cwk = sample_demand(demand, P, 500, rng.integers(W, size=500))        # sample B: to judge the chosen plan
m = build_two_stage(d, econ, st, check, cwk, plan=plans[k])
solve(m)
lower, half_l = mean_ci(scenario_profits(m))
print(f"upper bound (average SAA optimum, {reps} samples of {n}): ${upper:,.2f} +/- ${half_u:,.2f}")
print(f"the best of the {reps} plans on sample A (500 weeks): ${values[k]:,.2f}; the worst ${min(values):,.2f}")
print(f"lower bound (the chosen plan on sample B, 500 other weeks): ${lower:,.2f} +/- ${half_l:,.2f}")
print(f"estimated gap: ${upper - lower:,.2f} +/- ${np.hypot(half_u, half_l):,.2f}; "
      f"true value of the chosen plan ${evaluate_plan(d, econ, st, demand, weeks, plans[k]):,.2f} vs optimum ${v_star:,.2f}")
```

```text
--- bounds
upper bound (average SAA optimum, 20 samples of 20): $112,590.77 +/- $518.96
the best of the 20 plans on sample A (500 weeks): $112,728.55; the worst $112,539.98
lower bound (the chosen plan on sample B, 500 other weeks): $112,276.94 +/- $476.57
estimated gap: $313.83 +/- $704.58; true value of the chosen plan $112,574.89 vs optimum $112,600.11
```

The estimated gap is $314 ± $705 a week: with 95% confidence, the chosen plan is within about $1,000 of the best possible (0.9%). The truth, here known, is $25. The intervals are wide because weekly profit has heavy tails. Tighter bounds need more replications or larger evaluation samples, not cleverer optimization.

> **Decision it supports:** planning with many possible futures, when you can only sample them; and knowing how close to optimal the plan is.
>
> **Other decisions you can support the same way:** portfolio choices over simulated returns · capacity plans over simulated demand · any two-stage model whose scenarios come from a simulator.

### The planted bug

`day54/bug_independent.py` builds 103 scenarios by drawing a historical week **for each product separately** ("more combinations, more realistic"). Run it.

```python
mixed = {(p, k): demand[p, int(rng.integers(W))] for p in P for k in range(W)}       # every product its own week
plan, inside = solve_plan(d, econ, st, mixed, weeks)
m = build_two_stage(d, econ, st, mixed, weeks, plan=plan)
solve(m)
print(f"in its own scenarios: ${inside:,.2f} a week, worst week ${scenario_profits(m).min():,.2f}")
m = build_two_stage(d, econ, st, demand, weeks, plan=plan)
solve(m)
print(f"on the real weeks:    ${scenario_profits(m).mean():,.2f} a week, worst week ${scenario_profits(m).min():,.2f}")
best, v_star = solve_plan(d, econ, st, demand, weeks)
print(f"Day 53's plan on the real weeks: ${v_star:,.2f}")
```

<details><summary>What happens, and why</summary>

```text
in its own scenarios: $112,662.79 a week, worst week $108,690.39
on the real weeks:    $112,588.08 a week, worst week $79,042.97
Day 53's plan on the real weeks: $112,600.11
```

The plan earns almost the same on average. But in its own scenarios, the worst week is $108,690, a reassuring picture. On the real weeks, the worst is **$79,043**. Mixing weeks product by product breaks the Christmas and January weeks into pieces that cancel out (Day 44's lesson). The average survives; the **risk** disappears from view, and Day 58 needs that risk. Sample whole scenarios, never their pieces separately.

</details>

## Mini-project: would SAA still lease the ovens?

Day 53 decided to lease the extra ovens using all 103 weeks. With $$N$$ = 5, 10, 20 or 40 sampled weeks, solve the lease model 20 times each. How often does it lease? What is the decision (lease and plan) really worth, judged on all 103 weeks?

<details><summary>Solution</summary>

```python
"""Day 54 mini-project: Day 53 decided to lease the extra ovens using all 103 historical weeks. With fewer scenarios,
would SAA still say "lease"? For N = 5, 10, 20, 40 sampled weeks, solve the lease model 20 times and count how often
it leases, and what the decision is really worth (judged on all 103 weeks).

Run from the repository root (after day06/load_tables.py):  python day54/mini_project_lease_saa.py   (about 2 minutes)
"""
import sys

import duckdb
import numpy as np
import pandas as pd
import pyomo.environ as pyo

for folder in ("day06", "day42", "day44", "day53", "day54"):
    sys.path.insert(0, folder)
from bakery import solve
from saa import sample_demand
from two_stage import build_two_stage, load_two_stage

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
hours, cost = con.sql("SELECT extra_capacity, cost_per_week FROM bakery_investments WHERE option = 'lease_ovens'").fetchall()[0]
d, econ, st, demand, weeks = load_two_stage(con)
P, W = d["products"], len(weeks)


def true_value(lease, plan):
    m = build_two_stage(d, econ, st, demand, weeks, plan=plan, lease=(hours, cost))
    m.lease.fix(lease)
    solve(m)
    return pyo.value(m.expected_profit)


m = build_two_stage(d, econ, st, demand, weeks, lease=(hours, cost))
solve(m)
best = pyo.value(m.expected_profit)
print(f"all 103 weeks: lease = {pyo.value(m.lease):.0f}, ${best:,.2f} a week")
rng = np.random.default_rng(54)
rows = []
for n in (5, 10, 20, 40):
    leases, values = 0, []
    for _ in range(20):
        dm, wk = sample_demand(demand, P, n, rng.integers(W, size=n))
        s = build_two_stage(d, econ, st, dm, wk, lease=(hours, cost))
        solve(s)
        lease = int(round(pyo.value(s.lease)))
        leases += lease
        values.append(true_value(lease, {p: max(0.0, pyo.value(s.x[p])) for p in P}))
    rows.append((n, f"{leases}/20", round(np.mean(values), 2), round(best - np.mean(values), 2), round(best - min(values), 2)))
print(pd.DataFrame(rows, columns=["N", "times_leased", "true_value_mean", "average_loss", "worst_loss"]).to_string(index=False))
```

```text
all 103 weeks: lease = 1, $113,084.92 a week
 N times_leased  true_value_mean  average_loss  worst_loss
 5        20/20        112882.07        202.85      692.99
10        20/20        112980.67        104.26      223.01
20        20/20        112987.12         97.80      438.56
40        20/20        113060.13         24.79       55.35
```

- **The yes/no decision is stable**: every sample, even 5 weeks, says lease. The lease is worth about $485 a week net, well clear of zero, so sampling noise doesn't flip it.
- **The plan that goes with it** is less stable: with 5 weeks it loses $203 a week on average, and up to $693. Forty weeks bring that to $25.
- In general, a decision that is far from its break-even is easy to get right with few scenarios. The fine details of a plan need more.

</details>

## Commit

```bash
git add .
git commit -m "Day 54: sample average approximation"
git push
```

### Day 54 checklist

- [ ] I can solve a two-stage model on N sampled scenarios and evaluate the plan out of sample.
- [ ] I know an SAA optimum is optimistic and noisy, and never report it as the expected value.
- [ ] I can bound the optimality gap with replications and a fresh evaluation sample.
- [ ] I choose plans on one sample and judge them on another.
- [ ] I sample whole scenarios, so dependence between products survives.

---

# DAY 55 — Mini-project: VSS and EVPI

Day 53's stochastic program earned more than the plan for an average week. Is it worth building stochastic models in general, or only sometimes? And what would a perfect forecast be worth? Two numbers answer these questions:

- The **value of the stochastic solution** (VSS): what the stochastic model adds over planning for the average.
- The **expected value of perfect information** (EVPI): what knowing the future would add.

Today's mini-project computes both for the bakery and finds out when they are large.

| Time | Block |
|---|---|
| 0:00 – 0:30 | 55.1 The brief |
| 0:30 – 1:30 | 55.2 Four numbers |
| 1:30 – 1:45 | Break |
| 1:45 – 2:45 | 55.3 When does the stochastic model pay?; 55.4 December |
| 2:45 – 3:30 | 55.5 What is a better forecast worth? |
| 3:30 – 3:45 | Break |
| 3:45 – 4:45 | 55.6 The memo |
| 4:45 – 5:00 | Commit with Git + checklist |

## 55.1 The brief

The bakery's planning team asks two questions. Should we keep using the stochastic model, or is planning for the average week good enough? And the forecasting team wants budget for a better forecast: what is it worth?

Four numbers, all expected weekly profit:

| Number | Meaning |
|---|---|
| **EV** | The plan for the average week, as promised by its own model |
| **EEV** | That plan's expected profit when real weeks vary (fixed first stage, recourse in each scenario) |
| **RP** | The stochastic program's expected profit (Day 53) |
| **WS** | Wait-and-see: a plan made knowing each week in advance (Day 53's planted bug, used on purpose) |

$$\text{VSS} = \text{RP} - \text{EEV}, \qquad \text{EVPI} = \text{WS} - \text{RP}$$

For a maximization, $$\text{EEV} \le \text{RP} \le \text{WS}$$ always holds. A stochastic plan is never worse than the average-week plan, and foresight never hurts. Create `day55/values.py`:

```python
"""Day 55 mini-project: what is modelling uncertainty worth (VSS), and what would perfect information be worth (EVPI)?

Import it:  from values import scale_demand, four_numbers
"""
import pyomo.environ as pyo

from bakery import solve
from two_stage import build_two_stage


def scale_demand(demand, cap, k):
    """Stretch every scenario's error by k: demand = forecast x (1 + k (ratio - 1)). k = 0.5 is a forecast with half
    the error, k = 2 twice the error."""
    return {(p, w): cap[p] * max(0.0, 1 + k * (v / cap[p] - 1)) for (p, w), v in demand.items()}


def four_numbers(d, econ, st, demand, weeks):
    """EV: the plan for the average week; EEV: that plan's expected profit when weeks vary; RP: the stochastic
    program's expected profit; WS: wait-and-see, a plan made knowing each week in advance.
    VSS = RP - EEV (what the stochastic model adds), EVPI = WS - RP (what perfect forecasts would add)."""
    n = len(weeks)
    mean = {(p, 0): sum(demand[p, w] for w in range(n)) / n for p in d["products"]}
    ev = build_two_stage(d, econ, st, mean, ["average"])
    solve(ev)
    plan = {p: max(0.0, pyo.value(ev.x[p])) for p in ev.P}
    out = {"EV": pyo.value(ev.expected_profit)}
    for name, kw in [("EEV", {"plan": plan}), ("RP", {}), ("WS", {"anticipate": True})]:
        m = build_two_stage(d, econ, st, demand, weeks, **kw)
        solve(m)
        out[name] = pyo.value(m.expected_profit)
    out["VSS"], out["EVPI"] = out["RP"] - out["EEV"], out["WS"] - out["RP"]
    return out
```

## 55.2 Four numbers

```python
# ---------- 55.2 Four numbers ----------
print("--- four numbers")
base = four_numbers(d, econ, st, demand, weeks)
for k in ("EV", "EEV", "RP", "WS", "VSS", "EVPI"):
    print(f"{k:>5}: ${base[k]:>12,.2f}")
print(f"VSS is {base['VSS'] / base['RP']:.2%} of profit (${base['VSS'] * 52:,.0f} a year); "
      f"EVPI {base['EVPI'] / base['RP']:.2%} (${base['EVPI'] * 52:,.0f} a year)")
```

```text
--- four numbers
   EV: $  113,816.84
  EEV: $  111,661.74
   RP: $  112,600.11
   WS: $  113,648.14
  VSS: $      938.37
 EVPI: $    1,048.03
VSS is 0.83% of profit ($48,795 a year); EVPI 0.93% ($54,498 a year)
```

- **EV** promises $113,817, more than even perfect foresight delivers: planning for the average week is **overconfident** about what it will earn.
- Its plan really earns $111,662 (EEV), $2,155 less than promised. The stochastic plan earns $938 more: **VSS ≈ $49,000 a year**.
- Perfect forecasts would add another $1,048 a week: **EVPI ≈ $54,000 a year**. That is the ceiling on what any forecasting project could deliver for this decision.

## 55.3 When does the stochastic model pay?

Stretch the forecast errors (×0.5 to ×2) and give the bakery more or less flexibility (0, 40 or 80 top-up oven hours):

```python
# ---------- 55.3 When does the stochastic model pay? ----------
print("--- uncertainty and flexibility")
rows = []
for k in (0.5, 1.0, 1.5, 2.0):
    for hours in (0, 40, 80):
        r = four_numbers(d, econ, {**st, "topup_oven_hours": hours}, scale_demand(demand, d["cap"], k), weeks)
        rows.append((k, hours, round(r["RP"], 2), round(r["VSS"], 2), round(r["EVPI"], 2)))
grid = pd.DataFrame(rows, columns=["error_scale", "topup_hours", "RP", "VSS", "EVPI"])
print(grid.to_string(index=False))
```

```text
--- uncertainty and flexibility
 error_scale  topup_hours        RP     VSS    EVPI
         0.5            0 107688.73  434.46  653.84
         0.5           40 113225.19  479.79  530.83
         0.5           80 117259.38  856.77  184.23
         1.0            0 106972.86  843.91 1243.38
         1.0           40 112600.11  938.37 1048.03
         1.0           80 116829.54 1449.49  500.09
         1.5            0 106219.36 1216.43 1698.62
         1.5           40 111915.25 1367.95 1416.79
         1.5           80 116282.07 1957.54  732.95
         2.0            0 105442.78 1565.86 2126.95
         2.0           40 111209.93 1794.59 1758.16
         2.0           80 115674.21 2422.17  990.95
```

- **VSS grows with uncertainty**: $480 a week with half today's errors, $1,795 with twice the errors (40 h). If the future is nearly certain, the average is a fine plan.
- **VSS grows with flexibility**, too: more top-up hours give the stochastic plan more ways to use what it knows about the spread. A plan with a second chance needs to know the odds to use it well.
- **EVPI falls with flexibility**: with 80 top-up hours, perfect information is worth $500 instead of $1,048. **Flexibility and information are substitutes**: a bakery that can react doesn't need to know in advance.

## 55.4 December

```python
# ---------- 55.4 December ----------
print("--- december")
dd, de, ds, ddem, dweeks = load_two_stage(con, december_only=True)
dec = four_numbers(dd, de, ds, ddem, dweeks)
print(", ".join(f"{k} ${dec[k]:,.2f}" for k in ("EEV", "RP", "WS", "VSS", "EVPI")))
```

```text
--- december
EEV $108,757.11, RP $113,773.36, WS $114,863.36, VSS $5,016.24, EVPI $1,090.00
```

In December's weeks, VSS is **$5,016 a week**, five times the annual average. December's errors are large and one-sided (demand above forecast), and the average-week plan misses that. In December the stochastic model is not a refinement; it is the plan.

## 55.5 What is a better forecast worth?

EVPI is a ceiling. A real forecasting project halves the errors at best. Its value is the stochastic plan's profit with smaller errors, against today's:

```python
# ---------- 55.5 What is a better forecast worth? ----------
print("--- better forecasts")
rp = grid.set_index(["error_scale", "topup_hours"]).RP
for k in (0.5, 1.5, 2.0):
    print(f"forecast errors x{k}: the stochastic plan earns ${rp[k, 40] - rp[1.0, 40]:+,.2f} a week against today's errors")

# the yule logs (Day 51): one product, no second chance
hist = con.sql("SELECT forecast_logs, demand_logs FROM yule_log_history").df()
e = dict(con.sql("SELECT * FROM yule_log_economics").fetchall())
salvage = e["boxing_day_price"] * e["boxing_day_sell_share"]
ratios = (hist.demand_logs / hist.forecast_logs).to_numpy()
fc = con.sql("SELECT sum(forecast_logs) FROM yule_log_forecast").fetchall()[0][0] / 40      # an average store


def profit(q, dem):
    sold = np.minimum(q, dem)
    return e["price"] * sold - e["cost"] * q + salvage * (q - sold)


dem = fc * ratios
q_star = q_empirical(dem, critical_ratio(e["price"], e["cost"], salvage))
eev, rp_y, ws = profit(fc, dem).mean(), profit(q_star, dem).mean(), profit(dem, dem).mean()
print(f"yule logs, an average store (forecast {fc:.1f}): ordering the forecast ${eev:,.2f}, newsvendor ${rp_y:,.2f}, "
      f"perfect information ${ws:,.2f} -> VSS ${rp_y - eev:,.2f} ({(rp_y - eev) / rp_y:.1%}), "
      f"EVPI ${ws - rp_y:,.2f} ({(ws - rp_y) / rp_y:.1%})")
```

```text
--- better forecasts
forecast errors x0.5: the stochastic plan earns $+625.08 a week against today's errors
forecast errors x1.5: the stochastic plan earns $-684.86 a week against today's errors
forecast errors x2.0: the stochastic plan earns $-1,390.18 a week against today's errors
yule logs, an average store (forecast 48.0): ordering the forecast $866.73, newsvendor $949.50, perfect information $1,013.32 -> VSS $82.78 (8.7%), EVPI $63.81 (6.7%)
```

- **Halving the forecast errors is worth about $625 a week**, $32,500 a year, for the bakery plan alone. Worse forecasts cost about as much.
- For the yule logs (one product, no second chance), VSS is 8.7% of the profit and EVPI 6.7%. Without recourse, the stochastic model (here, the newsvendor) matters far more in relative terms.

## 55.6 The memo

<details><summary>A model memo</summary>

*To: bakery planning and the forecasting team. Re: what modelling uncertainty and better forecasts are worth.*

"**Keep the stochastic plan.** Planning for the average week promises $113,800 a week but earns $111,700. The stochastic plan earns $112,600: **about $49,000 a year more**. In December it earns **$5,000 a week more**, so it must be used for the holiday weeks above all.

**A better forecast has a value we can now state.** Perfect forecasts would be worth at most $54,000 a year to the bakery plan. Halving today's forecast errors would be worth about **$32,500 a year**. That is the budget ceiling for a forecasting project aimed at this decision; other decisions it improves would add to it.

**Flexibility is an alternative to forecasting.** Doubling the mid-week top-up from 40 to 80 oven hours would raise profit by about $4,200 a week and cut the value of perfect forecasts in half. Compare the cost of that capacity with the cost of better forecasts before choosing."

</details>

## Commit

```bash
git add .
git commit -m "Day 55: VSS and EVPI mini-project"
git push
```

### Day 55 checklist

- [ ] I can compute EV, EEV, RP and WS, and from them VSS and EVPI.
- [ ] I know that planning for the average is overconfident, and by how much here.
- [ ] I can show how VSS and EVPI change with uncertainty and with flexibility.
- [ ] I can put a value on a better forecast, and know that EVPI is its ceiling.
- [ ] I can tell a team when a stochastic model is worth the trouble.

---

# DAY 56 — Chance Constraints and Service Levels

So far the objective has carried every concern: shortages cost lost margin, and the plan balanced it against waste. But some promises are not trade-offs to management: "the core breads are always on the shelf", "the trucks start within 30 minutes". A **chance constraint** turns such a promise into a rule: the plan must satisfy it **with probability at least α**. Today you add one to the bakery's two-stage model. You will see:

- why one promise for each product is not one promise for all;
- what a promise costs;
- how a fill-rate target differs;
- why a promise kept in sample can fail out of sample.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 56.1 A promise as a chance constraint |
| 0:45 – 1:45 | 56.2 The price of service |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 56.3 A fill rate instead; 56.4 in sample and out of sample |
| 2:45 – 3:45 | A planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: the union's overtime cap |
| 4:45 – 5:00 | Commit with Git + checklist |

## 56.1 A promise as a chance constraint

Store Operations wants the week's plan to **cover** demand for the three core products (sourdough, whole wheat, croissants) in at least 90% of weeks. The mid-week top-up stays free for other surprises.

- **Individual chance constraint**: $$P(x_p \ge D_p) \ge \alpha$$ for each product. Its deterministic equivalent is just $$x_p \ge$$ the α-quantile of $$D_p$$, a newsvendor-style bound.
- **Joint chance constraint**: $$P(x_p \ge D_p \text{ for all core } p) \ge \alpha$$. There is no simple formula. With scenarios, give each week a yes/no variable $$z_w$$ (1 = this week may fall short), and write
  $$x_p \ge D_{p,w} - M_p z_w, \qquad \sum_w z_w \le \lfloor (1-\alpha) N \rfloor.$$
  This is a MIP with one binary per scenario.

Create `day56/chance.py`:

```python
"""Day 56: chance constraints on the two-stage bakery model. The promise: "the week's plan covers the core products'
demand in at least alpha of the weeks" (the mid-week top-up is kept for surprises elsewhere). One yes/no variable per
scenario (and product) marks a week the plan may fall short.

Import it:  from chance import add_service, covered
"""
import numpy as np
import pyomo.environ as pyo


def add_service(m, demand, core, alpha, joint=True, fill=None):
    """Add a service promise to a two-stage model m (Day 53).
    joint=True: the plan covers all core products together in at least alpha of the weeks (one binary per week).
    joint=False: each core product separately (one binary per product and week).
    fill: instead, an expected fill rate: units short <= (1 - fill) x units wanted, over plan + top-up (an LP)."""
    W = list(m.W)
    if fill is not None:
        m.fill_rate = pyo.Constraint(core, rule=lambda m, p: sum(demand[p, w] - m.s[p, w] for w in W)
                                     <= (1 - fill) * sum(demand[p, w] for w in W))
        return m
    big = {p: max(demand[p, w] for w in W) for p in core}             # a miss can't be worse than all demand
    misses = np.floor((1 - alpha) * len(W) + 1e-9)
    if joint:
        m.miss = pyo.Var(m.W, within=pyo.Binary)                       # 1: the plan may fall short this week
        z = lambda p, w: m.miss[w]
        m.budget = pyo.Constraint(expr=sum(m.miss[w] for w in W) <= misses)
    else:
        m.miss = pyo.Var(core, m.W, within=pyo.Binary)
        z = lambda p, w: m.miss[p, w]
        m.budget = pyo.Constraint(core, rule=lambda m, p: sum(m.miss[p, w] for w in W) <= misses)
    m.cover = pyo.Constraint(core, m.W, rule=lambda m, p, w: m.x[p] >= demand[p, w] - big[p] * z(p, w))
    return m


def covered(plan, demand, core, n_weeks):
    """Week by week: does the plan cover every core product's demand? Also product by product."""
    per = {p: np.array([plan[p] >= demand[p, w] - 1e-6 for w in range(n_weeks)]) for p in core}
    return np.all(np.vstack(list(per.values())), axis=0), per
```

Then `day56/run_chance.py`:

```python
# ---------- 56.1 A promise as a chance constraint ----------
print("--- individual and joint")
from newsvendor import q_empirical
print("90% quantiles of demand (an individual chance constraint's answer): "
      + ", ".join(f"{p} {q_empirical([demand[p, w] for w in range(W)], 0.90):.2f}" for p in CORE))
rows = [describe("no promise (Day 53)", run()),
        describe("each core product 90% of weeks", run(0.90, joint=False)),
        describe("all core products together 90%", run(0.90, joint=True))]
print(pd.DataFrame(rows).to_string(index=False))
```

```text
--- individual and joint
90% quantiles of demand (an individual chance constraint's answer): BK-SD 36.15, BK-WW 46.48, BK-CR 32.25
                       promise    profit  all_core_covered  BK-SD_covered  BK-WW_covered  BK-CR_covered  worst_fill_rate  x_BK-SD  x_BK-WW  x_BK-CR
           no promise (Day 53) 112600.11             0.029          0.029          0.184          0.184           0.8813    30.84    44.11    30.11
each core product 90% of weeks 111629.70             0.854          0.903          0.903          0.903           0.9887    36.15    46.48    32.25
all core products together 90% 111392.60             0.903          0.951          0.951          0.913           0.9890    36.49    46.91    32.37
```

- The individual promise puts each product exactly at its 90% quantile, the deterministic equivalent. But all three are covered together in only **85% of weeks**: each product misses different weeks.
- The joint promise covers all three in 90% of weeks for $237 a week more.
- The gap is small because the misses mostly fall in the **same** weeks (Christmas again). For independent products, three 90% promises would give about 73% jointly.
- Without any promise, Day 53's plan covers the core products in only 3% of weeks. It plans sourdough low on purpose and relies on the top-up.

## 56.2 The price of service

```python
# ---------- 56.2 The price of service ----------
print("--- price of service")
rp = rows[0]["profit"]
rows = []
for alpha in (0.80, 0.85, 0.90, 0.95, 0.98, 1.00):
    r = describe(f"joint {alpha:.0%}", run(alpha, joint=True))
    rows.append((alpha, r["profit"], round(rp - r["profit"], 2), r["all_core_covered"]))
print(pd.DataFrame(rows, columns=["alpha", "profit", "cost_of_promise", "weeks_covered"]).to_string(index=False))
```

```text
--- price of service
 alpha    profit  cost_of_promise  weeks_covered
  0.80 111832.78           767.33          0.806
  0.85 111661.86           938.25          0.854
  0.90 111392.60          1207.51          0.903
  0.95 111090.76          1509.35          0.951
  0.98 107586.55          5013.56          0.981
  1.00 104786.30          7813.81          1.000
```

Each extra point of service costs more than the last. 80% costs $767 a week, 95% costs $1,509, and **98% costs $5,014**: the last weeks to cover are the Christmas weeks, where demand is far above forecast. A promise of 100% costs $7,800 a week ($406,000 a year). Show management this curve before they choose α.

## 56.3 A fill rate instead

A **fill rate** (Day 52) is not a chance constraint. It limits the **expected** units short, and it is linear:

$$\sum_w \pi_w (D_{p,w} - s_{p,w}) \le (1 - \beta) \sum_w \pi_w D_{p,w}$$

```python
# ---------- 56.3 A fill rate instead: an LP ----------
print("--- fill rate")
rows = [describe(f"expected fill rate {f:.1%}", run(fill=f)) for f in (0.99, 0.995)]
print(pd.DataFrame(rows).to_string(index=False))
```

```text
--- fill rate
                 promise    profit  all_core_covered  BK-SD_covered  BK-WW_covered  BK-CR_covered  worst_fill_rate  x_BK-SD  x_BK-WW  x_BK-CR
expected fill rate 99.0% 112488.22             0.058          0.175          0.175          0.252            0.990    34.29    44.09    30.44
expected fill rate 99.5% 112461.52             0.058          0.175          0.175          0.252            0.995    34.29    44.09    30.44
```

A 99.5% fill rate costs only $139 a week. But the plan covers the core products in **6% of weeks**: the top-up covers the shortfalls on average, and the remaining shortfalls are small. Customers feel the fill rate; store managers feel the empty-shelf weeks. **Which promise is made matters more than its number.**

## 56.4 In sample and out of sample

A chance constraint kept on the scenarios can fail on new weeks. Solve the joint promise on 40 sampled weeks, then check the plan on all 103:

```python
# ---------- 56.4 In sample and out of sample ----------
print("--- out of sample")
rng = np.random.default_rng(56)
rows = []
for alpha in (0.90, 0.95, 0.98):
    out = []
    for _ in range(10):
        dm, wk = sample_demand(demand, d["products"], 40, rng.integers(W, size=40))
        m = run(alpha, joint=True, dem=dm, wk=wk)
        allc, _ = covered(plan_of(m), demand, CORE, W)                   # the same plan on all 103 weeks
        out.append(allc.mean())
    rows.append((alpha, round(np.mean(out), 3), round(min(out), 3), round(np.mean(np.array(out) >= 0.90), 2)))
print(pd.DataFrame(rows, columns=["alpha_in_40_weeks", "covered_on_103_mean", "covered_worst", "share_meeting_90"])
      .to_string(index=False))
```

```text
--- out of sample
 alpha_in_40_weeks  covered_on_103_mean  covered_worst  share_meeting_90
              0.90                0.820          0.738               0.0
              0.95                0.884          0.864               0.3
              0.98                0.974          0.951               1.0
```

A 90% promise kept on 40 weeks covers only **82%** of real weeks, and none of the 10 plans reaches 90%. The plan sits exactly on the edge of its sample, and a new sample has a few more weeks over the edge. To **keep** 90%, promise **98%** in a sample of 40 (97% covered on 103 weeks, every plan above 90%). With few scenarios, tighten the promise or use more scenarios, then check out of sample.

> **Decision it supports:** plans that must keep a promise with high probability, not just on average.
>
> **Other decisions you can support the same way:** reserve generation in power grids (P(supply ≥ demand) ≥ 99.9%) · bank capital (value at risk) · emergency-room staffing · emissions caps that must hold most days.

### The planted bug

`day56/bug_budget.py` is a colleague's joint 90% promise, written by hand. Run it.

```python
m.miss = pyo.Var(m.W, within=pyo.Binary)
m.budget = pyo.Constraint(expr=sum(m.miss[w] for w in m.W) <= np.floor(alpha * len(weeks)))
m.cover = pyo.Constraint(CORE, m.W, rule=lambda m, p, w: m.x[p] >= demand[p, w] - 1_000 * m.miss[w])
solve(m)
allc, _ = covered({p: pyo.value(m.x[p]) for p in m.P}, demand, CORE, len(weeks))
print(f"profit ${pyo.value(m.expected_profit):,.2f}; promise 'kept'; weeks the plan covers all core products: {allc.mean():.1%}")
```

<details><summary>What happens, and why</summary>

```text
profit $112,538.17; promise 'kept'; weeks the plan covers all core products: 10.7%
```

The promise is "kept" for $62 a week, but the plan covers the core products in **11%** of weeks. The budget lets the model **miss** $$\lfloor 0.9 \times 103 \rfloor = 92$$ weeks, where it should allow $$\lfloor 0.1 \times 103 \rfloor = 10$$. The binaries count the weeks you may miss, so their budget is $$1 - \alpha$$. Check every chance constraint by counting the covered weeks afterwards, as `covered` does.

</details>

## Mini-project: the union's overtime cap

The bakers' union proposes capping the mid-week top-up at 20 overtime oven hours, not 40. Management counters: "at most 20 hours in at least 90% of weeks; up to 40 in the others." Compare three rules: 40 hours always (today), 20 hours always (the union), and the counter-proposal, which is a chance constraint on the **second stage**. How often does the counter-proposal use more than 20 hours?

<details><summary>Solution</summary>

```python
"""Day 56 mini-project: the bakers' union proposes a cap on the mid-week top-up: 20 overtime oven hours, not 40.
Management counters: "at most 20 hours in at least 90% of weeks; up to 40 in the others". Compare the expected
profit of: 40 hours always (today), 20 hours always (the union), and the counter-proposal (a chance constraint on the
second stage). How often does the counter-proposal actually use more than 20 hours?

Run from the repository root (after day06/load_tables.py):  python day56/mini_project_overtime.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd
import pyomo.environ as pyo

for folder in ("day06", "day42", "day44", "day53"):
    sys.path.insert(0, folder)
from bakery import solve
from two_stage import build_two_stage, load_two_stage

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
d, econ, st, demand, weeks = load_two_stage(con)
oven = lambda p: d["use"].get(("oven_hours", p), 0.0)


def hours_used(m):
    return np.array([sum(oven(p) * pyo.value(m.y[p, w]) for p in m.P) for w in m.W])


rows = []
for name, hours, chance in [("40 hours always (today)", 40, None), ("20 hours always (union)", 20, None),
                            ("20 hours in 90% of weeks, 40 in the rest", 40, (20, 0.90))]:
    m = build_two_stage(d, econ, {**st, "topup_oven_hours": hours}, demand, weeks)
    if chance:
        cap, alpha = chance
        m.over = pyo.Var(m.W, within=pyo.Binary)                          # 1: this week may use up to 40 hours
        m.soft_cap = pyo.Constraint(m.W, rule=lambda m, w: sum(oven(p) * m.y[p, w] for p in m.P)
                                    <= cap + (hours - cap) * m.over[w])
        m.how_often = pyo.Constraint(expr=sum(m.over[w] for w in m.W) <= np.floor((1 - alpha) * len(weeks)))
    solve(m)
    h = hours_used(m)
    rows.append((name, round(pyo.value(m.expected_profit), 2), round(h.mean(), 1), round((h > 20 + 1e-6).mean(), 3)))
t = pd.DataFrame(rows, columns=["rule", "expected_profit", "topup_hours_used", "weeks_over_20h"])
t["vs_today"] = (t.expected_profit - t.expected_profit[0]).round(2)
print(t.to_string(index=False))
```

```text
                                    rule  expected_profit  topup_hours_used  weeks_over_20h  vs_today
                 40 hours always (today)        112600.11              40.0           1.000      0.00
                 20 hours always (union)        109921.84              20.0           0.000  -2678.27
20 hours in 90% of weeks, 40 in the rest        110204.15              21.9           0.097  -2395.96
```

- The union's cap costs **$2,678 a week**. The counter-proposal recovers only **$282** of that, using more than 20 hours in exactly the 10 weeks it is allowed.
- Why so little? The top-up is used to the full **every week** (40 hours today, 20 under the cap). It isn't a reserve for bad weeks; it's capacity the plan counts on. A chance constraint helps when the resource is needed only sometimes.
- The real question is whether 20 overtime oven hours are worth $2,678 a week, about $134 an hour against the $55 premium. That is a negotiation about capacity, not about bad weeks.

*Note for the negotiation:* "Every hour of mid-week top-up earns about $134 against a $55 overtime premium. The counter-proposal (20 hours in 90% of weeks) saves little. If the cap is agreed, plan the lost capacity elsewhere (Day 53's oven lease)."

</details>

## Commit

```bash
git add .
git commit -m "Day 56: chance constraints and service levels"
git push
```

### Day 56 checklist

- [ ] I can write an individual chance constraint as a quantile and a joint one as a scenario MIP.
- [ ] I know three individual promises are not one joint promise, and how correlation narrows the gap.
- [ ] I can draw the price-of-service curve and explain its steep end.
- [ ] I know a fill rate and a chance constraint promise different things.
- [ ] I check a chance constraint out of sample and tighten it if needed.

---

# DAY 57 — Robust Optimization

Stochastic programming needs probabilities. Sometimes you have only **ranges**: "an oven batch can take up to 10% longer", "the supplier delivers in 3 to 17 days". Sometimes a constraint must hold for **every** case in a range, not on average. **Robust optimization** plans for the worst case within an **uncertainty set**. The art is choosing a set that is protective without being absurd. Today you make the bakery's oven constraint robust with a **budget of uncertainty**, set a robust reorder point, and order yule logs without any probabilities at all.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 57.1 How long the ovens really take |
| 0:45 – 1:45 | 57.2 The budget of uncertainty |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 57.3 A robust reorder point |
| 2:45 – 3:45 | Robust or stochastic?; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: yule logs without probabilities |
| 4:45 – 5:00 | Commit with Git + checklist |

## 57.1 How long the ovens really take

Day 6 assumed each batch takes exactly its standard oven time. `bakery_oven_log` records 26 weeks of actual oven hours. Create `day57/robust.py`:

```python
"""Day 57: robust optimization. Oven hours per batch vary (bakery_oven_log); plan so the oven constraint holds even
if up to Gamma products take their longest observed time (Bertsimas and Sim's budget of uncertainty).

Import it:  from robust import oven_deviations, build_robust, oven_overrun
"""
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from bakery import build_model


def oven_deviations(con, quantile=0.95):
    """Actual / standard oven hours, per product and logged week; and each product's deviation: the given quantile
    of (ratio - 1). The deviation is how much longer than standard a batch can take, as a share."""
    log = con.sql("""SELECT week_start, sku, actual_oven_hours / standard_oven_hours AS ratio
                     FROM bakery_oven_log""").df()
    table = log.pivot(index="week_start", columns="sku", values="ratio")
    eps = (table - 1).quantile(quantile).clip(lower=0)
    return table, eps


def build_robust(d, eps, gamma, resource="oven_hours"):
    """Day 6's model, with the oven row protected: sum a x + Gamma lam + sum mu <= H, lam + mu_p >= eps_p a_p x_p.
    The dual of 'the worst Gamma products run long' (gamma = 0: the nominal model; gamma = all: every product long)."""
    m = build_model(d)
    a = {p: d["use"].get((resource, p), 0.0) for p in d["products"]}
    m.lam = pyo.Var(within=pyo.NonNegativeReals)
    m.mu = pyo.Var(m.P, within=pyo.NonNegativeReals)
    m.capacity_limit[resource].deactivate()
    m.robust_oven = pyo.Constraint(expr=sum(a[p] * m.x[p] for p in m.P) + gamma * m.lam + sum(m.mu[p] for p in m.P)
                                   <= d["capacity"][resource])
    m.protect = pyo.Constraint(m.P, rule=lambda m, p: m.lam + m.mu[p] >= eps[p] * a[p] * m.x[p])
    return m


def oven_overrun(plan, d, table, resource="oven_hours"):
    """For each logged week (its real ratios, all products together): hours beyond capacity if this plan were baked."""
    a = pd.Series({p: d["use"].get((resource, p), 0.0) for p in d["products"]})
    x = pd.Series(plan)
    used = (table[a.index] * a * x).sum(axis=1)
    return np.maximum(used - d["capacity"][resource], 0.0).to_numpy()
```

Then `day57/run_robust.py`:

```python
# ---------- 57.1 How long the ovens really take ----------
print("--- oven times")
table, eps = oven_deviations(con)
cat = pd.Series(d["category"])
summary = pd.DataFrame({"mean_ratio": table.mean(), "deviation_95": eps}).join(cat.rename("category"))
print(summary.groupby("category").agg(products=("mean_ratio", "size"), mean_ratio=("mean_ratio", "mean"),
                                      deviation_95=("deviation_95", "mean")).round(3).to_string())
week_total = (table * pd.Series({p: d["use"].get(("oven_hours", p), 0.0) for p in d["products"]})).sum(axis=1) / \
             sum(d["use"].get(("oven_hours", p), 0.0) for p in d["products"])
print(f"a whole week's oven time vs standard: {week_total.min():.3f} to {week_total.max():.3f} (26 weeks)")
```

```text
--- oven times
              products  mean_ratio  deviation_95
category
bread               10       1.012         0.065
cake                 6       0.997         0.147
cookie               6       1.023         0.087
viennoiserie         8       1.012         0.106
a whole week's oven time vs standard: 0.962 to 1.053 (26 weeks)
```

Batches take about 1% longer than standard on average. In a bad week (the 95th percentile), a cake takes 15% longer and a bread 6.5% longer. A whole week's oven time ranges from 4% under to 5% over standard: Day 6's plan uses all 600 hours, so most weeks it **overruns**.

## 57.2 The budget of uncertainty

Protect the oven row against products running long. Each product's oven time per batch lies in $$[a_p, a_p(1 + \varepsilon_p)]$$.

- Protecting against **every** product at its worst (a **box** set) is too pessimistic: they never all run long at once.
- **Bertsimas and Sim's budget** $$\Gamma$$ protects against the worst $$\Gamma$$ products running long at once:

$$\sum_p a_p x_p + \max_{|S| \le \Gamma} \sum_{p \in S} \varepsilon_p a_p x_p \le H$$

The inner maximum is an LP. Its dual turns the constraint into linear rows, the **robust counterpart**:

$$\sum_p a_p x_p + \Gamma \lambda + \sum_p \mu_p \le H, \qquad \lambda + \mu_p \ge \varepsilon_p a_p x_p, \qquad \lambda, \mu \ge 0$$

$$\Gamma = 0$$ is Day 6's model; $$\Gamma = 30$$ is the box.

```python
# ---------- 57.2 The budget of uncertainty ----------
print("--- gamma")
rows, curve = [], []
for gamma in (0, 1, 2, 3, 5, 8, 12, 30):
    m = build_robust(d, eps, gamma)
    solve(m)
    plan = {p: pyo.value(m.x[p]) for p in m.P}
    over = oven_overrun(plan, d, table)
    profit = pyo.value(m.total_profit)
    net = profit - overrun_cost * over.mean()
    rows.append((gamma, round(profit, 2), round(sum(d["use"].get(("oven_hours", p), 0) * x for p, x in plan.items()), 1),
                 round((over > 0).mean(), 3), round(over.mean(), 2), round(over.max(), 1), round(net, 2)))
    curve.append((gamma, profit, net, (over > 0).mean()))
t = pd.DataFrame(rows, columns=["gamma", "planned_profit", "standard_oven_hours", "weeks_overrun", "mean_overrun_h",
                                "worst_overrun_h", "profit_after_overruns"])
print(t.to_string(index=False))
base = t.iloc[0]
for r in t.iloc[1:].itertuples():
    print(f"gamma {r.gamma:>2}: beats the nominal plan if an overrun hour costs more than "
          f"${(base.planned_profit - r.planned_profit) / (base.mean_overrun_h - r.mean_overrun_h):,.0f}")
c = np.array(curve)
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(c[:, 3] * 100, c[:, 1], "o-", color="#3b82f6", label="planned profit")
ax.plot(c[:, 3] * 100, c[:, 2], "o-", color="#e5484d", label="after overrun costs")
for g, x, y in zip(c[:, 0], c[:, 3] * 100, c[:, 1]):
    ax.annotate(f"Γ={g:.0f}", (x, y), textcoords="offset points", xytext=(4, 6), fontsize=7, color="#6b7280")
ax.set(xlabel="weeks the ovens overrun (% of 26 logged weeks)", ylabel="profit a week ($)",
       title="The price of robustness: oven times that vary")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/day57_robust.png", dpi=150)
print("saved figures/day57_robust.png")
```

```text
--- gamma
 gamma  planned_profit  standard_oven_hours  weeks_overrun  mean_overrun_h  worst_overrun_h  profit_after_overruns
     0       108351.28                600.0          0.692           11.98             49.7              106913.64
     1       107719.21                593.6          0.538            8.07             42.7              106750.41
     2       107194.26                588.5          0.462            5.57             37.1              106525.96
     3       106731.12                586.2          0.385            4.07             33.8              106242.29
     5       106049.26                579.3          0.192            1.97             26.0              105812.60
     8       105256.51                570.2          0.077            0.71             16.0              105171.57
    12       104443.85                563.7          0.038            0.31              8.1              104406.57
    30       102829.64                549.2          0.000            0.00              0.0              102829.64
gamma  1: beats the nominal plan if an overrun hour costs more than $162
gamma  2: beats the nominal plan if an overrun hour costs more than $181
gamma  3: beats the nominal plan if an overrun hour costs more than $205
gamma  5: beats the nominal plan if an overrun hour costs more than $230
gamma  8: beats the nominal plan if an overrun hour costs more than $275
gamma 12: beats the nominal plan if an overrun hour costs more than $335
gamma 30: beats the nominal plan if an overrun hour costs more than $461
saved figures/day57_robust.png
```

![The price of robustness](figures/day57_robust.png)

- **The price of robustness**: each step of $$\Gamma$$ gives up profit to overrun less often. Day 6's plan overruns in 69% of weeks (12 hours on average). $$\Gamma = 5$$ overruns in 19% of weeks for $2,302 a week less, and $$\Gamma = 30$$ never overruns, for $5,522 less.
- **Whether that's worth it depends on what an overrun costs.** At $120 an overrun hour (late bread, overtime), the nominal plan still earns the most after overruns. $$\Gamma = 5$$ wins only if an overrun hour costs more than **$230**: for example, if a late run means trucks leave without bread.
- Robust protection is insurance. Price it against what it protects.

## 57.3 A robust reorder point

The olive oil supplier (SUP-B) delivered in 3 to 17 days. A robust reorder point covers the **longest** lead time in the record, plus $$\Gamma$$ "high" days of demand (the 90th percentile of daily demand above the mean). Compare with Day 52's stochastic reorder point:

```python
# ---------- 57.3 A robust reorder point ----------
print("--- robust reorder point")
items, settings = load_items(con)
item = "AM-04"
i = items.loc[item]
dem, lead = demand_history(con, item), lead_history(con, item)
q = np.round(eoq(i.mean_day * 365, settings["order_cost_per_po"], i.hold_per_case_year))
l_max = lead.max()
high = np.quantile(dem, 0.90) - dem.mean()                            # a "high" day: 90th percentile above the mean
print(f"{item}: lead time {lead.min()} to {l_max} days (mean {lead.mean():.2f}); a high day is +{high:.1f} cases")
rows = []
for name, s in [("Day 52 (stochastic, 98% fill)", 266.0), ("nominal: mean x mean lead", np.round(i.mean_day * lead.mean())),
                ("robust, Gamma = 0", np.round(i.mean_day * l_max)),
                ("robust, Gamma = sqrt(Lmax)", np.round(i.mean_day * l_max + np.sqrt(l_max) * high)),
                ("robust, Gamma = Lmax (box)", np.round(i.mean_day * l_max + l_max * high))]:
    rng = np.random.default_rng(57)
    sims = pd.DataFrame([simulate_policy(dem, lead, rng, 730, reorder_point=s, order_qty=q) for _ in range(20)]).mean()
    rows.append((name, s, round(sims.fill_rate, 4), round(sims.avg_on_hand, 1)))
print(pd.DataFrame(rows, columns=["reorder_point", "s", "fill_rate", "avg_on_hand"]).to_string(index=False))
```

```text
--- robust reorder point
AM-04: lead time 3 to 17 days (mean 11.77); a high day is +11.3 cases
                reorder_point     s  fill_rate  avg_on_hand
Day 52 (stochastic, 98% fill) 266.0     0.9868        262.5
    nominal: mean x mean lead 209.0     0.9464        215.3
            robust, Gamma = 0 302.0     0.9971        303.3
   robust, Gamma = sqrt(Lmax) 348.0     0.9997        346.5
   robust, Gamma = Lmax (box) 493.0     1.0000        491.5
```

- Covering the **mean** lead time with **mean** demand fills only 94.6%.
- The robust point for the worst lead time ($$\Gamma = 0$$) already fills 99.7%, with 41 more cases on hand than Day 52's. One 17-day delivery in the record sets the policy.
- More $$\Gamma$$ piles on stock for almost nothing.
- **Robust policies are only as sensible as their sets**: a set built from the single worst observation is driven by that observation.

### Robust or stochastic?

| Use **stochastic** models when | Use **robust** models when |
|---|---|
| you can estimate probabilities (history, a simulator) | you know ranges but not probabilities |
| shortfalls cost money you can price | a constraint must hold (safety, contracts, physics) |
| you care about the average (or a tail: Day 58) | the worst case matters, or an adversary chooses it |
| the model with scenarios stays solvable | scenarios would make the model too big |

Both can be combined: robust constraints inside a stochastic objective, or an uncertainty set sized so the constraint holds with probability α. That is how Bertsimas and Sim relate $$\Gamma$$ to a chance constraint.

> **Decision it supports:** plans that must stay feasible for every case in a range, and a clear price for that protection.
>
> **Other decisions you can support the same way:** power-grid dispatch under uncertain renewables · portfolio limits under uncertain returns · network capacity against link failures · schedules with uncertain task times.

### The planted bug

`day57/bug_protection.py` writes the robust oven row by hand, with $$\Gamma = 5$$. Run it.

```python
m.lam = pyo.Var(within=pyo.NonNegativeReals)
m.mu = pyo.Var(m.P, within=pyo.NonNegativeReals)
m.capacity_limit["oven_hours"].deactivate()
m.robust_oven = pyo.Constraint(expr=sum(a[p] * m.x[p] for p in m.P) + 5 * m.lam + sum(m.mu[p] for p in m.P) <= 600)
m.protect = pyo.Constraint(m.P, rule=lambda m, p: m.lam + m.mu[p] >= eps[p] * a[p])
solve(m)
over = oven_overrun({p: pyo.value(m.x[p]) for p in m.P}, d, table)
print(f"gamma 5: profit ${pyo.value(m.total_profit):,.2f}; weeks the ovens overrun {(over > 0).mean():.1%}")
```

<details><summary>What happens, and why</summary>

```text
gamma 5: profit $108,203.88; weeks the ovens overrun 65.4%
```

"Robust" at $$\Gamma = 5$$ for only $147 of profit, yet the ovens overrun in 65% of weeks, almost as often as with no protection. The protection rows read $$\lambda + \mu_p \ge \varepsilon_p a_p$$: they protect against one **batch** of each product running long, not $$x_p$$ batches. The protection must scale with the decision: $$\varepsilon_p a_p x_p$$. Check a robust model by simulating its plan against the data that motivated it.

</details>

## Mini-project: yule logs without probabilities

Suppose the yule log history were lost, and all you knew was that a store sells between the smallest and the largest share of its forecast ever seen. How many logs should an average store order? Compare three robust rules with Day 51's newsvendor, judged on the real history:

- **Max-min profit**: the best worst case.
- **Min-max regret**: regret is what you lose against the best order for the demand that happens. Minimize its worst case.
- **The middle of the range.**

<details><summary>Solution</summary>

```python
"""Day 57 mini-project: suppose the yule log history were lost, and all we knew was that a store sells between
the smallest and largest share of its forecast ever seen. Without probabilities, how many logs should an average
store order? Compare three robust rules (max-min profit, min-max regret, and the middle of the range) with Day 51's
newsvendor, judged on the real history.

Run from the repository root (after day06/load_tables.py):  python day57/mini_project_regret.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, "day51")
from newsvendor import critical_ratio, q_empirical

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
e = dict(con.sql("SELECT * FROM yule_log_economics").fetchall())
price, cost, salvage = e["price"], e["cost"], e["boxing_day_price"] * e["boxing_day_sell_share"]
hist = con.sql("SELECT demand_logs / forecast_logs AS ratio FROM yule_log_history").df().ratio.to_numpy()
fc = con.sql("SELECT avg(forecast_logs) FROM yule_log_forecast").fetchall()[0][0]
low, high = fc * hist.min(), fc * hist.max()
print(f"an average store: forecast {fc:.1f} logs; demand somewhere in [{low:.1f}, {high:.1f}]")


def profit(q, dem):
    sold = np.minimum(q, dem)
    return price * sold - cost * q + salvage * (q - sold)


qs = np.arange(0, int(high) + 2)
grid = np.linspace(low, high, 401)                                   # possible demands in the range
table = np.array([[profit(q, x) for x in grid] for q in qs])
regret = np.array([[profit(x, x) - profit(q, x) for x in grid] for q in qs])
rules = {"max-min profit (the worst case)": qs[table.min(axis=1).argmax()],
         "min-max regret": qs[regret.max(axis=1).argmin()],
         "the middle of the range": round((low + high) / 2),
         "the forecast": round(fc),
         "newsvendor on the history (Day 51)": np.ceil(q_empirical(fc * hist, critical_ratio(price, cost, salvage)))}
dem = fc * hist
rows = [(name, q, round(profit(q, low), 2), round(regret[int(q)].max(), 2), round(profit(q, dem).mean(), 2))
        for name, q in rules.items()]
print(pd.DataFrame(rows, columns=["rule", "logs", "profit_if_lowest", "worst_regret", "expected_on_history"])
      .to_string(index=False))
r = critical_ratio(price, cost, salvage)
print(f"min-max regret formula: low + (high - low) x critical ratio = {low + (high - low) * r:.1f}")
```

```text
an average store: forecast 48.0 logs; demand somewhere in [20.8, 93.8]
                              rule  logs  profit_if_lowest  worst_regret  expected_on_history
   max-min profit (the worst case)  21.0            425.29       1492.38               430.46
                    min-max regret  88.0            297.99        127.74               938.67
           the middle of the range  57.0            356.89        754.38               923.40
                      the forecast  48.0            373.99        938.88               866.48
newsvendor on the history (Day 51)  76.0            320.79        364.88               949.50
min-max regret formula: low + (high - low) x critical ratio = 87.6
```

- **Max-min profit** orders 21 logs, the lowest demand. It protects the worst case and earns **half** the newsvendor's profit on the real history. Pure worst-case thinking is too timid when shortages are what cost money.
- **Min-max regret** orders 88 logs. For a newsvendor with demand in $$[a, b]$$, the answer has a formula, $$a + (b - a) \times$$ the critical ratio. It earns $939, within 1.1% of the newsvendor's $950, **without any probabilities**.
- Regret is the robust criterion to use when the downside of the worst case isn't the real concern. Here, missed sales hurt more than leftovers.

*Note:* "If the history is unreliable, order 88 logs per average store (the min-max regret rule). It is almost as good as the newsvendor with the history (76), and much better than planning for the worst case (21)."

</details>

## Commit

```bash
git add .
git commit -m "Day 57: robust optimization"
git push
```

Then run the tests: `tests/test_uncertainty_models.py` checks Days 54–57. It covers:

- sampling scenarios, and error scaling;
- EEV ≤ RP ≤ WS;
- the individual chance constraint kept for each product;
- Γ = 0 equal to the nominal model, and the box set keeping every product 10% long within the ovens.

Add `"day54"` to `"day57"` to the folders in `conftest.py`.

```python
"""Known answers for Days 54-57: SAA samples, VSS and EVPI, chance constraints, robust budgets."""
import numpy as np
import pyomo.environ as pyo
import pytest

from bakery import build_model, load_data, solve
from chance import add_service, covered
from robust import build_robust
from saa import sample_demand
from two_stage import build_two_stage, load_two_stage
from values import four_numbers, scale_demand


def test_sample_demand():
    demand = {("A", 0): 1.0, ("A", 1): 2.0, ("A", 2): 3.0}
    dm, wk = sample_demand(demand, ["A"], 4, [2, 2, 0, 1])
    assert [dm["A", k] for k in range(4)] == [3.0, 3.0, 1.0, 2.0] and wk == ["2", "2", "0", "1"]
    assert scale_demand(demand, {"A": 2.0}, 0.0) == {k: 2.0 for k in demand}      # no error: every week = forecast


def test_value_ordering(con):
    d, econ, st, demand, weeks = load_two_stage(con)
    r = four_numbers(d, econ, st, demand, weeks)
    assert r["EEV"] <= r["RP"] + 1e-6 <= r["WS"] + 2e-6                  # a stochastic plan is never worse; foresight never hurts


def test_each_product_covered(con):
    d, econ, st, demand, weeks = load_two_stage(con)
    core = ["BK-SD", "BK-CR"]
    m = build_two_stage(d, econ, st, demand, weeks)
    add_service(m, demand, core, 0.90, joint=False)
    solve(m)
    _, per = covered({p: pyo.value(m.x[p]) for p in m.P}, demand, core, len(weeks))
    assert all(per[p].mean() >= 0.90 for p in core)


def test_robust_budget(con):
    d = load_data(con)
    nominal = build_model(d)
    solve(nominal)
    eps = {p: 0.1 for p in d["products"]}
    m0 = build_robust(d, eps, 0)
    solve(m0)
    assert pyo.value(m0.total_profit) == pytest.approx(pyo.value(nominal.total_profit))   # gamma 0 = nominal
    m = build_robust(d, eps, len(d["products"]))
    solve(m)
    used = sum(1.1 * d["use"].get(("oven_hours", p), 0) * pyo.value(m.x[p]) for p in m.P)
    assert used <= d["capacity"]["oven_hours"] + 1e-6                     # every product 10% long still fits
```

```text
84 passed
```

### Day 57 checklist

- [ ] I can write a robust counterpart for a budget-of-uncertainty set.
- [ ] I can draw the price of robustness and price it against the cost of a violation.
- [ ] I know robust policies inherit the quirks of their uncertainty sets.
- [ ] I can choose between stochastic and robust models for a decision.
- [ ] I can use max-min profit and min-max regret, and know when each makes sense.
