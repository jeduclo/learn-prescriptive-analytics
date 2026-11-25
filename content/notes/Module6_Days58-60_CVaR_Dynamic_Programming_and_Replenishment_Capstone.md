# DAY 58 — Risk Measures: CVaR

Every model since Day 53 maximized **expected** profit. A bakery that earns $112,600 on average but has a few terrible weeks may prefer a plan that earns a little less on average and much more in its bad weeks. To plan for that, you need a way to measure the bad weeks:

- **Value at risk (VaR)**: how bad the worst weeks get.
- **Conditional value at risk (CVaR)**: how bad they are on average. CVaR fits into a linear program.

Today you measure both, put CVaR into the two-stage model, trace the trade-off between the average and the tail, and check that it holds a year later.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 58.1 VaR and CVaR of a plan |
| 0:45 – 1:45 | 58.2 CVaR in the model |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 58.3 The mean-CVaR frontier |
| 2:45 – 3:45 | 58.4 First year in, second year out; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: the CFO's risk limit |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Days 59–60:

- `mdp_settings`: olive oil from an overnight wholesaler.
- `replenishment_scenarios`: stress tests for the capstone.

## 58.1 VaR and CVaR of a plan

For a level α (say 5%):

- **VaR** is the profit the worst α of weeks fall below.
- **CVaR** is the **average** profit in those worst α of weeks.

VaR says where the tail starts; CVaR says how deep it is. CVaR also has the properties a risk measure should have: diversifying never makes it look worse, and VaR doesn't always pass that test. Create `day58/cvar.py`:

```python
"""Day 58: risk measures. Value at risk (VaR) and conditional value at risk (CVaR) of weekly profit, and CVaR inside
the two-stage bakery model (Rockafellar and Uryasev's linear formulation).

Import it:  from cvar import var_cvar, add_cvar
"""
import numpy as np
import pyomo.environ as pyo


def var_cvar(profits, alpha=0.05, weights=None):
    """VaR: the profit the worst alpha of weeks fall below. CVaR: the average profit in those worst alpha of weeks
    (the fraction of a week at the boundary counts in part). Equal weights unless given."""
    x = np.asarray(profits, dtype=float)
    w = np.full(len(x), 1 / len(x)) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum = np.cumsum(w)
    k = int(np.searchsorted(cum, alpha - 1e-12))                     # the week where the worst alpha ends
    var = x[k]
    take = np.minimum(w, np.maximum(alpha - (cum - w), 0))           # how much of each week is in the tail
    return var, float((take * x).sum() / alpha)


def add_cvar(m, alpha=0.05, weight=0.5):
    """Replace a two-stage model's objective by (1 - weight) x expected profit + weight x CVaR_alpha(profit).
    CVaR_alpha = max over eta of  eta - (1 / alpha) E[(eta - profit)+]  (linear with one extra variable per week)."""
    n = len(m.W)
    m.eta = pyo.Var()                                                 # at the optimum: the VaR
    m.below = pyo.Var(m.W, within=pyo.NonNegativeReals)               # how far each week falls below eta
    m.tail = pyo.Constraint(m.W, rule=lambda m, w: m.below[w] >= m.eta - m.week_profit[w])
    m.cvar = pyo.Expression(expr=m.eta - sum(m.below[w] for w in m.W) / (alpha * n))
    m.mean = pyo.Expression(expr=sum(m.week_profit[w] for w in m.W) / n)
    m.expected_profit.deactivate()
    m.risk_objective = pyo.Objective(expr=(1 - weight) * m.mean + weight * m.cvar, sense=pyo.maximize)
    return m
```

Then `day58/run_cvar.py`:

```python
# ---------- 58.1 VaR and CVaR of a plan ----------
print("--- var and cvar")
rp = build_two_stage(d, econ, st, demand, weeks)
solve(rp)
prof = scenario_profits(rp)
for a in (0.05, 0.10, 0.20):
    v, c = var_cvar(prof, a)
    print(f"alpha {a:.0%}: VaR ${v:,.2f} (the worst {a:.0%} of weeks earn less), CVaR ${c:,.2f} (their average)")
worst = pd.Series(prof, index=weeks).nsmallest(6)
print("the worst weeks: " + ", ".join(f"{w} ${p:,.0f}" for w, p in worst.items()))
```

```text
--- var and cvar
alpha 5%: VaR $111,962.72 (the worst 5% of weeks earn less), CVaR $98,574.43 (their average)
alpha 10%: VaR $112,636.39 (the worst 10% of weeks earn less), CVaR $105,481.90 (their average)
alpha 20%: VaR $113,109.97 (the worst 20% of weeks earn less), CVaR $109,179.07 (their average)
the worst weeks: 2025-01-06 $79,043, 2026-01-05 $83,584, 2026-07-06 $106,916, 2025-07-07 $109,635, 2025-06-30 $111,687, 2025-05-19 $111,963
```

The VaR at 5% is $111,963, barely below the average: on most weeks the plan does fine. But two weeks, the first weeks of January 2025 and 2026, earn **$79,000 and $84,000**, and they drag the CVaR down to **$98,574**. VaR doesn't see how bad those two weeks are; CVaR does.

## 58.2 CVaR in the model

Rockafellar and Uryasev showed that CVaR is the optimum of a small problem:

$$\text{CVaR}_\alpha = \max_\eta \Big\{ \eta - \frac{1}{\alpha} \sum_w \pi_w (\eta - \text{profit}_w)^+ \Big\}$$

The positive part $$(\cdot)^+$$ becomes one variable per week, $$u_w \ge \eta - \text{profit}_w,\ u_w \ge 0$$, so the problem is linear and fits inside the two-stage model. At the optimum, $$\eta$$ is the VaR. The objective weights expected profit against CVaR:

$$\max\ (1 - w)\,\mathbb{E}[\text{profit}] + w\, \text{CVaR}_{5\%}[\text{profit}]$$

```python
# ---------- 58.2 CVaR in the model ----------
print("--- cvar model")


def solve_risk(weight, alpha=0.05, dem=demand, wk=weeks):
    m = add_cvar(build_two_stage(d, econ, st, dem, wk), alpha, weight)
    solve(m)
    return m


m = solve_risk(1.0)
print(f"maximize CVaR 5%: eta = ${pyo.value(m.eta):,.2f} (the VaR), CVaR ${pyo.value(m.cvar):,.2f}, "
      f"mean ${pyo.value(m.mean):,.2f}")
print(f"check from its weeks: VaR, CVaR = " + ", ".join(f"${x:,.2f}" for x in var_cvar(scenario_profits(m), 0.05)))
```

```text
--- cvar model
maximize CVaR 5%: eta = $107,461.63 (the VaR), CVaR $103,807.19, mean $107,278.91
check from its weeks: VaR, CVaR = $107,461.63, $103,807.19
```

The model's CVaR matches the one computed from its own weeks: the formulation is right.

## 58.3 The mean-CVaR frontier

```python
# ---------- 58.3 The mean-CVaR frontier ----------
print("--- frontier")
rows, plans = [], {}
for weight in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
    m = solve_risk(weight)
    p = scenario_profits(m)
    plans[weight] = {q: max(0.0, pyo.value(m.x[q])) for q in m.P}
    rows.append((weight, round(p.mean(), 2), round(var_cvar(p, 0.05)[1], 2), round(p.min(), 2),
                 round(sum(plans[weight].values()), 1)))
front = pd.DataFrame(rows, columns=["cvar_weight", "mean", "cvar_5", "worst_week", "batches_planned"])
print(front.to_string(index=False))
change = pd.DataFrame({"risk_neutral": pd.Series(plans[0.0]), "max_cvar": pd.Series(plans[1.0])}).round(2)
change["change"] = (change.max_cvar - change.risk_neutral).round(2)
print("largest changes, risk-neutral -> max CVaR:")
print(change[change.change.abs() > 1].sort_values("change").to_string())
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(front.cvar_5, front["mean"], "o-", color="#3b82f6")
for r in front.itertuples():
    ax.annotate(f"w={r.cvar_weight:g}", (r.cvar_5, r.mean), textcoords="offset points", xytext=(5, 4), fontsize=7,
                color="#6b7280")
ax.set(xlabel="CVaR 5%: average of the worst 5% of weeks ($)", ylabel="expected profit a week ($)",
       title="The mean-CVaR frontier of the bakery's week")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/day58_cvar.png", dpi=150)
print("saved figures/day58_cvar.png")
```

```text
--- frontier
 cvar_weight      mean    cvar_5  worst_week  batches_planned
        0.00 112600.11  98574.43    79042.69            429.2
        0.10 112510.87 100537.09    83038.36            426.2
        0.25 112340.22 101545.08    84484.10            425.7
        0.50 111967.78 102179.33    85095.31            423.7
        0.75 109657.44 103597.52    91938.58            414.9
        0.90 108897.13 103766.97    93499.06            413.7
        1.00 107278.91 103807.19    96206.33            409.6
largest changes, risk-neutral -> max CVaR:
       risk_neutral  max_cvar  change
BK-03         21.37      1.46  -19.91
BK-18         28.40     19.73   -8.67
BK-13         26.40     17.76   -8.64
BK-14         24.86     17.10   -7.76
BK-23         19.14     13.15   -5.99
BK-29         18.08     12.50   -5.58
BK-CR         30.11     24.65   -5.46
BK-12         14.50      9.87   -4.63
BK-WW         44.11     39.51   -4.60
BK-25         14.99     10.52   -4.47
BK-16          8.76      5.92   -2.84
BK-10         19.71     17.62   -2.09
BK-04         17.81     15.86   -1.95
BK-22          7.40      5.61   -1.79
BK-07          7.79     12.38    4.59
BK-15          0.00      4.99    4.99
BK-21          0.00     11.23   11.23
BK-20          0.00     14.94   14.94
BK-08          0.00     16.24   16.24
BK-24          0.00     16.66   16.66
saved figures/day58_cvar.png
```

![The mean-CVaR frontier](figures/day58_cvar.png)

- The frontier is **steep at first**: a weight of 0.5 gives up $632 a week on average and gains **$3,605** in the worst 5% of weeks. Most of the tail can be protected cheaply. The last bit costs a lot: going from 0.5 to CVaR alone gives up another $4,689 a week for $1,628 more CVaR.
- **How the plan protects itself:** it bakes less of the products that fall furthest in January (rye BK-03, croissants, BK-13, BK-14, BK-18), and **spreads** into products the risk-neutral plan didn't make at all (BK-08, BK-20, BK-21, BK-24). This is diversification, as in a portfolio.
- Which point to choose is a judgement about risk appetite, not a calculation. The frontier shows management what each choice costs.

## 58.4 First year in, second year out

Each year in the history has one Christmas and one January. Plan on the first year, then test on the second:

```python
# ---------- 58.4 First year in, second year out ----------
print("--- year out")
first, second = list(range(52)), list(range(52, len(weeks)))
pick = lambda idx: ({(p, k): demand[p, w] for p in d["products"] for k, w in enumerate(idx)}, [weeks[w] for w in idx])
d1, w1 = pick(first)
d2, w2 = pick(second)
rows = []
for weight in (0.0, 0.5, 1.0):
    m = solve_risk(weight, dem=d1, wk=w1)
    plan = {q: max(0.0, pyo.value(m.x[q])) for q in m.P}
    ev = build_two_stage(d, econ, st, d2, w2, plan=plan)
    solve(ev)
    p_in, p_out = scenario_profits(m), scenario_profits(ev)
    rows.append((weight, round(p_in.mean(), 2), round(var_cvar(p_in, 0.05)[1], 2), round(p_out.mean(), 2),
                 round(var_cvar(p_out, 0.05)[1], 2)))
print(f"trained on {w1[0]}..{w1[-1]}, tested on {w2[0]}..{w2[-1]}")
print(pd.DataFrame(rows, columns=["cvar_weight", "mean_train", "cvar_train", "mean_test", "cvar_test"]).to_string(index=False))
```

```text
--- year out
trained on 2024-12-02..2025-11-24, tested on 2025-12-01..2026-11-16
 cvar_weight  mean_train  cvar_train  mean_test  cvar_test
         0.0   112588.33    98366.96  112598.93   98692.36
         0.5   112006.62   101807.24  112021.45  102260.56
         1.0   107772.35   103298.80  107833.23  103311.92
```

The trade-off holds a year later: the risk-averse plans keep their better CVaR on weeks they have never seen, at the same cost in mean. The tail has repeating causes here (the January drop happens every year), so protecting it generalizes. Tails with **unique** causes would not; check before trusting a CVaR plan.

> **Decision it supports:** plans that trade a little average profit for much better bad weeks, at a price everyone can see.
>
> **Other decisions you can support the same way:** portfolio allocation (CVaR is a standard risk limit) · capacity plans that must survive a bad season · energy contracts against price spikes · insurance purchases.

### The planted bug

`day58/bug_tail.py` adds CVaR to the model by hand. Run it.

```python
alpha, n = 0.05, len(weeks)
m.eta = pyo.Var()
m.below = pyo.Var(m.W, within=pyo.NonNegativeReals)
m.tail = pyo.Constraint(m.W, rule=lambda m, w: m.below[w] >= m.eta - m.week_profit[w])
m.expected_profit.deactivate()
m.risk = pyo.Objective(expr=m.eta - sum(m.below[w] for w in m.W) / n, sense=pyo.maximize)
solve(m)
p = scenario_profits(m)
print(f"the model's 'CVaR 5%': ${pyo.value(m.risk):,.2f} (eta ${pyo.value(m.eta):,.2f})")
print(f"the plan's mean ${p.mean():,.2f}; its real CVaR 5% ${var_cvar(p, 0.05)[1]:,.2f}; worst week ${p.min():,.2f}")
```

<details><summary>What happens, and why</summary>

```text
the model's 'CVaR 5%': $112,600.11 (eta $114,269.81)
the plan's mean $112,600.11; its real CVaR 5% $98,574.43; worst week $79,042.69
```

The "CVaR" equals the **mean**, $112,600, and the plan is the risk-neutral one: its real CVaR is $98,574. The objective forgot to divide by α:

- **Without $$1/\alpha$$**, raising $$\eta$$ always pays until $$\eta$$ reaches the best week, so the formula collapses to the expected profit.
- **With $$1/\alpha$$**, each unit of $$\eta$$ costs $$1/\alpha$$ times the share of weeks below it. $$\eta$$ then stops at the α-quantile, the VaR.

Check a risk model by recomputing its risk measure from its own scenario profits, as 58.2 does.

</details>

## Mini-project: the CFO's risk limit

The CFO sets a limit: "the worst 5% of weeks must average at least $X". Maximize expected profit subject to $$\text{CVaR}_{5\%} \ge X$$, for X = $100,000, $102,000, $103,000 and $103,500. What does each limit cost a year? Does it still hold in the second year if the plan is made on the first?

<details><summary>Solution</summary>

```python
"""Day 58 mini-project: the CFO sets a risk limit: "the worst 5% of weeks must average at least $X". Maximize expected
profit subject to CVaR_5% >= X for X = $100,000, $102,000, $103,000 and $103,500. What does each limit cost a year?
Does the limit still hold in the second year if the plan is made on the first?

Run from the repository root (after day06/load_tables.py):  python day58/mini_project_cfo.py
"""
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

for folder in ("day06", "day42", "day44", "day53", "day58"):
    sys.path.insert(0, folder)
from bakery import solve
from cvar import add_cvar, var_cvar
from two_stage import build_two_stage, load_two_stage, scenario_profits

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
d, econ, st, demand, weeks = load_two_stage(con)
idx1, idx2 = range(52), range(52, len(weeks))
pick = lambda idx: ({(p, k): demand[p, w] for p in d["products"] for k, w in enumerate(idx)}, [weeks[w] for w in idx])
rows, base = [], None
for limit in (None, 100_000, 102_000, 103_000, 103_500):
    m = add_cvar(build_two_stage(d, econ, st, demand, weeks), 0.05, weight=0.0)     # objective: expected profit
    if limit:
        m.cfo_limit = pyo.Constraint(expr=m.cvar >= limit)
    solve(m)
    p = scenario_profits(m)
    base = base or p.mean()
    # the same limit, planned on the first year and checked on the second
    d1, w1 = pick(idx1)
    m1 = add_cvar(build_two_stage(d, econ, st, d1, w1), 0.05, weight=0.0)
    if limit:
        m1.cfo_limit = pyo.Constraint(expr=m1.cvar >= limit)
    try:
        solve(m1)
        d2, w2 = pick(idx2)
        ev = build_two_stage(d, econ, st, d2, w2, plan={q: max(0.0, pyo.value(m1.x[q])) for q in m1.P})
        solve(ev)
        second = round(var_cvar(scenario_profits(ev), 0.05)[1], 2)
    except RuntimeError:
        second = "no plan meets it in year 1"
    rows.append(("none" if limit is None else f"${limit:,}", round(p.mean(), 2), round(var_cvar(p, 0.05)[1], 2),
                 round((base - p.mean()) * 52), second))
print(pd.DataFrame(rows, columns=["cvar_limit", "expected_profit", "cvar_5", "cost_a_year", "cvar_5_second_year"])
      .to_string(index=False))
```

```text
cvar_limit  expected_profit    cvar_5  cost_a_year         cvar_5_second_year
      none        112600.11  98574.43            0                   98692.36
  $100,000        112548.37 100000.00         2690                   100361.2
  $102,000        112118.18 102000.00        25060                  102502.19
  $103,000        110776.65 103000.00        94820                  103277.06
  $103,500        109866.47 103500.00       142149 no plan meets it in year 1
```

- A **$100,000** limit costs only $2,690 a year, and **$102,000** costs $25,000. Both hold in the second year.
- **$103,000** costs $95,000 a year. **$103,500** costs $142,000, and no plan meets it using the first year alone: it is at the edge of what is possible.
- A CVaR **constraint** is easier to agree on than a weight. Executives can set a floor in dollars; the model finds the cheapest way to keep it.

*Memo to the CFO:* "A floor of $102,000 on the average of the worst 5% of weeks (today: $98,600) costs about $25,000 a year in expected profit and held up on a year of data the plan hadn't seen. Floors above $103,000 cost $95,000 a year or more."

</details>

## Commit

```bash
git add .
git commit -m "Day 58: risk measures and CVaR"
git push
```

### Day 58 checklist

- [ ] I can compute VaR and CVaR from scenario profits, and explain why CVaR sees the depth of the tail.
- [ ] I can add CVaR to an LP with Rockafellar and Uryasev's formulation.
- [ ] I can draw a mean-CVaR frontier and describe how the plan diversifies.
- [ ] I can impose a CVaR floor as a constraint, and check it out of sample.
- [ ] I verify a risk model by recomputing its measure from its scenarios.

---

# DAY 59 — MDPs, Dynamic Programming and Reinforcement Learning

So far each decision was made once (a week's plan), or by a formula repeated (a reorder point). Many decisions are made **again and again**, and each one changes the situation for the next: today's order is tomorrow's stock. A **Markov decision process** (MDP) models that:

- **States**: what you know now.
- **Actions**: what you can do.
- **Transitions**: what happens next, with probabilities.
- **Costs**: what each step costs.

**Dynamic programming** solves an MDP exactly when the model is known. **Reinforcement learning** learns a policy from experience when it isn't. Today you do both for olive oil at DC-MTL, and see why a known model beats learning when you have one.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 59.1 The MDP |
| 0:45 – 1:45 | 59.2 Value iteration |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 59.3 Against the formulas |
| 2:45 – 3:45 | 59.4 Reinforcement learning: Q-learning; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: what is the order cost worth? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 59.1 The MDP

`mdp_settings`: olive oil (AM-04) can also be bought from an overnight wholesaler. The DC orders in the evening and the cases arrive before the next day's demand, so the lead time doesn't need a place in the state.

| Element | Here |
|---|---|
| State $$x$$ | cases on hand in the evening, 0 to 600 |
| Action $$a$$ | cases to order tonight |
| Transition | tomorrow evening: $$\max(x + a - D, 0)$$, with $$D$$ drawn from the 182 days of history |
| Cost of a day | $150 if it orders, $15.10 per case short (lost margin), $0.0345 per case left over (holding) |

The **Bellman equation** for the average cost per day $$g$$ and the relative values $$v$$:

$$g + v(x) = \min_{a \ge 0}\Big\{ K\,[a > 0] + \mathbb{E}\big[c(x + a, D) + v(\max(x + a - D, 0))\big] \Big\}$$

The value of being at $$x$$ is the best action's cost today plus the value of where it leads. Create `day59/mdp.py`:

```python
"""Day 59: an inventory Markov decision process for one DC item. Each evening the DC sees its stock x and orders a
cases from an overnight wholesaler; they arrive before the next day's demand. Unmet demand is lost.
Cost of a day: K if it orders + lost margin m per case short + holding h per case left at the end of the day.

Import it:  from mdp import load_mdp, value_iteration, policy_cost, q_learning
"""
import numpy as np


def load_mdp(con):
    s = dict(con.sql("SELECT * FROM mdp_settings").fetchall())
    item, cap = s["item_id"], int(s["max_stock_cases"])
    i = con.sql(f"SELECT cost_per_case, margin_per_case FROM dc_items WHERE item_id = '{item}'").fetchall()[0]
    st = dict(con.sql("SELECT * FROM inventory_settings").fetchall())
    hist = con.sql(f"SELECT cases FROM dc_daily_demand WHERE item_id = '{item}'").df().cases.to_numpy()
    pmf = np.bincount(hist) / len(hist)                               # demand distribution: the history itself
    return {"item": item, "cap": cap, "step": int(s["order_step_cases"]), "pmf": pmf, "history": hist,
            "K": float(st["order_cost_per_po"]), "m": float(i[1]),
            "h": float(i[0]) * float(st["holding_rate_per_year"]) / float(st["days_per_year"])}


def day_cost_and_next(p, pmf=None):
    """For every stock level y after delivery (0..cap): the expected cost of the day (lost margin + holding) and the
    transition matrix to the next evening's stock max(y - D, 0)."""
    pmf = p["pmf"] if pmf is None else pmf
    n, dem = p["cap"] + 1, np.arange(len(pmf))
    cost, trans = np.zeros(n), np.zeros((n, n))
    for y in range(n):
        short, left = np.maximum(dem - y, 0), np.maximum(y - dem, 0)
        cost[y] = pmf @ (p["m"] * short + p["h"] * left)
        np.add.at(trans[y], left, pmf)
    return cost, trans


def value_iteration(p, tol=1e-6, max_iter=20_000, pmf=None):
    """Relative value iteration for the average cost a day. Returns the order-up-to level for each stock x
    (the policy), the average cost a day, and the number of iterations."""
    cost, trans = day_cost_and_next(p, pmf)
    n = p["cap"] + 1
    v = np.zeros(n)
    for it in range(max_iter):
        g = cost + trans @ v                                          # value of starting the day at level y
        best_up = np.minimum.accumulate(g[::-1])[::-1]                # the best y >= x
        arg_up = np.array([x + int(np.argmin(g[x:])) for x in range(n)])
        new = np.minimum(g, p["K"] + best_up)                         # don't order, or order up to the best level
        avg = new[0] - v[0]
        new -= new[0]
        if np.max(np.abs(new - v)) < tol:
            v = new
            break
        v = new
    order_up = np.where(p["K"] + best_up < g - 1e-9, arg_up, np.arange(n))
    return order_up, avg, it + 1


def policy_cost(p, order_up, pmf=None):
    """Exact average cost a day of a stationary policy (order up to order_up[x] at stock x): the Markov chain's
    stationary distribution. Also: orders a year and fill rate."""
    pmf = p["pmf"] if pmf is None else pmf
    cost, trans = day_cost_and_next(p, pmf)
    n = p["cap"] + 1
    y = np.asarray(order_up)
    chain = trans[y]                                                  # evening x -> next evening, via y = order_up[x]
    vals, vecs = np.linalg.eig(chain.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    pi = pi / pi.sum()
    orders = pi @ (y > np.arange(n))
    dem = np.arange(len(pmf))
    short = np.array([pmf @ np.maximum(dem - yy, 0) for yy in y])
    return {"cost_a_day": float(pi @ (cost[y] + p["K"] * (y > np.arange(n)))), "orders_a_year": float(orders * 365),
            "fill_rate": float(1 - (pi @ short) / (pmf @ dem)), "avg_stock": float(pi @ y)}


def q_learning(p, episodes, rng, days=365, gamma=0.999, alpha=0.05, eps=0.1, check_every=None):
    """Tabular Q-learning from simulated days (demand resampled from history). States are stock levels in buckets
    of `step` cases; actions: order 0, step, 2 step, ... cases. It learns costs to go (minimize), with exploration
    and learning rates that shrink as experience grows. Returns the greedy policy (order-up-to level for every
    stock level) and a learning curve: the exact cost a day of the greedy policy every `check_every` years."""
    step = p["step"]
    buckets = p["cap"] // step + 1
    actions = np.arange(buckets) * step
    q = np.zeros((buckets, len(actions)))
    allowed = np.array([[b * step + a <= p["cap"] for a in actions] for b in range(buckets)])
    q[~allowed] = np.inf
    curve, x = [], 0
    for ep in range(episodes):
        explore, rate = max(0.01, eps * (1 - ep / episodes)), alpha / (1 + ep / 50)
        for _ in range(days):
            b = x // step
            ok = np.flatnonzero(allowed[b])
            k = rng.choice(ok) if rng.random() < explore else ok[np.argmin(q[b, ok])]
            y = x + actions[k]
            d = p["history"][rng.integers(len(p["history"]))]
            nxt = max(y - d, 0)
            c = (p["K"] if k > 0 else 0.0) + p["m"] * max(d - y, 0) + p["h"] * nxt
            q[b, k] += rate * (c + gamma * q[nxt // step].min() - q[b, k])
            x = nxt
        if check_every and (ep + 1) % check_every == 0:
            curve.append((ep + 1, policy_cost(p, greedy(q, actions, p))["cost_a_day"]))
    return greedy(q, actions, p), curve


def greedy(q, actions, p):
    up = []
    for x in range(p["cap"] + 1):
        a = actions[int(np.argmin(q[x // p["step"]]))]
        up.append(x + a if x + a <= p["cap"] else x)
    return np.array(up)
```

Then `day59/run_mdp.py`:

```python
# ---------- 59.1 The MDP ----------
print("--- the mdp")
mean, sd = p["history"].mean(), p["history"].std(ddof=1)
print(f"{p['item']}: demand {mean:.2f} +/- {sd:.2f} cases a day (0 to {p['history'].max()}); states 0..{p['cap']} cases; "
      f"K ${p['K']:.0f} an order, lost margin ${p['m']:.2f} a case, holding ${p['h']:.4f} a case a day")
```

```text
--- the mdp
AM-04: demand 17.74 +/- 8.72 cases a day (0 to 53); states 0..600 cases; K $150 an order, lost margin $15.10 a case, holding $0.0345 a case a day
```

## 59.2 Value iteration

**Relative value iteration** applies the Bellman equation repeatedly, from any starting values, until the values stop changing. For each level $$y$$ after delivery it computes $$G(y)$$: the day's expected cost plus the expected value of tomorrow. Then at each stock level it chooses between ordering nothing ($$G(x)$$) and paying $$K$$ to reach the best level above $$x$$.

```python
# ---------- 59.2 Value iteration ----------
print("--- value iteration")
order_up, avg, iters = value_iteration(p)
orders_at = np.flatnonzero(order_up > np.arange(len(order_up)))
s_star, S_star = orders_at.max(), int(order_up[0])
print(f"converged in {iters} iterations; average cost ${avg:,.2f} a day")
print(f"the optimal policy: when stock is {s_star} or less, order up to {S_star}; otherwise order nothing  "
      f"(an (s, S) policy, s = {s_star}, S = {S_star})")
print("order-up-to level by evening stock: " + ", ".join(f"{x}->{order_up[x]}" for x in (0, 10, 20, s_star, s_star + 1, 60)))
```

```text
--- value iteration
converged in 1572 iterations; average cost $14.09 a day
the optimal policy: when stock is 27 or less, order up to 415; otherwise order nothing  (an (s, S) policy, s = 27, S = 415)
order-up-to level by evening stock: 0->415, 10->415, 20->415, 27->415, 28->28, 60->60
```

Nobody told the algorithm what shape the policy should have, yet it found an **(s, S) policy**: when stock falls to 27 cases or fewer, order up to 415; otherwise order nothing. Scarf proved in 1960 that with a fixed order cost, (s, S) is optimal for problems like this. Dynamic programming finds the right $$s$$ and $$S$$ without a formula.

## 59.3 Against the formulas

```python
# ---------- 59.3 Against the formulas ----------
print("--- policies compared")
n = p["cap"] + 1
q_eoq = int(round(eoq(mean * 365, p["K"], p["h"] * 365)))
z = stats.norm.ppf(0.98)
policies = {
    "optimal (s, S) by DP": order_up,
    f"EOQ: s = 98% of a day's demand, Q = {q_eoq}": None,
    "order up to the 98% quantile every day": np.maximum(np.arange(n), int(np.quantile(p["history"], 0.98))),
    "order up to the average every day": np.maximum(np.arange(n), int(round(mean))),
}
s_eoq = int(round(mean + z * sd))
policies[f"EOQ: s = 98% of a day's demand, Q = {q_eoq}"] = np.where(np.arange(n) <= s_eoq,
                                                                   np.minimum(np.arange(n) + q_eoq, p["cap"]), np.arange(n))
rows = []
for name, pol in policies.items():
    r = policy_cost(p, pol)
    rows.append((name, round(r["cost_a_day"], 2), round(r["orders_a_year"], 1), round(r["fill_rate"], 4), round(r["avg_stock"], 1)))
print(pd.DataFrame(rows, columns=["policy", "cost_a_day", "orders_a_year", "fill_rate", "avg_stock_after_delivery"])
      .to_string(index=False))
```

```text
--- policies compared
                                 policy  cost_a_day  orders_a_year  fill_rate  avg_stock_after_delivery
                   optimal (s, S) by DP       14.09           16.2     0.9993                     226.7
EOQ: s = 98% of a day's demand, Q = 393       14.24           16.5     0.9999                     233.0
 order up to the 98% quantile every day      152.65          365.0     0.9929                      39.0
      order up to the average every day      199.24          365.0     0.8166                      18.0
```

- The **EOQ policy** from Day 52 costs $14.24 a day, only 1% above the optimum: the classic formulas are excellent here. Dynamic programming earns its keep where formulas don't exist, such as perishable stock, capacity limits or prices that change.
- **Ordering every day** costs ten times as much, because the order cost is paid 365 times a year. Ordering only up to the average every day is worse still: it pays the order cost daily **and** runs short (fill rate 82%).

## 59.4 Reinforcement learning: Q-learning

Suppose nobody knew the demand distribution, only how to act and see what happens. **Q-learning** learns $$Q(x, a)$$, the cost to go after taking action $$a$$ in state $$x$$, from experience. After each simulated day:

$$Q(x, a) \leftarrow Q(x, a) + \text{rate}\,\big[c + \gamma \min_{a'} Q(x', a') - Q(x, a)\big]$$

It explores (a random action 10% of the time at first) and acts greedily otherwise. Stock is grouped in buckets of 20 cases, and orders are multiples of 20, so the table stays small.

```python
# ---------- 59.4 Reinforcement learning: Q-learning ----------
print("--- q-learning")
rng = np.random.default_rng(59)
learned, curve = q_learning(p, 1_000, rng, check_every=50)
c = np.array(curve)
for ep, cost in curve[1::2]:
    print(f"  after {ep:>4} simulated years: greedy policy costs ${cost:,.2f} a day")
print(f"DP optimum ${avg:,.2f}; Q-learning's greedy policy over the last 500 years: median ${np.median(c[10:, 1]):,.2f}, "
      f"best ${c[:, 1].min():,.2f}, worst ${c[10:, 1].max():,.2f}")
print(f"its final policy: orders when stock <= {np.flatnonzero(learned > np.arange(n)).max()}, "
      f"up to {int(learned[0])} from empty")
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(c[:, 0], c[:, 1], "o-", color="#3b82f6", label="Q-learning's greedy policy")
ax.axhline(avg, ls="--", color="#e5484d", label="dynamic programming (optimal)")
ax.set(xlabel="simulated years of experience", ylabel="average cost a day ($)", yscale="log",
       title="Q-learning learns from experience, unevenly")
ax.set_yticks([15, 20, 30, 50, 100, 150], ["15", "20", "30", "50", "100", "150"])
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("figures/day59_qlearning.png", dpi=150)
print("saved figures/day59_qlearning.png")
```

```text
--- q-learning
  after  100 simulated years: greedy policy costs $14.36 a day
  after  200 simulated years: greedy policy costs $14.63 a day
  after  300 simulated years: greedy policy costs $18.73 a day
  after  400 simulated years: greedy policy costs $15.18 a day
  after  500 simulated years: greedy policy costs $142.79 a day
  after  600 simulated years: greedy policy costs $22.65 a day
  after  700 simulated years: greedy policy costs $34.69 a day
  after  800 simulated years: greedy policy costs $16.40 a day
  after  900 simulated years: greedy policy costs $17.46 a day
  after 1000 simulated years: greedy policy costs $43.98 a day
DP optimum $14.09; Q-learning's greedy policy over the last 500 years: median $22.09, best $14.36, worst $142.79
its final policy: orders when stock <= 99, up to 580 from empty
saved figures/day59_qlearning.png
```

![Q-learning against dynamic programming](figures/day59_qlearning.png)

- After 100 simulated years, the greedy policy costs $14.36 a day, close to the optimal $14.09.
- But it **doesn't stay there**. A few states that are rarely visited keep wrong values, and when the greedy choice flips in one of them (say, order nothing when empty), the policy costs $143 a day. Over the last 500 years the median is $22.
- **Dynamic programming used the model and solved it exactly in seconds. Q-learning needed a thousand simulated years and still isn't reliable.**
- When you can write down the model, solve it. Reinforcement learning is for problems whose model is unknown or far too large: then the noise, the tuning and the need for a simulator are the price of having no model. Even then, test the learned policy, as here, before trusting it.

> **Decision it supports:** a decision repeated over time, where each choice shapes the next situation.
>
> **Other decisions you can support the same way:** when to replace a machine or a truck · pricing perishable goods as the sell-by date nears · admitting patients or bookings with limited capacity · when to rebalance a portfolio.

### The planted bug

`day59/bug_mean_demand.py` runs value iteration with the **average** daily demand, "because the whole distribution is too slow". Run it.

```python
mean = int(round(p["history"].mean()))
certain = np.zeros(mean + 1)
certain[mean] = 1.0                                                   # every day sells exactly the average
order_up, avg, _ = value_iteration(p, pmf=certain)
s = np.flatnonzero(order_up > np.arange(len(order_up))).max()
print(f"policy: when stock is {s} or less, order up to {order_up[0]}; the model says ${avg:,.2f} a day")
r = policy_cost(p, order_up)                                          # judged with the real demand distribution
print(f"with real demand: ${r['cost_a_day']:,.2f} a day, fill rate {r['fill_rate']:.4f}")
```

<details><summary>What happens, and why</summary>

```text
policy: when stock is 17 or less, order up to 396; the model says $13.67 a day
with real demand: $14.51 a day, fill rate 0.9964
```

The model promises $13.67 a day, **less than the true optimum** ($14.09). Its policy, judged on real demand, costs $14.51. With demand certain, the model sees no reason for safety stock, so it reorders at 17 cases instead of 27 and runs short on bad days. This is Day 41's **flaw of averages**, inside a dynamic program. A DP is exactly as good as its transition probabilities; replacing them with averages throws away what the method is for.

</details>

## Mini-project: what is the order cost worth?

The $150 cost of placing an order drives the policy. Purchasing could combine olive oil with coffee (same supplier) in one order, roughly halving the order cost per item, or automate ordering. Re-solve the DP for order costs of $0, $25, $75, $150 and $300. How do $$s$$ and $$S$$ move, and what is each reduction worth a year?

<details><summary>Solution</summary>

```python
"""Day 59 mini-project: the cost of placing an order ($150) drives olive oil's policy. Purchasing could combine
olive oil with coffee (same supplier) in one order, roughly halving the order cost per item, or automate ordering.
Re-solve the DP for an order cost of $0, $25, $75, $150 and $300. How do s and S move, and what is each
reduction worth a year?

Run from the repository root (after day06/load_tables.py):  python day59/mini_project_order_cost.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, "day59")
from mdp import load_mdp, policy_cost, value_iteration

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
p = load_mdp(con)
rows = []
for k in (0, 25, 75, 150, 300):
    q = {**p, "K": float(k)}
    order_up, avg, _ = value_iteration(q)
    orders = np.flatnonzero(order_up > np.arange(len(order_up)))
    r = policy_cost(q, order_up)
    rows.append((k, int(orders.max()), int(order_up[0]), round(avg, 2), round(r["orders_a_year"], 1),
                 round(r["avg_stock"], 1), round(r["fill_rate"], 4)))
t = pd.DataFrame(rows, columns=["order_cost", "s", "S", "cost_a_day", "orders_a_year", "avg_stock", "fill_rate"])
t["saving_a_year_vs_150"] = ((t.cost_a_day[t.order_cost == 150].item() - t.cost_a_day) * 365).round(0)
print(t.to_string(index=False))
```

```text
 order_cost  s   S  cost_a_day  orders_a_year  avg_stock  fill_rate  saving_a_year_vs_150
          0 52  53        1.22          365.0       53.0     1.0000                4698.0
         25 32 187        6.23           39.1      115.0     0.9993                2869.0
         75 29 302       10.19           22.8      171.1     0.9993                1424.0
        150 27 415       14.09           16.2      226.7     0.9993                   0.0
        300 25 575       19.63           11.5      305.7     0.9993               -2022.0
```

- **S moves with the order cost** (53 → 575 cases), roughly like EOQ's $$\sqrt{K}$$. **s barely moves** (52 → 25): the reorder point protects against demand, and demand doesn't change.
- **Halving the order cost to $75** (joint ordering with coffee) saves about **$1,400 a year** for olive oil alone, with the same service. Automating to $25 saves $2,900.
- With a free order, the DP orders every day up to 53 cases (a base-stock policy) and never runs short.

*Note to Purchasing:* "Ordering olive oil with coffee in one purchase order would save about $1,400 a year on this item alone; across the six items, Day 60 finds more."

</details>

## Commit

```bash
git add .
git commit -m "Day 59: MDPs, dynamic programming and Q-learning"
git push
```

### Day 59 checklist

- [ ] I can define states, actions, transitions and costs of an MDP, and write its Bellman equation.
- [ ] I can solve an MDP with (relative) value iteration and read the policy's structure.
- [ ] I can evaluate a policy exactly through its stationary distribution.
- [ ] I can train a tabular Q-learning agent, and judge it against the exact solution.
- [ ] I know when to use dynamic programming and when reinforcement learning is worth its cost.

---

# DAY 60 — Capstone: A Replenishment Policy That Holds Up

Module 6's capstone brings the module's ideas to DC-MTL's six ambient items:

- formulas and safety stock (Day 52);
- simulation-optimization (Day 49);
- robust sets (Day 57);
- validation on held-out data (Days 54 and 58);
- stress tests (Day 50).

The question from the DC manager: **which replenishment policy should we use, and will it hold up if things get worse?** Fit every policy on the first four months of history, judge it on the last two, then stress it.

| Time | Block |
|---|---|
| 0:00 – 0:30 | 60.1 The brief and the split |
| 0:30 – 1:30 | 60.2 Five policies, fitted on training data |
| 1:30 – 1:45 | Break |
| 1:45 – 2:45 | 60.3 On the held-out months; 60.4 under stress |
| 2:45 – 3:30 | 60.5 The memo |
| 3:30 – 3:45 | Break |
| 3:45 – 4:30 | Module 6 wrap-up |
| 4:30 – 5:00 | Commit, tag, tests + checklist |

## 60.1 The brief and the split

Create `day60/replenish.py`:

```python
"""Day 60 capstone: replenishment policies for DC-MTL's six ambient items, fitted on the first four months and
judged on the last two, as tested and under stress.

Import it:  from replenish import split_history, fit_policies, simulate_item, evaluate
"""
import numpy as np
import pandas as pd
from scipy import stats

from inventory import eoq, k_for_fill_rate, sigma_lead_time


def split_history(con, train_days=120):
    """Daily demand and purchase-order lead times, split by date: the first train_days days to fit, the rest to test."""
    dem = con.sql("SELECT item_id, demand_date, cases FROM dc_daily_demand ORDER BY demand_date").df()
    cut = sorted(dem.demand_date.unique())[train_days]
    po = con.sql("""SELECT item_id, supplier_id, ordered_on, received_on - ordered_on AS lead FROM purchase_orders""").df()
    part = lambda df, col, train: df[(df[col] < cut) if train else (df[col] >= cut)]
    out = {}
    for name, train in (("train", True), ("test", False)):
        out[name] = {"demand": {i: g.cases.to_numpy() for i, g in part(dem, "demand_date", train).groupby("item_id")},
                     "lead": {i: g.lead.to_numpy() for i, g in part(po, "ordered_on", train).groupby("item_id")}}
    return out, pd.Timestamp(cut).date()


def fit_policies(items, settings, train, sim_opt=None):
    """Five policies per item, all from training data only:
    weekly       today: every 7 days order up to S (98% fill-rate formula), one order per item
    weekly_joint the same S, but one order per supplier each week (items share the order cost)
    sQ_formula   Day 52: EOQ and a 98% fill-rate reorder point
    sQ_tuned     Day 49: EOQ, reorder point searched by simulation (cheapest with simulated fill >= 98%)
    sQ_robust    Day 57: EOQ, reorder point for the longest lead time + sqrt(L) high days"""
    K, fill, R = settings["order_cost_per_po"], settings["target_fill_rate"], int(settings["review_period_days"])
    rows = []
    for item, i in items.iterrows():
        d, l = train["demand"][item], train["lead"][item]
        mean, sd, lm, ls = d.mean(), d.std(ddof=1), l.mean(), l.std(ddof=1)
        q = float(np.round(eoq(mean * 365, K, i.hold_per_case_year)))
        sig = sigma_lead_time(mean, sd, lm, ls)
        s_formula = float(np.round(mean * lm + k_for_fill_rate(fill, q, sig) * sig))
        sig_r = sigma_lead_time(mean, sd, lm + R, ls)
        big_s = float(np.round(mean * (R + lm) + k_for_fill_rate(fill, mean * R, sig_r) * sig_r))
        high = np.quantile(d, 0.90) - mean
        s_robust = float(np.round(mean * l.max() + np.sqrt(l.max()) * high))
        s_tuned = sim_opt(item, q, s_formula) if sim_opt else s_formula
        rows.append((item, i.supplier_id, q, s_formula, s_tuned, s_robust, big_s))
    return pd.DataFrame(rows, columns=["item", "supplier", "Q", "s_formula", "s_tuned", "s_robust", "S_weekly"]).set_index("item")


def simulate_item(dem, lead, rng, days, s=None, q=None, review=None, up_to=None, demand_factor=1.0, extra_lead=0):
    """Day 52's simulation (lost sales), also returning the days on which orders were placed (for joint ordering)."""
    on_hand = up_to if up_to is not None else s + q
    pipeline, order_days = {}, []
    wanted = shipped = stock = 0.0
    for day in range(days):
        on_hand += pipeline.pop(day, 0)
        dd = round(dem[rng.integers(len(dem))] * demand_factor)
        ship = min(dd, on_hand)
        on_hand -= ship
        wanted += dd
        shipped += ship
        stock += on_hand
        position = on_hand + sum(pipeline.values())
        qty = (q if position <= s else 0) if review is None else (max(0, up_to - position) if day % review == 0 else 0)
        if qty > 0:
            arrive = day + int(lead[rng.integers(len(lead))]) + extra_lead
            pipeline[arrive] = pipeline.get(arrive, 0) + qty
            order_days.append(day)
    return {"wanted": wanted, "shipped": shipped, "avg_stock": stock / days, "order_days": order_days}


def evaluate(policies, items, settings, data, policy, reps=20, days=730, seed=60, demand_factor=1.0, late=None, extra=0):
    """Cost a year (holding + ordering + lost margin), fill rate and stock of one policy for all six items.
    Every policy uses the same seed for each item: common random numbers."""
    K = settings["order_cost_per_po"]
    totals = {"holding": 0.0, "ordering": 0.0, "lost_margin": 0.0, "wanted": 0.0, "shipped": 0.0, "stock": 0.0}
    item_fill = {}
    joint_days = {}
    for item, pol in policies.iterrows():
        i = items.loc[item]
        rng = np.random.default_rng([seed, int(item[-2:])])
        ex = extra if late == i.supplier_id else 0
        res = []
        for r in range(reps):
            if policy.startswith("weekly"):
                out = simulate_item(data["demand"][item], data["lead"][item], rng, days, review=int(settings["review_period_days"]),
                                    up_to=pol.S_weekly, demand_factor=demand_factor, extra_lead=ex)
            else:
                out = simulate_item(data["demand"][item], data["lead"][item], rng, days, s=pol[f"s_{policy[3:]}"], q=pol.Q,
                                    demand_factor=demand_factor, extra_lead=ex)
            res.append(out)
            if policy == "weekly_joint":
                joint_days.setdefault((pol.supplier, r), set()).update(out["order_days"])
        year = 365 / days / reps
        wanted, shipped = sum(o["wanted"] for o in res), sum(o["shipped"] for o in res)
        totals["holding"] += np.mean([o["avg_stock"] for o in res]) * i.hold_per_case_year
        totals["lost_margin"] += (wanted - shipped) * year * i.margin_per_case
        if policy != "weekly_joint":
            totals["ordering"] += sum(len(o["order_days"]) for o in res) * year * K
        totals["wanted"] += wanted
        totals["shipped"] += shipped
        totals["stock"] += np.mean([o["avg_stock"] for o in res])
        item_fill[item] = shipped / wanted
    if policy == "weekly_joint":
        totals["ordering"] = sum(len(v) for v in joint_days.values()) * 365 / days / reps * K
    return {"cost_a_year": totals["holding"] + totals["ordering"] + totals["lost_margin"],
            "holding": totals["holding"], "ordering": totals["ordering"], "lost_margin": totals["lost_margin"],
            "fill_rate": totals["shipped"] / totals["wanted"], "worst_item_fill": min(item_fill.values()),
            "stock": totals["stock"]}
```

Then `day60/run_capstone.py`:

```python
# ---------- 60.1 The brief and the split ----------
print("--- split")
data, cut = split_history(con)
for name in ("train", "test"):
    n_days = len(next(iter(data[name]["demand"].values())))
    n_po = sum(len(v) for v in data[name]["lead"].values())
    print(f"{name}: {n_days} days of demand, {n_po} purchase orders")
print(f"split on {cut}")
means = pd.DataFrame({n: {i: v.mean() for i, v in data[n]["demand"].items()} for n in ("train", "test")}).round(2)
means["lead_train"] = {i: v.mean() for i, v in data["train"]["lead"].items()}
means["lead_test"] = {i: v.mean() for i, v in data["test"]["lead"].items()}
print(means.round(2).to_string())
```

```text
--- split
train: 120 days of demand, 105 purchase orders
test: 62 days of demand, 51 purchase orders
split on 2026-09-22
        train    test  lead_train  lead_test
AM-01  120.38  117.34        5.65       6.56
AM-02   86.69   87.73        6.12       5.78
AM-03   25.84   27.39       10.61      10.75
AM-04   17.83   17.55       11.94      11.44
AM-05   42.27   43.31        8.00       8.62
AM-06   63.27   59.34        8.06       8.12
```

Demand changes little between the periods. Lead times move by up to a day: SUP-A's pasta orders took 6.6 days in the test months against 5.7 before. A policy fitted on the past meets a slightly different future, as always.

## 60.2 Five policies, fitted on training data

| Policy | From | Rule |
|---|---|---|
| `weekly` | today | every 7 days, order up to S (98% fill formula), one order per item |
| `weekly_joint` | Day 52's note | the same, but one order per **supplier** each week |
| `sQ_formula` | Day 52 | EOQ and the 98% fill-rate reorder point |
| `sQ_tuned` | Day 49 | EOQ; reorder point searched by simulation on training data (cheapest with fill ≥ 98%) |
| `sQ_robust` | Day 57 | EOQ; reorder point covering the longest lead time + √L high days |

```python
# ---------- 60.2 Five policies, fitted on training data ----------
print("--- policies")
K, fill = settings["order_cost_per_po"], settings["target_fill_rate"]


def tune(item, q, s0):
    """Day 49: try reorder points around the formula's on the training data (same random numbers for each), keep the
    cheapest with simulated fill rate >= 98%."""
    i = items.loc[item]
    best = None
    for s in np.round(s0 * np.arange(0.70, 1.41, 0.05)):
        rng = np.random.default_rng([600, int(item[-2:])])
        res = [simulate_item(data["train"]["demand"][item], data["train"]["lead"][item], rng, 730, s=s, q=q)
               for _ in range(20)]
        f = sum(o["shipped"] for o in res) / sum(o["wanted"] for o in res)
        cost = (np.mean([o["avg_stock"] for o in res]) * i.hold_per_case_year
                + sum(o["wanted"] - o["shipped"] for o in res) / 40 * i.margin_per_case)
        if f >= fill and (best is None or cost < best[0]):
            best = (cost, s)
    return float(best[1]) if best else float(s0)


pol = fit_policies(items, settings, data["train"], sim_opt=tune)
print(pol.to_string())
```

```text
--- policies
      supplier       Q  s_formula  s_tuned  s_robust  S_weekly
item
AM-01    SUP-A  1999.0      729.0    875.0    1109.0    1693.0
AM-02    SUP-A  1291.0      568.0    710.0     793.0    1249.0
AM-03    SUP-B   439.0      350.0    420.0     462.0     577.0
AM-04    SUP-B   394.0      274.0    315.0     350.0     448.0
AM-05    SUP-C   857.0      370.0    463.0     530.0     726.0
AM-06    SUP-C   837.0      609.0    731.0     776.0    1110.0
```

The tuned reorder points are about 20% above the formula's: in simulation, the formula falls short of 98% with lost sales and the real, skewed lead times. The robust points are higher still.

## 60.3 On the held-out months

```python
# ---------- 60.3 On the held-out months ----------
print("--- test")
rows = []
for p in POLICIES:
    r = evaluate(pol, items, settings, data["test"], p)
    rows.append((p, round(r["cost_a_year"]), round(r["holding"]), round(r["ordering"]), round(r["lost_margin"]),
                 round(r["fill_rate"], 4), round(r["worst_item_fill"], 4), round(r["stock"])))
test = pd.DataFrame(rows, columns=["policy", "cost_a_year", "holding", "ordering", "lost_margin", "fill_rate",
                                   "worst_item_fill", "stock_cases"])
print(test.to_string(index=False))
```

```text
--- test
      policy  cost_a_year  holding  ordering  lost_margin  fill_rate  worst_item_fill  stock_cases
      weekly        70224    16258     47250         6716     0.9927           0.9839         2178
weekly_joint        46599    16258     23625         6716     0.9927           0.9839         2178
  sQ_formula        55315    23502     18915        12899     0.9847           0.9720         3300
    sQ_tuned        49122    27856     19125         2141     0.9973           0.9951         3869
   sQ_robust        50308    31027     19155          126     0.9999           0.9997         4369
```

- **Ordering weekly item by item, as today, is the most expensive policy** ($70,200 a year): $47,250 of it is order costs.
- **Joint weekly orders** cut that in half and are the cheapest policy on the test months ($46,600), with the same stock and service.
- **The formula (s, Q) misses its target** on the test months: 97.2% for the worst item, with $12,900 of lost margin. A policy fitted to the past met a future with slower suppliers.
- **The tuned and robust policies** keep every item above 99.5%, for $49,100 and $50,300.

## 60.4 Under stress

`replenishment_scenarios`: demand 20% higher, SUP-B three days later than usual, or both.

```python
# ---------- 60.4 Under stress ----------
print("--- stress")
scen = con.sql("SELECT * FROM replenishment_scenarios").df()
rows = []
for s in scen.itertuples():
    for p in POLICIES:
        r = evaluate(pol, items, settings, data["test"], p, demand_factor=s.demand_factor,
                     late=None if s.late_supplier == "none" else s.late_supplier, extra=int(s.extra_lead_days))
        rows.append((s.scenario, p, round(r["cost_a_year"]), round(r["fill_rate"], 4), round(r["worst_item_fill"], 4)))
stress = pd.DataFrame(rows, columns=["scenario", "policy", "cost_a_year", "fill_rate", "worst_item_fill"])
print(stress.pivot(index="policy", columns="scenario", values="cost_a_year").loc[POLICIES, scen.scenario].to_string())
print(stress.pivot(index="policy", columns="scenario", values="worst_item_fill").loc[POLICIES, scen.scenario].to_string())

fig, axes = plt.subplots(1, len(scen), figsize=(12, 3.4), sharey=True)
colours = dict(zip(POLICIES, ["#9aa4b2", "#6b7280", "#3b82f6", "#22a06b", "#e5484d"]))
for ax, s in zip(axes, scen.scenario):
    g = stress[stress.scenario == s]
    for r in g.itertuples():
        ax.scatter(r.worst_item_fill * 100, r.cost_a_year / 1000, color=colours[r.policy], s=40, label=r.policy)
    ax.axvline(98, ls="--", color="#c0c6d0", lw=1)
    ax.set_title(s.replace("_", " "), fontsize=9)
    ax.set_xlabel("worst item's fill rate (%)", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("cost a year ($ thousands)")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig("figures/day60_policies.png", dpi=150)
print("saved figures/day60_policies.png")
```

```text
--- stress
scenario      as_tested  demand_up_20  sup_b_late_3_days    both
policy
weekly            70224        125218              76997  150411
weekly_joint      46599        101593              53372  126786
sQ_formula        55315         97411              66755  127927
sQ_tuned          49122         64680              51581   82930
sQ_robust         50308         54845              49860   62831
scenario      as_tested  demand_up_20  sup_b_late_3_days    both
policy
weekly           0.9839        0.9100             0.9510  0.8442
weekly_joint     0.9839        0.9100             0.9510  0.8442
sQ_formula       0.9720        0.9282             0.9287  0.8208
sQ_tuned         0.9951        0.9720             0.9801  0.9089
sQ_robust        0.9997        0.9904             0.9937  0.9574
saved figures/day60_policies.png
```

![Policies under stress](figures/day60_policies.png)

- **The weekly policies collapse**: with 20% more demand, the worst item's fill rate falls to 91% and lost margin doubles the cost. Their order-up-to levels have no room for surprises between reviews.
- **The robust policy barely moves**: $62,800 a year with both stresses at once (worst item 95.7%), against $82,900 for the tuned policy and $126,800–150,400 for the weekly ones.
- **The tuned policy** is the best compromise if stress is unlikely. The robust one costs $1,200 a year more as tested and saves $1,700–$20,000 a year if SUP-B's delays or demand get worse.

## 60.5 The memo

<details><summary>A model memo</summary>

*To: the DC-MTL manager and Purchasing. Re: how we reorder ambient items.*

"We tested five reordering policies for our six ambient items. Each was fitted on May to September and judged on September to November, then stressed with 20% more demand and SUP-B three days late.

1. **Stop ordering every item weekly, one order each.** It is our most expensive policy ($70,000 a year), mostly order costs, and it fails under stress (91% fill for the worst item when demand rises 20%).
2. **Reorder each item when its stock reaches a reorder point**, with order quantities from the EOQ. Set the reorder points robustly: cover the longest lead time we have seen, plus a margin for high-demand days. As tested it costs about $50,000 a year with 99.9% fill. With higher demand and SUP-B late, it still fills 96% for the worst item at $63,000, while weekly ordering would cost $127,000–150,000.
3. **Next step: combine orders per supplier.** Joint weekly ordering showed that sharing order costs saves about $23,000 a year. A policy that reorders by reorder point **and** adds other items from the same supplier to the truck when they are close to theirs (a 'can-order' policy) should get both savings.
4. **Watch the suppliers.** SUP-A and SUP-B deliver one to two days later than quoted. Lead times, not demand, are our biggest risk."

</details>

## Module 6 wrap-up

| Day | Idea | Northbound decision |
|---|---|---|
| 51 | The newsvendor | Croissant batches, yule logs, pooling |
| 52 | Inventory policies, safety stock | Reorder points at DC-MTL; a faster supplier? |
| 53 | Two-stage stochastic programs | The bakery's week with a top-up bake; an oven lease |
| 54 | Sample average approximation | How many scenarios; bounds on the gap |
| 55 | VSS and EVPI | What modelling uncertainty and better forecasts are worth |
| 56 | Chance constraints | Service promises; the union's overtime cap |
| 57 | Robust optimization | Oven times that vary; ordering without probabilities |
| 58 | CVaR | The bakery's bad weeks; the CFO's floor |
| 59 | MDPs, DP, Q-learning | Olive oil from an overnight wholesaler |
| 60 | Capstone | A replenishment policy that holds up |

Module 7 moves from uncertainty to **trade-offs**: decision trees and the value of information, several objectives at once, and how to present a choice between options that are each best at something.

## Commit and tag

```bash
git add .
git commit -m "Day 60: capstone, a replenishment policy that holds up"
git tag v6.0-module6
git push
git push --tags
```

Then run the tests: `tests/test_risk_and_dp.py` checks Days 58–60:

- VaR and CVaR on known numbers;
- the CVaR model against its own weeks;
- a small DP that no (s, S) policy beats;
- weekly review ordering every week;
- the train/test split.

Add `"day58"`, `"day59"` and `"day60"` to the folders in `conftest.py`.

```python
"""Known answers for Days 58-60: VaR and CVaR, the CVaR model, dynamic programming, the capstone's simulation."""
import numpy as np
import pyomo.environ as pyo
import pytest

from bakery import solve
from cvar import add_cvar, var_cvar
from mdp import policy_cost, value_iteration
from replenish import simulate_item, split_history
from two_stage import build_two_stage, load_two_stage, scenario_profits


def test_var_cvar():
    v, c = var_cvar(np.arange(1, 101), 0.05)
    assert v == 5 and c == pytest.approx(3.0)                         # worst 5 of 100: 1..5, average 3
    assert var_cvar([10.0, 20.0], 0.25)[1] == 10.0


def test_cvar_model_matches_its_weeks(con):
    d, econ, st, demand, weeks = load_two_stage(con)
    m = add_cvar(build_two_stage(d, econ, st, demand, weeks), 0.05, weight=1.0)
    solve(m)
    assert pyo.value(m.cvar) == pytest.approx(var_cvar(scenario_profits(m), 0.05)[1], abs=1e-4)


def test_dynamic_programming_small():
    p = {"cap": 30, "pmf": np.full(6, 1 / 6), "K": 10.0, "m": 5.0, "h": 0.1, "step": 5}
    order_up, avg, _ = value_iteration(p)
    assert policy_cost(p, order_up)["cost_a_day"] == pytest.approx(avg, abs=1e-4)
    n = np.arange(31)
    for s, S in [(2, 10), (5, 15), (8, 20), (0, 30)]:                  # no (s, S) policy beats the DP
        other = np.where(n <= s, S, n)
        assert policy_cost(p, other)["cost_a_day"] >= avg - 1e-6


def test_weekly_review_orders_weekly():
    out = simulate_item(np.array([5]), np.array([2]), np.random.default_rng(0), 70, review=7, up_to=50)
    assert len(out["order_days"]) == 10 and out["shipped"] == out["wanted"]


def test_split(con):
    data, cut = split_history(con)
    assert len(data["train"]["demand"]["AM-01"]) == 120 and len(data["test"]["demand"]["AM-01"]) == 62
```

```text
89 passed
```

### Day 60 checklist

- [ ] I fit policies on one period and judge them on another.
- [ ] I compare policies by full cost (holding, ordering, lost margin) and by the worst item's service.
- [ ] I stress-test policies and choose one that holds up, at a price I can state.
- [ ] I can combine formulas, simulation-optimization and robust sets in one comparison.
- [ ] I can write a memo that says what to change, what it costs, and what to do next.
