# DAY 21 — Binary Variables and Business Logic

Module 3 is about decisions that are yes or no: lease the ovens or not, make a product this week or not, open a site or not. On Day 13 you answered such questions by trying every combination; on Day 15 you used binary variables without looking inside. Today you put yes/no decisions into the model properly, turn business rules ("only if", "not both", "at most two") into constraints, and see what the solver does differently once a model has them.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 21.1 One yes/no decision |
| 0:45 – 1:45 | 21.2 Four investments in one model |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 21.3 Business rules as constraints |
| 2:45 – 3:45 | 21.4 Relaxations and branch and bound; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: what each budget buys |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Days 21–23: `bakery_investments` (four options for the full bakery, each adding capacity at a weekly cost), `bakery_range_setup` (each product's weekly set-up cost and minimum run, Day 22), `candidate_sites` and `site_store_lanes` (three existing DCs and seven candidate sites, Day 23).

## 21.1 One yes/no decision

A **binary variable** takes the value 0 or 1. It turns a decision into a number the model can multiply: "lease two ovens" becomes $$b \in \{0, 1\}$$, the oven capacity becomes $$600 + 100\,b$$, and the cost becomes $$900\,b$$. When $$b = 0$$ nothing changes; when $$b = 1$$ the bakery has 700 oven hours and pays $900 a week.

A model with binary (or integer) variables is a **mixed-integer program** (MIP): some variables must be whole numbers, others (the batches) may be fractions. In Pyomo the only change is the domain: `within=pyo.Binary`.

## 21.2 Four investments in one model

The plant manager's menu, `bakery_investments`:

```text
                          resource  extra_capacity  cost_per_week
option
lease_ovens             oven_hours             100            900
hire_baker             baker_hours              40           1600
starter_tank     sourdough_starter              12            250
packing_upgrade      packing_hours              60           1400
```

On Day 13 you tried all 8 combinations of three options with a loop. With four options that is 16 solves; with 20 options a million. Instead, one model with one binary per option. Create `day21/investments.py`:

```python
import itertools
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day06")
from bakery import build_model, load_data, solve


def build_with_investments(d, menu, domain=pyo.Binary):
    """The Day 6 model plus one yes/no variable per investment: buy[o] = 1 adds its capacity and its cost."""
    m = build_model(d)
    m.O = pyo.Set(initialize=menu.index.tolist())
    m.buy = pyo.Var(m.O, within=domain)
    for r in m.R:
        options = [o for o in m.O if menu.resource[o] == r]
        m.capacity_limit[r].set_value(
            sum(m.use[r, p] * m.x[p] for p in m.P) <= m.capacity[r] + sum(menu.extra_capacity[o] * m.buy[o] for o in options))
    m.investment_cost = pyo.Expression(expr=sum(menu.cost_per_week[o] * m.buy[o] for o in m.O))
    m.total_profit.deactivate()
    m.net_profit = pyo.Objective(expr=sum(m.profit[p] * m.x[p] for p in m.P) - m.investment_cost, sense=pyo.maximize)
    return m


def bought(m):
    return [o for o in m.O if pyo.value(m.buy[o]) > 0.5]
```

Each capacity constraint gains the extra capacity of the options that add to it; the objective subtracts their cost. `domain` defaults to binary; we'll use other domains in a moment.

```python
if __name__ == "__main__":
    pd.set_option("display.width", 140)
    con = duckdb.connect("northbound.duckdb", read_only=True)
    d = load_data(con)
    menu = con.sql("SELECT * FROM bakery_investments").df().set_index("option")
    print(menu.to_string())
    base = build_model(d)
    solve(base)
    print(f"no investments: ${pyo.value(base.total_profit):,.2f}")

    # ---------- 21.2 All four options in one model ----------
    m = build_with_investments(d, menu)
    solve(m)
    print(f"best: buy {bought(m)}; net profit ${pyo.value(m.net_profit):,.2f}")

    # Check: try all 16 combinations, as on Day 13
    best = None
    for choice in itertools.product([0, 1], repeat=len(menu)):
        e = build_with_investments(d, menu)
        for o, c in zip(menu.index, choice):
            e.buy[o].fix(c)
        solve(e)
        if best is None or pyo.value(e.net_profit) > best[0] + 1e-6:
            best = (pyo.value(e.net_profit), choice)
    print(f"enumeration of 16 combinations: best ${best[0]:,.2f} with {dict(zip(menu.index, best[1]))}")
```

```text
no investments: $108,351.28
best: buy ['lease_ovens', 'hire_baker', 'starter_tank', 'packing_upgrade']; net profit $113,488.16
enumeration of 16 combinations: best $113,488.16 with {'lease_ovens': 1, 'hire_baker': 1, 'starter_tank': 1, 'packing_upgrade': 1}
```

**Buy all four**: the bakery earns $113,488.16 a week after paying for them, **$5,136.88 more than today**. The single MIP gives the same answer as trying all 16 combinations, in one solve.

> **Decision it supports:** which of several investments to make, given that their values depend on each other.
>
> **Other decisions you can support the same way:** which projects to fund · which machines to buy · which marketing campaigns to run.

## 21.3 Business rules as constraints

Binaries make logic expressible as linear constraints. With $$b_A, b_B \in \{0,1\}$$:

| Business rule | Constraint |
|---|---|
| A only if B | $$b_A \le b_B$$ |
| Not both A and B | $$b_A + b_B \le 1$$ |
| At least one of A, B | $$b_A + b_B \ge 1$$ |
| Exactly one of A, B, C | $$b_A + b_B + b_C = 1$$ |
| At most k of the options | $$\sum b \le k$$ |
| A if and only if B | $$b_A = b_B$$ |
| A only if B is **not** chosen | $$b_A \le 1 - b_B$$ |
| Budget | $$\sum \text{cost} \cdot b \le \text{budget}$$ |

Each row is worth checking with all four combinations of 0 and 1: the constraint must allow exactly the combinations the rule allows. Six rules, added one at a time:

```python
    # ---------- 21.3 Business rules as constraints on binaries ----------
    b = lambda m, o: m.buy[o]
    rules = {
        "budget $2,500 a week": lambda m: m.investment_cost <= 2500,
        "at most 2 options": lambda m: sum(m.buy[o] for o in m.O) <= 2,
        "baker only with ovens": lambda m: b(m, "hire_baker") <= b(m, "lease_ovens"),
        "ovens or packing, not both": lambda m: b(m, "lease_ovens") + b(m, "packing_upgrade") <= 1,
        "exactly one option": lambda m: sum(m.buy[o] for o in m.O) == 1,
        "packing only if no new baker": lambda m: b(m, "packing_upgrade") <= 1 - b(m, "hire_baker"),
    }
    rows = []
    for name, rule in rules.items():
        e = build_with_investments(d, menu)
        e.rule = pyo.Constraint(expr=rule(e))
        solve(e)
        rows.append((name, ", ".join(bought(e)), round(pyo.value(e.net_profit), 2)))
    print(pd.DataFrame(rows, columns=["rule", "buy", "net_profit"]).to_string(index=False))
```

```text
                        rule                                                    buy  net_profit
        budget $2,500 a week                              lease_ovens, starter_tank   109623.37
           at most 2 options                              lease_ovens, starter_tank   109623.37
       baker only with ovens lease_ovens, hire_baker, starter_tank, packing_upgrade   113488.16
  ovens or packing, not both                              lease_ovens, starter_tank   109623.37
          exactly one option                                           starter_tank   108706.52
packing only if no new baker             lease_ovens, starter_tank, packing_upgrade   110918.02
```

- **"Baker only with ovens"** doesn't bind: the best plan buys both anyway.
- **A $2,500 budget, "at most 2" and "ovens or packing, not both"** all give the same answer: ovens and starter tank, $109,623.37.
- **"Exactly one"**: the starter tank, the best single option.
- **"Packing only if no new baker"** gives up the baker and keeps the rest: $110,918.02.

Notice what didn't happen: nobody had to write "if this then that" code around the solver. The rules sit in the model, and the solver finds the best plan that obeys them all.

## 21.4 Relaxations and branch and bound

What if we let the binaries take any value between 0 and 1? That is the model's **LP relaxation**:

```python
    # The LP relaxation: let buy take any value between 0 and 1
    r = build_with_investments(d, menu, domain=pyo.UnitInterval)
    solve(r)
    print(f"relaxation: ${pyo.value(r.net_profit):,.2f} with buy = {({o: round(pyo.value(r.buy[o]), 3) + 0.0 for o in r.O})}")
```

```text
relaxation: $114,456.97 with buy = {'lease_ovens': 0.658, 'hire_baker': 1.0, 'starter_tank': 1.0, 'packing_upgrade': 0.528}
```

The relaxation leases 0.658 of the ovens and 0.528 of the packing upgrade: meaningless as a decision, but useful as a **bound**. Every yes/no plan is also a plan of the relaxation, so the relaxation's $114,456.97 is an upper limit on what any combination can earn. The best combination ($113,488.16) is within 0.85% of it.

Rounding the relaxation is not a method: here rounding up happens to give the right answer, but in general rounded solutions can be infeasible or far from the best. MIP solvers use **branch and bound** instead (Day 26 in detail): solve the relaxation; pick a fractional variable (say, ovens = 0.658); solve two new problems, one with ovens = 0 and one with ovens = 1; repeat on each, and drop any branch whose relaxation bound is worse than the best plan already found. The log shows it:

```python
    # ---------- 21.4 What the MIP solver does ----------
    print("--- HiGHS log for the investment model")
    pyo.SolverFactory("highs").solve(build_with_investments(d, menu), tee=True)
```

```text
        Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints |       Work
Src  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time

 J       0       0         0   0.00%   inf             112599.622222      Large        0      0      0         0     0.0s
 S       0       0         0   0.00%   146199.2        113488.16         28.82%        0      0      0         0     0.0s

50.0% inactive integer columns, restarting
Model after restart has 4 rows, 32 cols (2 bin., 0 int., 0 impl., 30 cont., 0 dom.fix.), and 96 nonzeros

         0       0         0   0.00%   114456.974667   113488.16          0.85%        0      0      0         5     0.0s
         0       0         0   0.00%   114456.974667   113488.16          0.85%        0      0      0         5     0.0s
         1       0         1 100.00%   113488.16       113488.16          0.00%        6      1      1        14     0.0s

Solving report
  Status            Optimal
  Primal bound      113488.16
  Dual bound        113488.16
  Gap               0% (tolerance: 0.01%)
```

(Your log's times and thread counts will differ.) Read the columns **BestBound** (the best the relaxation says is possible) and **BestSol** (the best plan found so far). The solver found $113,488.16 almost immediately with a heuristic, then proved it optimal by raising the bound to meet it: **gap 0%**. On big models, the gap is what tells you how good the current plan is while the solver is still working.

### The planted bug

`day21/bug_bounds.py` builds the investment model "with the right bounds". Run it.

```python
con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_data(con)
menu = con.sql("SELECT * FROM bakery_investments").df().set_index("option")
m = build_with_investments(d, menu, domain=pyo.NonNegativeReals)
for o in m.O:
    m.buy[o].setub(1)
solve(m)
print(f"net profit ${pyo.value(m.net_profit):,.2f}")
print({o: round(pyo.value(m.buy[o]), 3) + 0.0 for o in m.O})
```

<details><summary>What happens, and why</summary>

```text
net profit $114,456.97
{'lease_ovens': 0.658, 'hire_baker': 1.0, 'starter_tank': 1.0, 'packing_upgrade': 0.528}
```

`within=NonNegativeReals` with an upper bound of 1 is not binary: it is the relaxation. The model happily leases 0.658 of two ovens and reports a profit no real decision can earn. Bounds of 0 and 1 don't make a variable a yes/no decision; only the domain does. Whenever a model's "decisions" come out as fractions, check the domain first.

</details>

## Mini-project: what each budget buys

Finance asks what each weekly investment budget from $0 to $5,000 would buy, in steps of $500. The plant manager adds a rule: no new baker unless the ovens are leased. Report each budget at which the best choice changes, with what it spends and what it gains.

<details><summary>Solution</summary>

```python
import sys

import duckdb
import pyomo.environ as pyo

sys.path.insert(0, "day06")
sys.path.insert(0, "day21")
from bakery import build_model, load_data, solve
from investments import bought, build_with_investments

con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_data(con)
menu = con.sql("SELECT * FROM bakery_investments").df().set_index("option")
base = build_model(d)
solve(base)
base_profit = pyo.value(base.total_profit)

previous = None
for budget in range(0, 5001, 500):
    m = build_with_investments(d, menu)
    m.budget = pyo.Constraint(expr=m.investment_cost <= budget)
    m.baker_needs_ovens = pyo.Constraint(expr=m.buy["hire_baker"] <= m.buy["lease_ovens"])
    solve(m)
    choice = bought(m)
    if choice != previous:
        spent = pyo.value(m.investment_cost) + 0.0
        gain = pyo.value(m.net_profit) - base_profit
        print(f"budget ${budget:>5,}: buy {choice or 'nothing'}; spends ${spent:,.0f}, net gain ${gain:,.2f} a week")
        previous = choice
```

```text
budget $    0: buy nothing; spends $0, net gain $0.00 a week
budget $  500: buy ['starter_tank']; spends $250, net gain $355.24 a week
budget $1,500: buy ['lease_ovens', 'starter_tank']; spends $1,150, net gain $1,272.09 a week
budget $3,000: buy ['lease_ovens', 'starter_tank', 'packing_upgrade']; spends $2,550, net gain $2,566.74 a week
budget $4,000: buy ['lease_ovens', 'hire_baker', 'packing_upgrade']; spends $3,900, net gain $3,516.80 a week
budget $4,500: buy ['lease_ovens', 'hire_baker', 'starter_tank', 'packing_upgrade']; spends $4,150, net gain $5,136.88 a week
```

The best set does not simply grow with the budget. At $4,000 the plan **drops the starter tank** to afford the baker: ovens + baker + packing gain more than ovens + starter + packing, but the starter tank no longer fits. At $4,500 everything fits. This is why investment choices need a model: a ranking of options by return ("buy the best ones until the money runs out") misses combinations like this.

*Memo to finance:* "Each budget step buys something different. $1,150 a week (ovens and starter tank) returns $1,272; $2,550 returns $2,567; the full $4,150 returns $5,137. The baker only pays together with the ovens and the packing upgrade: approve all three or none of them."

</details>

## Commit

```bash
git add .
git commit -m "Day 21: binary investments and business rules"
git push
```

### Day 21 checklist

- [ ] I can model a yes/no decision with a binary variable that adds capacity and cost.
- [ ] One MIP gives the same answer as enumerating every combination.
- [ ] I can write "only if", "not both", "at least one", "at most k" and budget rules as linear constraints.
- [ ] I know the LP relaxation gives a bound, and why rounding it is not a method.
- [ ] I can read BestBound, BestSol and Gap in a MIP log.
- [ ] I know that bounds of 0 and 1 don't make a variable binary.

---

# DAY 22 — Fixed Costs and Big-M

Some costs come with doing something at all, whatever the amount: cleaning a line before a product runs, a truck that goes out half empty, a supplier's minimum order. They are **fixed costs**, and an LP can't see them: it would happily make 1.6 batches of a product and pay nothing extra. Today you add fixed costs and minimum runs to the bakery with a binary per product, and meet the **big-M** constraint that links them, and the trap of choosing M badly.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 22.1 What the LP misses |
| 0:45 – 1:45 | 22.2 Set-up costs and minimum runs |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 22.3 How big should M be? |
| 2:45 – 3:45 | 22.4 A planted bug: M too small |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: the cost of keeping each product |
| 4:45 – 5:00 | Commit with Git + checklist |

## 22.1 What the LP misses

`bakery_range_setup` gives each product a weekly **set-up cost** (cleaning and changeover, paid in any week the product is made) and a **minimum run** (the fewest batches worth starting). Start with Day 6's plan and pay its set-ups afterwards. Create `day22/setups.py`:

```python
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day06")
sys.path.insert(0, "day22")
from bakery import build_model, load_data, solve
from setup_model import build_with_setups

pd.set_option("display.width", 140)
con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_data(con)
setup = con.sql("SELECT * FROM bakery_range_setup").df().set_index("sku")

# ---------- 22.1 The Day 6 plan, then pay its set-ups ----------
lp = build_model(d)
solve(lp)
made = [p for p in lp.P if pyo.value(lp.x[p]) > 1e-6]
too_small = [p for p in made if pyo.value(lp.x[p]) < setup.min_batches_if_made[p] - 1e-6]
lp_net = pyo.value(lp.total_profit) - setup.setup_cost_per_week[made].sum()
print(f"Day 6 plan: {len(made)} products made, profit ${pyo.value(lp.total_profit):,.2f}, "
      f"set-ups ${setup.setup_cost_per_week[made].sum():,.0f}, net ${lp_net:,.2f}")
print("made below their minimum run:", {p: round(pyo.value(lp.x[p]), 2) for p in too_small})
```

```text
Day 6 plan: 22 products made, profit $108,351.28, set-ups $6,330, net $102,021.28
made below their minimum run: {'BK-26': 1.6}
```

The Day 6 plan makes 22 products and pays $6,330 of set-ups. Worse, it makes **1.6 batches of chocolate chip cookies**, below the 4-batch minimum: the plan can't run as written.

## 22.2 Set-up costs and minimum runs

One binary per product, $$y_p = 1$$ if product $$p$$ is made this week, and two constraints:

$$
\begin{aligned}
x_p &\le M_p\, y_p && \text{no batches unless made} \\
x_p &\ge \text{min run}_p\, y_p && \text{if made, at least the minimum run} \\
\text{net profit} &= \sum_p \text{profit}_p\, x_p - \sum_p \text{set-up}_p\, y_p
\end{aligned}
$$

When $$y_p = 0$$ both constraints force $$x_p = 0$$. When $$y_p = 1$$ they say $$\text{min run}_p \le x_p \le M_p$$. $$M_p$$ is a number big enough never to cut off a good plan: the **big-M**. A variable that is either 0 or between a minimum and a maximum is called **semi-continuous**: a pattern you will use again and again (minimum order sizes, minimum truck loads, minimum shift lengths). Create `day22/setup_model.py`:

```python
"""Day 22: fixed costs. A product made in a week pays a set-up cost and must run at least a minimum number of batches.

Import it:  from setup_model import build_with_setups
"""
import sys

import pyomo.environ as pyo

sys.path.insert(0, "day06")
from bakery import build_model


def build_with_setups(d, setup, big_m=None, domain=pyo.Binary):
    """make[p] = 1 if product p is made this week.
    x[p] <= M[p] * make[p]: no batches unless made. By default M[p] is the product's demand cap, the smallest
    value that never cuts off a good plan. x[p] >= min_run[p] * make[p]: if made, at least the minimum run."""
    m = build_model(d)
    M = {p: (big_m if big_m is not None else d["cap"][p]) for p in d["products"]}
    m.make = pyo.Var(m.P, within=domain)
    m.only_if_made = pyo.Constraint(m.P, rule=lambda m, p: m.x[p] <= M[p] * m.make[p])
    m.min_run = pyo.Constraint(m.P, rule=lambda m, p: m.x[p] >= setup.min_batches_if_made[p] * m.make[p])
    m.setup_cost = pyo.Expression(expr=sum(setup.setup_cost_per_week[p] * m.make[p] for p in m.P))
    m.total_profit.deactivate()
    m.net_profit = pyo.Objective(expr=sum(m.profit[p] * m.x[p] for p in m.P) - m.setup_cost, sense=pyo.maximize)
    return m
```

```python
# ---------- 22.2 Plan with the set-ups in the model ----------
m = build_with_setups(d, setup)
solve(m)
mip_made = [p for p in m.P if pyo.value(m.make[p]) > 0.5]
print(f"with set-ups: {len(mip_made)} products made, profit before set-ups "
      f"${pyo.value(m.net_profit + m.setup_cost):,.2f}, set-ups ${pyo.value(m.setup_cost):,.0f}, net ${pyo.value(m.net_profit):,.2f}")
changes = pd.DataFrame({
    "product": [d["name"][p] for p in m.P],
    "day6_batches": [round(pyo.value(lp.x[p]), 2) + 0.0 for p in m.P],
    "now": [round(pyo.value(m.x[p]), 2) + 0.0 for p in m.P],
    "setup_cost": [setup.setup_cost_per_week[p] for p in m.P],
    "min_run": [setup.min_batches_if_made[p] for p in m.P],
}, index=list(m.P))
print(changes[changes.day6_batches != changes.now].to_string())
```

```text
with set-ups: 21 products made, profit before set-ups $108,262.38, set-ups $6,020, net $102,242.38
                              product  day6_batches    now  setup_cost  min_run
BK-18              Butter tart 6-pack          25.0  25.94       430.0        4
BK-26  Chocolate chip cookies 12-pack           1.6   0.00       310.0        4
```

The plan drops chocolate chip cookies (saving their $310 set-up) and uses the time on butter tarts: **$102,242.38 net**, $221 better than the Day 6 plan with its set-ups paid, and actually runnable. A small change here because the bakery already made few small products; in a range with more tail products, set-ups decide which products survive.

> **Decision it supports:** which products to make this week, once each product costs something just to start.
>
> **Other decisions you can support the same way:** which production lines to run · which suppliers to order from when each order has a delivery fee · which routes to drive when each truck has a fixed daily cost.

## 22.3 How big should M be?

Any $$M_p$$ at least as large as the most product $$p$$ could ever make gives the same best plan. So why not just write 1,000,000? Because M changes the **relaxation**:

```python
# ---------- 22.3 How big should M be? ----------
rows = []
for big_m in (None, 100, 10_000, 1_000_000):
    relaxed = build_with_setups(d, setup, big_m, domain=pyo.UnitInterval)
    solve(relaxed)
    exact = build_with_setups(d, setup, big_m)
    solve(exact)
    rows.append(("demand cap" if big_m is None else f"{big_m:,}", round(pyo.value(relaxed.net_profit), 2),
                 round(pyo.value(exact.net_profit), 2),
                 round((pyo.value(relaxed.net_profit) - pyo.value(exact.net_profit)) / pyo.value(exact.net_profit) * 100, 2)))
print(pd.DataFrame(rows, columns=["M", "relaxation_bound", "best_plan", "gap_pct"]).to_string(index=False))
```

```text
         M  relaxation_bound  best_plan  gap_pct
demand cap         102372.08  102242.38     0.13
       100         107076.82  102242.38     4.73
    10,000         108338.54  102242.38     5.96
 1,000,000         108351.15  102242.38     5.97
```

Every M finds the same plan ($102,242.38). But the relaxation's bound weakens as M grows: from 0.13% above the best plan with M = demand cap, to 6% with M = 1,000,000. With a huge M, $$y_p = 0.000002$$ allows 2 batches in the relaxation, so the set-up costs almost vanish and the bound says little. On this small model HiGHS finds the answer anyway; on a big one, a weak bound means a huge search tree and hours instead of seconds (Day 27 measures it).

The rule: **M should be the smallest number that is always valid**, usually a capacity or a demand cap you already have, per variable, not one number for the whole model. Very large values also invite numerical trouble (Day 9): with tolerances around $$10^{-6}$$, $$M = 10^9$$ lets "$$y = 10^{-6}$$" pass for zero while allowing 1,000 batches.

## 22.4 A planted bug: M too small

`day22/bug_small_m.py` sets one M for every product: 20 batches, "plenty, since most products sell fewer than 20". Run it.

```python
con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_data(con)
setup = con.sql("SELECT * FROM bakery_range_setup").df().set_index("sku")
m = build_with_setups(d, setup, big_m=20)
result = solve(m)
print(result.solver.termination_condition, f"net profit ${pyo.value(m.net_profit):,.2f}")
```

<details><summary>What happens, and why</summary>

```text
optimal net profit $95,349.70
```

"Optimal", and $6,893 a week worse than the real best plan. M = 20 is a **hidden capacity**: every product whose demand cap is above 20 can now make at most 20 batches, however profitable it is. Too large an M weakens the relaxation; too small an M silently changes the problem. To catch it, list the variables sitting exactly at their M: any of them is a sign M is cutting off part of the plan.

</details>

## Mini-project: the cost of keeping each product

The plan drops 9 products: 8 that Day 6 already dropped, plus chocolate chip cookies. Marketing wants to know what it would cost to keep each one on the shelf at its minimum run. Fix each dropped product's $$y_p$$ to 1 in turn, re-solve, and rank them. What would it cost to keep all 30?

<details><summary>Solution</summary>

```python
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day06")
sys.path.insert(0, "day22")
from bakery import load_data, solve
from setup_model import build_with_setups

pd.set_option("display.width", 140)
con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_data(con)
setup = con.sql("SELECT * FROM bakery_range_setup").df().set_index("sku")
best = build_with_setups(d, setup)
solve(best)
dropped = [p for p in best.P if pyo.value(best.make[p]) < 0.5]

rows = []
for p in dropped:
    m = build_with_setups(d, setup)
    m.make[p].fix(1)                         # keep this product on the shelf, at least its minimum run
    solve(m)
    rows.append((p, d["name"][p], d["category"][p], round(pyo.value(m.x[p]), 2),
                 round(pyo.value(best.net_profit) - pyo.value(m.net_profit), 2)))
keep = pd.DataFrame(rows, columns=["sku", "product", "category", "batches_if_kept", "cost_a_week"])
print(keep.sort_values("cost_a_week").to_string(index=False))

everything = build_with_setups(d, setup)
for p in everything.P:
    everything.make[p].fix(1)
solve(everything)
print(f"keep all 30 products: ${pyo.value(everything.net_profit):,.2f}, "
      f"${pyo.value(best.net_profit) - pyo.value(everything.net_profit):,.2f} a week less than the best plan")
```

```text
  sku                        product     category  batches_if_kept  cost_a_week
BK-07                Olive sourdough        bread             3.00       108.61
BK-26 Chocolate chip cookies 12-pack       cookie             4.00       256.05
BK-22                     Cheesecake         cake             4.71       321.06
BK-24               Maple layer cake         cake             3.00       416.73
BK-08               Seeded sourdough        bread             4.00       422.71
BK-21              Black forest cake         cake             4.00       472.51
BK-15                  Danish 4-pack viennoiserie             5.00       505.88
BK-20                Lemon loaf cake         cake             4.00       714.96
BK-19                    Carrot cake         cake             4.00       745.76
keep all 30 products: $97,744.09, $4,498.29 a week less than the best plan
```

Keeping **olive sourdough costs only $109 a week**, chocolate chip cookies $256, cheesecake $321; carrot cake and lemon loaf cost over $700 each. Keeping all 30 products costs $4,498 a week. That is the number marketing must beat with the sales a full range brings (customers who leave when their cake is missing). The ranked list lets them choose: keep the cheap ones, drop the costly ones.

`fix(1)` on a binary is the MIP version of Day 7's "what does this rule cost?": force the decision, re-solve, compare. There are no reduced costs for binaries; forcing and re-solving is how you price a yes/no decision.

</details>

## Commit

```bash
git add .
git commit -m "Day 22: set-up costs, minimum runs, big-M"
git push
```

### Day 22 checklist

- [ ] I can model a fixed cost with a binary and a big-M constraint.
- [ ] I can model a minimum run (a semi-continuous variable).
- [ ] I choose M as the smallest always-valid value, per variable.
- [ ] I can show that a large M weakens the relaxation, and a small M changes the problem.
- [ ] I check for variables sitting exactly at their M.
- [ ] I priced each yes/no decision by fixing it and re-solving.

---

# DAY 23 — Facility Location

Should Northbound's distribution network look different? Three DCs serve 40 stores today; seven other sites are available, from Hamilton to Quebec City. Each site has a weekly fixed cost (rent and staff, or for a new site, the lease or build cost spread per week) and a capacity. Opening a site costs money every week but shortens deliveries. Today's model, **facility location**, weighs the two, and is the core of every network design study.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 23.1 Today's network as a location model |
| 0:45 – 1:45 | 23.2 Uncapacitated, capacitated, single-sourced |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 23.3 The relaxation, and why it can mislead |
| 2:45 – 3:45 | 23.4 How close is each candidate?; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: should Ottawa close? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 23.1 The model

`candidate_sites` lists 10 sites; `site_store_lanes` the cost per pallet from each to each store, computed like Day 14's lanes. Decisions: $$o_i = 1$$ if site $$i$$ runs; $$s_{ij}$$ the share of store $$j$$'s pallets that come from site $$i$$.

$$
\begin{aligned}
\min\quad & \sum_i \text{fixed}_i\, o_i + \sum_{i,j} \text{cost}_{ij}\, \text{pallets}_j\, s_{ij} \\
\text{s.t.}\quad & \textstyle\sum_i s_{ij} = 1 && \forall j \text{ (every store fully served)} \\
& s_{ij} \le o_i && \forall i, j \text{ (only from open sites)} \\
& \textstyle\sum_j \text{pallets}_j\, s_{ij} \le \text{capacity}_i\, o_i && \forall i \\
& o_i \in \{0,1\},\; s_{ij} \in \{0, 1\} \text{ (single sourcing) or } [0, 1]
\end{aligned}
$$

The French-label and six-hour rules from Day 15 fix the forbidden $$s_{ij}$$ to 0. Create `day23/facility.py`:

```python
"""Day 23: facility location. Which sites to run, and which stores each one serves.

Import it:  from facility import load_sites, build_location, summary
"""
import pandas as pd
import pyomo.environ as pyo

MAX_MINUTES = 360


def load_sites(con):
    sites = con.sql("SELECT * FROM candidate_sites ORDER BY site_id").df().set_index("site_id")
    stores = con.sql("""SELECT s.store_id, s.city, s.province, l.pallets_per_week
                        FROM stores s JOIN store_logistics l USING (store_id) ORDER BY s.store_id""").df().set_index("store_id")
    lanes = con.sql("SELECT * FROM site_store_lanes").df().set_index(["site_id", "store_id"])
    return sites, stores, lanes


def build_location(sites, stores, lanes, capacitated=True, single_source=True, domain=pyo.Binary, fixed_scale=1.0):
    """open[i] = 1 if site i runs (and pays its weekly fixed cost);
    serve[i, j] = share of store j's pallets that come from site i (0 or 1 with single sourcing)."""
    m = pyo.ConcreteModel(name="facility_location")
    m.I = pyo.Set(initialize=sites.index.tolist())
    m.J = pyo.Set(initialize=stores.index.tolist())
    m.open = pyo.Var(m.I, within=domain)
    serve_domain = pyo.Binary if (single_source and domain is pyo.Binary) else pyo.UnitInterval
    m.serve = pyo.Var(m.I, m.J, within=serve_domain)
    m.every_store = pyo.Constraint(m.J, rule=lambda m, j: sum(m.serve[i, j] for i in m.I) == 1)
    m.only_if_open = pyo.Constraint(m.I, m.J, rule=lambda m, i, j: m.serve[i, j] <= m.open[i])
    if capacitated:
        m.capacity = pyo.Constraint(m.I, rule=lambda m, i: sum(stores.pallets_per_week[j] * m.serve[i, j] for j in m.J)
                                    <= sites.capacity_pallets[i] * m.open[i])
    for i in m.I:
        for j in m.J:
            if (stores.province[j] == "QC" and not sites.ships_french_labels[i]) or lanes.minutes[i, j] > MAX_MINUTES:
                m.serve[i, j].fix(0)
    m.fixed = pyo.Expression(expr=fixed_scale * sum(sites.fixed_cost_per_week[i] * m.open[i] for i in m.I))
    m.transport = pyo.Expression(expr=sum(lanes.cost_per_pallet[i, j] * stores.pallets_per_week[j] * m.serve[i, j]
                                          for i in m.I for j in m.J))
    m.cost = pyo.Objective(expr=m.fixed + m.transport)
    return m


def summary(m, sites, stores):
    rows = []
    for i in m.I:
        if pyo.value(m.open[i]) > 1e-6:
            load = sum(stores.pallets_per_week[j] * pyo.value(m.serve[i, j]) for j in m.J)
            n = sum(pyo.value(m.serve[i, j]) > 1e-6 for j in m.J)
            rows.append((i, sites.place[i], round(pyo.value(m.open[i]), 3) + 0.0, n, round(load, 1), sites.capacity_pallets[i]))
    return pd.DataFrame(rows, columns=["site", "place", "open", "stores", "pallets", "capacity"])
```

The constraint `only_if_open` is a big-M constraint with M = 1, the tightest possible: a store can take at most all of its pallets from a site, and only if it's open.

Then `day23/run_location.py`, starting with today's network: open the three DCs, close the candidates, and let the model assign stores.

```python
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day23")
from facility import build_location, load_sites, summary

pd.set_option("display.width", 140)
con = duckdb.connect("northbound.duckdb", read_only=True)
sites, stores, lanes = load_sites(con)
solver = pyo.SolverFactory("highs")
print(sites.to_string())


def run(label, **kw):
    m = build_location(sites, stores, lanes, **kw)
    solver.solve(m)
    print(f"--- {label}: ${pyo.value(m.cost):,.2f} a week = fixed ${pyo.value(m.fixed):,.0f} + transport ${pyo.value(m.transport):,.2f}")
    print(summary(m, sites, stores).to_string(index=False))
    return m


# ---------- 23.1 Today's network as a location model ----------
today = build_location(sites, stores, lanes)
for i in today.I:
    today.open[i].fix(1 if i.startswith("DC-") else 0)
solver.solve(today)
print(f"--- today's three DCs, single sourcing: ${pyo.value(today.cost):,.2f} a week = fixed ${pyo.value(today.fixed):,.0f} "
      f"+ transport ${pyo.value(today.transport):,.2f}")
```

```text
                 place         kind  fixed_cost_per_week  capacity_pallets  ships_french_labels
site_id
DC-MIS     Mississauga  existing DC                21000              1600                    0
DC-MTL        Montreal  existing DC                19000              1500                    1
DC-OTT          Ottawa  existing DC                11000               700                    1
NEW-BRA       Brampton    candidate                23000              1800                    0
NEW-COR       Cornwall    candidate                12000              1000                    1
NEW-DRU  Drummondville    candidate                12500               900                    1
NEW-HAM       Hamilton    candidate                16000              1100                    0
NEW-KIN       Kingston    candidate                12500               900                    0
NEW-LAV          Laval    candidate                20000              1300                    1
NEW-QUE    Quebec City    candidate                13500               700                    1
--- today's three DCs, single sourcing: $92,492.76 a week = fixed $51,000 + transport $41,492.76
```

Today's three DCs, with single sourcing: **$92,492.76 a week**: $51,000 of fixed costs and Day 15's $41,492.76 of transport.

## 23.2 Uncapacitated, capacitated, single-sourced

Now let the model choose the sites, in three versions of increasing realism:

```python
# ---------- 23.2 Uncapacitated, capacitated, single-sourced ----------
run("no capacity limits", capacitated=False, single_source=False)
run("capacity limits, stores may be split", single_source=False)
best = run("capacity limits, one site per store")
```

```text
--- no capacity limits: $85,450.50 a week = fixed $46,000 + transport $39,450.50
   site    place  open  stores  pallets  capacity
 DC-MTL Montreal   1.0      15   1427.0      1500
 DC-OTT   Ottawa   1.0      10    923.0       700
NEW-HAM Hamilton   1.0      15   1195.0      1100
--- capacity limits, stores may be split: $90,089.81 a week = fixed $51,000 + transport $39,089.81
  site       place  open  stores  pallets  capacity
DC-MIS Mississauga   1.0      18   1345.0      1600
DC-MTL    Montreal   1.0      16   1500.0      1500
DC-OTT      Ottawa   1.0       8    700.0       700
--- capacity limits, one site per store: $92,492.76 a week = fixed $51,000 + transport $41,492.76
  site       place  open  stores  pallets  capacity
DC-MIS Mississauga   1.0      17   1365.0      1600
DC-MTL    Montreal   1.0      16   1480.0      1500
DC-OTT      Ottawa   1.0       7    700.0       700
```

- **Without capacities**, Hamilton replaces Mississauga: it is cheaper to run ($16,000 against $21,000), and Toronto stores are close to both. But Hamilton then ships 1,195 pallets with room for 1,100, and Ottawa ships 923 with room for 700. The **uncapacitated** model is the classic textbook one, and its answer here is impossible.
- **With capacities**, the model keeps today's three DCs: $90,089.81 with split stores, **$92,492.76** with single sourcing. None of the seven candidates pays for itself.

The finding: **today's network is the right one** at today's costs and volumes. A network study that confirms the current network is a real result: it saves the cost of a move, and tells you what would have to change for a move to pay (23.4).

> **Decision it supports:** which sites to run, and which customers each serves.
>
> **Other decisions you can support the same way:** where to put ambulance stations, schools, cell towers or charging stations · which warehouses to keep after a merger.

## 23.3 The relaxation, and why it can mislead

```python
# ---------- 23.3 The relaxation, and why it can mislead ----------
run("relaxation: open and serve may be fractions", domain=pyo.UnitInterval)
```

```text
--- relaxation: open and serve may be fractions: $89,717.62 a week = fixed $51,243 + transport $38,474.27
   site       place  open  stores  pallets  capacity
 DC-MIS Mississauga  1.00      17   1326.3      1600
 DC-MTL    Montreal  1.00      16   1500.0      1500
 DC-OTT      Ottawa  1.00       8    700.0       700
NEW-COR    Cornwall  0.02      10     18.7      1000
```

The relaxation "opens 2% of Cornwall" for $243 a week and splits ten stores between Cornwall and their DC. Its bound, $89,717.62, is 3% below the best plan. For this model the relaxation is fairly tight because `only_if_open` links each store to each site separately (the **strong formulation**). Day 27 compares it with a weaker but common alternative (one constraint per site instead of one per store and site) and shows the difference in search effort.

## 23.4 How close is each candidate?

A MIP has no reduced costs for its binaries. To learn how far each candidate is from being worth opening, **force it open** and re-solve: the extra cost is what it would take to justify it.

```python
# ---------- 23.4 How close is each candidate? Force it open ----------
rows = []
for i in sites.index[sites.kind == "candidate"]:
    m = build_location(sites, stores, lanes)
    m.open[i].fix(1)
    solver.solve(m)
    extra = pyo.value(m.cost) - pyo.value(best.cost)
    rows.append((i, sites.place[i], sites.fixed_cost_per_week[i], round(extra, 2),
                 round(sites.fixed_cost_per_week[i] - extra, 2), ", ".join(x for x in m.I if pyo.value(m.open[x]) > 0.5)))
force = pd.DataFrame(rows, columns=["candidate", "place", "fixed_cost", "extra_cost_if_opened", "break_even_fixed_cost", "network"])
print(force.sort_values("extra_cost_if_opened").to_string(index=False))
```

```text
candidate         place  fixed_cost  extra_cost_if_opened  break_even_fixed_cost                          network
  NEW-QUE   Quebec City       13500               2656.20               10843.80  DC-MIS, DC-MTL, DC-OTT, NEW-QUE
  NEW-COR      Cornwall       12000               3880.35                8119.65          DC-MIS, DC-MTL, NEW-COR
  NEW-BRA      Brampton       23000               4200.99               18799.01          DC-MTL, DC-OTT, NEW-BRA
  NEW-DRU Drummondville       12500               4708.70                7791.30  DC-MIS, DC-MTL, DC-OTT, NEW-DRU
  NEW-KIN      Kingston       12500               5558.45                6941.55  DC-MIS, DC-MTL, DC-OTT, NEW-KIN
  NEW-LAV         Laval       20000               5588.02               14411.98 DC-MIS, DC-OTT, NEW-LAV, NEW-QUE
  NEW-HAM      Hamilton       16000               7102.24                8897.76 DC-MTL, DC-OTT, NEW-HAM, NEW-KIN
```

- **Quebec City is closest**: opening it costs $2,656 a week more than today's network. If its weekly cost fell from $13,500 to **$10,844** (a cheaper lease, a shared building), it would pay.
- **Cornwall** is the interesting one: forced open, the model **closes Ottawa** and serves the Ottawa region from Cornwall, which has room for all 923 pallets. It costs $3,880 a week more than today, but fixes Ottawa's capacity problem: see the mini-project.
- **Brampton**, forced open, replaces Mississauga; **Laval**, forced open, replaces Montreal and brings Quebec City along. Forcing one site open shows the whole network reorganizing around it.

### The planted bug

`day23/bug_link.py` drops one constraint from the uncapacitated model, "because without capacity rows we don't need it". Run it.

```python
con = duckdb.connect("northbound.duckdb", read_only=True)
sites, stores, lanes = load_sites(con)
m = build_location(sites, stores, lanes, capacitated=False, single_source=False)
m.del_component(m.only_if_open)      # "with no capacity rows, we don't need it"
pyo.SolverFactory("highs").solve(m)
print(f"${pyo.value(m.cost):,.2f} a week = fixed ${pyo.value(m.fixed) + 0.0:,.0f} + transport ${pyo.value(m.transport):,.2f}")
opened = summary(m, sites, stores)
print("open sites:", opened.site.tolist())
served_by_closed = sum(pyo.value(m.serve[i, j]) > 1e-6 and pyo.value(m.open[i]) < 0.5 for i in m.I for j in m.J)
print(f"store deliveries from sites that are closed: {served_by_closed}")
```

<details><summary>What happens, and why</summary>

```text
$20,648.86 a week = fixed $0 + transport $20,648.86
open sites: []
store deliveries from sites that are closed: 40
```

No site is open, no fixed cost is paid, and every store is served, from closed sites. Without `only_if_open`, nothing ties deliveries to open sites; in the capacitated model the capacity row happens to do that job ($$\le \text{capacity}_i\, o_i$$), which is why the bug only shows when capacities are removed. Every link between a yes/no decision and the quantities it controls must be in the model explicitly.

</details>

## Mini-project: should Ottawa close?

The Ottawa DC is over capacity (Day 14), and a board member asks whether to close it. Compare: the best network; Ottawa closed with no new site; Ottawa closed with any candidate allowed; Ottawa replaced by Cornwall; Ottawa replaced by Kingston. Recommend.

<details><summary>Solution</summary>

```python
import sys

import duckdb
import pyomo.environ as pyo

sys.path.insert(0, "day23")
from facility import build_location, load_sites, summary

con = duckdb.connect("northbound.duckdb", read_only=True)
sites, stores, lanes = load_sites(con)
solver = pyo.SolverFactory("highs")


def scenario(label, fixes):
    m = build_location(sites, stores, lanes)
    for i, v in fixes.items():
        m.open[i].fix(v)
    result = solver.solve(m, load_solutions=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        print(f"--- {label}: {result.solver.termination_condition}")
        return None
    m.solutions.load_from(result)
    print(f"--- {label}: ${pyo.value(m.cost):,.2f} a week (fixed ${pyo.value(m.fixed):,.0f}, transport ${pyo.value(m.transport):,.2f})")
    print(summary(m, sites, stores).to_string(index=False))
    return pyo.value(m.cost)


candidates = {i: 0 for i in sites.index if sites.kind[i] == "candidate"}
best = scenario("best network", {})
scenario("close Ottawa, no new site", {"DC-OTT": 0, **candidates})
scenario("close Ottawa, any candidate may open", {"DC-OTT": 0})
scenario("close Ottawa, open Cornwall", {"DC-OTT": 0, **candidates, "NEW-COR": 1})
scenario("close Ottawa, open Kingston", {"DC-OTT": 0, **candidates, "NEW-KIN": 1})
```

```text
--- best network: $92,492.76 a week (fixed $51,000, transport $41,492.76)
  site       place  open  stores  pallets  capacity
DC-MIS Mississauga   1.0      17   1365.0      1600
DC-MTL    Montreal   1.0      16   1480.0      1500
DC-OTT      Ottawa   1.0       7    700.0       700
--- close Ottawa, no new site: infeasible
--- close Ottawa, any candidate may open: $96,373.11 a week (fixed $52,000, transport $44,373.11)
   site       place  open  stores  pallets  capacity
 DC-MIS Mississauga   1.0      15   1195.0      1600
 DC-MTL    Montreal   1.0      15   1427.0      1500
NEW-COR    Cornwall   1.0      10    923.0      1000
--- close Ottawa, open Cornwall: $96,373.11 a week (fixed $52,000, transport $44,373.11)
   site       place  open  stores  pallets  capacity
 DC-MIS Mississauga   1.0      15   1195.0      1600
 DC-MTL    Montreal   1.0      15   1427.0      1500
NEW-COR    Cornwall   1.0      10    923.0      1000
--- close Ottawa, open Kingston: infeasible
```

- **Closing Ottawa without a replacement is impossible**: Mississauga and Montreal can't take 3,545 pallets.
- **Kingston can't replace Ottawa either**: it has no French-labelled stock, so the Ottawa region's Quebec stores would have to come from Montreal, which has only 20 pallets of room.
- **Cornwall can**: it takes exactly the 10 Ottawa-region stores (923 pallets), with room to spare. The network costs **$96,373.11**, $3,880 a week more than keeping Ottawa.

*Memo to the board:* "Keep the Ottawa DC. Replacing it with a new Cornwall site would cost about $3,900 a week more ($200,000 a year), before any move costs. Ottawa's real problem is capacity, not location: Day 15 found that 100 more pallets of capacity in Ottawa saves $4,800 a week. Expand Ottawa before considering a new site. Revisit if Cornwall's cost falls below about $8,100 a week."

</details>

## Commit

```bash
git add .
git commit -m "Day 23: facility location"
git push
```

Then run the tests: `tests/test_integer.py` checks Days 21–23 (the investment MIP against enumeration, the relaxation bound, minimum runs, and the location model). Add `"day21"`, `"day22"`, `"day23"` to the folders in `conftest.py`.

```python
"""Module 3, Days 21-23: binary decisions, fixed costs and facility location."""
import itertools

import pyomo.environ as pyo
import pytest
from bakery import load_data, solve
from facility import build_location, load_sites
from investments import build_with_investments
from setup_model import build_with_setups

TOL = 1e-6


@pytest.fixture(scope="module")
def range_data(con):
    return load_data(con)


def test_investment_mip_matches_enumeration(con, range_data):
    menu = con.sql("SELECT * FROM bakery_investments").df().set_index("option")
    m = build_with_investments(range_data, menu)
    solve(m)
    best = -1
    for choice in itertools.product([0, 1], repeat=len(menu)):
        e = build_with_investments(range_data, menu)
        for o, c in zip(menu.index, choice):
            e.buy[o].fix(c)
        solve(e)
        best = max(best, pyo.value(e.net_profit))
    assert pyo.value(m.net_profit) == pytest.approx(best, abs=0.01) == pytest.approx(113_488.16, abs=0.01)


def test_relaxation_is_an_upper_bound(con, range_data):
    menu = con.sql("SELECT * FROM bakery_investments").df().set_index("option")
    exact, relaxed = build_with_investments(range_data, menu), build_with_investments(range_data, menu, pyo.UnitInterval)
    solve(exact)
    solve(relaxed)
    assert pyo.value(relaxed.net_profit) >= pyo.value(exact.net_profit) - TOL


def test_setups_respect_minimum_runs(con, range_data):
    setup = con.sql("SELECT * FROM bakery_range_setup").df().set_index("sku")
    m = build_with_setups(range_data, setup)
    solve(m)
    assert pyo.value(m.net_profit) == pytest.approx(102_242.38, abs=0.01)
    for p in m.P:
        x = pyo.value(m.x[p])
        assert x < TOL or x >= setup.min_batches_if_made[p] - TOL


def test_location_keeps_todays_network_and_serves_every_store_once(con):
    sites, stores, lanes = load_sites(con)
    m = build_location(sites, stores, lanes)
    pyo.SolverFactory("highs").solve(m)
    assert pyo.value(m.cost) == pytest.approx(92_492.76, abs=0.01)
    assert sorted(i for i in m.I if pyo.value(m.open[i]) > 0.5) == ["DC-MIS", "DC-MTL", "DC-OTT"]
    for j in m.J:
        assert sum(round(pyo.value(m.serve[i, j])) for i in m.I) == 1
```

```text
40 passed
```

### Day 23 checklist

- [ ] I can write a facility location model with fixed costs, capacities and single sourcing.
- [ ] I know why the uncapacitated model can give impossible answers.
- [ ] I can say when a location study confirming today's network is a real result.
- [ ] I priced each candidate by forcing it open, and computed its break-even cost.
- [ ] I know every yes/no decision must be linked explicitly to what it controls.
- [ ] I answered the board's question with feasible and infeasible scenarios, and a recommendation.
