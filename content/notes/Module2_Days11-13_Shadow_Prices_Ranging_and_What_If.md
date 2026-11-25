# DAY 11 — Shadow Prices and Duality

Module 2 starts by asking the solver a better question. Module 1 asked "what is the best plan?". Today we ask "**what is each limit worth?**": what one more oven hour, packing hour or batch of starter would add to profit. On Day 2 you found $133.33 for an oven hour by re-solving. The solver already knows these numbers; they are a by-product of every LP solve, called **shadow prices** or **dual values**.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 11.1 Asking the solver for shadow prices |
| 0:45 – 1:45 | 11.2 The dual problem: renting out the bakery |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 11.3 Shadow prices for the full range |
| 2:45 – 3:45 | 11.4 When shadow prices mislead: degenerate corners |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: what is an hour worth, week by week? |
| 4:45 – 5:00 | Commit with Git + checklist |

Module 2 works with the code and database from Module 1. Start from your `v1.0-module1` tag, with `northbound.duckdb` built by `day06/load_tables.py`.

## 11.1 Asking the solver for shadow prices

The **shadow price** of a constraint is how much the objective improves per unit of extra right-hand side, with everything else unchanged. For a capacity, it is what one more hour is worth. Pyomo receives it from the solver if you declare a **suffix** called `dual` on the model before solving:

```python
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
```

After the solve, `m.dual[constraint]` holds that constraint's shadow price. Create `day11/shadow_prices.py`. It reuses Day 6's `bakery.py`, and compares each shadow price with the Day 2 method, re-solving with one more unit:

```python
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day06")
from bakery import build_model, load_data, solve

pd.set_option("display.width", 120)
con = duckdb.connect("northbound.duckdb", read_only=True)
con.sql("CREATE TEMP VIEW module1_products AS SELECT *, 'module 1' AS category FROM bakery_products")
con.sql("CREATE TEMP VIEW module1_forecast AS SELECT sku, demand_cap_batches FROM forecast_next_week")


def solve_with_duals(m):
    """Ask the solver to send back a dual value (shadow price) for every constraint."""
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solve(m)
    return m


# ---------- 11.1 The 3-product bakery: re-solving versus asking the solver ----------
d = load_data(con, "module1_products", "bakery_recipe", "bakery_resources", "module1_forecast")
m = solve_with_duals(build_model(d))
base = pyo.value(m.total_profit)
print(f"profit ${base:,.2f}")
print(f"{'resource':<18}{'dual':>10}{'re-solve +1':>14}")
for r in m.R:
    more = dict(d, capacity=dict(d["capacity"], **{r: d["capacity"][r] + 1}))
    m1 = build_model(more)
    solve(m1)
    print(f"{r:<18}{m.dual[m.capacity_limit[r]]:>10.2f}{pyo.value(m1.total_profit) - base:>14.2f}")
print("demand caps:", {p: round(m.dual[m.demand[p]], 2) for p in m.P})
```

```text
profit $17,500.00
resource                dual   re-solve +1
baker_hours            33.33         33.33
oven_hours            133.33        133.33
sourdough_starter       0.00          0.00
demand caps: {'BK-CR': 0.0, 'BK-SD': 0.0, 'BK-WW': 33.33}
```

The solver's numbers match the re-solves exactly, at the cost of one solve instead of four:

- **Oven $133.33 an hour, bakers $33.33**: Day 2's numbers.
- **Starter $0**: 20 batches of starter are spare, and more of something spare is worth nothing.
- **The whole-wheat cap is worth $33.33 a batch**: one more batch of whole-wheat demand would add $33.33. Croissant and sourdough caps are worth nothing; the bakery doesn't reach them.

A constraint with slack has a shadow price of zero; a constraint with a positive shadow price is binding. That pairing is called **complementary slackness**, and it is a quick way to read a plan.

> **Decision it supports:** which limit to relax first, and the most to pay for relaxing it.
>
> **Other decisions you can support the same way:** what one more truck, nurse shift or warehouse door is worth · the value of a tighter or looser regulation.

## 11.2 The dual problem: renting out the bakery

Where do shadow prices come from? Every LP has a twin, its **dual**, and the shadow prices are the dual's solution.

Imagine a competitor wants to rent the whole bakery for a week: the ovens, the bakers, the starter, and the right to sell up to each product's demand cap. They offer a price per oven hour $$y_\text{oven}$$, per baker hour $$y_\text{baker}$$, per batch of starter $$y_\text{starter}$$, and per batch of demand $$z_p$$. Northbound will only accept a **fair** offer: for every product, the rent on what one batch uses must be at least the profit Northbound would make baking it itself. The competitor wants the cheapest fair offer:

$$
\begin{aligned}
\min\quad & 100\,y_\text{oven} + 80\,y_\text{baker} + 40\,y_\text{starter} + 35\,z_\text{SD} + 45\,z_\text{WW} + 31\,z_\text{CR} \\
\text{s.t.}\quad & 2\,y_\text{oven} + 1\,y_\text{baker} + 1\,y_\text{starter} + z_\text{SD} \ge 300 && \text{(sourdough)} \\
& 1\,y_\text{oven} + 1\,y_\text{baker} + z_\text{WW} \ge 200 && \text{(whole-wheat)} \\
& 1.5\,y_\text{oven} + 1.5\,y_\text{baker} + z_\text{CR} \ge 250 && \text{(croissant)} \\
& y, z \ge 0
\end{aligned}
$$

Every number of the original model appears, transposed: the original's capacities become the dual's costs, its profits become the dual's limits, and each column of the recipe becomes a row.

```python
# ---------- 11.2 The dual problem: what would someone pay to rent the bakery? ----------
dual = pyo.ConcreteModel()
dual.R = pyo.Set(initialize=d["resources"])
dual.P = pyo.Set(initialize=d["products"])
dual.y = pyo.Var(dual.R, within=pyo.NonNegativeReals)    # rent per unit of each resource
dual.z = pyo.Var(dual.P, within=pyo.NonNegativeReals)    # rent per batch of each product's demand
dual.rent = pyo.Objective(expr=sum(d["capacity"][r] * dual.y[r] for r in dual.R)
                          + sum(d["cap"][p] * dual.z[p] for p in dual.P))      # minimize
dual.fair = pyo.Constraint(dual.P, rule=lambda dm, p: sum(d["use"].get((r, p), 0) * dm.y[r] for r in dm.R) + dm.z[p] >= d["profit"][p])
pyo.SolverFactory("highs").solve(dual)
print(f"cheapest fair rent ${pyo.value(dual.rent):,.2f}")
print({r: round(pyo.value(dual.y[r]), 2) for r in dual.R}, {p: round(pyo.value(dual.z[p]), 2) for p in dual.P})

# Complementary slackness, checked product by product
for p in m.P:
    resource_cost = sum(d["use"].get((r, p), 0) * m.dual[m.capacity_limit[r]] for r in m.R)
    print(f"  {p}: makes {pyo.value(m.x[p]):5.1f}, profit {d['profit'][p]:6.1f}, "
          f"resources worth {resource_cost:6.2f}, cap worth {m.dual[m.demand[p]]:5.2f}")
```

```text
cheapest fair rent $17,500.00
{'baker_hours': 33.33, 'oven_hours': 133.33, 'sourdough_starter': 0.0} {'BK-CR': 0.0, 'BK-SD': 0.0, 'BK-WW': 33.33}
  BK-CR: makes  10.0, profit  250.0, resources worth 250.00, cap worth  0.00
  BK-SD: makes  20.0, profit  300.0, resources worth 300.00, cap worth  0.00
  BK-WW: makes  45.0, profit  200.0, resources worth 166.67, cap worth 33.33
```

Two results, both true of every LP:

- **Strong duality:** the cheapest fair rent, $17,500, equals the best profit. Northbound is indifferent between baking and renting out at those prices.
- **The rents are the shadow prices.** The competitor's prices for oven, bakers and starter are exactly $133.33, $33.33 and $0.

The product-by-product check is complementary slackness again. Every product the bakery makes "pays for" the resources it uses at shadow prices: croissant and sourdough exactly ($250 = $250, $300 = $300), whole-wheat with its cap's value ($166.67 + $33.33 = $200). A product whose resources cost more than its profit would not be made at all; Day 12 calls that gap its reduced cost.

### Math corner (optional): why strong duality holds

For any plan $$x$$ that fits and any fair rent $$(y, z)$$, profit ≤ rent: each batch earns at most what its resources rent for, and the bakery uses at most what it has. This is **weak duality**, and it takes two lines of algebra: multiply each capacity constraint by its $$y$$, each cap by its $$z$$, add them up, and compare with the fairness rows. Strong duality says the gap closes at the optimum; the simplex method proves it by construction, because its final basis gives a fair rent equal to the plan's profit. The practical meaning: if someone shows you a plan and a set of prices with equal value, both are optimal, and you can check it by hand.

## 11.3 Shadow prices for the full range

The same suffix on the 30-product model:

```python
# ---------- 11.3 The full range ----------
d = load_data(con)
m = solve_with_duals(build_model(d))
print(f"full range profit ${pyo.value(m.total_profit):,.2f}")
res = pd.DataFrame([(r, pyo.value(m.capacity_limit[r].body), d["capacity"][r], m.dual[m.capacity_limit[r]]) for r in m.R],
                   columns=["resource", "used", "capacity", "shadow_price"]).round(2)
print(res.to_string(index=False))
caps = pd.DataFrame([(p, d["name"][p], pyo.value(m.x[p]), d["cap"][p], m.dual[m.demand[p]]) for p in m.P],
                    columns=["sku", "product", "batches", "cap", "shadow_price"]).round(2)
print("Most valuable demand caps (one more batch of demand is worth):")
print(caps.sort_values("shadow_price", ascending=False).head(6).to_string(index=False))
```

```text
full range profit $108,351.28
         resource   used  capacity  shadow_price
      baker_hours 539.62       560          0.00
       oven_hours 600.00       600         99.34
    packing_hours 300.00       300         94.46
sourdough_starter  60.00        60         54.09
Most valuable demand caps (one more batch of demand is worth):
  sku                 product  batches  cap  shadow_price
BK-10            Raisin bread     20.0   20        134.38
BK-04         Multigrain loaf     18.0   18        107.21
BK-17   Apple turnover 4-pack      9.0    9         73.87
BK-06      Country white loaf      8.0    8         70.60
BK-16            Brioche loaf      9.0    9         60.50
BK-13 Almond croissant 4-pack     27.0   27         57.30
```

For the bakery manager:

- **An oven hour is worth $99.34, a packing hour $94.46, a batch of starter $54.09.** Baker hours are worth nothing: 20 are spare.
- Packing overtime costs **$60 an hour**, less than $94.46: worth booking, at least for the first few hours. That is the Module 1 capstone's overtime decision, read straight from one number.
- **The most valuable demand cap is raisin bread** ($134.38 a batch): if marketing can sell more of one product, raisin bread is the one. Cakes, absent from this list, have caps worth nothing: the bakery doesn't bake them.

A shadow price is a **marginal** value: true for a small change, not necessarily for a large one. How small? Day 12 answers that.

### Try it 11.1

Which products' demand caps have a shadow price of zero even though the product is made? What does that tell you?

<details><summary>Solution</summary>

```python
print(caps[(caps.batches > 0) & (caps.shadow_price == 0)])
```

```text
      sku                         product  batches  cap  shadow_price
14  BK-18              Butter tart 6-pack     25.0   30           0.0
22  BK-26  Chocolate chip cookies 12-pack      1.6    9           0.0
28  BK-SD                  Sourdough loaf     35.0   35           0.0
```

Butter tarts (25 of 30) and chocolate chip cookies (1.6 of 9) are below their caps, so more demand is worth nothing: the bakery can't bake more of them profitably. These are the **marginal** products, which absorb any extra or missing capacity: the first to move in any what-if.

Sourdough is the odd one: it **is** at its cap of 35, yet the cap is "worth" nothing. More sourdough demand would need more starter, which is also full, and the solver has put the value on the starter instead. Two limits bind on the same product at once: a degenerate corner, the subject of 11.4, and the first sign that sourdough is fragile (Day 12).

</details>

## 11.4 When shadow prices mislead: degenerate corners

On Day 3 you met a degenerate corner: three lines through one point. With croissants fixed at 10 and whole-wheat capped at 45, the best plan (20, 45) sits on the oven line, the baker line and the cap at once. What are the shadow prices there?

```python
# ---------- 11.4 Degenerate corners: the dual is not unique ----------
deg = pyo.ConcreteModel()
deg.S = pyo.Var(bounds=(0, 35))
deg.W = pyo.Var(bounds=(0, 45))
deg.oven = pyo.Constraint(expr=2 * deg.S + deg.W <= 85)        # croissants fixed at 10 (Day 3, Try it 3.2)
deg.bakers = pyo.Constraint(expr=deg.S + deg.W <= 65)
deg.profit = pyo.Objective(expr=300 * deg.S + 200 * deg.W, sense=pyo.maximize)
deg.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
pyo.SolverFactory("highs").solve(deg)
print(f"degenerate corner S={pyo.value(deg.S):g}, W={pyo.value(deg.W):g}: "
      f"oven dual {deg.dual[deg.oven]:.2f}, bakers dual {deg.dual[deg.bakers]:.2f}")
for change in (+1, -1):
    for name, con_ in (("oven", deg.oven), ("bakers", deg.bakers)):
        rhs = con_.upper
        con_.set_value(con_.body <= rhs + change)
        pyo.SolverFactory("highs").solve(deg)
        print(f"  {name} {change:+d} hour: profit changes by {pyo.value(deg.profit) - 15000:+.2f}")
        con_.set_value(con_.body <= rhs)
```

```text
degenerate corner S=20, W=45: oven dual 150.00, bakers dual 0.00
  oven +1 hour: profit changes by +100.00
  bakers +1 hour: profit changes by +0.00
  oven -1 hour: profit changes by -150.00
  bakers -1 hour: profit changes by -100.00
```

The solver says an oven hour is worth $150 and a baker hour nothing. But **one more oven hour adds only $100**, one less costs $150, and one less baker hour costs $100 although its "price" is zero. At a degenerate corner, the value of a resource depends on the direction of the change, and the solver reports just one of several valid dual solutions.

The rule: shadow prices are exact for small changes at a **non-degenerate** corner. When a plan has more binding constraints than usual, or when a decision rides on a shadow price, check it with a re-solve in the direction you care about. It takes a second.

## Mini-project: what is an hour worth, week by week?

In the Module 1 capstone, the plan booked packing overtime in 11 of 12 weeks, the full 40 hours in only 3. Shadow prices explain exactly why. Add the dual suffix to the capstone model and report, for each week, the shadow price of oven hours, packing hours (after overtime), starter and freezer, next to the packing overtime booked.

<details><summary>Solution</summary>

```python
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day08")
sys.path.insert(0, "day10")
from capstone import build_capstone
from planner import solve

pd.set_option("display.width", 120)
con = duckdb.connect("northbound.duckdb", read_only=True)
d, m = build_capstone(con)
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
solve(m)

rows = []
for t in m.T:
    rows.append((t,
                 m.dual[m.capacity["oven_hours", t]],
                 m.dual[m.capacity["packing_hours", t]],
                 pyo.value(m.overtime["packing_hours", t]),
                 m.dual[m.capacity["sourdough_starter", t]],
                 m.dual[m.freezer[t]]))
weekly = pd.DataFrame(rows, columns=["week", "oven_$_per_h", "packing_$_per_h", "packing_overtime_h",
                                     "starter_$_per_batch", "freezer_$_per_slot"]).set_index("week").round(2) + 0.0
print(weekly.to_string())
print("Weeks where packing overtime is between 0 and 40 hours have a packing shadow price of exactly $60:")
print(weekly[(weekly.packing_overtime_h > 0) & (weekly.packing_overtime_h < 40)][["packing_$_per_h"]].T.to_string())
```

```text
      oven_$_per_h  packing_$_per_h  packing_overtime_h  starter_$_per_batch  freezer_$_per_slot
week
1           126.23            75.88               40.00                 9.60                 0.0
2           132.57            71.90               40.00                 0.00                 0.0
3           139.95            66.79               40.00                18.99                 0.0
4           149.06            60.00               13.36                12.13                 0.0
5           137.65            60.00               22.22                 0.00                 0.0
6           114.63            60.00               32.80                40.74                 0.0
7           117.36            60.00               26.06                35.28                 0.0
8           119.36            60.00                6.99                46.70                 0.0
9           122.81            60.00               29.37                24.38                 0.0
10          125.54            60.00               39.11                40.35                 0.0
11          128.27            60.00               26.89                13.47                 0.0
12          134.10            56.73                0.00                 3.25                 0.0
Weeks where packing overtime is between 0 and 40 hours have a packing shadow price of exactly $60:
week               4     5     6     7     8     9     10    11
packing_$_per_h  60.0  60.0  60.0  60.0  60.0  60.0  60.0  60.0
```

Three patterns, each a lesson:

1. **Packing is worth exactly $60 in every week where overtime is between 0 and 40 hours.** The model books overtime until one more hour is worth exactly what it costs. In weeks 1–3, overtime is at its 40-hour limit, and an hour is worth more ($66.79 to $75.88): the bakery would book more if it could. In week 12, no overtime is booked, and an hour is worth $56.73, less than $60. Complementary slackness, in business terms.
2. **The oven is worth most in Christmas week ($149.06)**, when capacity drops to 60% and demand peaks. The busiest hour of the year is the most valuable one.
3. **The freezer is worth $0 every week**: it never fills (Day 8's finding, now in one column).

</details>

## Commit

```bash
git add .
git commit -m "Day 11: shadow prices, the dual, degenerate corners"
git push
```

### Day 11 checklist

- [ ] I can get shadow prices from Pyomo with a `dual` suffix.
- [ ] I can explain a shadow price as the value of one more unit of a limit.
- [ ] I can write the dual of a small LP and check that its value equals the plan's profit.
- [ ] I use complementary slackness to read a plan: binding limits have prices, slack ones don't.
- [ ] I check shadow prices with a re-solve at degenerate corners.
- [ ] I explained the capstone's overtime pattern with weekly shadow prices.

---

# DAY 12 — Reduced Costs and Ranging

Yesterday's shadow prices valued the limits. Today's tools value the products and the plan's stability: **reduced costs** say how much more a product must earn before it is worth making, and **ranging** says how far any number can move before the plan changes. Together they answer the planner's most common question: "if this changes, do I need a new plan?"

| Time | Block |
|---|---|
| 0:00 – 0:45 | 12.1 Reduced costs: what an unbaked product is missing |
| 0:45 – 1:45 | 12.2 Ranging with HiGHS |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 12.3 Objective ranges: how stable is the plan? |
| 2:45 – 3:45 | 12.4 Right-hand-side ranges: how far a shadow price holds |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: which prices would change the plan? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 12.1 Reduced costs

On Day 6, eight products were not baked. Black forest cake earns $256 a batch, more than many baked products. Why not bake it, and how much would its profit need to rise?

The **reduced cost** of a product is its profit minus the value, at shadow prices, of everything one batch uses:

$$
\text{reduced cost}_p = \text{profit}_p - \sum_r \text{use}_{r,p}\, y_r
$$

For a product that is baked (and below its cap), it is zero: it pays exactly for its resources (Day 11). For a product that is not baked, it is negative: the shortfall. Pyomo receives reduced costs through a suffix called `rc`. Create `day12/reduced_costs.py`:

```python
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day06")
sys.path.insert(0, "day12")
from bakery import build_model, load_data, solve
from ranging import ranging

pd.set_option("display.width", 140)
con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_data(con)
m = build_model(d)
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
m.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
solve(m)
base = pyo.value(m.total_profit)

# ---------- 12.1 Reduced costs: how far is each unbaked product from being worth baking? ----------
rows = []
for p in m.P:
    if pyo.value(m.x[p]) > 1e-6:
        continue
    resources = sum(d["use"].get((r, p), 0) * m.dual[m.capacity_limit[r]] for r in m.R)
    rows.append((p, d["name"][p], d["profit"][p], round(resources, 2), round(m.rc[m.x[p]], 2)))
rc = pd.DataFrame(rows, columns=["sku", "product", "profit_per_batch", "resources_at_shadow_prices", "reduced_cost"])
print(rc.sort_values("reduced_cost", ascending=False).to_string(index=False))
```

```text
  sku           product  profit_per_batch  resources_at_shadow_prices  reduced_cost
BK-07   Olive sourdough             227.0                      229.97         -2.97
BK-22        Cheesecake             245.4                      253.89         -8.49
BK-15     Danish 4-pack             208.0                      225.07        -17.07
BK-21 Black forest cake             256.0                      273.27        -17.27
BK-08  Seeded sourdough             248.0                      299.51        -51.51
BK-24  Maple layer cake             193.8                      262.36        -68.56
BK-20   Lemon loaf cake             219.2                      313.50        -94.30
BK-19       Carrot cake             199.0                      322.94       -123.94
```

Black forest cake's oven and packing time is worth $273.27 at shadow prices; it earns $256, so it is $17.27 short. **Olive sourdough is only $2.97 short**: the slightest price rise or cost cut would bring it into the plan. Carrot cake is $123.94 short: no realistic price change will bake it.

Check one, the reduced cost of seeded sourdough ($51.51):

```python
# Check one: give seeded sourdough just over its reduced cost, and it enters the plan
gap = -m.rc[m.x["BK-08"]]
for extra in (gap - 1, gap + 1):
    m2 = build_model(dict(d, profit=dict(d["profit"], **{"BK-08": d["profit"]["BK-08"] + extra})))
    solve(m2)
    print(f"BK-08 profit +${extra:.2f}: bakes {pyo.value(m2.x['BK-08']):.2f} batches")
```

```text
BK-08 profit +$50.51: bakes 0.00 batches
BK-08 profit +$52.51: bakes 6.67 batches
```

One dollar below its reduced cost, still not baked; one dollar above, and 6.67 batches enter the plan.

> **Decision it supports:** which products are close to earning a place in the plan, and which are far away.
>
> **Other decisions you can support the same way:** which suppliers are close to winning business · which routes or shifts are nearly worth adding.

## 12.2 Ranging with HiGHS

Reduced costs and shadow prices are values at the current plan. **Ranging** adds how far they hold:

- **Objective ranging**, for each product: the lowest and highest profit per batch for which the current plan stays optimal (Day 3's "$200 to $400" for sourdough).
- **Right-hand-side ranging**, for each limit: the range of capacity over which the shadow price stays the same.

HiGHS computes both, but Pyomo does not pass them back. So we write the model to an LP file, load it into HiGHS directly through its Python package `highspy` (installed with HiGHS on Day 1), and use Pyomo's **symbol map** to translate HiGHS's names (`x(BK_03)`) back to Pyomo's (`x[BK-03]`). Create `day12/ranging.py`:

```python
"""Day 12: ranging with HiGHS. How far can a number move before the plan changes?

Pyomo does not pass ranging back, so we write the model as an LP file, load it into
HiGHS directly (highspy), and match HiGHS's names back to the Pyomo components.

Import it:  from ranging import ranging
"""
import os
import tempfile

import highspy
import pandas as pd


def ranging(m):
    """Return (costs, limits): ranging for every variable's objective coefficient
    and every constraint's right-hand side, as DataFrames indexed by Pyomo name."""
    path = os.path.join(tempfile.mkdtemp(), "model.lp")
    _, symbol_map_id = m.write(path, io_options={"symbolic_solver_labels": True})
    by_symbol = m.solutions.symbol_map[symbol_map_id].bySymbol

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(path)
    h.run()
    status, r = h.getRanging()
    lp, sol = h.getLp(), h.getSolution()

    n = lp.num_col_
    costs = pd.DataFrame({
        "name": [by_symbol[c].name for c in lp.col_names_],
        "value": sol.col_value,
        "coefficient": lp.col_cost_,
        "lowest": r.col_cost_dn.value_[:n],
        "highest": r.col_cost_up.value_[:n],
    }).set_index("name")
    limits = pd.DataFrame({
        "name": [by_symbol[c].name for c in lp.row_names_],
        "activity": sol.row_value,
        "limit": lp.row_upper_,
        "shadow_price": sol.row_dual,
        "lowest": r.row_bound_dn.value_,
        "highest": r.row_bound_up.value_,
        "profit_at_lowest": r.row_bound_dn.objective_,
        "profit_at_highest": r.row_bound_up.objective_,
    }).set_index("name")
    # A limit with spare capacity can fall to what is used, or rise without end, and nothing changes.
    # (HiGHS reports a different quantity for such rows, so we state the simple answer ourselves.)
    spare = (limits.limit - limits.activity > 1e-7) & (limits.shadow_price.abs() < 1e-9)
    objective = h.getInfo().objective_function_value
    limits.loc[spare, ["lowest", "highest"]] = list(zip(limits.activity[spare], [float("inf")] * spare.sum()))
    limits.loc[spare, ["profit_at_lowest", "profit_at_highest"]] = objective
    return costs, limits
```

Two details:

- `getRanging()` returns arrays for columns (variables) and rows (constraints). Each has a `value_` (the limit of the range) and an `objective_` (the profit at that limit).
- For a constraint with spare capacity, the answer is simple and we state it ourselves: the limit can fall to what is used, or rise forever, and nothing changes.

## 12.3 Objective ranges: how stable is the plan?

```python
# ---------- 12.2 Objective ranging: how far can each product's profit move? ----------
costs, limits = ranging(m)
costs = costs.rename(index=lambda n: n[2:-1])          # "x[BK-03]" -> "BK-03"
costs["product"] = [d["name"][p] for p in costs.index]
costs["batches"] = costs.value.round(2)
show = costs[["product", "batches", "coefficient", "lowest", "highest"]].round(2)
print(show.to_string())
```

```text
                              product  batches  coefficient  lowest  highest
name
BK-03                        Rye loaf     25.0        267.0  229.97      inf
BK-04                 Multigrain loaf     18.0        294.0  186.79      inf
BK-05                 Baguette 2-pack      7.0        317.0  265.29      inf
BK-06              Country white loaf      8.0        246.0  175.40      inf
BK-07                 Olive sourdough      0.0        227.0    -inf   229.97
BK-08                Seeded sourdough      0.0        248.0    -inf   299.51
BK-09                 Ciabatta 4-pack      8.0        214.0  186.31      inf
BK-10                    Raisin bread     20.0        350.0  215.62      inf
BK-12         Pain au chocolat 4-pack     15.0        282.0  244.45      inf
BK-13         Almond croissant 4-pack     27.0        262.5  205.20      inf
BK-14            Cinnamon roll 4-pack     26.0        238.5  225.07      inf
BK-15                   Danish 4-pack      0.0        208.0    -inf   225.07
BK-16                    Brioche loaf      9.0        295.5  235.00      inf
BK-17           Apple turnover 4-pack      9.0        289.0  215.13      inf
BK-18              Butter tart 6-pack     25.0        235.0  224.38   238.30
BK-19                     Carrot cake      0.0        199.0    -inf   322.94
BK-20                 Lemon loaf cake      0.0        219.2    -inf   313.50
BK-21               Black forest cake      0.0        256.0    -inf   273.27
BK-22                      Cheesecake      0.0        245.4    -inf   253.89
BK-23                    Banana bread     20.0        285.6  263.83      inf
BK-24                Maple layer cake      0.0        193.8    -inf   262.36
BK-25         Oatmeal cookies 12-pack     16.0        196.2  174.42      inf
BK-26  Chocolate chip cookies 12-pack      1.6        193.8  189.73   208.36
BK-27             Nanaimo bars 6-pack     25.0        237.0  203.73      inf
BK-28             Date squares 6-pack     20.0        227.4  184.35      inf
BK-29              Shortbread 12-pack     19.0        177.0  164.97      inf
BK-30                 Brownies 6-pack     20.0        229.2  213.18      inf
BK-CR                Croissant 6-pack     31.0        250.0  205.69      inf
BK-SD                  Sourdough loaf     35.0        300.0  297.03   337.03
BK-WW                Whole-wheat loaf     45.0        200.0  146.57      inf
```

How to read a row: as long as raisin bread's profit stays between $215.62 and infinity, the bakery should keep baking its full 20 batches. For an unbaked product, the highest value is its profit plus its reduced cost (olive sourdough: $229.97).

The plan is **robust** for most products: their profit can fall $30 to $100 a batch before anything changes. Three are **fragile**:

- **Sourdough** ($300; range $297.03 to $337.03): a $2.97 drop in profit per batch, **3 cents a loaf**, changes the plan. The reason: olive sourdough, $2.97 short of entering, competes with it for starter. This is Day 6's and the capstone's sourdough surprise, now measured.
- **Butter tarts** ($224.38 to $238.30) and **chocolate chip cookies** ($189.73 to $208.36): the two partial products, which move with any change.

### Try it 12.1

In the 3-product bakery from Module 1, what is the range for each product's profit? Compare sourdough's range with Day 3's two-product answer ($200 to $400).

<details><summary>Solution</summary>

```python
d = load_data(con, "module1_products", "bakery_recipe", "bakery_resources", "module1_forecast")
m = build_model(d)
solve(m)
costs, limits = ranging(m)
print(costs.round(2))
```

```text
          value  coefficient  lowest  highest
name                                         
x[BK-CR]   10.0        250.0  225.00   300.00
x[BK-SD]   20.0        300.0  166.67   333.33
x[BK-WW]   45.0        200.0  166.67      inf
```

(Create the two `module1_` views as in `day11/shadow_prices.py`.) With croissants in the model, sourdough's range is $166.67 to $333.33, not $200 to $400: a third product changes which corner neighbours the current one. Whole-wheat can rise without limit (it is already at its cap).

</details>

## 12.4 Right-hand-side ranges: how far a shadow price holds

```python
# ---------- 12.3 Right-hand-side ranging: over what range does each shadow price hold? ----------
res = limits[limits.index.str.startswith("capacity")].round(2)
print(res[["activity", "limit", "shadow_price", "lowest", "highest", "profit_at_lowest", "profit_at_highest"]].to_string())

for oven in (589, 590, 600, 602, 603, 610):
    m3 = build_model(dict(d, capacity=dict(d["capacity"], oven_hours=oven)))
    solve(m3)
    predicted = base + 99.34 * (oven - 600)
    print(f"oven {oven}: re-solved ${pyo.value(m3.total_profit):,.2f}, shadow price predicts ${predicted:,.2f}")
```

```text
                                   activity  limit  shadow_price  lowest  highest  profit_at_lowest  profit_at_highest
name
capacity_limit[baker_hours]          539.62  560.0          0.00  539.62      inf         108351.28          108351.28
capacity_limit[oven_hours]           600.00  600.0         99.34  589.43   602.29         107301.11          108578.34
capacity_limit[packing_hours]        300.00  300.0         94.46  299.06   304.35         108262.38          108762.46
capacity_limit[sourdough_starter]     60.00   60.0         54.09   57.09    60.00         108193.93          108351.28
oven 589: re-solved $107,257.32, shadow price predicts $107,258.54
oven 590: re-solved $107,357.88, shadow price predicts $107,357.88
oven 600: re-solved $108,351.28, shadow price predicts $108,351.28
oven 602: re-solved $108,549.96, shadow price predicts $108,549.96
oven 603: re-solved $108,642.02, shadow price predicts $108,649.30
oven 610: re-solved $109,182.01, shadow price predicts $109,344.68
```

Day 11's oven shadow price, **$99.34, holds only between 589.43 and 602.29 oven hours**. Inside that range, shadow price × change is exact (oven +2: $198.68 both ways). Outside it, the plan changes and the prediction drifts: at +10 hours the shadow price overstates the gain by $163.

The starter's range is **57.09 to 60.00**: its price holds downward but not upward at all. Another degenerate corner (Day 11): one more batch of starter adds $51.12, not $54.09 (re-solve and see).

Ranges are narrow in a 30-product model: each extra hour shifts some product to its cap or off it, and the next hour is worth a little less. For anything bigger than a few hours, re-solve. Day 13 makes that cheap.

## Mini-project: which prices would change the plan?

The pricing team reviews transfer prices next month. Before they do, tell them which products' plans would change with a small price move. For each product, convert the objective range into **cents per loaf or pack** (profit per batch ÷ units per batch), in both directions, and rank products by the smallest move that changes the plan, as a share of their price.

<details><summary>Solution</summary>

```python
import sys

import duckdb
import pandas as pd

sys.path.insert(0, "day06")
sys.path.insert(0, "day12")
from bakery import build_model, load_data, solve
from ranging import ranging

pd.set_option("display.width", 140)
con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_data(con)
m = build_model(d)
solve(m)
costs, _ = ranging(m)
costs = costs.rename(index=lambda n: n[2:-1])

units = dict(con.sql("SELECT sku, units_per_batch FROM bakery_range").fetchall())
price = dict(con.sql("SELECT sku, transfer_price FROM bakery_range").fetchall())
t = pd.DataFrame({
    "product": [d["name"][p] for p in costs.index],
    "batches": costs.value.round(2),
    # how many cents per loaf or pack the price can fall, or must rise, before the plan changes
    "price_can_fall_cents": ((costs.coefficient - costs.lowest) / pd.Series(units) * 100).round(1),
    "price_must_rise_cents": ((costs.highest - costs.coefficient) / pd.Series(units) * 100).round(1),
    "price": pd.Series(price),
}, index=costs.index)
t["smallest_move_pct"] = (t[["price_can_fall_cents", "price_must_rise_cents"]].min(axis=1) / t.price).round(2)
print(t.sort_values("smallest_move_pct").head(8).to_string())
```

```text
                              product  batches  price_can_fall_cents  price_must_rise_cents  price  smallest_move_pct
name
BK-SD                  Sourdough loaf     35.0                   3.0                   37.0   4.50               0.67
BK-07                 Olive sourdough      0.0                   inf                    3.0   3.55               0.85
BK-18              Butter tart 6-pack     25.0                  21.2                    6.6   6.95               0.95
BK-26  Chocolate chip cookies 12-pack      1.6                   6.8                   24.3   5.00               1.36
BK-22                      Cheesecake      0.0                   inf                   42.5  19.90               2.14
BK-14            Cinnamon roll 4-pack     26.0                  26.9                    inf   7.15               3.76
BK-21               Black forest cake      0.0                   inf                   86.4  21.05               4.10
BK-30                 Brownies 6-pack     20.0                  26.7                    inf   6.35               4.20
```

*Memo to pricing:* "Three price moves would change next week's bakery plan by themselves: sourdough down 3 cents a loaf (0.7%), olive sourdough up 3 cents (0.9%), butter tarts up 7 cents a pack (1%). Every other product's price can move several percent without changing the plan. Sourdough and olive sourdough compete for the same starter, so treat their prices together."

`pd.Series(units)` lines up the units per batch with the products by SKU; the division works product by product.

</details>

## Tests for Days 11 and 12

Shadow prices, reduced costs and ranges are claims you can test against re-solving. Add `"day12"` to the folders in `tests/conftest.py`, then create `tests/test_sensitivity.py`:

```python
"""Module 2, Days 11-12: shadow prices, reduced costs and ranging agree with re-solving."""
import pyomo.environ as pyo
import pytest
from bakery import build_model, load_data, solve
from ranging import ranging


@pytest.fixture(scope="module")
def full(con):
    d = load_data(con)
    m = build_model(d)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solve(m)
    return d, m


def resolve(d, **capacity):
    m = build_model(dict(d, capacity=dict(d["capacity"], **capacity)))
    solve(m)
    return pyo.value(m.total_profit)


def test_shadow_price_matches_a_small_resolve(full):
    d, m = full
    base = pyo.value(m.total_profit)
    assert resolve(d, oven_hours=601) - base == pytest.approx(m.dual[m.capacity_limit["oven_hours"]], abs=1e-6)
    assert resolve(d, packing_hours=301) - base == pytest.approx(m.dual[m.capacity_limit["packing_hours"]], abs=1e-6)


def test_complementary_slackness(full):
    d, m = full
    for r in m.R:
        slack = d["capacity"][r] - pyo.value(m.capacity_limit[r].body)
        assert slack * m.dual[m.capacity_limit[r]] == pytest.approx(0, abs=1e-6)


def test_reduced_cost_is_profit_minus_resources_at_shadow_prices(full):
    d, m = full
    for p in m.P:
        if pyo.value(m.x[p]) < 1e-6:
            worth = sum(d["use"].get((r, p), 0) * m.dual[m.capacity_limit[r]] for r in m.R)
            assert m.rc[m.x[p]] == pytest.approx(d["profit"][p] - worth, abs=1e-6)


def test_oven_range_holds_inside_and_breaks_outside(full):
    d, m = full
    _, limits = ranging(m)
    oven = limits.loc["capacity_limit[oven_hours]"]
    price, base = oven.shadow_price, pyo.value(m.total_profit)
    inside = oven.highest - 0.5
    outside = oven.highest + 5
    assert resolve(d, oven_hours=inside) - base == pytest.approx(price * (inside - 600), abs=1e-4)
    assert resolve(d, oven_hours=outside) - base < price * (outside - 600) - 1
```

```text
26 passed
```

Each test states a theorem from today and yesterday as a check on your data: a shadow price equals a small re-solve; slack times price is zero; a reduced cost is profit minus resources at shadow prices; a shadow price holds inside its range and fails outside it.

## Commit

```bash
git add .
git commit -m "Day 12: reduced costs, ranging with highspy, price fragility"
git push
```

### Day 12 checklist

- [ ] I can get reduced costs from Pyomo with an `rc` suffix and explain them as profit minus resources at shadow prices.
- [ ] I verified a reduced cost by re-solving just below and above it.
- [ ] I can compute objective and right-hand-side ranges with HiGHS.
- [ ] I know a shadow price holds only inside its range, and that ranges can be narrow.
- [ ] I found the fragile products in the plan, and why sourdough is one of them.
- [ ] I can translate a range into cents per loaf for the pricing team.

---

# DAY 13 — What-If Analysis at Scale

Shadow prices and ranges answer small questions about one number at a time. Managers ask big ones: "what if we lose two ovens?", "what if butter goes up 20%?", "what if we lease ovens **and** add a shift?". Today you answer them by re-solving, many times, cheaply: sweeps over one or two limits, a scenario table stored in the database, and an honest comparison of prediction against re-solving.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 13.1 Sweeps: one limit, fifty values |
| 0:45 – 1:45 | 13.2 Two limits at once |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 13.3 A scenario table |
| 2:45 – 3:45 | 13.4 Predict with shadow prices, or re-solve? |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: which investments to make |
| 4:45 – 5:00 | Commit with Git + checklist |

## 13.1 Sweeps: one limit, fifty values

A **sweep** solves the model for a list of values of one parameter. Two ways to do it: rebuild the model each time, or build it once and change one constraint with `set_value` (Day 7). Create `day13/sweeps.py`:

```python
import os
import sys
import time

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo

sys.path.insert(0, "day06")
from bakery import build_model, load_data, solve

os.makedirs("figures", exist_ok=True)
con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_data(con)
hours = list(range(400, 901, 10))

# ---------- 13.1 Rebuild for every value, or build once and change one limit ----------
start = time.perf_counter()
rebuilt = []
for h in hours:
    m = build_model(dict(d, capacity=dict(d["capacity"], oven_hours=h)))
    solve(m)
    rebuilt.append(pyo.value(m.total_profit))
t_rebuild = time.perf_counter() - start

start = time.perf_counter()
m = build_model(d)
oven = m.capacity_limit["oven_hours"]
changed = []
for h in hours:
    oven.set_value(oven.body <= h)
    solve(m)
    changed.append(pyo.value(m.total_profit))
t_change = time.perf_counter() - start
print(f"{len(hours)} solves: rebuild {t_rebuild:.2f} s, change one limit {t_change:.2f} s, "
      f"same answers: {np.allclose(rebuilt, changed)}")

profit = np.array(changed)
slope = np.diff(profit) / 10                      # $ per extra oven hour, between sweep points
for h, p, s in zip(hours[::10], profit[::10], list(slope[::10]) + [None]):
    print(f"  oven {h} h: ${p:,.2f}" + (f"  (next 10 h worth ${s:,.2f}/h)" if s is not None else ""))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(hours, profit / 1000, "-", color="#4f7bd9")
a1.axvline(600, color="grey", ls=":")
a1.set_xlabel("oven hours a week")
a1.set_ylabel("profit ($ thousand a week)")
a1.set_title("Profit as oven hours change")
a2.step(hours[1:], slope, where="pre", color="#e08a2c")
a2.axvline(600, color="grey", ls=":")
a2.set_xlabel("oven hours a week")
a2.set_ylabel("$ per extra oven hour")
a2.set_title("Value of one more oven hour")
fig.tight_layout()
fig.savefig("figures/day13_oven_sweep.png", dpi=120)
```

```text
51 solves: rebuild 0.35 s, change one limit 0.23 s, same answers: True
  oven 400 h: $79,802.73  (next 10 h worth $165.94/h)
  oven 500 h: $95,577.40  (next 10 h worth $149.07/h)
  oven 600 h: $108,351.28  (next 10 h worth $83.07/h)
  oven 700 h: $109,434.90  (next 10 h worth $0.00/h)
  oven 800 h: $109,434.90  (next 10 h worth $0.00/h)
  oven 900 h: $109,434.90
```

(Your times will differ.) Fifty-one solves of the 30-product model take a third of a second either way. On models this size, rebuild if it keeps the code simple; on big models (Module 3), changing the model in place matters.

![Oven sweep](figures/day13_oven_sweep.png)

The left chart is profit against oven hours: a line made of straight pieces, each steeper than the next one to its right. The right chart is its slope, the **value of one more oven hour**: a staircase going down. Each step is a range from Day 12; each riser is a product reaching its cap or leaving the plan. Three things to see:

- **Below 600 hours** an oven hour is worth $100 to $165: ovens are the bakery's main bottleneck.
- **Just above 600** the value drops fast, to $83 and then $25.
- **Beyond about 620 hours an oven hour is worth nothing**: packing and starter bind instead. Leasing ovens beyond that point would buy idle capacity.

Every LP's value curve has this shape (concave and piecewise linear): **diminishing returns** are built into linear programming.

## 13.2 Two limits at once

Limits interact. Sweep oven and packing hours together:

```python
# ---------- 13.2 Two limits at once: oven hours x packing hours ----------
ovens = list(range(500, 801, 50))
packs = list(range(240, 381, 20))
grid = np.zeros((len(packs), len(ovens)))
pack = m.capacity_limit["packing_hours"]
for i, ph in enumerate(packs):
    pack.set_value(pack.body <= ph)
    for j, oh in enumerate(ovens):
        oven.set_value(oven.body <= oh)
        solve(m)
        grid[i, j] = pyo.value(m.total_profit)
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(grid / 1000, origin="lower", cmap="viridis", aspect="auto",
               extent=[ovens[0] - 25, ovens[-1] + 25, packs[0] - 10, packs[-1] + 10])
for i, ph in enumerate(packs):
    for j, oh in enumerate(ovens):
        light = grid[i, j] > grid.min() + 0.6 * (grid.max() - grid.min())
        ax.text(oh, ph, f"{grid[i, j] / 1000:.1f}", ha="center", va="center", fontsize=8,
                color="black" if light else "white")
ax.add_patch(plt.Rectangle((575, 290), 50, 20, fill=False, edgecolor="red", lw=2))
ax.set_xlabel("oven hours a week")
ax.set_ylabel("packing hours a week")
ax.set_title("Weekly profit ($ thousand); red box = today")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig("figures/day13_oven_packing.png", dpi=120)
print("row = packing hours, column = oven hours, profit in $ thousand")
print("      " + "".join(f"{o:>7}" for o in ovens))
for i, ph in enumerate(packs):
    print(f"{ph:>5} " + "".join(f"{grid[i, j] / 1000:>7.1f}" for j in range(len(ovens))))
```

```text
row = packing hours, column = oven hours, profit in $ thousand
          500    550    600    650    700    750    800
  240    92.1   96.1   96.4   96.4   96.4   96.4   96.4
  260    94.2   99.1  101.3  101.3  101.3  101.3  101.3
  280    95.6  101.5  105.6  105.7  105.7  105.7  105.7
  300    95.6  103.0  108.4  109.4  109.4  109.4  109.4
  320    95.6  103.0  109.9  111.7  111.7  111.7  111.7
  340    95.6  103.0  109.9  111.7  111.7  111.7  111.7
  360    95.6  103.0  109.9  111.7  111.7  111.7  111.7
  380    95.6  103.0  109.9  111.7  111.7  111.7  111.7
```

![Oven and packing hours](figures/day13_oven_packing.png)

Read the grid from today's plan (the red box, $108.4 thousand):

- **More oven hours alone** (moving right) stop paying at 650: $109.4 thousand, whatever the ovens.
- **More packing alone** (moving up) stops at 320: $109.9 thousand.
- **Both together** reach $111.7 thousand: more than either alone, and more than the sum of the two gains. Ovens and packing are **complements**: each is worth more when you have more of the other.

A one-at-a-time analysis would miss this. Shadow prices are one-at-a-time by construction.

## 13.3 A scenario table

Real what-ifs change several numbers at once, and managers want to compare many of them. Keep **scenarios as data**: a CSV with one row per change, read into the database, so anyone can add a scenario without touching code. Create `day13/scenarios.csv`:

```text
scenario,change,target,value,description
base,none,,,today's plan
lose_two_ovens,capacity_add,oven_hours,-100,two deck ovens out for repair
extra_ovens,capacity_add,oven_hours,100,lease two more deck ovens
extra_baker,capacity_add,baker_hours,40,hire one more baker
packing_shift,capacity_add,packing_hours,40,third packing shift on one line
starter_tank,capacity_add,sourdough_starter,12,second sourdough starter tank
ovens_and_packing,capacity_add,oven_hours,100,lease two ovens ...
ovens_and_packing,capacity_add,packing_hours,40,... and add the packing shift
butter_up_20,cost_multiply,viennoiserie,1.20,butter price rises 20% (viennoiserie ingredients)
cookie_demand_up,demand_multiply,cookie,1.5,cookie promotion lifts demand 50%
demand_down_10,demand_multiply,all,0.90,a weak week: all demand down 10%
```

`ovens_and_packing` has two rows: both changes apply. Three kinds of change cover these scenarios: add to a capacity, multiply demand caps (for a category, or `all`), multiply ingredient costs for a category (which lowers profit per batch). Then `day13/run_scenarios.py`:

```python
import copy
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day06")
from bakery import build_model, load_data, solve

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb")
con.sql("CREATE OR REPLACE TABLE scenarios AS SELECT * FROM read_csv('day13/scenarios.csv')")
base_data = load_data(con)
ingredients = dict(con.sql("SELECT sku, units_per_batch * ingredient_cost FROM bakery_range").fetchall())


def apply(d, change, target, value):
    """Return a copy of the model data with one change applied."""
    d = copy.deepcopy(d)
    in_target = [p for p in d["products"] if target == "all" or d["category"][p] == target]
    if change == "capacity_add":
        d["capacity"][target] += value
    elif change == "demand_multiply":
        for p in in_target:
            d["cap"][p] *= value
    elif change == "cost_multiply":
        for p in in_target:
            d["profit"][p] -= ingredients[p] * (value - 1)
    elif change != "none":
        raise ValueError(f"unknown change {change}")
    return d


def run(d):
    m = build_model(d)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solve(m)
    binding = [r for r in m.R if m.dual[m.capacity_limit[r]] > 1e-6]
    return m, pyo.value(m.total_profit), binding


results = []
base_m, base_profit, _ = run(base_data)
for name, rows in con.sql("SELECT * FROM scenarios").df().groupby("scenario", sort=False):
    d = base_data
    for r in rows.itertuples():
        d = apply(d, r.change, r.target, r.value)
    m, profit, binding = run(d)
    moved = sum(abs(pyo.value(m.x[p]) - pyo.value(base_m.x[p])) > 1e-6 for p in m.P)
    results.append((name, round(profit, 2), round(profit - base_profit, 2), moved, ", ".join(b.replace("_hours", "") for b in binding)))

out = pd.DataFrame(results, columns=["scenario", "profit", "change_vs_base", "products_changed", "binding_limits"])
con.sql("CREATE OR REPLACE TABLE scenario_results AS SELECT * FROM out")
print(con.sql("SELECT * FROM scenario_results ORDER BY change_vs_base DESC"))
con.close()
```

```text
┌───────────────────┬───────────┬────────────────┬──────────────────┬─────────────────────────────────────────┐
│     scenario      │  profit   │ change_vs_base │ products_changed │             binding_limits              │
│      varchar      │  double   │     double     │      int64       │                 varchar                 │
├───────────────────┼───────────┼────────────────┼──────────────────┼─────────────────────────────────────────┤
│ ovens_and_packing │  111657.2 │        3305.92 │                3 │ baker, sourdough_starter                │
│ packing_shift     │  109906.6 │        1555.32 │                5 │ baker, oven, sourdough_starter          │
│ extra_ovens       │  109434.9 │        1083.62 │                5 │ baker, packing, sourdough_starter       │
│ cookie_demand_up  │ 108993.33 │         642.05 │                7 │ oven, packing, sourdough_starter        │
│ starter_tank      │ 108956.52 │         605.24 │                4 │ oven, packing, sourdough_starter        │
│ base              │ 108351.28 │            0.0 │                0 │ oven, packing, sourdough_starter        │
│ extra_baker       │ 108351.28 │            0.0 │                0 │ oven, packing, sourdough_starter        │
│ demand_down_10    │ 106183.95 │       -2167.33 │               27 │ baker, oven, packing, sourdough_starter │
│ butter_up_20      │ 105165.89 │       -3185.39 │                4 │ oven, packing, sourdough_starter        │
│ lose_two_ovens    │   95577.4 │      -12773.88 │                5 │ oven                                    │
└───────────────────┴───────────┴────────────────┴──────────────────┴─────────────────────────────────────────┘
  10 rows                                                                                           5 columns
```

`copy.deepcopy` gives each scenario its own copy of the data, so changes never leak into the next scenario: a classic source of silent errors in what-if code. `products_changed` counts products whose batches moved; `binding_limits` lists limits with a positive shadow price.

What the table says:

- **Losing two ovens costs $12,774 a week**, 12% of profit. Ovens become the only binding limit.
- **Ovens and packing together** add $3,306, more than ovens ($1,084) plus packing ($1,555): the complementarity from 13.2.
- **An extra baker adds nothing** on its own: bakers are not a bottleneck. Remember this for the mini-project.
- **A 20% butter rise costs $3,185**, but changes only 4 products: the viennoiserie mostly stays in the plan, earning less.
- **A weak week (demand −10%)** costs $2,167 and moves 27 of 30 products: with less demand, caps bind, and every limit binds, bakers included.

The results are stored in `scenario_results`, so the next step (a dashboard, a memo, Module 8's app) reads them from the database.

> **Decision it supports:** a ranked list of what could go right or wrong, with a dollar value on each.
>
> **Other decisions you can support the same way:** budget scenarios for a hospital · supply disruption plans · regulatory change impact studies.

## 13.4 Predict with shadow prices, or re-solve?

Shadow prices give instant predictions; re-solves give answers. How far apart are they?

```python
import sys

import duckdb
import pyomo.environ as pyo

sys.path.insert(0, "day06")
from bakery import build_model, load_data, solve

con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_data(con)
m = build_model(d)
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
solve(m)
base = pyo.value(m.total_profit)
price = {r: m.dual[m.capacity_limit[r]] for r in m.R}

print(f"{'change':<26}{'predicted':>12}{'re-solved':>12}{'error':>10}")
for r, delta in [("oven_hours", 2), ("oven_hours", 10), ("oven_hours", 100), ("oven_hours", -100),
                 ("packing_hours", 4), ("packing_hours", 40), ("sourdough_starter", 12), ("baker_hours", 40)]:
    m2 = build_model(dict(d, capacity=dict(d["capacity"], **{r: d["capacity"][r] + delta})))
    solve(m2)
    predicted = price[r] * delta
    actual = pyo.value(m2.total_profit) - base
    print(f"{r + f' {delta:+d}':<26}{predicted:>12,.2f}{actual:>12,.2f}{actual - predicted:>10,.2f}")
```

```text
change                       predicted   re-solved     error
oven_hours +2                   198.68      198.68     -0.00
oven_hours +10                  993.40      830.73   -162.67
oven_hours +100               9,934.00    1,083.62 -8,850.38
oven_hours -100              -9,934.00  -12,773.88 -2,839.88
packing_hours +4                377.84      377.84     -0.00
packing_hours +40             3,778.40    1,555.32 -2,223.08
sourdough_starter +12           649.08      605.24    -43.84
baker_hours +40                   0.00        0.00      0.00
```

Inside the ranges from Day 12 (oven +2, packing +4) the prediction is exact. Beyond them it is always **too optimistic** for increases (it ignores diminishing returns) and **too optimistic** for losses too (losing 100 oven hours costs $12,774, not the predicted $9,934). For 100 extra oven hours the prediction is off by a factor of nine.

| Use shadow prices to… | Re-solve to… |
|---|---|
| Rank which limits matter | Value any change bigger than its range |
| Explain a plan to a manager | Answer any question about money |
| Spot complementary slackness (overtime at $60) | Combine several changes |

## Mini-project: which investments to make

The plant manager has three options, each with a weekly cost, plus packing overtime by the hour:

| Option | Adds | Cost a week |
|---|---|---|
| Lease 2 deck ovens | 100 oven hours | $900 |
| Hire a baker | 40 baker hours | $1,600 |
| Second starter tank | 12 batches of starter | $250 |
| Packing overtime | up to 40 hours | $60 an hour |

Evaluate **all 8 combinations** of the three yes/no options. In each, let the model choose packing overtime hours (a variable, as in the capstone). Rank the combinations by net weekly profit after costs, and recommend.

<details><summary>Solution</summary>

```python
import itertools
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day06")
from bakery import build_model, load_data, solve

pd.set_option("display.width", 140)
con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_data(con)

OPTIONS = {   # name: (resource, extra capacity a week, cost a week in $)
    "lease 2 ovens": ("oven_hours", 100, 900),
    "hire a baker": ("baker_hours", 40, 1600),
    "starter tank": ("sourdough_starter", 12, 250),
}
OVERTIME_PACKING = (40, 60)   # up to 40 hours a week at $60 an hour

rows = []
for choice in itertools.product([False, True], repeat=len(OPTIONS)):
    cap = dict(d["capacity"])
    fixed = 0
    for take, (name, (r, extra, cost)) in zip(choice, OPTIONS.items()):
        if take:
            cap[r] += extra
            fixed += cost
    m = build_model(dict(d, capacity=cap))
    # packing overtime as a variable: the model books as many hours as pay
    m.ot = pyo.Var(bounds=(0, OVERTIME_PACKING[0]))
    pk = m.capacity_limit["packing_hours"]
    pk.set_value(pk.body <= cap["packing_hours"] + m.ot)
    m.total_profit.deactivate()
    m.net = pyo.Objective(expr=sum(m.profit[p] * m.x[p] for p in m.P) - OVERTIME_PACKING[1] * m.ot, sense=pyo.maximize)
    solve(m)
    rows.append((" + ".join(n for t, n in zip(choice, OPTIONS) if t) or "nothing", round(pyo.value(m.ot), 2),
                 round(pyo.value(m.net), 2), fixed, round(pyo.value(m.net) - fixed, 2)))

t = pd.DataFrame(rows, columns=["investments", "packing_overtime_h", "profit_after_overtime", "weekly_cost", "net"])
t["vs_nothing"] = (t.net - t.net[t.investments == "nothing"].iloc[0]).round(2)
print(t.sort_values("net", ascending=False).to_string(index=False))
```

```text
                                investments  packing_overtime_h  profit_after_overtime  weekly_cost       net  vs_nothing
lease 2 ovens + hire a baker + starter tank               31.66              115738.56         2750 112988.56     4151.27
               lease 2 ovens + hire a baker               28.78              114041.28         2500 111541.28     2703.99
               lease 2 ovens + starter tank               16.62              112470.58         1150 111320.58     2483.29
                              lease 2 ovens               13.25              110862.50          900 109962.50     1125.21
                               starter tank               12.92              109359.46          250 109109.46      272.17
                                    nothing               17.66              108837.29            0 108837.29        0.00
                hire a baker + starter tank               12.92              109359.46         1850 107509.46    -1327.83
                               hire a baker               17.66              108837.29         1600 107237.29    -1600.00
```

*Recommendation:* "Do all three: lease the ovens, hire a baker and install the starter tank, and book about 32 hours of packing overtime a week. Net gain **$4,151 a week**. The baker is the surprise: alone he loses $1,600 a week, because bakers are not a bottleneck today; with two more ovens, bakers become one, and adding the baker to the oven lease raises the net gain by a further $1,579 a week after his cost."

`itertools.product([False, True], repeat=3)` lists the 8 combinations. Eight solves are fine; with 30 options there would be a billion combinations, and the yes/no choices belong inside the model as **binary variables**. That is Module 3.

</details>

## Commit

```bash
git add .
git commit -m "Day 13: sweeps, scenario table, investments"
git push
```

### Day 13 checklist

- [ ] I can sweep a parameter by rebuilding the model or by changing one constraint.
- [ ] I can read a value curve and its staircase of marginal values.
- [ ] I found complementary limits with a two-way sweep.
- [ ] My scenarios are data in a CSV and a table, and the results are stored in the database.
- [ ] I know when a shadow price predicts well and when to re-solve.
- [ ] I evaluated every combination of three investments and found why a useless baker becomes useful.
