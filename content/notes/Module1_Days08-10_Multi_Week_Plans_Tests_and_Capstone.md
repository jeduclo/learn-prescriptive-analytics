# DAY 8 — Multi-Period Production and Inventory

Until now every plan covered one week. But Christmas is coming: demand for cookies and viennoiserie jumps 45–80%, and Christmas week has only 60% of the usual oven and baker hours. A bakery that plans one week at a time meets the peak empty-handed. Today you plan **six weeks at once**, with a freezer that lets products baked early be sold later.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 8.1 Two weeks, one product, by hand |
| 0:45 – 1:45 | 8.2 The model: make, sell, keep |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 8.3 Six weeks of the full range |
| 2:45 – 3:45 | 8.4 Horizons and end effects; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: the freezer is replaced in December |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page, then rebuild the data and the database:

```bash
python make_data.py
python day01/ladder.py
python day06/load_tables.py
```

Four new tables matter today:

| Table | What it holds |
|---|---|
| `plan_weeks` | 12 weeks from 30 November 2026, with each week's share of normal capacity (Christmas week 0.6, New Year's week 0.8, Family Day week 0.8) |
| `plan_demand` | Demand for each of the 30 products in each week, in batches, with the Christmas peak (bread ×1.15, viennoiserie ×1.45, cakes ×1.6, cookies ×1.8 in weeks 3 and 4) |
| `bakery_storage` | For each category: how many weeks a batch keeps, the cost of holding it a week, and the freezer slots it takes |
| `plan_settings` | Freezer size (120 slots), overtime limits and prices, the café contract, the shelf rule |

```text
category,shelf_life_weeks,holding_cost_per_batch_week,freezer_slots_per_batch,note
bread,0,0,0,baked and sold the same week
viennoiserie,3,8.0,1.0,"frozen unbaked, baked off in stores"
cake,2,12.0,1.5,frozen finished cakes
cookie,4,3.0,0.5,"packed, shelf-stable"
```

Bread cannot be stored: it is baked and sold the same week. Everything else can wait in the freezer or dry store for a few weeks, at a price.

## 8.1 Two weeks, one product, by hand

The smallest version of the idea. Croissants: the bakery can make 10 batches a week; demand is 5 batches in week 1 and 15 in week 2. A batch sells for $350, its ingredients cost $100, and a batch kept in the freezer for a week costs $8.

Week by week, the bakery makes 5 in week 1 (no point making more) and 10 in week 2 (its limit): 15 batches sold. Planning both weeks together, it makes 10 in week 1, sells 5 and **keeps 5**, then makes 10 more in week 2 and sells 15. That is 20 sold.

Three variables per product and week do the job:

- **make** $$m_{p,t}$$: batches baked in week $$t$$
- **sell** $$s_{p,t}$$: batches sold in week $$t$$
- **keep** $$k_{p,t}$$: batches in stock at the end of week $$t$$

tied together by a **balance** constraint, one per product and week: what you had, plus what you bake, equals what you sell plus what you keep.

$$
k_{p,t-1} + m_{p,t} = s_{p,t} + k_{p,t}
$$

Create `day08/toy.py`:

```python
import pyomo.environ as pyo

m = pyo.ConcreteModel()
m.T = pyo.Set(initialize=[1, 2], ordered=True)
demand = {1: 5, 2: 15}              # batches of croissants
m.make = pyo.Var(m.T, bounds=(0, 10))   # at most 10 batches a week
m.sell = pyo.Var(m.T, within=pyo.NonNegativeReals)
m.keep = pyo.Var(m.T, within=pyo.NonNegativeReals)
m.balance = pyo.Constraint(m.T, rule=lambda m, t: (m.keep[t - 1] if t > 1 else 0) + m.make[t] == m.sell[t] + m.keep[t])
m.sales = pyo.Constraint(m.T, rule=lambda m, t: m.sell[t] <= demand[t])
m.profit = pyo.Objective(expr=sum(350 * m.sell[t] - 100 * m.make[t] - 8 * m.keep[t] for t in m.T), sense=pyo.maximize)

pyo.SolverFactory("highs").solve(m)
for t in m.T:
    print(f"week {t}: make {pyo.value(m.make[t]):g}, sell {pyo.value(m.sell[t]):g}, keep {pyo.value(m.keep[t]):g}")
print(f"profit ${pyo.value(m.profit):,.2f}")
```

```text
week 1: make 10, sell 5, keep 5
week 2: make 10, sell 15, keep 0
profit $4,960.00
```

Check by hand: 20 batches sold × $350 = $7,000, minus 20 baked × $100 = $2,000, minus 5 kept × $8 = $40: **$4,960**.

Notice the profit is no longer "profit per batch × batches". Ingredients are paid **when you bake**, revenue arrives **when you sell**, and holding costs accrue **while you wait**. In a one-week model these moments coincide; across weeks they do not, so each gets its own term.

## 8.2 The model: make, sell, keep

For the full range we put the model in a module, `day08/planner.py`, built like Day 6's `bakery.py`: load the data, build, solve, report. On paper:

$$
\begin{aligned}
\max\quad & \sum_{p,t} \big(\text{revenue}_p\, s_{p,t} - \text{ingredients}_p\, m_{p,t} - \text{holding}_p\, k_{p,t}\big) \\
\text{s.t.}\quad & k_{p,t-1} + m_{p,t} = s_{p,t} + k_{p,t} && \text{(balance; } k_{p,0} = 0) \\
& s_{p,t} \le \text{demand}_{p,t} && \text{(sales)} \\
& \textstyle\sum_p \text{use}_{r,p}\, m_{p,t} \le \text{capacity}_{r,t} && \text{(capacity each week)} \\
& k_{p,t} \le \textstyle\sum_{t-L_p < j \le t} m_{p,j} && \text{(shelf life } L_p) \\
& \textstyle\sum_p \text{slots}_p\, k_{p,t} \le 120 && \text{(freezer)} \\
& m, s, k \ge 0
\end{aligned}
$$

The **shelf-life** rule says stock at the end of week $$t$$ must have been baked in the last $$L_p$$ weeks. For bread $$L = 0$$: the sum is empty and stock must be zero. For cookies $$L = 4$$: stock can't exceed what was baked in the last four weeks.

```python
"""Day 8: a multi-week bakery plan with storage. Used again on Days 9 and 10.

Import it:  from planner import load_plan_data, build, solve, plan_table, week_table
"""
import duckdb
import pandas as pd
import pyomo.environ as pyo


def load_plan_data(con, weeks=6):
    """Everything the multi-week model needs, as plain Python objects."""
    p = con.sql("""
        SELECT p.sku, p.product_name, p.category,
               p.units_per_batch * p.transfer_price AS revenue_per_batch,
               p.units_per_batch * p.ingredient_cost AS ingredients_per_batch,
               s.shelf_life_weeks, s.holding_cost_per_batch_week, s.freezer_slots_per_batch
        FROM bakery_range p JOIN bakery_storage s USING (category)
        ORDER BY p.sku""").df().set_index("sku")
    w = con.sql(f"SELECT week, week_start, capacity_factor FROM plan_weeks WHERE week <= {weeks} ORDER BY week").df()
    dem = con.sql(f"SELECT sku, week, demand_batches FROM plan_demand WHERE week <= {weeks}").fetchall()
    use = con.sql("SELECT resource, sku, use_per_batch FROM bakery_range_recipe").fetchall()
    cap = dict(con.sql("SELECT resource, weekly_capacity FROM bakery_range_resources").fetchall())
    settings = dict(con.sql("SELECT setting, value FROM plan_settings").fetchall())
    missing = set(p.index) - {s for s, _, _ in dem}
    if missing or len(dem) != len(p) * len(w):
        raise ValueError(f"demand must have one row per product and week; missing products: {sorted(missing)}")
    return {
        "products": p,
        "weeks": w,
        "demand": {(s, t): d for s, t, d in dem},
        "use": {(r, s): u for r, s, u in use},
        # Holidays cut oven, baker and packing hours; the starter culture works every day
        "capacity": {(r, t): c * (f if r.endswith("_hours") else 1)
                     for r, c in cap.items() for t, f in zip(w.week, w.capacity_factor)},
        "resources": list(cap),
        "freezer_slots": float(settings["freezer_slots"]),
    }


def build(d, storage=True):
    """make: batches baked in week t; sell: batches sold; keep: batches in stock at the end of week t."""
    P, products = list(d["products"].index), d["products"]
    m = pyo.ConcreteModel(name="bakery_weeks")
    m.P = pyo.Set(initialize=P)
    m.R = pyo.Set(initialize=d["resources"])
    m.T = pyo.Set(initialize=d["weeks"].week.tolist(), ordered=True)
    m.demand = pyo.Param(m.P, m.T, initialize=d["demand"])
    m.use = pyo.Param(m.R, m.P, initialize=d["use"], default=0)
    m.cap = pyo.Param(m.R, m.T, initialize=d["capacity"], mutable=True)
    m.freezer_slots = pyo.Param(initialize=d["freezer_slots"], mutable=True)

    m.make = pyo.Var(m.P, m.T, within=pyo.NonNegativeReals)
    m.sell = pyo.Var(m.P, m.T, within=pyo.NonNegativeReals)
    m.keep = pyo.Var(m.P, m.T, within=pyo.NonNegativeReals)

    def balance(m, p, t):
        before = m.keep[p, m.T.prev(t)] if t != m.T.first() else 0     # the plan starts with empty stores
        return before + m.make[p, t] == m.sell[p, t] + m.keep[p, t]

    def shelf_life(m, p, t):
        L = int(products.loc[p, "shelf_life_weeks"]) if storage else 0
        return m.keep[p, t] <= sum(m.make[p, k] for k in m.T if t - L < k <= t)

    m.balance = pyo.Constraint(m.P, m.T, rule=balance)
    m.sales_limit = pyo.Constraint(m.P, m.T, rule=lambda m, p, t: m.sell[p, t] <= m.demand[p, t])
    m.capacity = pyo.Constraint(m.R, m.T, rule=lambda m, r, t: sum(m.use[r, p] * m.make[p, t] for p in m.P) <= m.cap[r, t])
    m.shelf_life = pyo.Constraint(m.P, m.T, rule=shelf_life)
    m.freezer = pyo.Constraint(m.T, rule=lambda m, t: sum(
        products.loc[p, "freezer_slots_per_batch"] * m.keep[p, t] for p in m.P) <= m.freezer_slots)

    m.revenue = pyo.Expression(expr=sum(products.loc[p, "revenue_per_batch"] * m.sell[p, t] for p in m.P for t in m.T))
    m.ingredients = pyo.Expression(expr=sum(products.loc[p, "ingredients_per_batch"] * m.make[p, t] for p in m.P for t in m.T))
    m.holding = pyo.Expression(expr=sum(products.loc[p, "holding_cost_per_batch_week"] * m.keep[p, t] for p in m.P for t in m.T))
    m.profit = pyo.Objective(expr=m.revenue - m.ingredients - m.holding, sense=pyo.maximize)
    return m


def solve(m):
    result = pyo.SolverFactory("highs").solve(m, load_solutions=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"no optimal plan: {result.solver.termination_condition}")
    m.solutions.load_from(result)
    return result


def v(x):
    return round(pyo.value(x), 2) + 0.0          # + 0.0 turns -0.0 into 0.0


def plan_table(m, d):
    rows = [(t, p, d["products"].loc[p, "category"], v(m.make[p, t]), v(m.sell[p, t]), v(m.demand[p, t]), v(m.keep[p, t]))
            for t in m.T for p in m.P]
    df = pd.DataFrame(rows, columns=["week", "sku", "category", "make", "sell", "demand", "keep"])
    df["lost"] = (df.demand - df.sell).round(2) + 0.0
    return df


def week_table(m, d):
    slots = d["products"]["freezer_slots_per_batch"]
    rows = []
    for t in m.T:
        rows.append((t, d["weeks"].set_index("week").loc[t, "capacity_factor"],
                     v(sum(m.demand[p, t] for p in m.P)), v(sum(m.sell[p, t] for p in m.P)),
                     v(sum(slots[p] * m.keep[p, t] for p in m.P)),
                     v(sum(m.cap[r, t] - m.capacity[r, t].body for r in ["oven_hours"]))))
    return pd.DataFrame(rows, columns=["week", "capacity", "demand", "sold", "freezer_slots_used", "spare_oven_hours"])
```

Points to slow down for:

- `m.T` is an **ordered** set, so `m.T.prev(t)` and `m.T.first()` work. Writing `m.keep[p, t - 1]` instead fails in week 1 with a `KeyError`: there is no week 0. Every multi-period model needs a rule for the first period; here, the plan starts with empty stores.
- **Capacity per week:** `load_data` multiplies oven, baker and packing hours by the week's factor. The starter culture is alive every day, so it is not cut.
- `m.cap` and `m.freezer_slots` are **mutable**, so what-ifs change them without rebuilding.
- `week_table` reports the week as a whole: demand, sales, freezer use and spare oven hours.

> **Decision it supports:** how much of each product to bake each week, and what to build up ahead of a peak.
>
> **Other decisions you can support the same way:** staffing ahead of a busy season · building stock before a plant shutdown · hydro reservoir releases over a season.

## 8.3 Six weeks of the full range

Create `day08/run_weeks.py`:

```python
import sys

import duckdb
import pandas as pd

sys.path.insert(0, "day08")
from planner import build, load_plan_data, plan_table, solve, week_table

pd.set_option("display.width", 120)
con = duckdb.connect("northbound.duckdb", read_only=True)

d = load_plan_data(con, weeks=6)
m = build(d)
solve(m)
print(week_table(m, d).to_string(index=False))

plan = plan_table(m, d)
stock = plan[plan.keep > 0].pivot_table(index="sku", columns="week", values="keep", aggfunc="sum", fill_value=0)
print("Batches in stock at the end of each week:")
print(stock.join(d["products"]["product_name"]).to_string())
print(plan.groupby("category")[["make", "sell", "demand", "lost"]].sum().round(1).to_string())
print(f"profit ${m.profit():,.2f} = revenue ${m.revenue():,.2f} - ingredients ${m.ingredients():,.2f} - holding ${m.holding():,.2f}")

week_by_week = build(d, storage=False)
solve(week_by_week)
print(f"without storage: ${week_by_week.profit():,.2f}; planning ahead adds ${m.profit() - week_by_week.profit():,.2f}")
```

```text
 week  capacity  demand   sold  freezer_slots_used  spare_oven_hours
    1       1.0   602.6 365.29               43.27               0.0
    2       1.0   605.6 366.40               83.87               0.0
    3       1.0   865.9 427.01              106.36               0.0
    4       0.6   865.8 426.40                0.00               0.0
    5       0.8   552.3 349.41                0.00               0.0
    6       1.0   537.1 421.48                0.00               0.0
Batches in stock at the end of each week:
           1      2      3             product_name
sku                                                
BK-12   0.00  20.65  20.90  Pain au chocolat 4-pack
BK-16  24.67  26.60  13.40             Brioche loaf
BK-27  37.19  73.23  44.30      Nanaimo bars 6-pack
BK-28   0.00   0.00  23.84      Date squares 6-pack
BK-CR   0.00   0.00  37.98         Croissant 6-pack
                make    sell  demand   lost
category                                   
bread         1005.0  1005.0  1310.1  305.1
cake            97.4    97.4   862.7  765.3
cookie         527.9   527.9   813.7  285.8
viennoiserie   725.7   725.7  1042.8  317.1
profit $597,870.27 = revenue $909,487.71 - ingredients $309,928.06 - holding $1,689.37
without storage: $595,836.82; planning ahead adds $2,033.45
```

Read it as the bakery manager would:

- **The freezer fills before Christmas** (43, 84, then 106 slots) and is **empty after Christmas week**: everything built up is sold in week 4, when capacity drops to 60% while demand stays at its peak.
- **What gets stored** is Nanaimo bars, brioche, pain au chocolat, date squares and croissants: products that keep, earn well, and whose Christmas demand far exceeds what week 4 can bake.
- **Planning ahead adds $2,033.45** over six weeks against a week-by-week plan (`storage=False` sets every shelf life to 0). Small against a $600,000 total, because the bakery is short of capacity every single week: there is little spare time to bake ahead. Spare capacity is what makes inventory valuable.
- **Demand is far above what the bakery can make** in every week (602 batches of demand, 365 sold in week 1). That is not a planning failure; it is Days 1–7's finding, and the reason Module 3 will look at a bigger plant.

`m.profit()` is shorthand for `pyo.value(m.profit)`; Pyomo objectives and expressions can be called like functions.

### Try it 8.1

Which category loses the most demand, in batches and as a share of its demand? Using the numbers above, why does the model let it happen?

<details><summary>Solution</summary>

Cakes: 765.3 of 862.7 batches lost, **89% of cake demand**. Bread loses 305.1 of 1,310.1 (23%), cookies 35%, viennoiserie 30%. Cakes use the most baker and oven time per dollar of profit (Day 6, Try it 6.1), so when capacity is short, they are the first to go. Whether that is acceptable is a business question: Day 10 adds a rule that keeps every product on the shelf.

</details>

## 8.4 Horizons and end effects

The model plans six weeks. What happens after week 6 doesn't exist for it, so it ends with **empty stores**: nothing is kept for week 7, because stock left at the end earns nothing. That is the **end effect**: every finite plan behaves oddly near its last period.

Planning 12 weeks instead, with the same code:

```python
# Does a longer horizon change the first weeks?
d12 = load_plan_data(con, weeks=12)
m12 = build(d12)
solve(m12)
print(week_table(m12, d12).to_string(index=False))
```

```text
 week  capacity  demand   sold  freezer_slots_used  spare_oven_hours
    1       1.0   602.6 365.29               43.27               0.0
    2       1.0   605.6 366.40               83.87               0.0
    3       1.0   865.9 427.01              106.36               0.0
    4       0.6   865.8 426.40                0.00               0.0
    5       0.8   552.3 349.41                0.00               0.0
    6       1.0   537.1 414.32                7.87               0.0
    7       1.0   543.5 408.39               21.14               0.0
    8       1.0   617.3 442.70               10.13               0.0
    9       1.0   607.9 436.20                3.70               0.0
   10       1.0   613.6 431.34                3.23               0.0
   11       1.0   620.0 412.40               23.23               0.0
   12       0.8   613.2 378.66                0.00               0.0
```

Weeks 1–5 are identical. **Week 6 changes**: in the 12-week plan it sells 414.32 batches instead of 421.48 and keeps 7.87 freezer slots of stock for the weeks after. The six-week plan sold that stock in week 6 because, for it, there was no week 7.

The practical rule: plan further ahead than you act. Planners run a long horizon, carry out only the first week or two, and re-plan next week with fresh data: a **rolling horizon** (Day 39 and Module 8). The weeks you act on should be far from the end of the horizon.

### The planted bug

`day08/bug_balance.py` rebuilds the balance with a single character changed. Run it and look at the two numbers it prints.

```python
def balance(m, p, t):
    before = m.keep[p, m.T.prev(t)] if t != m.T.first() else 0
    return before + m.make[p, t] == m.sell[p, t] - m.keep[p, t]


m.del_component(m.balance)
m.balance = pyo.Constraint(m.P, m.T, rule=balance)
solve(m)
print(f"profit ${m.profit():,.2f}")
made = sum(pyo.value(m.make[p, t]) for p in m.P for t in m.T)
sold = sum(pyo.value(m.sell[p, t]) for p in m.P for t in m.T)
print(f"batches made {made:,.1f}, batches sold {sold:,.1f}")
```

<details><summary>What happens, and how to spot it</summary>

```text
profit $1,022,646.51
batches made 1,838.2, batches sold 3,400.7
```

The model sells 3,400 batches having made 1,838. The balance says `before + make == sell - keep`: a minus sign where a plus belongs, so every batch "kept" lets the bakery sell one more batch it never baked. Profit nearly doubles.

The check that catches it is a **conservation law**: over the horizon, batches sold plus final stock can never exceed batches made. Day 9 turns checks like this into automatic tests.

</details>

## Mini-project: the freezer is replaced in December

The freezer is being replaced. During weeks 1–3, a rented freezer gives only 40 slots. What does it cost, and how big a freezer is worth having?

1. Re-solve with 40 slots in weeks 1–3 (change those three constraints with `set_value`).
2. Solve with freezers of 0 to 150 slots.
3. Write two sentences for the plant manager.

<details><summary>Solution</summary>

```python
import sys

import duckdb

sys.path.insert(0, "day08")
from planner import build, load_plan_data, solve, week_table

con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_plan_data(con, weeks=6)

base = build(d)
solve(base)

# A temporary freezer with 40 slots in weeks 1-3: one constraint per week, so replace the
# single freezer size by a size per week
small = build(d)
for t in [1, 2, 3]:
    small.freezer[t].set_value(
        sum(d["products"].loc[p, "freezer_slots_per_batch"] * small.keep[p, t] for p in small.P) <= 40)
solve(small)
print(week_table(small, d).to_string(index=False))
print(f"full freezer ${base.profit():,.2f}; 40 slots in weeks 1-3 ${small.profit():,.2f}; "
      f"cost of the small freezer ${base.profit() - small.profit():,.2f}")

# How big a freezer is worth having?
for slots in (0, 20, 40, 60, 80, 100, 120, 150):
    m = build(d)
    m.freezer_slots = slots
    solve(m)
    print(f"  {slots:>3} slots: ${m.profit():,.2f}")
```

```text
 week  capacity  demand   sold  freezer_slots_used  spare_oven_hours
    1       1.0   602.6 429.24                 0.0               0.0
    2       1.0   605.6 366.51                40.0               0.0
    3       1.0   865.9 445.42                40.0               0.0
    4       0.6   865.8 338.62                 0.0               0.0
    5       0.8   552.3 349.41                 0.0               0.0
    6       1.0   537.1 421.48                 0.0               0.0
full freezer $597,870.27; 40 slots in weeks 1-3 $597,086.15; cost of the small freezer $784.12
    0 slots: $595,836.82
   20 slots: $596,542.40
   40 slots: $597,086.15
   60 slots: $597,479.63
   80 slots: $597,717.30
  100 slots: $597,856.98
  120 slots: $597,870.27
  150 slots: $597,870.27
```

`set_value` replaces one week's freezer constraint; `m.freezer_slots = slots` changes the mutable size for every week.

*Memo:* "Renting a 40-slot freezer for the first three weeks of December costs about $784 in lost profit over the Christmas period, so a larger rental for those weeks is worth up to $784 more than the 40-slot one. Beyond about 110 slots extra freezer space is worth nothing this season: the bakery is short of oven time, not freezer space."

Each extra 20 slots is worth less than the last ($706, $544, $393, $238, $140, $13): **diminishing returns**, the usual shape of the value of a resource.

</details>

## Commit

```bash
git add .
git commit -m "Day 8: multi-week plan with storage, horizons, freezer value"
git push
```

### Day 8 checklist

- [ ] I can write a stock balance with make, sell and keep variables, and a rule for the first week.
- [ ] I separate revenue (at sale), ingredients (at baking) and holding (while waiting) in the objective.
- [ ] I can model shelf life and a shared freezer.
- [ ] I measured what planning ahead is worth ($2,033.45 over six weeks).
- [ ] I can explain end effects and why planners use rolling horizons.
- [ ] I can value a resource by sweeping its size.

---

# DAY 9 — Units, Scaling and Known-Answer Tests

The models are growing, and so are the ways they can be wrong without saying so. Today is about trust: units that silently differ by a factor of 100, numbers so small the solver throws them away, and a test suite that catches yesterday's planted bug in under a second.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 9.1 Units: a forecast in loaves, a model in batches |
| 0:45 – 1:45 | 9.2 Scaling: when the solver ignores your constraint |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 9.3 Known-answer tests with pytest |
| 2:45 – 3:45 | 9.4 Property tests; a planted bug caught |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: tests for the flour blend |
| 4:45 – 5:00 | Commit with Git + checklist |

## 9.1 Units: a forecast in loaves, a model in batches

The model works in **batches** (100 loaves). The forecast file from Day 1 has both `forecast_units` (loaves and packs) and `demand_cap_batches`. Pick the wrong column and nothing crashes. Create `day09/units.py`:

```python
import sys

import duckdb

sys.path.insert(0, "day06")
from bakery import DataError, build_model, load_data, plan_table, solve

con = duckdb.connect()           # in memory: copies of the tables we need
for t in ("bakery_products", "bakery_recipe", "bakery_resources", "forecast_next_week", "bakery_sales_weekly"):
    con.sql(f"CREATE TABLE {t} AS SELECT * FROM 'data/{t}.csv'")
con.sql("CREATE VIEW products AS SELECT *, 'module 1' AS category FROM bakery_products")

# ---------- 9.1 The right forecast, and a forecast in the wrong unit ----------
con.sql("CREATE VIEW forecast_batches AS SELECT sku, demand_cap_batches FROM forecast_next_week")
con.sql("""CREATE VIEW forecast_loaves AS
           SELECT sku, forecast_units AS demand_cap_batches FROM forecast_next_week""")   # the mistake
for name in ("forecast_batches", "forecast_loaves"):
    d = load_data(con, "products", "bakery_recipe", "bakery_resources", name)
    m = build_model(d)
    solve(m)
    plan = plan_table(m, d)
    print(name, dict(zip(plan.sku, plan.batches)), f"profit ${plan.profit.sum():,.2f}")
```

```text
forecast_batches {'BK-CR': 10.0, 'BK-SD': 20.0, 'BK-WW': 45.0} profit $17,500.00
forecast_loaves {'BK-CR': 0.0, 'BK-SD': 20.0, 'BK-WW': 60.0} profit $18,000.00
```

With caps in loaves (3,500 instead of 35) the demand caps vanish in effect: the model bakes 60 batches of whole-wheat when the stores will take 45, and reports **$500 more profit than is possible**. The plan is wrong by 1,500 loaves a week, and every check from Day 6 passes: nothing is missing or negative.

Three habits prevent this:

1. **Put the unit in the name.** `demand_cap_batches`, `use_per_batch`, `weekly_capacity` in hours, `holding_cost_per_batch_week`. A column called `forecast` invites the mistake; `forecast_units` next to `demand_cap_batches` makes it visible in the SQL.
2. **Convert once, at the edge.** Raw data arrive in whatever unit the source uses; one query converts to model units, and nothing else converts.
3. **Check against history.** A forecast 100 times recent sales is a unit error until proven otherwise.

The third habit as code:

```python
# ---------- 9.2 A plausibility check: is the forecast in line with history? ----------
def check_forecast_against_history(con, forecast, low=0.5, high=2.0):
    """Compare each product's forecast (in batches) with its average weekly sales over the last 8 weeks."""
    rows = con.sql(f"""
        WITH hist AS (
            SELECT s.sku, sum(s.units_sold) / 8.0 / any_value(p.units_per_batch) AS avg_batches
            FROM bakery_sales_weekly s JOIN bakery_products p USING (sku)
            WHERE s.week_start > (SELECT max(week_start) FROM bakery_sales_weekly) - INTERVAL 8 WEEK
            GROUP BY s.sku
        )
        SELECT f.sku, f.demand_cap_batches, round(h.avg_batches, 1) AS avg_batches,
               round(f.demand_cap_batches / h.avg_batches, 2) AS ratio
        FROM {forecast} f JOIN hist h USING (sku)
        WHERE f.demand_cap_batches NOT BETWEEN {low} * h.avg_batches AND {high} * h.avg_batches
        ORDER BY f.sku""").fetchall()
    if rows:
        raise DataError("; ".join(f"{s}: forecast {f} vs about {a} a week ({r}x)" for s, f, a, r in rows))


for name in ("forecast_batches", "forecast_loaves"):
    try:
        check_forecast_against_history(con, name)
        print(name, "passes the history check")
    except DataError as e:
        print(name, "DataError:", e)
```

```text
forecast_batches passes the history check
forecast_loaves DataError: BK-CR: forecast 1551 vs about 30.3 a week (51.2x); BK-SD: forecast 3500 vs about 34.2 a week (102.4x); BK-WW: forecast 4499 vs about 43.2 a week (104.04x)
```

A factor of 50 or 100 is the fingerprint of a unit error (50 packs or 100 loaves per batch). The band of 0.5 to 2 times history is a judgement: wide enough for Christmas, narrow enough to catch units. Put such checks in `load_data`, next to Day 6's.

> **Decision it supports:** whether this week's inputs are in the units the model expects, before any plan leaves the building.
>
> **Other decisions you can support the same way:** prices in cents versus dollars · capacities per day versus per week · kilograms versus tonnes.

## 9.2 Scaling: when the solver ignores your constraint

Northbound Fresh starts a carbon pilot: the three Module 1 products emit 40, 25 and 30 kg of CO2 per batch (ovens and delivery), and the bakery's budget is 1,900 kg a week. Add one constraint and write it in four units: kilograms, tonnes, megatonnes and gigatonnes. Mathematically all four are the same model. Create `day09/scaling.py`:

```python
import pyomo.environ as pyo

CO2_KG = {"BK-SD": 40, "BK-WW": 25, "BK-CR": 30}    # kg of CO2 per batch (ovens and delivery)
CO2_CAP_KG = 1900                                     # the pilot carbon budget, kg a week
UNITS = {"kg": 1, "tonnes": 1e-3, "megatonnes": 1e-9, "gigatonnes": 1e-12}


def bakery(scale):
    m = pyo.ConcreteModel()
    m.P = pyo.Set(initialize=["BK-SD", "BK-WW", "BK-CR"])
    m.x = pyo.Var(m.P, bounds=lambda m, p: (0, {"BK-SD": 35, "BK-WW": 45, "BK-CR": 31}[p]))
    m.oven = pyo.Constraint(expr=2 * m.x["BK-SD"] + m.x["BK-WW"] + 1.5 * m.x["BK-CR"] <= 100)
    m.bakers = pyo.Constraint(expr=m.x["BK-SD"] + m.x["BK-WW"] + 1.5 * m.x["BK-CR"] <= 80)
    m.carbon = pyo.Constraint(expr=sum(CO2_KG[p] * scale * m.x[p] for p in m.P) <= CO2_CAP_KG * scale)
    m.profit = pyo.Objective(expr=300 * m.x["BK-SD"] + 200 * m.x["BK-WW"] + 250 * m.x["BK-CR"], sense=pyo.maximize)
    return m


for unit, scale in UNITS.items():
    m = bakery(scale)
    tee = unit in ("megatonnes", "gigatonnes")
    if tee:
        print(f"--- HiGHS log with carbon in {unit}")
    pyo.SolverFactory("highs").solve(m, tee=tee)
    kg = sum(CO2_KG[p] * pyo.value(m.x[p]) for p in m.P)
    plan = {p: round(pyo.value(m.x[p]), 2) for p in m.P}
    flag = "" if kg <= CO2_CAP_KG + 1e-6 else "   <-- OVER THE CARBON BUDGET"
    print(f"{unit:>10}: plan {plan}, profit ${pyo.value(m.profit):,.2f}, CO2 {kg:,.1f} kg{flag}")
```

```text
        kg: plan {'BK-SD': 3.75, 'BK-WW': 45.0, 'BK-CR': 20.83}, profit $15,333.33, CO2 1,900.0 kg
    tonnes: plan {'BK-SD': 3.75, 'BK-WW': 45.0, 'BK-CR': 20.83}, profit $15,333.33, CO2 1,900.0 kg
--- HiGHS log with carbon in megatonnes
Hessian has dimension 3 but no nonzeros, so is ignored
LP has 3 rows; 3 cols; 9 nonzeros
Coefficient ranges:
  Matrix  [3e-08, 2e+00]
  Cost    [2e+02, 3e+02]
  Bound   [3e+01, 4e+01]
  RHS     [2e-06, 1e+02]
WARNING: Problem has some excessively small bounds on constraints
…
Model status        : Optimal
…
megatonnes: plan {'BK-SD': 3.75, 'BK-WW': 45.0, 'BK-CR': 20.83}, profit $15,333.33, CO2 1,900.0 kg
--- HiGHS log with carbon in gigatonnes
WARNING: HighsSparseMatrix local_ar_matrix matrix contains 3 |value| in [2.5e-11, 4e-11] at/below tolerance of 1e-09: ignored
Hessian has dimension 3 but no nonzeros, so is ignored
LP has 3 rows; 3 cols; 6 nonzeros
Coefficient ranges:
  Matrix  [1e+00, 2e+00]
  Cost    [2e+02, 3e+02]
  Bound   [3e+01, 4e+01]
  RHS     [2e-09, 1e+02]
WARNING: Problem has some excessively small bounds on constraints
…
Model status        : Optimal
…
gigatonnes: plan {'BK-SD': 20.0, 'BK-WW': 45.0, 'BK-CR': 10.0}, profit $17,500.00, CO2 2,225.0 kg   <-- OVER THE CARBON BUDGET
```

(`…` marks log lines left out.) In kilograms and tonnes the carbon budget binds: profit falls to $15,333.33 and emissions sit at exactly 1,900 kg. In megatonnes HiGHS warns but still gets it right. In **gigatonnes** the carbon coefficients (4 × 10⁻¹¹) are below HiGHS's tolerance of 10⁻⁹, and the solver **drops them**: "LP has 6 nonzeros", not 9. The carbon constraint becomes 0 ≤ 1.9 × 10⁻⁹, always true. The status is "Optimal", the plan emits 2,225 kg, and the only clue is a warning in a log nobody reads.

Why it happens: solvers work in floating point with **absolute tolerances**. A number below about 10⁻⁹ is treated as zero; two constraints that differ by less than 10⁻⁷ are treated as equal. Those tolerances only make sense if the model's numbers are of everyday size.

Rules of thumb:

- **Choose units so that coefficients sit between about 10⁻³ and 10⁴.** Batches, hours, kilograms, thousands of dollars are good; seconds, grams and cents in the same model are not.
- **Read the "Coefficient ranges" lines** of the solver log. Ranges wider than about 10⁶ deserve a second look.
- **Recompute every constraint after solving, in real units**, as `scaling.py` does for CO2. The recheck is what flagged the gigatonne plan.

## 9.3 Known-answer tests with pytest

So far you checked models by running them and reading the output. **Tests** do the reading for you, every time, in a second. Add pytest to `requirements.txt` and install it:

```bash
echo "pytest==9.2.0" >> requirements.txt
pip install -r requirements.txt
```

A pytest test is a function whose name starts with `test_`, in a file whose name starts with `test_`, in a folder called `tests`. Inside, `assert` states what must be true. Create `tests/conftest.py`, which pytest reads first:

```python
"""Shared test set-up: an in-memory database built from data/*.csv, and the model folders on the path."""
import glob
import os
import sys

import duckdb
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in ("day06", "day08", "day09", "day10", "day12", "day14", "day15", "day16", "day18", "day19", "day20"):
    sys.path.insert(0, os.path.join(ROOT, folder))


@pytest.fixture(scope="session")
def con():
    """A fresh in-memory copy of the course data, so tests never touch northbound.duckdb."""
    db = duckdb.connect()
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "*.csv"))):
        table = os.path.splitext(os.path.basename(path))[0]
        db.sql(f"CREATE TABLE {table} AS SELECT * FROM read_csv('{path}')")
    db.sql("CREATE VIEW module1_products AS SELECT *, 'module 1' AS category FROM bakery_products")
    db.sql("CREATE VIEW module1_forecast AS SELECT sku, demand_cap_batches FROM forecast_next_week")
    return db
```

A **fixture** is something tests ask for by name: any test with a `con` argument gets this database. `scope="session"` builds it once for the whole run. The tests use a fresh in-memory copy of `data/`, so they never depend on, or damage, `northbound.duckdb`. (They do need `data/forecast_next_week.csv`, written by `day01/ladder.py`.)

Now the tests for Day 6's model, `tests/test_bakery.py`:

```python
"""Known-answer and property tests for the single-week bakery model (day06/bakery.py)."""
import pyomo.environ as pyo
import pytest
from bakery import DataError, build_model, load_data, plan_table, resource_table, solve


def small_data():
    """Day 3's two-product model, typed in: the answer is the corner (20, 60), $18,000."""
    return {
        "products": ["S", "W"], "resources": ["oven", "bakers", "starter"],
        "name": {"S": "sourdough", "W": "whole-wheat"}, "category": {"S": "bread", "W": "bread"},
        "profit": {"S": 300, "W": 200},
        "capacity": {"oven": 100, "bakers": 80, "starter": 40},
        "use": {("oven", "S"): 2, ("oven", "W"): 1, ("bakers", "S"): 1, ("bakers", "W"): 1, ("starter", "S"): 1},
        "cap": {"S": 1000, "W": 1000},
    }


def test_two_product_corner():
    m = build_model(small_data())
    solve(m)
    assert pyo.value(m.x["S"]) == pytest.approx(20)
    assert pyo.value(m.x["W"]) == pytest.approx(60)
    assert pyo.value(m.total_profit) == pytest.approx(18_000)


def test_three_products_known_answer(con):
    d = load_data(con, "module1_products", "bakery_recipe", "bakery_resources", "module1_forecast")
    m = build_model(d)
    solve(m)
    plan = dict(zip(plan_table(m, d).sku, plan_table(m, d).batches))
    assert plan == pytest.approx({"BK-SD": 20, "BK-WW": 45, "BK-CR": 10})
    assert pyo.value(m.total_profit) == pytest.approx(17_500)


def test_full_range_known_answer(con):
    d = load_data(con)
    m = build_model(d)
    solve(m)
    assert pyo.value(m.total_profit) == pytest.approx(108_351.28, abs=0.01)


def test_plan_respects_every_limit(con):
    d = load_data(con)
    m = build_model(d)
    solve(m)
    assert (resource_table(m).slack >= -1e-6).all()
    plan = plan_table(m, d)
    assert (plan.batches >= -1e-6).all()
    assert (plan.batches <= plan.demand_cap + 1e-6).all()


def test_missing_forecast_is_refused(con):
    con.sql("CREATE OR REPLACE TEMP VIEW short_forecast AS SELECT * FROM bakery_range_forecast WHERE sku <> 'BK-04'")
    with pytest.raises(DataError, match="BK-04 has no forecast"):
        load_data(con, forecast="short_forecast")


def test_infeasible_model_raises():
    m = build_model(small_data())
    m.impossible = pyo.Constraint(expr=m.x["S"] >= 41)     # the starter allows at most 40
    with pytest.raises(RuntimeError, match="infeasible"):
        solve(m)
```

Three kinds of test, each with a job:

| Kind | Example | What it protects |
|---|---|---|
| **Known answer, worked by hand** | Two products, corner (20, 60), $18,000 | The model's logic, on a case you can verify on paper (Day 3) |
| **Known answer, from a trusted run** | 3 products $17,500; 30 products $108,351.28 | Against any change that moves the answer (a *regression* test) |
| **Behaviour** | Missing forecast refused; infeasible model raises | The guard rails from Days 6 and 7 |

`pytest.approx` compares floating-point numbers with a tolerance: solver answers are rarely exactly 20, more often 19.999999999.

## 9.4 Property tests, and a bug caught

Known answers say *what* the answer is. **Property tests** say what must be true of *any* answer: stock balances, bread is never stored, limits hold. They keep working when the data change. `tests/test_planner.py` tests Day 8's planner:

```python
"""Known-answer and property tests for the multi-week planner (day08/planner.py)."""
import pandas as pd
import pyomo.environ as pyo
import pytest
from bakery import build_model, load_data
from planner import build, load_plan_data, solve

TOL = 1e-6


def toy_data():
    """Day 8's toy: one product, two weeks, 10 batches a week, demand 5 then 15. Answer: $4,960."""
    products = pd.DataFrame({"product_name": ["croissant"], "category": ["viennoiserie"],
                             "revenue_per_batch": [350.0], "ingredients_per_batch": [100.0],
                             "shelf_life_weeks": [3], "holding_cost_per_batch_week": [8.0],
                             "freezer_slots_per_batch": [1.0]}, index=pd.Index(["X"], name="sku"))
    return {"products": products, "weeks": pd.DataFrame({"week": [1, 2], "capacity_factor": [1.0, 1.0]}),
            "demand": {("X", 1): 5, ("X", 2): 15}, "use": {("oven_hours", "X"): 1.0},
            "capacity": {("oven_hours", 1): 10, ("oven_hours", 2): 10}, "resources": ["oven_hours"],
            "freezer_slots": 100.0}


@pytest.fixture(scope="module")
def six_weeks(con):
    d = load_plan_data(con, weeks=6)
    m = build(d)
    solve(m)
    return d, m


def test_toy_known_answer():
    m = build(toy_data())
    solve(m)
    assert [pyo.value(m.make["X", t]) for t in (1, 2)] == pytest.approx([10, 10])
    assert [pyo.value(m.keep["X", t]) for t in (1, 2)] == pytest.approx([5, 0])
    assert pyo.value(m.profit) == pytest.approx(4_960)


def test_six_weeks_known_answer(six_weeks):
    d, m = six_weeks
    assert pyo.value(m.profit) == pytest.approx(597_870.27, abs=0.01)


def test_stock_balances_every_week(six_weeks):
    d, m = six_weeks
    for p in m.P:
        before = 0
        for t in m.T:
            assert before + pyo.value(m.make[p, t]) == pytest.approx(pyo.value(m.sell[p, t]) + pyo.value(m.keep[p, t]), abs=TOL)
            before = pyo.value(m.keep[p, t])


def test_bread_is_never_stored(six_weeks):
    d, m = six_weeks
    bread = d["products"].index[d["products"].category == "bread"]
    assert max(pyo.value(m.keep[p, t]) for p in bread for t in m.T) <= TOL


def test_freezer_and_capacity_are_respected(six_weeks):
    d, m = six_weeks
    slots = d["products"].freezer_slots_per_batch
    for t in m.T:
        assert sum(slots[p] * pyo.value(m.keep[p, t]) for p in m.P) <= d["freezer_slots"] + TOL
        for r in m.R:
            assert pyo.value(m.capacity[r, t].body) <= pyo.value(m.cap[r, t]) + TOL


def test_storage_never_hurts(con, six_weeks):
    d, m = six_weeks
    no_storage = build(d, storage=False)
    solve(no_storage)
    assert pyo.value(m.profit) >= pyo.value(no_storage.profit) - TOL


def test_one_week_matches_the_single_week_model(con):
    """With one normal week and the Day 6 forecast as demand, the two models must agree."""
    d = load_plan_data(con, weeks=1)
    caps = dict(con.sql("SELECT sku, demand_cap_batches FROM bakery_range_forecast").fetchall())
    d["demand"] = {(p, 1): caps[p] for p in d["products"].index}
    m = build(d)
    solve(m)
    single = build_model(load_data(con))
    solve(single)
    assert pyo.value(m.profit) == pytest.approx(pyo.value(single.total_profit), abs=0.01)
```

The last test is a **consistency test**: with one normal week and Day 6's forecast as demand, the multi-week planner must give exactly the single-week answer, $108,351.28. Two models written on different days, by different reasoning, agreeing to the cent, is strong evidence both are right.

Run everything:

```bash
python -m pytest -v
```

```text
tests/test_bakery.py::test_two_product_corner PASSED
tests/test_bakery.py::test_three_products_known_answer PASSED
tests/test_bakery.py::test_full_range_known_answer PASSED
tests/test_bakery.py::test_plan_respects_every_limit PASSED
tests/test_bakery.py::test_missing_forecast_is_refused PASSED
tests/test_bakery.py::test_infeasible_model_raises PASSED
tests/test_planner.py::test_toy_known_answer PASSED
tests/test_planner.py::test_six_weeks_known_answer PASSED
tests/test_planner.py::test_stock_balances_every_week PASSED
tests/test_planner.py::test_bread_is_never_stored PASSED
tests/test_planner.py::test_freezer_and_capacity_are_respected PASSED
tests/test_planner.py::test_storage_never_hurts PASSED
tests/test_planner.py::test_one_week_matches_the_single_week_model PASSED
13 passed
```

### The planted bug

A colleague "fixes" the shelf-life rule in `day08/planner.py` so that it reads `if t - L <= k <= t)` instead of `if t - L < k <= t)`. Make that change, run the tests, then undo it.

<details><summary>What the tests say</summary>

```text
FAILED tests/test_planner.py::test_six_weeks_known_answer - assert 599469.586...
FAILED tests/test_planner.py::test_bread_is_never_stored - assert 52.0 <= 1e-06
2 failed, 11 passed
```

The known-answer test says *something* changed: profit rose by $1,599. The property test says *what*: bread is now kept in stock (52 batches). With `<=`, the window includes one extra week, so bread, with $$L = 0$$, may be kept for a week. Together the two failures point straight at the shelf-life rule.

This is why a suite needs both kinds of test. A known answer alone tells you the number moved; a property tells you which rule broke.

</details>

> **Decision it supports:** whether a changed model may be used for next week's plan.
>
> **Other decisions you can support the same way:** whether a new data feed can replace the old one · whether a solver upgrade changes any plan.

## Mini-project: tests for the flour blend

Day 5's flour blend is a script with top-level code, which tests cannot call. Turn it into a function `blend(flour, spec, min_share=None)` in `day09/flour.py` that returns `(status, cost_per_tonne, shares)` and does **not** raise when the blend is impossible. Then write `tests/test_flour.py` with at least four tests:

1. the known answer ($471.11; 5/9 Prairie, 4/9 Ontario);
2. a property: the blend meets every spec and the shares add to 1;
3. the Quebec label (Day 5, Try it 5.1) costs more, and its share is at least 20%;
4. ash at 0.50% is infeasible.

<details><summary>Solution</summary>

```python
"""Day 9 mini-project: the Day 5 flour blend as a function that tests can call.

Import it:  from flour import blend
"""
import pandas as pd
import pyomo.environ as pyo


def blend(flour, spec, min_share=None):
    """Least-cost shares of one tonne. Returns (status, cost_per_tonne, shares) and never raises on infeasible."""
    m = pyo.ConcreteModel()
    m.S = pyo.Set(initialize=flour.index.tolist())
    m.share = pyo.Var(m.S, bounds=(0, 1))
    m.one_tonne = pyo.Constraint(expr=sum(m.share[s] for s in m.S) == 1)
    m.spec = pyo.ConstraintList()
    for prop, row in spec.iterrows():
        quality = sum(flour.loc[s, prop] * m.share[s] for s in m.S)
        if pd.notna(row.min_value):
            m.spec.add(quality >= row.min_value)
        if pd.notna(row.max_value):
            m.spec.add(quality <= row.max_value)
    for s, share in (min_share or {}).items():
        m.spec.add(m.share[s] >= share)
    m.cost = pyo.Objective(expr=sum(flour.loc[s, "price_per_tonne"] * m.share[s] for s in m.S))
    result = pyo.SolverFactory("highs").solve(m, load_solutions=False)
    status = str(result.solver.termination_condition)
    if status != "optimal":
        return status, None, None
    m.solutions.load_from(result)
    return status, pyo.value(m.cost), {s: pyo.value(m.share[s]) for s in m.S}
```

```python
"""Day 9 mini-project: tests for the flour blend."""
import pytest
from flour import blend


@pytest.fixture(scope="module")
def flour(con):
    return con.sql("SELECT * FROM flour_suppliers").df().set_index("supplier_id")


@pytest.fixture(scope="module")
def spec(con):
    return con.sql("SELECT * FROM flour_spec").df().set_index("property")


def test_known_answer(flour, spec):
    status, cost, shares = blend(flour, spec)
    assert status == "optimal"
    assert cost == pytest.approx(471.11, abs=0.01)
    assert shares["FL-PHR"] == pytest.approx(5 / 9)
    assert shares["FL-OSW"] == pytest.approx(4 / 9)


def test_blend_meets_every_spec(flour, spec):
    _, _, shares = blend(flour, spec)
    assert sum(shares.values()) == pytest.approx(1)
    for prop, row in spec.iterrows():
        quality = sum(flour.loc[s, prop] * x for s, x in shares.items())
        if row.min_value == row.min_value:          # not NaN
            assert quality >= row.min_value - 1e-6
        if row.max_value == row.max_value:
            assert quality <= row.max_value + 1e-6


def test_quebec_label_costs_more(flour, spec):
    _, base, _ = blend(flour, spec)
    _, local, shares = blend(flour, spec, min_share={"FL-QCO": 0.2})
    assert local == pytest.approx(502.44, abs=0.01)
    assert local > base
    assert shares["FL-QCO"] >= 0.2 - 1e-6


def test_ash_050_is_impossible(flour, spec):
    tight = spec.copy()
    tight.loc["ash_pct", "max_value"] = 0.50
    status, cost, _ = blend(flour, tight)
    assert status == "infeasible" and cost is None
```

`row.min_value == row.min_value` is a quick NaN test: NaN is the only value not equal to itself. With these, `python -m pytest -q` runs 17 tests.

</details>

## Commit

```bash
git add .
git commit -m "Day 9: units, scaling, pytest suite"
git push
```

### Day 9 checklist

- [ ] My column names carry their units, and I convert units in one place.
- [ ] I check forecasts against history to catch unit errors.
- [ ] I read the coefficient ranges in a solver log and keep coefficients of everyday size.
- [ ] I recompute constraints in real units after solving.
- [ ] I have known-answer, property and behaviour tests, and they run in about a second.
- [ ] I watched a test suite catch a one-character bug.

---

# DAY 10 — Capstone: The 12-Week Bakery Plan

The bakery manager needs a plan from 30 November to 21 February: 30 products, 12 weeks, Christmas in the middle. Today you build it alone, from the pieces of Days 1–9, and defend it in a one-page memo. The model solution follows each part; try each part before opening it.

| Time | Block |
|---|---|
| 0:00 – 0:30 | 10.1 The brief |
| 0:30 – 1:45 | 10.2 Extend the planner: contract, shelf rule, overtime |
| 1:45 – 2:00 | Break |
| 2:00 – 3:00 | 10.3 Tests, then the plan |
| 3:00 – 3:45 | 10.4 Read the plan: overtime, lost sales, the sourdough question |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | 10.5 The memo |
| 4:45 – 5:00 | Commit, tag and checklist |

## 10.1 The brief

> **To:** Module 1 analysts **From:** Bakery manager, Northbound Fresh
>
> I need next quarter's plan by Monday: what to bake each week for all 30 products, from 30 November to the week of 15 February. Rules:
>
> 1. **Café Laurier** takes 50 batches of whole-wheat every week, on top of what the stores sell. The contract is signed: it must be met.
> 2. **Every product stays on the shelf**: at least 2 batches sold every week.
> 3. Christmas, New Year's and Family Day weeks have fewer oven, baker and packing hours (`plan_weeks`).
> 4. We may book **overtime**: up to 80 baker hours a week at $45 an hour, and up to 40 packing hours a week (a third shift) at $60 an hour. Ovens cannot run longer.
> 5. Use the freezer (120 slots) as you see fit.
>
> Tell me: the profit, how much overtime to book and when, which products customers will miss most, and anything I should push back on.

All the data are in `northbound.duckdb`: `plan_weeks`, `plan_demand`, `bakery_storage`, `plan_settings`, plus the range tables from Day 6.

### Deliverables and marking

| Deliverable | Weight | What earns full marks |
|---|---|---|
| `day10/capstone.py`: the model | 30% | Extends Day 8's planner without copying it; every rule from the brief; data from the database |
| `tests/test_capstone.py` | 20% | A known answer, every rule of the brief as a property, and one comparison test |
| The plan, stored and drawn | 15% | `capstone_plan` and `capstone_weeks` tables; one clear chart |
| Analysis | 15% | Overtime value, lost sales by category and week, the sourdough question |
| `day10/memo.md` | 20% | One page; every number from a solve; recommendations the manager can act on |

## 10.2 Extend the planner

Day 8's `build` already has the balance, shelf life, freezer and weekly capacity. You need three additions, made to the model `build` returns, without editing `planner.py`:

1. **The contract:** raise whole-wheat demand by 50 batches each week, and require at least 52 batches sold (50 for the café, 2 for the shelf rule).
2. **The shelf rule:** at least 2 batches of every product sold each week.
3. **Overtime:** a variable per week for baker and packing hours, bounded by the brief, added to those capacity constraints, and its cost taken off the objective.

Hints: change `d["demand"]` before calling `build`; replace capacity constraints with `set_value` (Day 7); deactivate `m.profit` and add a new objective.

<details><summary>Model solution: the model</summary>

```python
import os
import sys

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day08")
from planner import build, load_plan_data, plan_table, solve, v, week_table

pd.set_option("display.width", 140)
WEEKS = 12


def settings(con):
    return {k: float(x) for k, x in con.sql("SELECT setting, value FROM plan_settings").fetchall()}


def build_capstone(con, overtime=True):
    s = settings(con)
    d = load_plan_data(con, weeks=WEEKS)
    # The cafe's whole-wheat comes on top of store demand (Day 7, fix 1)
    for t in d["weeks"].week:
        d["demand"]["BK-WW", t] += s["cafe_ww_batches"]
    m = build(d)

    # Minimum sales each week: the shelf rule for every product, the contract for whole-wheat
    def minimum(p):
        return s["cafe_ww_batches"] + s["shelf_minimum_batches"] if p == "BK-WW" else s["shelf_minimum_batches"]

    m.min_sales = pyo.Constraint(m.P, m.T, rule=lambda m, p, t: m.sell[p, t] >= minimum(p))

    # Overtime: extra baker and packing hours at a price
    limit = {"baker_hours": s["overtime_baker_hours_max"] if overtime else 0,
             "packing_hours": s["overtime_packing_hours_max"] if overtime else 0}
    cost = {"baker_hours": s["overtime_baker_cost"], "packing_hours": s["overtime_packing_cost"]}
    m.OT = pyo.Set(initialize=list(limit))
    m.overtime = pyo.Var(m.OT, m.T, bounds=lambda m, r, t: (0, limit[r]))
    for r in m.OT:
        for t in m.T:
            m.capacity[r, t].set_value(sum(m.use[r, p] * m.make[p, t] for p in m.P) <= m.cap[r, t] + m.overtime[r, t])
    m.overtime_cost = pyo.Expression(expr=sum(cost[r] * m.overtime[r, t] for r in m.OT for t in m.T))
    m.profit.deactivate()
    m.net_profit = pyo.Objective(expr=m.revenue - m.ingredients - m.holding - m.overtime_cost, sense=pyo.maximize)
    return d, m
```

`settings` reads `plan_settings` once, so every number in the brief comes from the database, not the code. `build_capstone(con, overtime=False)` gives the same model with overtime limits of zero, for comparison.

</details>

## 10.3 Tests, then the plan

Write the tests **before** you read the plan. Each rule in the brief becomes a property test; the profit becomes a known answer once you trust it.

<details><summary>Model solution: the tests</summary>

```python
"""Tests for the Day 10 capstone plan."""
import pyomo.environ as pyo
import pytest
from capstone import build_capstone, settings
from planner import solve

TOL = 1e-6


@pytest.fixture(scope="module")
def capstone(con):
    d, m = build_capstone(con)
    solve(m)
    return d, m, settings(con)


def test_known_answer(capstone):
    _, m, _ = capstone
    assert pyo.value(m.net_profit) == pytest.approx(1_255_362.50, abs=0.01)


def test_cafe_contract_and_shelf_rule_hold_every_week(capstone):
    _, m, s = capstone
    for t in m.T:
        assert pyo.value(m.sell["BK-WW", t]) >= s["cafe_ww_batches"] + s["shelf_minimum_batches"] - TOL
        for p in m.P:
            assert pyo.value(m.sell[p, t]) >= s["shelf_minimum_batches"] - TOL


def test_overtime_within_limits(capstone):
    _, m, s = capstone
    for t in m.T:
        assert pyo.value(m.overtime["baker_hours", t]) <= s["overtime_baker_hours_max"] + TOL
        assert pyo.value(m.overtime["packing_hours", t]) <= s["overtime_packing_hours_max"] + TOL


def test_overtime_option_never_hurts(con, capstone):
    _, m, _ = capstone
    _, no_ot = build_capstone(con, overtime=False)
    solve(no_ot)
    assert pyo.value(m.net_profit) >= pyo.value(no_ot.net_profit) - TOL


def test_christmas_week_uses_less_than_a_normal_week(capstone):
    _, m, _ = capstone
    oven = {t: pyo.value(m.capacity["oven_hours", t].body) for t in m.T}
    assert oven[4] == pytest.approx(0.6 * 600)
    assert oven[3] == pytest.approx(600)
```

`conftest.py` from Day 9 already puts `day10` on the path. The Christmas test checks that week 4 uses exactly 60% of a normal week's oven hours: the ovens are full in both weeks, and the holiday cut is applied.

```text
22 passed
```

</details>

Then the plan itself:

<details><summary>Model solution: running the plan</summary>

```python
if __name__ == "__main__":
    con = duckdb.connect("northbound.duckdb")

    # ---------- 10.1 The plan ----------
    d, m = build_capstone(con)
    solve(m)
    weeks = week_table(m, d)
    weeks["overtime_baker_h"] = [v(m.overtime["baker_hours", t]) for t in m.T]
    weeks["overtime_packing_h"] = [v(m.overtime["packing_hours", t]) for t in m.T]
    print(weeks.to_string(index=False))
    print(f"12-week profit ${pyo.value(m.net_profit):,.2f} = revenue ${pyo.value(m.revenue):,.2f} "
          f"- ingredients ${pyo.value(m.ingredients):,.2f} - holding ${pyo.value(m.holding):,.2f} "
          f"- overtime ${pyo.value(m.overtime_cost):,.2f}")

    # ---------- 10.2 What overtime is worth ----------
    _, no_ot = build_capstone(con, overtime=False)
    solve(no_ot)
    print(f"without overtime ${pyo.value(no_ot.net_profit):,.2f}; overtime adds "
          f"${pyo.value(m.net_profit) - pyo.value(no_ot.net_profit):,.2f} for ${pyo.value(m.overtime_cost):,.2f} of overtime")
```

</details>

```text
 week  capacity  demand   sold  freezer_slots_used  spare_oven_hours  overtime_baker_h  overtime_packing_h
    1       1.0   652.6 421.20               35.79               0.0               0.0               40.00
    2       1.0   655.6 401.80               92.53               0.0               0.0               40.00
    3       1.0   915.9 497.32               95.12               0.0               0.0               40.00
    4       0.6   915.8 426.80                0.00               0.0               0.0               13.36
    5       0.8   602.3 384.46                0.00               0.0               0.0               22.22
    6       1.0   587.1 436.20               14.23               0.0               0.0               32.80
    7       1.0   593.5 443.90               25.71               0.0               0.0               26.06
    8       1.0   667.3 488.30                8.45               0.0               0.0                6.99
    9       1.0   657.9 458.80               13.60               0.0               0.0               29.37
   10       1.0   663.6 454.70               23.14               0.0               0.0               39.11
   11       1.0   670.0 461.00               29.20               0.0               0.0               26.89
   12       0.8   663.2 431.60                0.00               0.0               0.0                0.00
12-week profit $1,255,362.50 = revenue $1,962,410.20 - ingredients $685,737.42 - holding $2,302.34 - overtime $19,007.94
without overtime $1,244,588.77; overtime adds $10,773.73 for $19,007.94 of overtime
```

## 10.4 Read the plan

**Overtime.** The plan books **packing overtime in 11 of 12 weeks**, at the full 40 hours in the three weeks before Christmas, and **no baker overtime at all**. Packing overtime costs $19,007.94 and earns $29,781.67 of extra margin: a net gain of **$10,773.73**. Baker overtime earns nothing because bakers are not the bottleneck (Day 6 found 20 spare baker hours a week). The brief offered both; the model says book only one.

The rest of `capstone.py` looks at what customers will miss:

```python
    # ---------- 10.3 Lost sales by category, and the products to talk about ----------
    plan = plan_table(m, d)
    lost = plan.pivot_table(index="category", columns="week", values="lost", aggfunc="sum").round(0)
    print("Batches of demand not met, by category and week:")
    print(lost.to_string())
    by_product = plan.groupby("sku").agg(sold=("sell", "sum"), demand=("demand", "sum")).join(d["products"]["product_name"])
    by_product["fill_rate"] = (by_product.sold / by_product.demand).round(3)
    print("Lowest fill rates:")
    print(by_product.sort_values("fill_rate").head(6).to_string())

    # ---------- 10.4 Store the plan and draw it ----------
    con.sql("CREATE OR REPLACE TABLE capstone_plan AS SELECT * FROM plan")
    con.sql("CREATE OR REPLACE TABLE capstone_weeks AS SELECT * FROM weeks")
    print(con.sql("""
        SELECT category, round(sum(sell), 1) AS sold, round(sum(demand), 1) AS demand,
               round(sum(sell) / sum(demand), 3) AS fill_rate, round(max(keep), 1) AS max_stock_batches
        FROM capstone_plan GROUP BY category ORDER BY fill_rate"""))

    os.makedirs("figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.bar(weeks.week, weeks.demand, color="#d0d7e5", label="demand (batches)")
    ax1.bar(weeks.week, weeks.sold, color="#4f7bd9", label="sold (batches)")
    ax1.set_ylabel("batches")
    ax1.set_title("Northbound Fresh bakery: 12-week plan")
    ax1.legend()
    ax2.bar(weeks.week - 0.2, weeks.overtime_baker_h, width=0.4, color="#e08a2c", label="baker overtime (h)")
    ax2.bar(weeks.week + 0.2, weeks.overtime_packing_h, width=0.4, color="#8a5cd1", label="packing overtime (h)")
    ax2.plot(weeks.week, weeks.freezer_slots_used, "k-o", label="freezer slots used (of 120)")
    ax2.set_xlabel("week (week 4 = Christmas, capacity 60%)")
    ax2.legend()
    ax2.set_xticks(weeks.week)
    fig.tight_layout()
    fig.savefig("figures/day10_plan.png", dpi=120)
    print("saved figures/day10_plan.png")
    con.close()
```

```text
Batches of demand not met, by category and week:
week             1      2      3      4      5     6     7     8     9     10    11     12
category                                                                                  
bread          57.0   63.0   77.0   77.0   58.0  37.0  38.0  53.0  48.0  51.0  53.0   58.0
cake          110.0  114.0  184.0  182.0  101.0  84.0  81.0  94.0  96.0  97.0  96.0  113.0
cookie          7.0    7.0   46.0   78.0    6.0   0.0   0.0   0.0   0.0   0.0   0.0    0.0
viennoiserie   57.0   69.0  111.0  151.0   53.0  30.0  31.0  32.0  55.0  61.0  61.0   61.0
Lowest fill rates:
       sold  demand        product_name  fill_rate
sku                                               
BK-18  24.0   381.6  Butter tart 6-pack      0.063
BK-24  24.0   367.4    Maple layer cake      0.065
BK-20  24.0   317.4     Lemon loaf cake      0.076
BK-19  24.0   309.2         Carrot cake      0.078
BK-SD  42.6   431.1      Sourdough loaf      0.099
BK-08  24.0   233.8    Seeded sourdough      0.103
┌──────────────┬────────┬────────┬───────────┬───────────────────┐
│   category   │  sold  │ demand │ fill_rate │ max_stock_batches │
│   varchar    │ double │ double │  double   │      double       │
├──────────────┼────────┼────────┼───────────┼───────────────────┤
│ cake         │  249.4 │ 1602.8 │     0.156 │               4.0 │
│ viennoiserie │ 1198.7 │ 1969.9 │     0.608 │              26.6 │
│ bread        │ 2533.9 │ 3203.3 │     0.791 │               0.0 │
│ cookie       │ 1324.1 │ 1468.8 │     0.901 │              79.4 │
└──────────────┴────────┴────────┴───────────┴───────────────────┘
```

![The 12-week plan](figures/day10_plan.png)

- **Cookies are stocked ahead of Christmas** (up to 79 batches of a single cookie product in store) and lose almost nothing after week 5: shelf-stable products are what the freezer is for.
- **Cakes get only the shelf minimum** (2 batches a week each): 16% of cake demand is met. Every cake batch takes oven and baker time that earns more elsewhere.
- **Sourdough is the surprise**: the chain's best-known loaf, $300 profit a batch, and the plan meets **10% of its demand**. It needs 2 oven hours a batch, the most of any bread, and the ovens are the binding resource; olive sourdough shares its starter and needs only 1.2 oven hours.

That last finding is exactly what to push back on, so price it. `day10/sourdough_question.py` adds a minimum share of sourdough demand:

```python
import sys

import duckdb
import pyomo.environ as pyo

sys.path.insert(0, "day08")
sys.path.insert(0, "day10")
from capstone import build_capstone
from planner import solve

con = duckdb.connect("northbound.duckdb", read_only=True)
d, base = build_capstone(con)
solve(base)

for share in (0.5, 0.8, 1.0):
    _, m = build_capstone(con)
    m.sourdough = pyo.Constraint(m.T, rule=lambda m, t: m.sell["BK-SD", t] >= share * m.demand["BK-SD", t])
    solve(m)
    cost = pyo.value(base.net_profit) - pyo.value(m.net_profit)
    sold = sum(pyo.value(m.sell["BK-SD", t]) for t in m.T)
    print(f"sourdough at {share:.0%} of demand: {sold:,.1f} batches over 12 weeks, "
          f"profit ${pyo.value(m.net_profit):,.2f}, costs ${cost:,.2f} (${cost / 12:,.2f} a week)")
```

```text
sourdough at 50% of demand: 215.6 batches over 12 weeks, profit $1,251,051.12, costs $4,311.38 ($359.28 a week)
sourdough at 80% of demand: 344.9 batches over 12 weeks, profit $1,247,025.83, costs $8,336.67 ($694.72 a week)
sourdough at 100% of demand: 431.1 batches over 12 weeks, profit $1,242,195.09, costs $13,167.41 ($1,097.28 a week)
```

Keeping sourdough at 80% of demand costs **$695 a week**, 0.7% of profit. The model cannot know that shoppers who don't find their sourdough may shop elsewhere; the manager can. That trade-off belongs in the memo.

## 10.5 The memo

Write `day10/memo.md`: one page for the bakery manager. Use only numbers from your solves.

<details><summary>Model answer</summary>

**To:** Bakery manager **Re:** Bakery plan, 30 November to 21 February

**Recommendation.** Bake to the attached plan: **$1.26 million** of contribution over 12 weeks (revenue $1.96 million, ingredients $686,000, freezer $2,300, overtime $19,000). It meets the Café Laurier contract and keeps every product on the shelf every week.

**Overtime.** Book a **third packing shift** in 11 of 12 weeks, the full 40 hours in the three weeks before Christmas. It costs $19,000 and adds $29,800 of margin: **+$10,800 net**. Do **not** book baker overtime: bakers are not the bottleneck, and extra baker hours earn nothing.

**Christmas.** Build up stock in the first three weeks of December (up to 95 freezer slots, mostly cookies and viennoiserie) and sell it in Christmas week, when the ovens run at 60%.

**What customers will miss.** Demand exceeds what the ovens can bake in every week. The plan sells 16% of cake demand, 61% of viennoiserie, 79% of bread and 90% of cookies.

**Push back on: sourdough.** The plan meets only 10% of sourdough demand, because sourdough uses the most oven time of any bread. Keeping it at 80% of demand costs **$695 a week** (0.7% of profit). I recommend paying it: losing the chain's best-known loaf from the shelves risks customers the model cannot see.

**Next steps.** Oven hours limit everything; a business case for more ovens (Module 3) is the biggest lever. Re-plan every Monday with fresh demand (rolling horizon).

</details>

## Commit and tag

The end of Module 1. Commit, and tag the repository so you can always return to this state:

```bash
git add .
git commit -m "Day 10: capstone 12-week plan, tests and memo"
git tag v1.0-module1
git push --tags
```

### Day 10 checklist

- [ ] My capstone extends the Day 8 planner without copying it.
- [ ] Every rule in the brief is a constraint and a test.
- [ ] The plan is stored in `northbound.duckdb` and drawn in one chart.
- [ ] I measured what overtime is worth, and found which kind not to book.
- [ ] I priced the sourdough question rather than guessing.
- [ ] My memo uses only numbers from solves, and tells the manager what to do.
