# DAY 6 — Data-Driven Models from DuckDB

So far the numbers lived in Python dictionaries you typed. Real models read their data from the company's database, check it, solve, and write the plan back. Today you build that loop: one model, fed by SQL, that plans the 3-product bakery and the full 30-product range **without changing a line of model code**.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 6.1 One database for the course; the full bakery range |
| 0:45 – 1:45 | 6.2 Load, check, build, solve: the bakery module |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 6.3 Same code, 3 products or 30 |
| 2:45 – 3:45 | 6.4 Plans back into the database; a bad forecast file |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: is a third packing shift worth it? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 6.1 One database for the course

From today, the course data lives in one DuckDB database file, `northbound.duckdb`, next to your code. Add it to `.gitignore` (it is rebuilt from the CSV files, so it does not belong in Git):

```bash
printf "*.duckdb\n*.duckdb.wal\n" >> .gitignore
```

Create `day06/load_tables.py`:

```python
import glob
import os

import duckdb

con = duckdb.connect("northbound.duckdb")
for path in sorted(glob.glob("data/*.csv")):
    table = os.path.splitext(os.path.basename(path))[0]
    con.sql(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv('{path}')")
print(con.sql("""
    SELECT table_name, estimated_size AS n_rows
    FROM duckdb_tables() ORDER BY table_name"""))
con.close()
```

```text
┌──────────────────────┬────────┐
│      table_name      │ n_rows │
│       varchar        │ int64  │
├──────────────────────┼────────┤
│ approval_rules       │      3 │
│ bakery_current_plan  │      3 │
│ bakery_disruptions   │      2 │
│ bakery_investments   │      4 │
│ bakery_machines      │      8 │
│ bakery_night_jobs    │     18 │
│ bakery_operations    │     72 │
│ bakery_oven_log      │    780 │
│ bakery_products      │      3 │
│ bakery_range         │     30 │
│   ·                  │      · │
│   ·                  │      · │
│   ·                  │      · │
│ stores               │     40 │
│ stress_scenarios     │      6 │
│ trip_log             │   1872 │
│ truck_orders         │     24 │
│ truck_spec           │      2 │
│ yule_log_economics   │      5 │
│ yule_log_forecast    │     40 │
│ yule_log_history     │    118 │
│ yule_markdown_days   │      4 │
│ yule_markdown_prices │      4 │
└──────────────────────┴────────┘
```

Every CSV becomes a table with the same name (your list grows as later lessons add data). `CREATE OR REPLACE` makes the script safe to rerun.

### The full bakery range

The bakery actually makes 30 products in four categories (bread, viennoiserie, cakes, cookies), on 4 resources:

```text
resource,unit,weekly_capacity,note
oven_hours,hours,600,12 deck ovens x 50 usable hours
baker_hours,hours,560,14 bakers x 40 hours
packing_hours,hours,300,2 packing lines x 150 hours
sourdough_starter,batches,60,starter culture can feed at most 60 batches a week
```

`bakery_range.csv` has the products (the three Module 1 products keep their numbers), `bakery_range_recipe.csv` has 94 rows of use per batch (4 sourdough products use starter), and `bakery_range_forecast.csv` has next week's demand cap for each product. The model is the one from Day 4; only the data are bigger.

## 6.2 Load, check, build, solve: the bakery module

We put the model in a module, `day06/bakery.py`, with one function per step:

| Function | Job |
|---|---|
| `load_data(con, …)` | Read the input tables with SQL, **check them**, return dictionaries |
| `build_model(d)` | Build the Pyomo model from the dictionaries; no numbers inside |
| `solve(m)` | Solve with HiGHS; raise an error unless the result is optimal |
| `plan_table`, `resource_table` | Turn the solution into tables people can read |
| `save_run(con, …)` | Store the plan in the database |

```python
"""Day 6: a data-driven bakery model. The same code plans 3 products or 30.

Import it:  from bakery import load_data, build_model, solve, plan_table
"""
import duckdb
import pandas as pd
import pyomo.environ as pyo


class DataError(Exception):
    """The input tables are inconsistent: fix the data, don't solve."""


def load_data(con, products="bakery_range", recipe="bakery_range_recipe",
              resources="bakery_range_resources", forecast="bakery_range_forecast"):
    """Read the four input tables with SQL and return plain Python dictionaries."""
    p = con.sql(f"""
        SELECT sku, product_name, category,
               units_per_batch * (transfer_price - ingredient_cost) AS profit_per_batch
        FROM {products} ORDER BY sku""").df()
    r = con.sql(f"SELECT resource, weekly_capacity FROM {resources} ORDER BY resource").df()
    u = con.sql(f"SELECT sku, resource, use_per_batch FROM {recipe}").df()
    f = con.sql(f"SELECT sku, demand_cap_batches FROM {forecast}").df()

    problems = []
    skus, res = set(p.sku), set(r.resource)
    for sku in sorted(skus - set(f.sku)):
        problems.append(f"{sku} has no forecast")
    for sku in sorted(set(f.sku) - skus):
        problems.append(f"forecast row for unknown product {sku}")
    for sku in sorted(skus - set(u.sku)):
        problems.append(f"{sku} has no recipe rows (it would use no resources)")
    for sku in sorted(set(u.sku) - skus):
        problems.append(f"recipe row for unknown product {sku}")
    for rr in sorted(set(u.resource) - res):
        problems.append(f"recipe uses unknown resource {rr}")
    if (u.use_per_batch < 0).any() or (f.demand_cap_batches < 0).any() or (r.weekly_capacity < 0).any():
        problems.append("negative use, forecast or capacity")
    if problems:
        raise DataError("; ".join(problems))

    return {
        "products": p.sku.tolist(),
        "resources": r.resource.tolist(),
        "name": dict(zip(p.sku, p.product_name)),
        "category": dict(zip(p.sku, p.category)),
        "profit": dict(zip(p.sku, p.profit_per_batch)),
        "capacity": dict(zip(r.resource, r.weekly_capacity)),
        "use": {(row.resource, row.sku): row.use_per_batch for row in u.itertuples()},
        "cap": dict(zip(f.sku, f.demand_cap_batches)),
    }


def build_model(d):
    m = pyo.ConcreteModel(name="bakery_plan")
    m.P = pyo.Set(initialize=d["products"])
    m.R = pyo.Set(initialize=d["resources"])
    m.profit = pyo.Param(m.P, initialize=d["profit"])
    m.capacity = pyo.Param(m.R, initialize=d["capacity"])
    m.use = pyo.Param(m.R, m.P, initialize=d["use"], default=0)
    m.cap = pyo.Param(m.P, initialize=d["cap"])
    m.x = pyo.Var(m.P, within=pyo.NonNegativeReals)
    m.demand = pyo.Constraint(m.P, rule=lambda m, p: m.x[p] <= m.cap[p])
    m.capacity_limit = pyo.Constraint(m.R, rule=lambda m, r: sum(m.use[r, p] * m.x[p] for p in m.P) <= m.capacity[r])
    m.total_profit = pyo.Objective(expr=sum(m.profit[p] * m.x[p] for p in m.P), sense=pyo.maximize)
    return m


def solve(m, tee=False):
    """Solve with HiGHS; raise unless the answer is optimal."""
    result = pyo.SolverFactory("highs").solve(m, tee=tee, load_solutions=False)
    tc = result.solver.termination_condition
    if tc != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"no optimal plan: {tc}")
    m.solutions.load_from(result)
    return result


def plan_table(m, d):
    rows = []
    for p in m.P:
        x = pyo.value(m.x[p])
        status = "zero" if x < 1e-6 else ("at cap" if x > d["cap"][p] - 1e-6 else "partial")
        rows.append((p, d["name"][p], d["category"][p], round(x, 4), d["cap"][p], status, round(d["profit"][p] * x, 2)))
    return pd.DataFrame(rows, columns=["sku", "product_name", "category", "batches", "demand_cap", "status", "profit"])


def resource_table(m):
    rows = []
    for r in m.R:
        used = pyo.value(m.capacity_limit[r].body)
        rows.append((r, round(used, 4), pyo.value(m.capacity[r]), round(pyo.value(m.capacity[r]) - used, 4)))
    return pd.DataFrame(rows, columns=["resource", "used", "capacity", "slack"])


def save_run(con, run_name, plan, resources):
    """Keep every plan in the database, so plans can be compared and audited."""
    con.sql("CREATE SEQUENCE IF NOT EXISTS run_seq")
    con.sql("""CREATE TABLE IF NOT EXISTS plan_runs (
                 run_id INTEGER PRIMARY KEY, run_name VARCHAR, solved_at TIMESTAMP, total_profit DOUBLE)""")
    con.sql("""CREATE TABLE IF NOT EXISTS plan_lines (
                 run_id INTEGER, sku VARCHAR, batches DOUBLE, demand_cap DOUBLE, status VARCHAR, profit DOUBLE)""")
    con.sql("""CREATE TABLE IF NOT EXISTS plan_resources (
                 run_id INTEGER, resource VARCHAR, used DOUBLE, capacity DOUBLE, slack DOUBLE)""")
    run_id = con.sql("SELECT nextval('run_seq')").fetchall()[0][0]
    con.execute("INSERT INTO plan_runs VALUES (?, ?, now()::TIMESTAMP, ?)", [run_id, run_name, float(plan.profit.sum())])
    con.sql("INSERT INTO plan_lines SELECT $run_id, sku, batches, demand_cap, status, profit FROM plan", params={"run_id": run_id})
    con.sql("INSERT INTO plan_resources SELECT $run_id, resource, used, capacity, slack FROM resources", params={"run_id": run_id})
    return run_id
```

Points worth slowing down for:

- **Table names are arguments.** `load_data` defaults to the full range, but any four tables (or views) with the same columns work.
- **Checks before the model.** Every product needs a forecast and at least one recipe row; every recipe row needs a known product and resource; nothing may be negative. A `DataError` stops the run with a list of every problem, not just the first. A model fed bad data still returns "optimal": the checks are your only protection.
- **`solve` refuses non-optimal results.** `load_solutions=False` lets us look at the status first; only then `m.solutions.load_from(result)` puts the values into the variables. Code that calls `solve` never sees a half-answer.
- **`save_run`** gives every run an id from a sequence and stores the plan and resource use. Parameters (`$run_id`) pass values into SQL safely instead of pasting them into the string.

> **Decision it supports:** a weekly plan that runs from the database, every week, with the same checks.
>
> **Other decisions you can support the same way:** any model that must be rerun when the data change, which is every model in production.

## 6.3 Same code, 3 products or 30

Create `day06/run_plans.py`. First the 3-product bakery from Days 1–4: two small views give its tables the column names `load_data` expects.

```python
import sys

import duckdb
import pandas as pd

sys.path.insert(0, "day06")
from bakery import build_model, load_data, plan_table, resource_table, save_run, solve

pd.set_option("display.width", 120)
pd.set_option("display.max_rows", 40)
con = duckdb.connect("northbound.duckdb")

# The Module 1 bakery: forecast_next_week.csv (from Day 1) has the demand caps
con.sql("CREATE OR REPLACE VIEW bakery_products_view AS SELECT *, 'module 1' AS category FROM bakery_products")
con.sql("CREATE OR REPLACE VIEW forecast_view AS SELECT sku, demand_cap_batches FROM forecast_next_week")
small = load_data(con, "bakery_products_view", "bakery_recipe", "bakery_resources", "forecast_view")
m = build_model(small)
solve(m)
plan, res = plan_table(m, small), resource_table(m)
print(plan[["sku", "batches", "status", "profit"]])
print(f"3-product profit ${plan.profit.sum():,.2f}")
save_run(con, "three products", plan, res)
```

```text
     sku  batches   status  profit
0  BK-CR     10.0  partial  2500.0
1  BK-SD     20.0  partial  6000.0
2  BK-WW     45.0   at cap  9000.0
3-product profit $17,500.00
```

The same $17,500, now from SQL. Then the full range, with the defaults:

```python
# The full range: same functions, different tables
full = load_data(con)
m = build_model(full)
solve(m)
plan, res = plan_table(m, full), resource_table(m)
print(plan[["sku", "product_name", "batches", "demand_cap", "status"]].to_string(index=False))
print(res.to_string(index=False))
print(f"30-product profit ${plan.profit.sum():,.2f}")
```

```text
  sku                   product_name  batches  demand_cap  status
BK-03                       Rye loaf     25.0          25  at cap
BK-04                Multigrain loaf     18.0          18  at cap
BK-05                Baguette 2-pack      7.0           7  at cap
BK-06             Country white loaf      8.0           8  at cap
BK-07                Olive sourdough      0.0          29    zero
BK-08               Seeded sourdough      0.0          19    zero
BK-09                Ciabatta 4-pack      8.0           8  at cap
BK-10                   Raisin bread     20.0          20  at cap
BK-12        Pain au chocolat 4-pack     15.0          15  at cap
BK-13        Almond croissant 4-pack     27.0          27  at cap
BK-14           Cinnamon roll 4-pack     26.0          26  at cap
BK-15                  Danish 4-pack      0.0           8    zero
BK-16                   Brioche loaf      9.0           9  at cap
BK-17          Apple turnover 4-pack      9.0           9  at cap
BK-18             Butter tart 6-pack     25.0          30 partial
BK-19                    Carrot cake      0.0          24    zero
BK-20                Lemon loaf cake      0.0          24    zero
BK-21              Black forest cake      0.0          18    zero
BK-22                     Cheesecake      0.0           9    zero
BK-23                   Banana bread     20.0          20  at cap
BK-24               Maple layer cake      0.0          28    zero
BK-25        Oatmeal cookies 12-pack     16.0          16  at cap
BK-26 Chocolate chip cookies 12-pack      1.6           9 partial
BK-27            Nanaimo bars 6-pack     25.0          25  at cap
BK-28            Date squares 6-pack     20.0          20  at cap
BK-29             Shortbread 12-pack     19.0          19  at cap
BK-30                Brownies 6-pack     20.0          20  at cap
BK-CR               Croissant 6-pack     31.0          31  at cap
BK-SD                 Sourdough loaf     35.0          35  at cap
BK-WW               Whole-wheat loaf     45.0          45  at cap
         resource   used  capacity  slack
      baker_hours 539.62       560  20.38
       oven_hours 600.00       600   0.00
    packing_hours 300.00       300   0.00
sourdough_starter  60.00        60   0.00
30-product profit $108,351.28
```

The full plan earns **$108,351.28 a week**. Read it the way a planner would:

- **22 products are baked, 8 are not.** Oven, packing and starter are full; 20 baker hours are spare.
- **Only 2 products are "partial"** (butter tarts and chocolate chip cookies). With 4 resources, at most 4 products can sit strictly between 0 and their cap in a corner solution; the rest are at a bound. That is Day 3's geometry in 30 dimensions.
- **Olive and seeded sourdough are not baked**: the starter is full, and the starter goes to sourdough loaves and rye, which earn more per batch of starter.

Then the plan goes into the database, and SQL reports on it:

```python
run_id = save_run(con, "full range", plan, res)

print(con.sql(f"""
    SELECT p.category, count(*) AS n_products,
           count(*) FILTER (WHERE l.status = 'at cap') AS at_cap,
           count(*) FILTER (WHERE l.status = 'zero') AS not_baked,
           round(sum(l.profit), 2) AS profit
    FROM plan_lines l JOIN bakery_range p USING (sku)
    WHERE l.run_id = {run_id}
    GROUP BY p.category ORDER BY profit DESC"""))
print(con.sql("SELECT run_id, run_name, round(total_profit, 2) AS total_profit FROM plan_runs ORDER BY run_id"))
con.close()
```

```text
┌──────────────┬────────────┬────────┬───────────┬──────────┐
│   category   │ n_products │ at_cap │ not_baked │  profit  │
│   varchar    │   int64    │ int64  │   int64   │  double  │
├──────────────┼────────────┼────────┼───────────┼──────────┤
│ bread        │         10 │      8 │         2 │  44366.0 │
│ viennoiserie │          8 │      6 │         1 │  36404.0 │
│ cookie       │          6 │      5 │         0 │ 21869.28 │
│ cake         │          6 │      1 │         5 │   5712.0 │
└──────────────┴────────────┴────────┴───────────┴──────────┘

┌────────┬────────────────┬──────────────┐
│ run_id │    run_name    │ total_profit │
│ int32  │    varchar     │    double    │
├────────┼────────────────┼──────────────┤
│      1 │ three products │      17500.0 │
│      2 │ full range     │    108351.28 │
└────────┴────────────────┴──────────────┘
```

Five of six cakes are not baked. That is a finding for the category manager, not a bug: cakes need the most baker and oven time per dollar. If the cakes must stay on the shelf for other reasons, that is a rule to add (Day 7 does exactly that).

### Try it 6.1

Write a SQL query over `plan_lines` (run 2) and the recipe that lists the products **not** baked, with their profit per batch and their oven, packing and starter use. Why isn't black forest cake baked, with $256 profit per batch?

<details><summary>Solution</summary>

```python
print(con.sql("""
    SELECT l.sku, p.product_name,
           round(p.units_per_batch * (p.transfer_price - p.ingredient_cost), 1) AS profit_per_batch,
           any_value(u.use_per_batch) FILTER (WHERE u.resource = 'oven_hours') AS oven,
           any_value(u.use_per_batch) FILTER (WHERE u.resource = 'packing_hours') AS packing,
           any_value(u.use_per_batch) FILTER (WHERE u.resource = 'sourdough_starter') AS starter
    FROM plan_lines l JOIN bakery_range p USING (sku) JOIN bakery_range_recipe u USING (sku)
    WHERE l.run_id = 2 AND l.status = 'zero'
    GROUP BY ALL ORDER BY profit_per_batch DESC"""))
```

```text
┌─────────┬───────────────────┬──────────────────┬────────┬─────────┬─────────┐
│   sku   │   product_name    │ profit_per_batch │  oven  │ packing │ starter │
│ varchar │      varchar      │      double      │ double │ double  │ double  │
├─────────┼───────────────────┼──────────────────┼────────┼─────────┼─────────┤
│ BK-21   │ Black forest cake │            256.0 │    1.8 │     1.0 │    NULL │
│ BK-08   │ Seeded sourdough  │            248.0 │    1.9 │     0.6 │     1.0 │
│ BK-22   │ Cheesecake        │            245.4 │    1.7 │     0.9 │    NULL │
│ BK-07   │ Olive sourdough   │            227.0 │    1.2 │     0.6 │     1.0 │
│ BK-20   │ Lemon loaf cake   │            219.2 │    2.3 │     0.9 │    NULL │
│ BK-15   │ Danish 4-pack     │            208.0 │    1.6 │     0.7 │    NULL │
│ BK-19   │ Carrot cake       │            199.0 │    2.3 │     1.0 │    NULL │
│ BK-24   │ Maple layer cake  │            193.8 │    1.5 │     1.2 │    NULL │
└─────────┴───────────────────┴──────────────────┴────────┴─────────┴─────────┘
```

`FILTER (WHERE …)` picks one resource's value inside the aggregate, turning the long recipe into columns. Black forest cake earns $256 a batch but uses 1.8 oven hours and 1.0 packing hour, both full. The products that are baked earn more **per hour of the full resources** they use. Profit per batch alone never decides; Day 11's reduced costs measure exactly how far each product is from being worth baking.

</details>

## 6.4 A bad forecast file

Next Monday, the forecast file arrives with `bk-04` instead of `BK-04`: one typo in one row. `day06/drill_bad_forecast.py` builds the model twice: once the quick way, where a missing cap silently becomes "no cap" (`.get()` returns `None`), and once through `load_data`.

```python
con = duckdb.connect()   # in memory: the drill must not touch northbound.duckdb
for t in ("bakery_range", "bakery_range_recipe", "bakery_range_resources", "bakery_range_forecast"):
    con.sql(f"CREATE TABLE {t} AS SELECT * FROM 'data/{t}.csv'")
con.sql("UPDATE bakery_range_forecast SET sku = 'bk-04' WHERE sku = 'BK-04'")   # the typo

# 1. Without the checks: build the model the quick way, with .get() for missing caps
d = {
    "products": [r[0] for r in con.sql("SELECT sku FROM bakery_range").fetchall()],
    "cap": dict(con.sql("SELECT sku, demand_cap_batches FROM bakery_range_forecast").fetchall()),
}
profit = dict(con.sql("SELECT sku, units_per_batch * (transfer_price - ingredient_cost) FROM bakery_range").fetchall())
use = {(r, s): u for s, r, u in con.sql("SELECT sku, resource, use_per_batch FROM bakery_range_recipe").fetchall()}
capacity = dict(con.sql("SELECT resource, weekly_capacity FROM bakery_range_resources").fetchall())
m = pyo.ConcreteModel()
m.P = pyo.Set(initialize=d["products"])
m.R = pyo.Set(initialize=list(capacity))
m.x = pyo.Var(m.P, bounds=lambda m, p: (0, d["cap"].get(p)))       # None = no upper bound
m.c = pyo.Constraint(m.R, rule=lambda m, r: sum(use.get((r, p), 0) * m.x[p] for p in m.P) <= capacity[r])
m.obj = pyo.Objective(expr=sum(profit[p] * m.x[p] for p in m.P), sense=pyo.maximize)
print(pyo.SolverFactory("highs").solve(m).solver.termination_condition)
print(f"profit ${pyo.value(m.obj):,.2f}; BK-04 Multigrain: {pyo.value(m.x['BK-04']):.2f} batches (forecast was 18)")

# 2. With the checks in load_data
try:
    load_data(con)
except DataError as e:
    print("DataError:", e)
```

```text
optimal
profit $121,309.40; BK-04 Multigrain: 255.47 batches (forecast was 18)
DataError: BK-04 has no forecast; forecast row for unknown product bk-04
```

Without the checks the solver says "optimal" and plans **255 batches of multigrain** against a forecast of 18, for a profit $13,000 higher than reality. Nobody would see an error; the stores would see 23,700 extra loaves. With the checks, the run stops and names both halves of the problem. The drill uses an in-memory database, so `northbound.duckdb` stays clean.

> **Decision it supports:** whether this week's data are fit to plan with, before anyone acts on the plan.
>
> **Other decisions you can support the same way:** rejecting a price file with a missing product · stopping a roster built on last month's staff list.

### Try it 6.2

What other mistakes in the input tables would produce a plausible but wrong plan? Name two that `load_data` already catches and one it does not.

<details><summary>Solution</summary>

Caught: a recipe row for a product that doesn't exist (a typo in the recipe file); a product with no recipe rows (it would use no resources and be unbounded, Day 7); a negative forecast. **Not caught:** a wrong but valid number, like Day 4's 15 oven hours instead of 1.5, or a forecast of 180 instead of 18. Range checks help ("no product uses more than 5 oven hours a batch", "a forecast more than 3 times last week's sales needs a human"), and Day 9 adds known-answer tests.

</details>

## Mini-project: is a third packing shift worth it?

Packing is full. The plant manager can add a third shift on one line: 40 more packing hours, for 340 in total. Without changing `bakery.py`:

1. Plan with 340 packing hours, using a view over `bakery_range_resources` and `load_data(con, resources=…)`.
2. Save the run as "packing 340".
3. Compare it with the "full range" run in SQL: which products change, and by how much?
4. How much more profit per week? Are all 40 hours used? What is the most the shift could cost and still pay?

<details><summary>Solution</summary>

```python
import sys

import duckdb

sys.path.insert(0, "day06")
from bakery import build_model, load_data, plan_table, resource_table, save_run, solve

con = duckdb.connect("northbound.duckdb")
con.sql("""CREATE OR REPLACE TEMP VIEW resources_more_packing AS
           SELECT resource, CASE WHEN resource = 'packing_hours' THEN 340 ELSE weekly_capacity END AS weekly_capacity
           FROM bakery_range_resources""")
d = load_data(con, resources="resources_more_packing")
m = build_model(d)
solve(m)
plan, res = plan_table(m, d), resource_table(m)
print(res.to_string(index=False))
new_run = save_run(con, "packing 340", plan, res)
base_run = con.sql("SELECT max(run_id) FROM plan_runs WHERE run_name = 'full range'").fetchall()[0][0]

print(con.sql(f"""
    SELECT sku, p.product_name, round(a.batches, 2) AS before, round(b.batches, 2) AS after,
           round(b.batches - a.batches, 2) AS change
    FROM (SELECT * FROM plan_lines WHERE run_id = {base_run}) a
    FULL JOIN (SELECT * FROM plan_lines WHERE run_id = {new_run}) b USING (sku)
    JOIN bakery_range p USING (sku)
    WHERE abs(b.batches - a.batches) > 1e-6
    ORDER BY change"""))
print(con.sql(f"""
    SELECT run_id, run_name, round(total_profit, 2) AS total_profit,
           round(total_profit - lag(total_profit) OVER (ORDER BY run_id), 2) AS gain
    FROM plan_runs WHERE run_id IN ({base_run}, {new_run}) ORDER BY run_id"""))
con.close()
```

```text
         resource     used  capacity   slack
      baker_hours 560.0000       560  0.0000
       oven_hours 600.0000       600  0.0000
    packing_hours 317.8459       340 22.1541
sourdough_starter  60.0000        60  0.0000
┌─────────┬────────────────────────────────┬────────┬────────┬────────┐
│   sku   │          product_name          │ before │ after  │ change │
│ varchar │            varchar             │ double │ double │ double │
├─────────┼────────────────────────────────┼────────┼────────┼────────┤
│ BK-SD   │ Sourdough loaf                 │   35.0 │    6.0 │  -29.0 │
│ BK-18   │ Butter tart 6-pack             │   25.0 │  29.09 │   4.09 │
│ BK-22   │ Cheesecake                     │    0.0 │    5.2 │    5.2 │
│ BK-26   │ Chocolate chip cookies 12-pack │    1.6 │    9.0 │    7.4 │
│ BK-07   │ Olive sourdough                │    0.0 │   29.0 │   29.0 │
└─────────┴────────────────────────────────┴────────┴────────┴────────┘

┌────────┬─────────────┬──────────────┬─────────┐
│ run_id │  run_name   │ total_profit │  gain   │
│ int32  │   varchar   │    double    │ double  │
├────────┼─────────────┼──────────────┼─────────┤
│      2 │ full range  │    108351.28 │    NULL │
│      3 │ packing 340 │     109906.6 │ 1555.32 │
└────────┴─────────────┴──────────────┴─────────┘
```

1–3. The extra shift raises profit by **$1,555.32 a week**, and the plan changes in a way no one would guess: **sourdough drops from 35 to 6 batches and olive sourdough goes from 0 to 29**. Both use one batch of starter, but olive sourdough needs 1.2 oven hours against sourdough's 2.0. The 23.2 oven hours this frees go to butter tarts, cheesecake and cookies, which now have packing time. Sourdough earns the most per batch, yet it is the product the model cuts.

4. Only 17.8 of the 40 hours are used: the bakers (560 hours) are now full. The shift pays if it costs less than $1,555 a week. Better: book about 18 hours, not 40, and price an extra baker next.

The `FULL JOIN … USING (sku)` keeps a product that appears in only one of the two runs; `lag()` puts each run's profit next to the one before.

</details>

## Commit

```bash
git add .
git commit -m "Day 6: bakery module fed by DuckDB, data checks, plan runs"
git push
```

### Day 6 checklist

- [ ] I loaded every course CSV into `northbound.duckdb`.
- [ ] My model reads its data with SQL and contains no numbers.
- [ ] The same code plans 3 products ($17,500) and 30 products ($108,351.28).
- [ ] `load_data` refuses inconsistent data and lists every problem.
- [ ] Plans are stored with a run id, and I can compare two runs in SQL.
- [ ] I can explain why a product with high profit per batch may not be baked.

---

# DAY 7 — When Models Break: Infeasible and Unbounded

Marketing, sales and product development each add a rule to next week's plan. The solver answers: **infeasible**. Today you learn to find out why, and to report it in business terms ("the café contract and the forecast disagree", "the festival needs 12 more batches of starter than we have") rather than as a solver error. Then you meet the opposite failure: a plan that says profit is infinite.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 7.1 Three new rules, one infeasible model |
| 0:45 – 1:45 | 7.2 Cheap checks first; 7.3 an elastic model |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 7.4 Finding the smallest set of rules that conflict |
| 2:45 – 3:45 | 7.5 Unbounded models: the missing rule |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: the sourdough festival memo |
| 4:45 – 5:00 | Commit with Git + checklist |

## 7.1 Three new rules, one infeasible model

`bakery_range_minimums.csv` collects three new rules from three teams:

| Rule | From | What it says |
|---|---|---|
| Shelf presence | Marketing | Every product at least 2 batches, so it never disappears from the shelf |
| Café Laurier contract | Sales | At least 50 batches of whole-wheat a week for a new café customer |
| Sourdough festival | Product development | At least 18 batches of each of the 4 sourdough products |

Several rules can set a minimum for the same product (whole-wheat has the shelf rule and the contract), so the largest counts. We add one constraint per product, $$x_p \ge \min_p$$. Create `day07/diagnose.py`:

```python
import sys

import duckdb
import pyomo.environ as pyo

sys.path.insert(0, "day06")
from bakery import build_model, load_data

con = duckdb.connect("northbound.duckdb", read_only=True)
solver = pyo.SolverFactory("highs")


def status(m):
    return solver.solve(m, load_solutions=False).solver.termination_condition


def minimums(rules_sql="SELECT * FROM bakery_range_minimums"):
    """Several rules can set a minimum for the same product: the largest one counts."""
    return dict(con.sql(f"SELECT sku, max(min_batches) FROM ({rules_sql}) GROUP BY sku").fetchall())


def add_minimums(m, mins):
    m.min_batches = pyo.Param(m.P, initialize=mins, default=0)
    m.minimum = pyo.Constraint(m.P, rule=lambda m, p: m.x[p] >= m.min_batches[p])
    return m


# ---------- 7.1 The new rules make the model infeasible ----------
d = load_data(con)
m = add_minimums(build_model(d), minimums())
print("with all minimum rules:", status(m))
try:
    solver.solve(m)
except Exception as e:
    print(type(e).__name__, "-", str(e)[:70], "...")
```

```text
with all minimum rules: infeasible
NoFeasibleSolutionError - A feasible solution was not found, so no solution can be loaded. Pleas ...
```

`status()` solves with `load_solutions=False` and returns only the termination condition. Without that option, Pyomo raises `NoFeasibleSolutionError` as soon as it tries to load a solution that does not exist. Either way: **no plan satisfies all the rules**.

The solver cannot tell you which rules to drop: that is a business decision. Your job is to find which rules conflict, and what each way out costs.

## 7.2 Cheap checks first

Most infeasibilities are simple, and SQL finds them in seconds. Two questions:

1. Is any minimum above its own maximum?
2. Do the minimums alone already need more of a resource than we have?

```python
# ---------- 7.2 Cheap checks first: a minimum above its own maximum ----------
print(con.sql("""
    SELECT n.sku, n.min_batches, n.reason, f.demand_cap_batches
    FROM bakery_range_minimums n JOIN bakery_range_forecast f USING (sku)
    WHERE n.min_batches > f.demand_cap_batches"""))
print(con.sql("""
    SELECT r.resource, round(sum(u.use_per_batch * n.min_batches), 1) AS needed_by_minimums,
           any_value(r.weekly_capacity) AS capacity
    FROM (SELECT sku, max(min_batches) AS min_batches FROM bakery_range_minimums GROUP BY sku) n
    JOIN bakery_range_recipe u USING (sku) JOIN bakery_range_resources r USING (resource)
    GROUP BY r.resource ORDER BY r.resource"""))
```

```text
┌─────────┬─────────────┬───────────────────────┬────────────────────┐
│   sku   │ min_batches │        reason         │ demand_cap_batches │
│ varchar │    int64    │        varchar        │       int64        │
├─────────┼─────────────┼───────────────────────┼────────────────────┤
│ BK-WW   │          50 │ Cafe Laurier contract │                 45 │
└─────────┴─────────────┴───────────────────────┴────────────────────┘

┌───────────────────┬────────────────────┬──────────┐
│     resource      │ needed_by_minimums │ capacity │
│      varchar      │       double       │  int64   │
├───────────────────┼────────────────────┼──────────┤
│ baker_hours       │              196.0 │      560 │
│ oven_hours        │              239.0 │      600 │
│ packing_hours     │              106.4 │      300 │
│ sourdough_starter │               72.0 │       60 │
└───────────────────┴────────────────────┴──────────┘
```

Both checks fire:

- **The contract asks for 50 batches of whole-wheat, but the cap is 45.** The cap is the *stores'* forecast. The café's 50 batches come on top of it, so the cap was wrong for this new kind of demand. This is a modelling mistake, not a capacity problem.
- **The festival needs 4 × 18 = 72 batches of starter; there are 60.** This one is a real capacity conflict.

> **Decision it supports:** which teams to call, and about what, before anyone touches the model.
>
> **Other decisions you can support the same way:** checking a sales promise against capacity before signing · checking a staffing request against the budget.

## 7.3 An elastic model: how little must we break?

On bigger models, conflicts involve many constraints and no simple SQL finds them. A standard trick is the **elastic model**: give every rule a "bend" variable that lets it be broken, at a price, and minimize the total bending. The answer shows which rules must bend, and by how much, to make the plan possible.

```python
# ---------- 7.3 An elastic model: how little must we break? ----------
def elastic(m):
    """Let every rule bend, at a price; minimize total bending instead of maximizing profit."""
    m.total_profit.deactivate()
    m.bend_min = pyo.Var(m.P, within=pyo.NonNegativeReals)
    m.bend_cap = pyo.Var(m.P, within=pyo.NonNegativeReals)
    m.bend_res = pyo.Var(m.R, within=pyo.NonNegativeReals)
    for p in m.P:
        m.minimum[p].set_value(m.x[p] + m.bend_min[p] >= m.min_batches[p])
        m.demand[p].set_value(m.x[p] <= m.cap[p] + m.bend_cap[p])
    for r in m.R:
        m.capacity_limit[r].set_value(sum(m.use[r, p] * m.x[p] for p in m.P) <= m.capacity[r] + m.bend_res[r])
    m.bending = pyo.Objective(expr=sum(m.bend_min[p] + m.bend_cap[p] for p in m.P) + sum(m.bend_res[r] for r in m.R))
    solver.solve(m)
    for v in (m.bend_min, m.bend_cap, m.bend_res):
        for k in v:
            if pyo.value(v[k]) > 1e-6:
                print(f"  {v.name}[{k}] = {pyo.value(v[k]):g}")


print("Elastic model:")
elastic(add_minimums(build_model(d), minimums()))
```

```text
Elastic model:
  bend_min[BK-07] = 12
  bend_min[BK-WW] = 5
```

To get a plan, whole-wheat's minimum must bend by 5 batches (50 → 45) and some sourdough minimum by 12 batches. The total, 17, is the smallest total bending possible.

Note *which* sourdough product bends: olive sourdough, but any of the four would do, because each bends the starter use by the same amount. When several ways out are equally cheap, the solver picks one arbitrarily. If some rules matter more than others, give their bend variables bigger weights: that is your statement of priorities, and the business should set it.

`set_value` replaces a constraint's expression in place, so the same model can be turned into its elastic version.

### Try it 7.1

The café contract is signed and must not bend. Give `bend_min["BK-WW"]` a weight of 10 in the objective (all others 1) and re-solve. What bends now?

<details><summary>Solution</summary>

```python
w = {"BK-WW": 10}
m.bending = pyo.Objective(expr=sum(w.get(p, 1) * m.bend_min[p] + m.bend_cap[p] for p in m.P)
                          + sum(m.bend_res[r] for r in m.R))
```

(delete the old objective first with `m.del_component(m.bending)`.) Now `bend_cap[BK-WW] = 5` instead of `bend_min[BK-WW] = 5`: the model raises whole-wheat's *forecast cap* to 50 rather than cutting the contract. Same total, different story: with the weights, the elastic model points at the real fix for the contract (raise the cap by the café's volume).

</details>

## 7.4 The smallest set of rules that conflict

The elastic model says how much to bend. To say which rules are **in conflict with each other**, we want an **irreducible infeasible set** (IIS): a set of rules that cannot all hold, where dropping any single one of them makes the rest feasible.

The **deletion filter** finds one with nothing but repeated solves:

1. Take every rule.
2. For each rule in turn, switch it off and re-solve.
3. If the model is still infeasible, that rule wasn't needed for the conflict: leave it off.
4. If it becomes feasible, the rule is part of the conflict: switch it back on.
5. The rules still on at the end are an IIS.

```python
# ---------- 7.4 A deletion filter: the smallest set of rules that conflict ----------
def conflict_set(m):
    """Drop rules one at a time; keep a rule dropped if the model stays infeasible.
    What is left is an irreducible infeasible set (IIS): remove any one rule and it solves."""
    cons = [c for comp in (m.minimum, m.demand, m.capacity_limit) for c in comp.values()]
    for c in cons:
        c.deactivate()
        if status(m) != pyo.TerminationCondition.infeasible:
            c.activate()        # needed for the conflict: put it back
    return [c.name for c in cons if c.active]


print("Conflict 1:", conflict_set(add_minimums(build_model(d), minimums())))

# Fix 1: the cafe contract comes on top of store demand, so it raises the cap too
d_fixed = dict(d, cap=dict(d["cap"], **{"BK-WW": d["cap"]["BK-WW"] + 50}))
m = add_minimums(build_model(d_fixed), minimums())
print("after fix 1:", status(m))
print("Conflict 2:", conflict_set(add_minimums(build_model(d_fixed), minimums())))

# Fix 2: the festival asks for 15 batches of each sourdough, not 18 (4 x 15 = 60 = starter capacity)
mins = minimums("""SELECT sku, CASE WHEN reason = 'sourdough festival' THEN 15 ELSE min_batches END AS min_batches
                   FROM bakery_range_minimums""")
m = add_minimums(build_model(d_fixed), mins)
print("after fix 2:", status(m))
solver.solve(m)
print(f"profit with all rules ${pyo.value(m.total_profit):,.2f}")
for p in ("BK-SD", "BK-03", "BK-07", "BK-08", "BK-WW"):
    print(f"  {p} {d['name'][p]:<18} {pyo.value(m.x[p]):g} batches")
```

```text
Conflict 1: ['minimum[BK-WW]', 'demand[BK-WW]']
after fix 1: infeasible
Conflict 2: ['minimum[BK-03]', 'minimum[BK-07]', 'minimum[BK-08]', 'minimum[BK-SD]', 'capacity_limit[sourdough_starter]']
after fix 2: optimal
profit with all rules $108,825.58
  BK-SD Sourdough loaf     15 batches
  BK-03 Rye loaf           15 batches
  BK-07 Olive sourdough    15 batches
  BK-08 Seeded sourdough   15 batches
  BK-WW Whole-wheat loaf   95 batches
```

The filter needed about 65 solves of a 30-product model, a fraction of a second.

- **Conflict 1** is the whole-wheat minimum against its cap. Fix 1 adds the café's 50 batches to the stores' 45: cap 95.
- The model is **still infeasible**: one IIS at a time. **Conflict 2** is the four festival minimums against the starter. Fix 2, after talking to product development: 15 batches of each, which exactly fills the 60 batches of starter.

Then the plan is feasible, and every rule has a price:

```python
# What does each rule cost? Solve with and without it.
m0 = build_model(d_fixed)
solver.solve(m0)
m1 = add_minimums(build_model(d_fixed), minimums("SELECT * FROM bakery_range_minimums WHERE reason = 'shelf presence'"))
solver.solve(m1)
print(f"no minimums ${pyo.value(m0.total_profit):,.2f}; shelf rule only ${pyo.value(m1.total_profit):,.2f}; all rules ${pyo.value(m.total_profit):,.2f}")
```

```text
no minimums $110,936.85; shelf rule only $109,908.38; all rules $108,825.58
```

(All three include the café's 50 batches.) The shelf-presence rule costs **$1,028.47 a week**; the festival at 15 batches costs another **$1,082.80**. Marketing can now weigh those numbers against the sales the rules bring. That conversation is the point of the whole exercise.

Commercial solvers can compute an IIS directly (Gurobi's `computeIIS` is the best known). The deletion filter is worth knowing because it works with any solver and you can explain it to anyone.

> **Decision it supports:** which business rules to renegotiate, in which order, with what each costs.
>
> **Other decisions you can support the same way:** service-level promises that can't all be kept · a timetable where some requests must be refused.

## 7.5 Unbounded: the missing rule

Product development adds **focaccia** to the product list ($280 profit a batch), but the recipe and forecast files have not been updated yet. A model built without Day 6's checks:

```python
# ---------- 7.5 Unbounded: a product with no recipe and no forecast ----------
# Focaccia: 100 loaves a batch, $4.20 price, $1.40 ingredients -> $280 a batch.
# Someone added it to the product list, but not yet to the recipe or the forecast.
d2 = dict(d)
d2["products"] = d["products"] + ["BK-31"]
d2["profit"] = dict(d["profit"], **{"BK-31": 280.0})
d2["name"] = dict(d["name"], **{"BK-31": "Focaccia"})
m = pyo.ConcreteModel()
m.P = pyo.Set(initialize=d2["products"])
m.x = pyo.Var(m.P, bounds=lambda m, p: (0, d2["cap"].get(p)))
m.c = pyo.Constraint(d2["resources"], rule=lambda m, r: sum(d2["use"].get((r, p), 0) * m.x[p] for p in m.P) <= d2["capacity"][r])
m.obj = pyo.Objective(expr=sum(d2["profit"][p] * m.x[p] for p in m.P), sense=pyo.maximize)
print("Focaccia added without recipe or forecast:", status(m))

# Find the culprit: give every variable a huge bound and see which ones reach it
for p in m.P:
    if m.x[p].ub is None:
        m.x[p].setub(1e6)
solver.solve(m)
print("variables at the 1,000,000 bound:", [p for p in m.P if pyo.value(m.x[p]) > 1e6 - 1])
```

```text
Focaccia added without recipe or forecast: unbounded
variables at the 1,000,000 bound: ['BK-31']
```

With no recipe rows, focaccia uses no oven, bakers or packing; with no forecast, it has no cap. Every batch earns $280 and costs nothing, so profit can grow forever: **unbounded**.

Real businesses are never unbounded. An unbounded model always means a missing rule or a missing number. To find it, give every variable a huge bound (here a million batches) and solve: the variables that reach the bound point at the gap. `load_data` would have refused the data outright ("BK-31 has no forecast; BK-31 has no recipe rows").

| Status | What it means | First check |
|---|---|---|
| **Infeasible** | No plan satisfies every rule | Minimums above maximums; minimums alone above capacity; then an elastic model or a deletion filter |
| **Unbounded** | Profit can grow forever | A variable with no cap, a product with no resource use, a wrong sign; then big bounds |
| **Optimal but wrong** | The model solved, but not the model you meant | `pprint`, the LP file, data checks, recomputing the objective |

## Mini-project: the sourdough festival memo

Product development insists on 18 batches of each sourdough product. The starter culture can be expanded to 72 batches a week (one more starter tank). Compare four options, with the café contract fixed as in 7.4:

- **A:** no festival (shelf rule and café only)
- **B:** festival at 15 batches each, 60 starter batches
- **C:** festival at 18 batches each, 72 starter batches
- **D:** festival at 18 batches each, 60 starter batches

Write `day07/festival_memo.md`: the profit of each option, what the extra starter capacity is worth per week, and a recommendation in three sentences.

<details><summary>Solution</summary>

```python
import sys

import duckdb
import pyomo.environ as pyo

sys.path.insert(0, "day06")
from bakery import build_model, load_data

con = duckdb.connect("northbound.duckdb", read_only=True)
solver = pyo.SolverFactory("highs")
d = load_data(con)
d = dict(d, cap=dict(d["cap"], **{"BK-WW": d["cap"]["BK-WW"] + 50}))   # the cafe contract on top of store demand


def plan(festival_min, starter_capacity):
    rules = con.sql(f"""
        SELECT sku, max(CASE WHEN reason = 'sourdough festival' THEN {festival_min} ELSE min_batches END)
        FROM bakery_range_minimums GROUP BY sku""").fetchall()
    dd = dict(d, capacity=dict(d["capacity"], sourdough_starter=starter_capacity))
    m = build_model(dd)
    m.minimum = pyo.Constraint(m.P, rule=lambda m, p: m.x[p] >= dict(rules).get(p, 0))
    result = solver.solve(m, load_solutions=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return None
    m.solutions.load_from(result)
    return pyo.value(m.total_profit)


options = {
    "A: no festival (shelf rule and cafe only)": plan(0, 60),
    "B: festival at 15 batches each": plan(15, 60),
    "C: festival at 18 batches, 72 starter batches": plan(18, 72),
    "D: festival at 18 batches, 60 starter batches": plan(18, 60),
}
for name, profit in options.items():
    print(f"{name:<48} {'infeasible' if profit is None else f'${profit:,.2f}'}")
gain = options["C: festival at 18 batches, 72 starter batches"] - options["B: festival at 15 batches each"]
print(f"12 extra starter batches are worth ${gain:,.2f} a week, ${gain / 12:,.2f} per batch")
```

```text
A: no festival (shelf rule and cafe only)        $109,908.38
B: festival at 15 batches each                   $108,825.58
C: festival at 18 batches, 72 starter batches    $109,278.60
D: festival at 18 batches, 60 starter batches    infeasible
12 extra starter batches are worth $453.02 a week, $37.75 per batch
```

*Memo:* "The festival as proposed (18 batches of each sourdough) is impossible with today's starter: it needs 72 batches and we have 60. At 15 batches each it fits and costs $1,082.80 a week in profit against no festival; at 18 batches with a second starter tank it costs $629.78. The second tank is worth $453 a week, so it pays for itself if it costs less than about $23,500 a year; otherwise run the festival at 15 batches."

Every number in the memo comes from a solve, and option D is reported as what it is: not a worse plan, but no plan at all.

</details>

## Commit

```bash
git add .
git commit -m "Day 7: infeasible and unbounded models, elastic model, deletion filter"
git push
```

### Day 7 checklist

- [ ] I check the status with `load_solutions=False` before loading a solution.
- [ ] I run cheap SQL checks (minimum above maximum, minimums above capacity) first.
- [ ] I can build an elastic model and read which rules must bend, and by how much.
- [ ] I can find an irreducible infeasible set with a deletion filter.
- [ ] I can find the missing rule behind an unbounded model.
- [ ] I report infeasibility as a business conflict with a price on each way out.
