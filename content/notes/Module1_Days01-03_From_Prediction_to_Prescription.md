# DAY 1 — From Prediction to Prescription

Today you meet Northbound Fresh, look at two years of its bakery sales, make a first forecast, and then ask the question the rest of the course answers: **given the forecast and the rules, what exactly should we do?**

| Time | Block |
|---|---|
| 0:00 – 0:45 | 1.1 The analytics ladder and the course set-up |
| 0:45 – 1:45 | 1.2 Descriptive: what happened? (DuckDB on the sales history) |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 1.3 Predictive: what will happen next week? |
| 2:45 – 3:45 | 1.4 Prescriptive: what should we bake? Checking plans and trying every plan |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: write a decision statement |
| 4:45 – 5:00 | Commit with Git + checklist |

## 1.1 The analytics ladder

Analytics answers four kinds of question. Each step up is harder and worth more.

| Step | Question | Bakery example | Typical tool |
|---|---|---|---|
| Descriptive | What happened? | How many sourdough loaves did we sell last December? | SQL, dashboards |
| Diagnostic | Why did it happen? | Did the price change cause the drop? | Causal inference |
| Predictive | What will happen? | How many will we sell next week? | Forecasting, machine learning |
| **Prescriptive** | **What should we do?** | **How many batches of each product should we bake next week?** | **Optimization, simulation, decision analysis** |

A forecast alone does not tell the bakery what to bake. The ovens, the bakers and the sourdough starter all have limits, and the products compete for them. Prescriptive analytics takes the forecast **and** the rules, and computes the best action that respects every rule.

### Meet Northbound Fresh

Northbound Fresh is a fictional grocery chain with 40 stores in Ontario and Quebec, served by three distribution centres (Mississauga, Ottawa and Montreal). In Module 1 we work on its central bakery, which makes three private-label products:

| SKU | Product | One batch |
|---|---|---|
| BK-SD | Sourdough loaf | 100 loaves |
| BK-WW | Whole-wheat loaf | 100 loaves |
| BK-CR | Croissant 6-pack | 50 packs |

Every Friday the bakery planner decides how many batches of each product to bake next week. She has used the same plan since September 2025: **25 sourdough, 35 whole-wheat, 10 croissant**. By Day 3 you will know whether she can do better, and by how much.

### Set up your repository

You work in GitHub Codespaces for the whole course.

1. On GitHub, create a new private repository called `northbound-decisions` with a README.
2. Click **Code → Codespaces → Create codespace on main**.
3. In the Codespace terminal, create a Python environment and install the course packages:

```bash
python -m venv .venv
source .venv/bin/activate
cat > requirements.txt <<'EOF'
duckdb==2.0.1
pandas==3.1.1
numpy==2.5.5
scipy==1.18.1
matplotlib==3.11.2
pyomo==6.11.0
highspy==1.16.0
EOF
pip install -r requirements.txt
```

4. Open the **Course data** page on this site, copy the whole of `make_data.py`, and save it at the root of your repository as `make_data.py`.
5. Create a `.gitignore` so generated files stay out of Git:

```bash
printf ".venv/\ndata/\nfigures/\n__pycache__/\n" > .gitignore
```

6. Generate the data:

```bash
python make_data.py
```

```text
data/dcs.csv: 3 rows
data/stores.csv: 40 rows
data/bakery_products.csv: 3 rows
data/bakery_recipe.csv: 7 rows
data/bakery_resources.csv: 3 rows
data/bakery_current_plan.csv: 3 rows
data/bakery_sales_weekly.csv: 12,360 rows
```

Everyone's data is identical because the generator uses a fixed random seed (2026). If your row counts differ, you copied an older version of the script.

> **Decision it supports:** none yet. This is the workshop where every later decision is built.
>
> **Other uses of the same set-up:** rerun `make_data.py` any time you damage the data · share the repository with a classmate and get the same numbers.

## 1.2 Descriptive: what happened?

Create `day01/ladder.py`. We start by pointing DuckDB at the CSV files. DuckDB reads CSV files directly, so there is nothing to load.

```python
import os

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
con = duckdb.connect()
con.sql("CREATE VIEW sales AS SELECT * FROM 'data/bakery_sales_weekly.csv'")
con.sql("CREATE VIEW products AS SELECT * FROM 'data/bakery_products.csv'")
con.sql("CREATE VIEW stores AS SELECT * FROM 'data/stores.csv'")

print(con.sql("""
    SELECT count(*) AS n_rows, count(DISTINCT week_start) AS n_weeks,
           count(DISTINCT store_id) AS n_stores,
           min(week_start) AS first_week, max(week_start) AS last_week
    FROM sales"""))
```

```text
┌────────┬─────────┬──────────┬────────────┬────────────┐
│ n_rows │ n_weeks │ n_stores │ first_week │ last_week  │
│ int64  │  int64  │  int64   │    date    │    date    │
├────────┼─────────┼──────────┼────────────┼────────────┤
│  12360 │     104 │       40 │ 2024-11-25 │ 2026-11-16 │
└────────┴─────────┴──────────┴────────────┴────────────┘
```

Two years of weekly sales: 104 weeks, 40 stores, 3 products. That would be 12,480 rows, but there are 12,360. Keep that gap in mind for Try it 1.2.

Now the chain total per week and product, and a chart:

```python
weekly = con.sql("""
    SELECT week_start, sku, sum(units_sold)::INTEGER AS units
    FROM sales GROUP BY ALL ORDER BY week_start, sku""").df()

fig, ax = plt.subplots(figsize=(10, 4))
for sku, g in weekly.groupby("sku"):
    ax.plot(g["week_start"], g["units"], label=sku)
ax.set_title("Northbound Fresh bakery: units sold per week, all stores")
ax.set_ylabel("units per week")
ax.legend()
fig.tight_layout()
fig.savefig("figures/day01_weekly_sales.png", dpi=120)
```

![Weekly bakery sales](figures/day01_weekly_sales.png)

Three patterns stand out: a Christmas peak (largest for croissants), a summer dip in July and August, and slow growth from one year to the next.

```python
print(con.sql("""
    SELECT sku, arg_max(week_start, units) AS busiest_week, max(units) AS units
    FROM (SELECT week_start, sku, sum(units_sold)::INTEGER AS units FROM sales GROUP BY ALL)
    GROUP BY sku ORDER BY sku"""))
```

```text
┌─────────┬──────────────┬───────┐
│   sku   │ busiest_week │ units │
│ varchar │     date     │ int32 │
├─────────┼──────────────┼───────┤
│ BK-CR   │ 2025-12-22   │  2479 │
│ BK-SD   │ 2025-12-15   │  4149 │
│ BK-WW   │ 2025-12-22   │  5387 │
└─────────┴──────────────┴───────┘
```

`arg_max(week_start, units)` returns the `week_start` of the row where `units` is largest: one function instead of a sort and a join.

> **Decision it supports:** when to plan extra capacity (the Christmas weeks).
>
> **Other decisions you can support the same way:** which months to schedule oven maintenance (the summer dip) · how much growth to budget for next year.

### Try it 1.1

Which distribution centre's stores sold the most croissants over the two years? Join `sales` to `stores`.

<details><summary>Solution</summary>

```python
print(con.sql("""
    SELECT st.dc_id, sum(s.units_sold)::INTEGER AS croissants
    FROM sales s JOIN stores st USING (store_id)
    WHERE s.sku = 'BK-CR'
    GROUP BY ALL ORDER BY croissants DESC"""))
```

```text
┌─────────┬────────────┐
│  dc_id  │ croissants │
│ varchar │   int32    │
├─────────┼────────────┤
│ DC-MTL  │      60538 │
│ DC-MIS  │      49455 │
│ DC-OTT  │      40558 │
└─────────┴────────────┘
```

Montreal, with 15 stores, sells the most.

</details>

### Try it 1.2

The table has 120 rows fewer than 104 weeks × 40 stores × 3 products. Find the store responsible, and the reason.

<details><summary>Solution</summary>

```python
print(con.sql("""
    SELECT store_id, count(DISTINCT week_start) AS n_weeks, min(week_start) AS first_week
    FROM sales GROUP BY ALL ORDER BY n_weeks LIMIT 3"""))
print(con.sql("SELECT store_id, store_name, opened_date FROM stores WHERE store_id = 'S40'"))
```

```text
┌──────────┬─────────┬────────────┐
│ store_id │ n_weeks │ first_week │
│ varchar  │  int64  │    date    │
├──────────┼─────────┼────────────┤
│ S40      │      64 │ 2025-09-01 │
│ S03      │     104 │ 2024-11-25 │
│ S04      │     104 │ 2024-11-25 │
└──────────┴─────────┴────────────┘

┌──────────┬─────────────────────────┬─────────────┐
│ store_id │       store_name        │ opened_date │
│ varchar  │         varchar         │    date     │
├──────────┼─────────────────────────┼─────────────┤
│ S40      │ Northbound Sherbrooke 2 │ 2025-09-01  │
└──────────┴─────────────────────────┴─────────────┘
```

Store S40 opened on 1 September 2025: 40 weeks × 3 products = 120 missing rows. They are missing because the store did not exist, not because data were lost. Part of the growth you saw in the chart comes from this new store.

</details>

## 1.3 Predictive: what will happen next week?

We need a forecast for the week of 23 November 2026. We try two simple methods and test them on the last 26 weeks: for each week, forecast it from the weeks before, then measure the miss.

- **Last week:** next week sells what this week sold.
- **8-week average:** next week sells the average of the last 8 weeks.

```python
print(con.sql("""
    WITH w AS (
        SELECT week_start, sku, sum(units_sold) AS units FROM sales GROUP BY ALL
    ), f AS (
        SELECT week_start, sku, units,
               lag(units) OVER (PARTITION BY sku ORDER BY week_start) AS last_week,
               avg(units) OVER (PARTITION BY sku ORDER BY week_start
                                ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING) AS avg_8_weeks
        FROM w
    )
    SELECT sku,
           round(avg(abs(units - last_week)), 1) AS mae_last_week,
           round(avg(abs(units - avg_8_weeks)), 1) AS mae_avg_8_weeks
    FROM f
    WHERE week_start > (SELECT max(week_start) FROM w) - INTERVAL 26 WEEK
    GROUP BY sku ORDER BY sku"""))
```

```text
┌─────────┬───────────────┬─────────────────┐
│   sku   │ mae_last_week │ mae_avg_8_weeks │
│ varchar │    double     │     double      │
├─────────┼───────────────┼─────────────────┤
│ BK-CR   │          36.0 │            78.6 │
│ BK-SD   │          95.2 │           171.2 │
│ BK-WW   │          99.9 │           219.9 │
└─────────┴───────────────┴─────────────────┘
```

MAE is the mean absolute error: the average size of the miss, in units. "Last week" misses by about half as much. The 8-week average lags behind: the last 26 weeks include the summer dip and the September recovery, and an average of old weeks is slow to follow them. So "last week" is our forecast.

```python
forecast = con.sql("""
    SELECT s.sku, p.units_per_batch,
           sum(s.units_sold)::INTEGER AS forecast_units,
           round(sum(s.units_sold) / p.units_per_batch, 2) AS forecast_batches,
           round(sum(s.units_sold) / p.units_per_batch)::INTEGER AS demand_cap_batches
    FROM sales s JOIN products p USING (sku)
    WHERE s.week_start = (SELECT max(week_start) FROM sales)
    GROUP BY s.sku, p.units_per_batch ORDER BY s.sku""")
print(forecast)
forecast.write_csv("data/forecast_next_week.csv")
```

```text
┌─────────┬─────────────────┬────────────────┬──────────────────┬────────────────────┐
│   sku   │ units_per_batch │ forecast_units │ forecast_batches │ demand_cap_batches │
│ varchar │      int64      │     int32      │      double      │       int32        │
├─────────┼─────────────────┼────────────────┼──────────────────┼────────────────────┤
│ BK-CR   │              50 │           1551 │            31.02 │                 31 │
│ BK-SD   │             100 │           3500 │             35.0 │                 35 │
│ BK-WW   │             100 │           4499 │            44.99 │                 45 │
└─────────┴─────────────────┴────────────────┴──────────────────┴────────────────────┘
```

`demand_cap_batches` is the forecast rounded to whole batches. It becomes a rule tomorrow: **don't bake more than we expect to sell**. Day 2 reads this file, so keep it.

In real work you would use a better forecasting model (that is the predictive course). Here the point is the hand-off: **the forecast is an input to the decision, not the decision.**

> **Decision it supports:** choosing a forecasting method by testing it on past weeks, not by taste.
>
> **Other decisions you can support the same way:** whether a new model is worth replacing the old one · how far ahead a forecast stays useful.

## 1.4 Prescriptive: what should we bake?

The bakery has three limits each week:

```python
print(open("data/bakery_resources.csv").read())
print(open("data/bakery_recipe.csv").read())
```

```text
resource,unit,weekly_capacity,note
oven_hours,hours,100,2 deck ovens x 50 usable hours
baker_hours,hours,80,2 bakers x 40 hours
sourdough_starter,batches,40,starter culture can feed at most 40 batches a week

sku,resource,use_per_batch
BK-SD,oven_hours,2.0
BK-SD,baker_hours,1.0
BK-SD,sourdough_starter,1.0
BK-WW,oven_hours,1.0
BK-WW,baker_hours,1.0
BK-CR,oven_hours,1.5
BK-CR,baker_hours,1.5
```

And each product earns a profit per batch: units per batch × (transfer price − ingredient cost).

```python
print(con.sql("""
    SELECT sku, product_name,
           units_per_batch * (transfer_price - ingredient_cost) AS profit_per_batch
    FROM products ORDER BY sku"""))
```

```text
┌─────────┬──────────────────┬──────────────────┐
│   sku   │   product_name   │ profit_per_batch │
│ varchar │     varchar      │      double      │
├─────────┼──────────────────┼──────────────────┤
│ BK-CR   │ Croissant 6-pack │            250.0 │
│ BK-SD   │ Sourdough loaf   │            300.0 │
│ BK-WW   │ Whole-wheat loaf │            200.0 │
└─────────┴──────────────────┴──────────────────┘
```

### Checking a plan

A function that takes any plan and reports what it uses and earns:

```python
con.sql("CREATE VIEW recipe AS SELECT * FROM 'data/bakery_recipe.csv'")
con.sql("CREATE VIEW resources AS SELECT * FROM 'data/bakery_resources.csv'")
con.sql("CREATE VIEW plan AS SELECT * FROM 'data/bakery_current_plan.csv'")


def check_plan(con, plan_sql):
    """Resources used by a plan versus capacity, and the plan's weekly profit."""
    con.sql(f"CREATE OR REPLACE TEMP VIEW p AS {plan_sql}")
    use = con.sql("""
        SELECT r.resource, sum(r.use_per_batch * p.batches_per_week) AS used,
               any_value(c.weekly_capacity) AS capacity,
               used <= capacity AS ok
        FROM recipe r JOIN p USING (sku) JOIN resources c USING (resource)
        GROUP BY r.resource ORDER BY r.resource""")
    profit = con.sql("""
        SELECT sum(p.batches_per_week * units_per_batch * (transfer_price - ingredient_cost)) AS weekly_profit
        FROM p JOIN products USING (sku)""").fetchall()[0][0]
    return use, profit


use, profit = check_plan(con, "SELECT * FROM plan")
print(use)
print(f"weekly profit: ${profit:,.0f}")
```

```text
┌───────────────────┬────────┬──────────┬─────────┐
│     resource      │  used  │ capacity │   ok    │
│      varchar      │ double │  int64   │ boolean │
├───────────────────┼────────┼──────────┼─────────┤
│ baker_hours       │   75.0 │       80 │ true    │
│ oven_hours        │  100.0 │      100 │ true    │
│ sourdough_starter │   25.0 │       40 │ true    │
└───────────────────┴────────┴──────────┴─────────┘

weekly profit: $17,000
```

The current plan fits: the ovens are exactly full, with 5 baker hours and 15 starter batches spare. It earns $17,000 a week.

Why not simply bake what the forecast says we can sell?

```python
use, profit = check_plan(con, "SELECT sku, demand_cap_batches AS batches_per_week FROM 'data/forecast_next_week.csv'")
print(use)
print(f"weekly profit: ${profit:,.0f}  (but is it possible?)")
```

```text
┌───────────────────┬────────┬──────────┬─────────┐
│     resource      │  used  │ capacity │   ok    │
│      varchar      │ double │  int64   │ boolean │
├───────────────────┼────────┼──────────┼─────────┤
│ baker_hours       │  126.5 │       80 │ false   │
│ oven_hours        │  161.5 │      100 │ false   │
│ sourdough_starter │   35.0 │       40 │ true    │
└───────────────────┴────────┴──────────┴─────────┘

weekly profit: $27,250  (but is it possible?)
```

Baking to forecast needs 161.5 oven hours and 126.5 baker hours. We have 100 and 80. The $27,250 is a fantasy. This is the gap prescriptive analytics fills: **the forecast says what we could sell; capacity says what we can make; the decision has to respect both.**

### The anatomy of a decision

Every prescriptive model has the same four parts. Write them in words before any code.

| Part | Bakery |
|---|---|
| **Decisions** (what we choose) | Batches of sourdough, whole-wheat and croissant to bake next week |
| **Objective** (what we want) | Maximize weekly profit |
| **Constraints** (rules we must respect) | Oven hours ≤ 100; baker hours ≤ 80; starter ≤ 40 batches; no product above its demand cap; no negative batches |
| **Data** (numbers we are given) | Profit per batch, use per batch, capacities, forecast |

Then as mathematics. Call the batches $$S$$ (sourdough), $$W$$ (whole-wheat) and $$C$$ (croissant):

$$
\begin{aligned}
\max\quad & 300S + 200W + 250C \\
\text{s.t.}\quad & 2S + W + 1.5C \le 100 && \text{(oven hours)}\\
& S + W + 1.5C \le 80 && \text{(baker hours)}\\
& S \le 40 && \text{(starter)}\\
& S \le 35,\; W \le 45,\; C \le 31 && \text{(demand caps)}\\
& S, W, C \ge 0
\end{aligned}
$$

This is a **linear program**: the objective and every constraint are sums of numbers times decision variables. Days 2 and 3 solve it; Day 4 writes it in Pyomo.

### Trying every plan

With three products and caps of 35, 45 and 31 batches, there are only 36 × 46 × 32 = 52,992 whole-batch plans. A computer can try them all. Create `day01/brute_force.py`:

```python
import time

import numpy as np

profit = {"BK-SD": 300, "BK-WW": 200, "BK-CR": 250}
oven = {"BK-SD": 2.0, "BK-WW": 1.0, "BK-CR": 1.5}
baker = {"BK-SD": 1.0, "BK-WW": 1.0, "BK-CR": 1.5}
cap = {"BK-SD": 35, "BK-WW": 45, "BK-CR": 31}   # demand caps from the forecast

start = time.perf_counter()
sd, ww, cr = np.meshgrid(np.arange(cap["BK-SD"] + 1),
                         np.arange(cap["BK-WW"] + 1),
                         np.arange(cap["BK-CR"] + 1), indexing="ij")
sd, ww, cr = sd.ravel(), ww.ravel(), cr.ravel()

oven_used = oven["BK-SD"] * sd + oven["BK-WW"] * ww + oven["BK-CR"] * cr
baker_used = baker["BK-SD"] * sd + baker["BK-WW"] * ww + baker["BK-CR"] * cr
fits = (oven_used <= 100) & (baker_used <= 80) & (sd <= 40)
value = profit["BK-SD"] * sd + profit["BK-WW"] * ww + profit["BK-CR"] * cr
value = np.where(fits, value, -1)

best = value.max()
winners = np.flatnonzero(value == best)
seconds = time.perf_counter() - start

print(f"plans tried:     {sd.size:,}")
print(f"plans that fit:  {fits.sum():,}")
print(f"best profit:     ${best:,}")
for i in winners:
    print(f"  best plan: sourdough {sd[i]}, whole-wheat {ww[i]}, croissant {cr[i]}")
print(f"time:            {seconds:.3f} s")

print()
print("Plans to try if each product can take 0..40 batches:")
for n_products in (3, 10, 30, 120):
    print(f"  {n_products:>4} products: 41^{n_products} = {41 ** n_products:.2e}")
```

```text
plans tried:     52,992
plans that fit:  37,559
best profit:     $17,500
  best plan: sourdough 20, whole-wheat 45, croissant 10
time:            0.008 s

Plans to try if each product can take 0..40 batches:
     3 products: 41^3 = 6.89e+04
    10 products: 41^10 = 1.34e+16
    30 products: 41^30 = 2.42e+48
   120 products: 41^120 = 3.42e+193
```

(Your time will differ.) The best plan, **20 sourdough, 45 whole-wheat, 10 croissant**, earns **$17,500 a week: $500 more than the current plan**, or about $26,000 a year, from the same ovens and bakers.

Brute force works for 3 products. Northbound Fresh's full bakery has 30 products; at a billion plans a second, 41^30 plans would take far longer than the age of the universe. That is why the rest of the course uses **solvers**, which find the best plan without trying them all.

> **Decision it supports:** next week's bake plan, and how much the current rule of thumb costs.
>
> **Other decisions you can support the same way:** a weekly staffing mix within a budget · how many trucks of each size to rent.

### Try it 1.3

Bakers ask for overtime: 90 baker hours instead of 80. Change one number in `brute_force.py`. What is the best profit now, and is the best plan unique?

<details><summary>Solution</summary>

Change `(baker_used <= 80)` to `(baker_used <= 90)`:

```text
best profit:     $17,800
  best plan: sourdough 10, whole-wheat 44, croissant 24
  best plan: sourdough 11, whole-wheat 45, croissant 22
```

Ten extra baker hours add $300 a week. Two different plans tie: the best plan is not always unique. Day 3 shows why from the geometry.

</details>

## Mini-project: write a decision statement

Northbound Fresh must decide which of its 3 distribution centres supplies each of its 40 stores. Without any code, write in `day01/decision_statement.md`:

1. The **decisions**, in one sentence.
2. The **objective**, and its unit.
3. At least four **constraints**, in plain words.
4. The **data** you would need, and which file in `data/` already has part of it.
5. How many possible assignments there are, if every store can be served by any DC.

<details><summary>Model answer</summary>

1. **Decisions:** for each store, which DC supplies it.
2. **Objective:** minimize total weekly transport cost, in dollars per week.
3. **Constraints:** every store is served by exactly one DC; each DC ships no more than its weekly capacity; delivery time from DC to store is under a maximum (fresh products); Quebec stores need French-labelled stock that only some DCs carry.
4. **Data:** store demand (from `bakery_sales_weekly.csv`, and later all products), DC capacities, cost per kilometre or per pallet for each DC–store pair, travel times. `stores.csv` and `dcs.csv` have the locations and the current DC.
5. 3^40 = 12,157,665,459,056,928,801 (about 1.2 × 10^19) assignments. Brute force is hopeless; this is the Day 15 mini-project, solved in under a second.

</details>

## Commit

```bash
git add .
git commit -m "Day 1: analytics ladder, forecast, plan checks, brute force"
git push
```

### Day 1 checklist

- [ ] I can place a question on the analytics ladder (descriptive, diagnostic, predictive, prescriptive).
- [ ] I generated the course data and got 12,360 sales rows.
- [ ] I chose a forecast by testing it on past weeks.
- [ ] I can explain why "bake to forecast" is impossible next week.
- [ ] I can name the four parts of a decision model: decisions, objective, constraints, data.
- [ ] I found the best plan by brute force ($17,500) and know why brute force will not scale.

---

# DAY 2 — Spreadsheet First: The Product Mix

Yesterday brute force found the best bake plan. Today you build the same model three ways: as a spreadsheet, by hand with algebra, and with a real solver. By the end you know what a solver does, and how to catch it when you have told it the wrong thing.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 2.1 The model as a spreadsheet |
| 0:45 – 1:45 | 2.2 Trial and error, and a greedy rule that fails |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 2.3 By hand: which constraints are tight? |
| 2:45 – 3:45 | 2.4 Your first solver, and a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: should the bakery add a rye loaf? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 2.1 The model as a spreadsheet

Planners have optimized with spreadsheets for decades. A spreadsheet model has three kinds of cells:

- **Decision cells:** the numbers you change (batches of each product).
- **Formula cells:** what follows from the decisions (hours used, profit).
- **Limit cells:** what the formulas must not exceed (capacities, demand caps).

If you have Excel on Windows, you can build this in a sheet: decisions in `B2:D2`, the recipe in `B5:D7`, hours used in `E5` as `=SUMPRODUCT(B5:D5,$B$2:$D$2)`, and profit as `=SUMPRODUCT(B3:D3,B2:D2)`. Excel's Solver add-in (**Data → Solver**) then changes the decision cells to maximize profit subject to `E5:E7 <= F5:F7`.

We build the same spreadsheet in pandas, so that it lives in Git and reads the course data. Create `day02/product_mix.py`:

```python
import numpy as np
import pandas as pd
from scipy.optimize import linprog

products = pd.read_csv("data/bakery_products.csv").set_index("sku")
products["profit_per_batch"] = products.units_per_batch * (products.transfer_price - products.ingredient_cost)
recipe = pd.read_csv("data/bakery_recipe.csv").pivot(index="resource", columns="sku", values="use_per_batch").fillna(0)
capacity = pd.read_csv("data/bakery_resources.csv").set_index("resource")["weekly_capacity"]
demand = pd.read_csv("data/forecast_next_week.csv").set_index("sku")["demand_cap_batches"]

skus = list(products.index)          # BK-SD, BK-WW, BK-CR
recipe = recipe[skus]
print("Recipe (use per batch):")
print(recipe)
```

```text
Recipe (use per batch):
sku                BK-SD  BK-WW  BK-CR
resource                              
baker_hours          1.0    1.0    1.5
oven_hours           2.0    1.0    1.5
sourdough_starter    1.0    0.0    0.0
```

`pivot` turns the long recipe file into a resource × product table. `fillna(0)` says whole-wheat and croissants use no starter. This table is the left-hand side of the constraints you wrote yesterday.

Now the spreadsheet itself, as a function:

```python
def evaluate(plan):
    """The spreadsheet: decision cells in, used / slack / profit out."""
    x = pd.Series(plan, dtype=float)[skus]
    used = recipe @ x
    sheet = pd.DataFrame({"used": used, "capacity": capacity[used.index]})
    sheet["slack"] = sheet.capacity - sheet.used
    over_demand = (x - demand[skus]).clip(lower=0)
    feasible = bool((sheet.slack >= -1e-9).all() and (over_demand <= 1e-9).all() and (x >= 0).all())
    profit = float(products.profit_per_batch[skus] @ x)
    return sheet, profit, feasible


plan = {"BK-SD": 25, "BK-WW": 35, "BK-CR": 10}
sheet, profit, ok = evaluate(plan)
print("Current plan:", plan)
print(sheet)
print(f"profit ${profit:,.0f}, feasible: {ok}")
```

```text
Current plan: {'BK-SD': 25, 'BK-WW': 35, 'BK-CR': 10}
                    used  capacity  slack
resource                                 
baker_hours         75.0        80    5.0
oven_hours         100.0       100    0.0
sourdough_starter   25.0        40   15.0
profit $17,000, feasible: True
```

`recipe @ x` is matrix multiplication: for every resource, the sum of use per batch × batches. It is `SUMPRODUCT` for all rows at once.

**Slack** is capacity left over. A constraint with zero slack is **binding** (or tight): it is what stops us earning more. In the current plan only the oven binds.

The `1e-9` tolerances matter: computers store 0.1 + 0.2 as 0.30000000000000004, so "exactly full" can look a hair over.

> **Decision it supports:** checking any plan a manager proposes, in seconds, before anyone bakes it.
>
> **Other decisions you can support the same way:** checking a staff roster against hour limits · checking a truck load against weight and volume.

### Try it 2.1

A manager proposes 20 sourdough, 40 whole-wheat and 15 croissant. Is it feasible? Which constraint fails, and by how much?

<details><summary>Solution</summary>

```python
sheet, profit, ok = evaluate({"BK-SD": 20, "BK-WW": 40, "BK-CR": 15})
print(sheet); print(profit, ok)
```

Oven hours used = 2 × 20 + 40 + 1.5 × 15 = 102.5, slack −2.5, so it is infeasible: 2.5 oven hours over. Baker hours: 20 + 40 + 22.5 = 82.5, also 2.5 over. Profit would be $17,750, but the plan can't be baked.

</details>

## 2.2 Trial and error, and a greedy rule that fails

Try a few plans:

```python
for plan in [{"BK-SD": 30, "BK-WW": 30, "BK-CR": 10},
             {"BK-SD": 20, "BK-WW": 45, "BK-CR": 10},
             {"BK-SD": 35, "BK-WW": 30, "BK-CR": 0}]:
    sheet, profit, ok = evaluate(plan)
    print(f"{plan}  profit ${profit:,.0f}  feasible {ok}  oven slack {sheet.slack['oven_hours']:.1f}  baker slack {sheet.slack['baker_hours']:.1f}")
```

```text
{'BK-SD': 30, 'BK-WW': 30, 'BK-CR': 10}  profit $17,500  feasible False  oven slack -5.0  baker slack 5.0
{'BK-SD': 20, 'BK-WW': 45, 'BK-CR': 10}  profit $17,500  feasible True  oven slack 0.0  baker slack 0.0
{'BK-SD': 35, 'BK-WW': 30, 'BK-CR': 0}  profit $16,500  feasible True  oven slack 0.0  baker slack 15.0
```

The third plan is the "make the most profitable product first" rule: sourdough earns most per batch ($300), so make all 35, then fill with whole-wheat. It earns $16,500, less than the current plan.

A smarter-sounding rule: the oven is the bottleneck, so rank products by **profit per oven hour** and fill the oven in that order.

```python
per_oven_hour = (products.profit_per_batch / recipe.loc["oven_hours"]).sort_values(ascending=False)
print("Profit per oven hour:")
print(per_oven_hour)
x = {s: 0.0 for s in skus}
for sku in per_oven_hour.index:
    # largest amount of this product that still fits every constraint
    room = [demand[sku]]
    used = recipe @ pd.Series(x)[skus]
    for r in recipe.index:
        if recipe.loc[r, sku] > 0:
            room.append((capacity[r] - used[r]) / recipe.loc[r, sku])
    x[sku] = max(0.0, min(room))
sheet, profit, ok = evaluate(x)
print("Greedy plan:", {k: round(float(v), 2) for k, v in x.items()}, f"profit ${profit:,.0f}")
```

```text
Profit per oven hour:
sku
BK-WW    200.000000
BK-CR    166.666667
BK-SD    150.000000
dtype: float64
Greedy plan: {'BK-SD': 0.0, 'BK-WW': 45.0, 'BK-CR': 23.33} profit $14,833
```

$14,833: worse still. The rule looks only at the oven, but croissants use 1.5 baker hours each and run the bakers out first (45 + 1.5 × 23.33 = 80). **With two or more binding constraints, no single ranking is right.** Products compete for several resources at once, and the best mix balances them. That balancing is exactly what a solver does.

### Try it 2.2

Rank by profit per **baker** hour instead and fill in that order. What profit do you get?

<details><summary>Solution</summary>

Per baker hour: sourdough $300, whole-wheat $200, croissant $166.67. Fill sourdough to its cap of 35 (70 oven hours, 35 baker hours). Whole-wheat then has room for 30 more oven hours, so 30 batches. The oven is now full: no croissants. Profit 300 × 35 + 200 × 30 = **$16,500**, the same plan as "most profitable first". Neither ranking finds $17,500.

</details>

## 2.3 By hand: which constraints are tight?

Brute force said the best plan uses the oven and the bakers fully, and whole-wheat sits at its demand cap of 45. If we **guess** those three constraints are binding, they become three equations in three unknowns:

$$
\begin{aligned}
1.5C + 2S + W &= 100 && \text{(oven full)}\\
1.5C + S + W &= 80 && \text{(bakers full)}\\
W &= 45 && \text{(whole-wheat at its cap)}
\end{aligned}
$$

```python
A = np.array([[1.5, 2, 1], [1.5, 1, 1], [0, 0, 1]])
b = np.array([100, 80, 45])
cr, sd, ww = np.linalg.solve(A, b)
print(f"By hand: croissant {cr:g}, sourdough {sd:g}, whole-wheat {ww:g}")
sheet, profit, ok = evaluate({"BK-CR": cr, "BK-SD": sd, "BK-WW": ww})
print(f"profit ${profit:,.0f}, feasible {ok}")
```

```text
By hand: croissant 10, sourdough 20, whole-wheat 45
profit $17,500, feasible True
```

Subtracting the second equation from the first gives $$S = 20$$ straight away: each sourdough batch uses one more oven hour than baker hour, and the oven has 20 more hours than the bakers.

This is the key idea behind linear programming: **the best plan sits where some set of constraints are all tight at once.** A solver's job is to find the right set without guessing.

### Try it 2.3

Guess a different set: oven full, bakers full, and croissants at their cap of 31. Solve it. Is the result feasible? Is it better?

<details><summary>Solution</summary>

```python
A = np.array([[1.5, 2, 1], [1.5, 1, 1], [1, 0, 0]])
print(np.linalg.solve(A, [100, 80, 31]))
```

```text
[31.  20.  13.5]
```

Croissant 31, sourdough 20, whole-wheat 13.5. Every constraint holds, so it is a feasible corner, but profit is 250 × 31 + 300 × 20 + 200 × 13.5 = **$16,450**. Feasible is not the same as best. (And a guess of "oven full, sourdough at 35, whole-wheat at 45" gives croissant = −10: a corner outside the feasible plans.)

</details>

## 2.4 Your first solver

`scipy.optimize.linprog` solves linear programs. With `method="highs"` (the default) it calls **HiGHS**, the same open-source solver we use through Pyomo from Day 4. It expects the problem in one fixed shape:

$$
\min_x\; c^\top x \quad \text{s.t.}\quad A_{ub}\,x \le b_{ub},\quad \text{lower} \le x \le \text{upper}
$$

It **minimizes**. To maximize profit, minimize minus profit.

```python
c = -products.profit_per_batch[skus].to_numpy()      # linprog minimises, so flip the sign
A_ub = recipe.to_numpy()
b_ub = capacity[recipe.index].to_numpy()
bounds = [(0, demand[s]) for s in skus]
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
print("linprog status:", res.message)
print("plan:", {s: round(float(v), 4) for s, v in zip(skus, res.x)})
print(f"profit ${-res.fun:,.2f}")
print("slack by resource:", {r: round(float(v), 4) for r, v in zip(recipe.index, res.slack)})
```

```text
linprog status: Optimization terminated successfully. (HiGHS Status 7: Optimal)
plan: {'BK-SD': 20.0, 'BK-WW': 45.0, 'BK-CR': 10.0}
profit $17,500.00
slack by resource: {'baker_hours': 0.0, 'oven_hours': 0.0, 'sourdough_starter': 20.0}
```

Same answer as brute force and as the algebra, without trying 52,992 plans or guessing. The demand caps went in as **bounds** (upper limits on single variables), which solvers handle more cheaply than full constraints.

Three checks every time you run a solver:

1. **Status:** "Optimal". Anything else (infeasible, unbounded, time limit) means the numbers are not an answer.
2. **Sign of the objective:** `res.fun` is −17,500 because we minimized minus profit.
3. **Sense check:** recompute profit and slack yourself (our `evaluate` function) and compare.

> **Decision it supports:** the weekly bake plan, recomputed in milliseconds whenever the forecast or capacity changes.
>
> **Other decisions you can support the same way:** a media mix across channels with budget and audience limits · a feed or fertilizer blend at least cost.

### Try it 2.4: what is one more hour worth?

Re-solve with 110 oven hours, and separately with 81 baker hours. How much does profit change per extra hour?

<details><summary>Solution</summary>

```python
for oven_h, baker_h in [(110, 80), (100, 81)]:
    r = linprog(c, A_ub=A_ub, b_ub=[baker_h, oven_h, 40], bounds=bounds, method="highs")
    print(oven_h, baker_h, r.x.round(4), -r.fun)
```

(The order of `b_ub` follows `recipe.index`: baker, oven, starter.)

```text
110 80 [30.     45.      3.3333] 18833.333333333332
100 81 [19.     45.     11.3333] 17533.333333333332
```

Ten more oven hours add $1,333.33: **$133.33 per oven hour**. One more baker hour adds **$33.33**. If an oven hour of overtime costs less than $133, it pays; a baker hour pays only if it costs under $33. These numbers are **shadow prices**, and the solver can report them without re-solving. That is Day 11.

</details>

### The planted bug

Open `day02/bug_sign.py`. **Before running it**, read it and predict what it will print.

```python
from scipy.optimize import linprog

profit = [300, 200, 250]                 # sourdough, whole-wheat, croissant
A_ub = [[2.0, 1.0, 1.5],                 # oven hours
        [1.0, 1.0, 1.5],                 # baker hours
        [1.0, 0.0, 0.0]]                 # sourdough starter
b_ub = [100, 80, 40]
bounds = [(0, 35), (0, 45), (0, 31)]

res = linprog(profit, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
print(res.message)
print("plan:", [round(float(v), 4) for v in res.x], "objective:", res.fun)
```

<details><summary>What happens, and the fix</summary>

```text
Optimization terminated successfully. (HiGHS Status 7: Optimal)
plan: [0.0, 0.0, 0.0] objective: 0.0
```

The solver reports success and recommends baking nothing. It did exactly what it was told: **minimize** profit, and the least profit is zero. The fix is `linprog([-p for p in profit], ...)`, then report `-res.fun`.

The lesson: an "Optimal" status means the solver solved *your* model, not that your model is right. Always run the sense check.

</details>

## Mini-project: should the bakery add a rye loaf?

Product development proposes a rye loaf: 100 loaves per batch, transfer price $4.00, ingredients $1.60 per loaf, 1.5 oven hours and 1 baker hour per batch, and half a batch of sourdough starter. Marketing expects to sell at most 15 batches a week.

1. Compute the rye profit per batch. Is it more or less than whole-wheat?
2. Solve the four-product model with `linprog` in `day02/mini_project_rye.py`.
3. How much does rye add to weekly profit? What does the bakery make less of?
4. Write two sentences for the planner: add rye or not?

<details><summary>Solution</summary>

```python
from scipy.optimize import linprog

skus = ["BK-SD", "BK-WW", "BK-CR", "BK-RY"]
profit = [300, 200, 250, 100 * (4.00 - 1.60)]          # rye: 100 loaves, $4.00 price, $1.60 ingredients
oven = [2.0, 1.0, 1.5, 1.5]
baker = [1.0, 1.0, 1.5, 1.0]
starter = [1.0, 0.0, 0.0, 0.5]                          # rye uses half a batch of starter
demand = [35, 45, 31, 15]

res = linprog([-p for p in profit], A_ub=[oven, baker, starter], b_ub=[100, 80, 40],
              bounds=[(0, d) for d in demand], method="highs")
print(res.message)
for s, x in zip(skus, res.x):
    print(f"  {s}: {x:g} batches")
print(f"profit ${-res.fun:,.2f}  (was $17,500.00 without rye)")
print("slack oven, baker, starter:", [round(float(v), 4) for v in res.slack])
```

```text
Optimization terminated successfully. (HiGHS Status 7: Optimal)
  BK-SD: 12.5 batches
  BK-WW: 45 batches
  BK-CR: 5 batches
  BK-RY: 15 batches
profit $17,600.00  (was $17,500.00 without rye)
slack oven, baker, starter: [0.0, 0.0, 20.0]
```

1. Rye earns 100 × ($4.00 − $1.60) = **$240 a batch**, more than whole-wheat ($200).
2. The solver bakes all 15 rye batches, cutting sourdough from 20 to 12.5 and croissants from 10 to 5.
3. Profit rises by only **$100 a week**. Rye's $240 is mostly paid for with oven and baker hours taken from sourdough and croissants.
4. *Memo:* "Rye adds about $100 a week ($5,200 a year) with current ovens and bakers, because it takes capacity from sourdough and croissants. Add it only if the launch costs less than that, or together with extra oven hours, which are worth $133 each."

Notice the plan has half a batch of sourdough. Real batches are whole; Module 3 adds integer variables.

</details>

## Commit

```bash
git add .
git commit -m "Day 2: spreadsheet model, greedy rules, by-hand solution, linprog"
git push
```

### Day 2 checklist

- [ ] I can build a plan checker that reports used, capacity and slack for each resource.
- [ ] I can explain why ranking products by profit per batch, or per hour of one resource, misses the best plan.
- [ ] I can find a corner by guessing which constraints are binding and solving the equations.
- [ ] I solved the product mix with `linprog` and got $17,500.
- [ ] I always check status, objective sign and a recomputed profit.
- [ ] I know what an extra oven hour ($133.33) and an extra baker hour ($33.33) are worth.

---

# DAY 3 — The Geometry of Linear Programs

Today you draw linear programs. With two products, every plan is a point on a chart, every constraint is a line, and the best plan is a corner you can see. The pictures explain everything the solver did yesterday, including the four ways a model can go wrong.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 3.1 Plans as points, constraints as lines |
| 0:45 – 1:45 | 3.2 Corner points and iso-profit lines |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 3.3 Adding demand caps; how far can a price move? |
| 2:45 – 3:45 | 3.4 Four special cases: infeasible, unbounded, many optima, degenerate |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: a December week with overtime |
| 4:45 – 5:00 | Commit with Git + checklist |

## 3.1 Plans as points, constraints as lines

Next week the croissant line is down for maintenance, so the bakery makes only sourdough ($$S$$) and whole-wheat ($$W$$). We start without demand caps, to see the shape clearly:

$$
\begin{aligned}
\max\quad & 300S + 200W \\
\text{s.t.}\quad & 2S + W \le 100 && \text{(oven)}\\
& S + W \le 80 && \text{(bakers)}\\
& S \le 40 && \text{(starter)}\\
& S, W \ge 0
\end{aligned}
$$

Each plan $$(S, W)$$ is a point. Each constraint, with $$\le$$ replaced by $$=$$, is a straight line; the plans that satisfy it lie on one side. The oven line runs from $$(0, 100)$$ to $$(50, 0)$$: all oven hours on whole-wheat, or all on sourdough.

The plans that satisfy **every** constraint form the **feasible region**. Because every constraint is a straight line, the region is a polygon with straight sides and no dents (it is **convex**).

Create `day03/geometry.py`. We write each constraint as three numbers $$(a_S, a_W, b)$$ meaning $$a_S S + a_W W \le b$$; the non-negativity rules become $$-S \le 0$$ and $$-W \le 0$$.

```python
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

os.makedirs("figures", exist_ok=True)

# Every constraint written as  a_S * S + a_W * W <= b
BASE = {
    "oven: 2S + W <= 100": (2, 1, 100),
    "bakers: S + W <= 80": (1, 1, 80),
    "starter: S <= 40": (1, 0, 40),
    "S >= 0": (-1, 0, 0),
    "W >= 0": (0, -1, 0),
}
```

## 3.2 Corner points and iso-profit lines

A **corner** (vertex) is where two constraint lines cross and every other constraint still holds. We find them all: intersect every pair of lines, keep the points that are feasible.

```python
def corners(cons):
    """Intersect every pair of constraint lines; keep the points that satisfy all constraints."""
    pts = []
    for (n1, (a1, b1, c1)), (n2, (a2, b2, c2)) in itertools.combinations(cons.items(), 2):
        M = np.array([[a1, b1], [a2, b2]], dtype=float)
        if abs(np.linalg.det(M)) < 1e-12:
            continue                                  # parallel lines never meet
        s, w = np.linalg.solve(M, [c1, c2])
        if all(a * s + b * w <= c + 1e-9 for a, b, c in cons.values()):
            pts.append((round(s, 6) + 0.0, round(w, 6) + 0.0))
    return sorted(set(pts))
```

(`+ 0.0` turns a "negative zero" −0.0 into 0.0, so the same corner isn't kept twice.)

The drawing function plots each line, shades the region, labels every corner with its profit, and draws a dashed **iso-profit line** through the best corner: all the plans that earn the same profit.

```python
def order_polygon(pts):
    """Sort corner points around their centre so they draw as a polygon."""
    cx, cy = np.mean(pts, axis=0)
    return sorted(pts, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))


def draw(ax, cons, profit, title, lim=(0, 100), unbounded=False):
    pts = corners(cons)
    xs = np.linspace(*lim, 200)
    lines = [(n, abc) for n, abc in cons.items() if not n.endswith(">= 0")]
    for i, (name, (a, b, c)) in enumerate(lines):
        color = f"C{i}"
        if b == 0:
            ax.axvline(c / a, lw=1.2, color=color, label=name)
        else:
            ax.plot(xs, (c - a * xs) / b, lw=1.2, color=color, label=name)
    if unbounded:
        # the region runs off the top of the chart: profit can grow forever
        ax.fill([0, 40, 40, 0], [0, 0, lim[1], lim[1]], alpha=0.18, color="tab:green", label="feasible region (never ends)")
        ax.annotate("", xy=(30, 98), xytext=(30, 60), arrowprops=dict(arrowstyle="->", lw=2))
        ax.text(31, 75, "more W is always\nallowed and earns more", fontsize=8)
        pts = []
    elif not pts:
        ax.text(5, 5, "no point satisfies every constraint", fontsize=9)
    if len(pts) >= 3:
        poly = order_polygon(pts)
        ax.fill(*zip(*poly), alpha=0.18, color="tab:green", label="feasible region")
    best = None
    for s, w in pts:
        v = profit[0] * s + profit[1] * w
        ax.plot(s, w, "ko", ms=4)
        ax.annotate(f"({s:g}, {w:g})\n${v:,.0f}", (s, w), textcoords="offset points", xytext=(5, 5), fontsize=8)
        if best is None or v > best[2]:
            best = (s, w, v)
    if best:
        # iso-profit line through the best corner
        ax.plot(xs, (best[2] - profit[0] * xs) / profit[1], "k--", lw=1, label=f"profit = ${best[2]:,.0f}")
        ax.plot(best[0], best[1], "r*", ms=14)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("S: sourdough batches")
    ax.set_ylabel("W: whole-wheat batches")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    return pts, best


profit = (300, 200)
fig, ax = plt.subplots(figsize=(6, 6))
pts, best = draw(ax, BASE, profit, "Sourdough and whole-wheat: profit 300 S + 200 W")
fig.tight_layout()
fig.savefig("figures/day03_feasible_region.png", dpi=120)
print("Corner points and their profit:")
for s, w in pts:
    print(f"  S = {s:>5g}, W = {w:>5g}: ${profit[0] * s + profit[1] * w:>8,.0f}")
print(f"Best corner: S = {best[0]:g}, W = {best[1]:g}, profit ${best[2]:,.0f}")

res = linprog([-300, -200], A_ub=[[2, 1], [1, 1], [1, 0]], b_ub=[100, 80, 40], method="highs")
print(f"linprog agrees: S = {res.x[0]:g}, W = {res.x[1]:g}, profit ${-res.fun:,.0f}")
```

```text
Corner points and their profit:
  S =     0, W =     0: $       0
  S =     0, W =    80: $  16,000
  S =    20, W =    60: $  18,000
  S =    40, W =     0: $  12,000
  S =    40, W =    20: $  16,000
Best corner: S = 20, W = 60, profit $18,000
linprog agrees: S = 20, W = 60, profit $18,000
```

![Feasible region](figures/day03_feasible_region.png)

Read the picture:

- The green polygon holds every plan the bakery can bake. It has 5 corners.
- The dashed line is every plan earning $18,000. Lines for other profits are parallel to it: lower profits sit closer to the origin, higher ones further out.
- Slide the dashed line outward as far as it goes while still touching the region. It last touches at **(20, 60)**, where the oven and baker lines cross.

That is why the best plan of a linear program is always at a corner (or along an edge between two equally good corners): the iso-profit line leaves the region last at a corner. The **simplex method**, the classic LP algorithm, walks from corner to corner along the edges, improving profit at each step, and stops when no neighbouring corner is better.

> **Decision it supports:** seeing, not just being told, why the best plan uses the oven and bakers fully and leaves starter spare.
>
> **Other decisions you can support the same way:** explaining a two-product trade-off to a manager with one chart · checking a solver's answer on a small version of a big model.

### Math corner (optional): why a corner?

Profit is linear, so along any straight segment between two plans it changes at a constant rate. Take any plan inside the region and move in the direction that raises profit: profit keeps rising until you hit a constraint line. Then move along that line in the direction that raises profit, until you hit another. In two dimensions you are now at a corner. If profit is exactly flat along the line you are on, every point of that edge is equally good, including both its end corners. So some corner is always among the best plans, provided the region is not empty and profit cannot grow forever. Those two exceptions are 3.4.

## 3.3 Adding the demand caps; how far can a price move?

Add the Day 1 demand caps, $$S \le 35$$ and $$W \le 45$$:

```python
DEMAND = dict(BASE)
DEMAND["demand: S <= 35"] = (1, 0, 35)
DEMAND["demand: W <= 45"] = (0, 1, 45)
fig, ax = plt.subplots(figsize=(6, 6))
pts, best = draw(ax, DEMAND, profit, "With demand caps S <= 35, W <= 45")
fig.tight_layout()
fig.savefig("figures/day03_with_demand.png", dpi=120)
print(f"With demand caps the best corner is S = {best[0]:g}, W = {best[1]:g}, profit ${best[2]:,.0f}")
```

```text
With demand caps the best corner is S = 27.5, W = 45, profit $17,250
```

![With demand caps](figures/day03_with_demand.png)

The whole-wheat cap cuts off the old corner (20, 60). The new best corner is where the oven line meets $$W = 45$$: $$2S + 45 = 100$$, so $$S = 27.5$$. The bakers now have 7.5 hours spare. Note the **half batch**: linear programs happily return fractions. Rounding down to 27 costs $150; Module 3 handles whole batches properly.

### How far can a price move before the plan changes?

Go back to the model without demand caps. The best corner (20, 60) stays best as long as the dashed iso-profit line is steeper than the baker line and flatter than the oven line. The baker line has slope ratio 1 : 1 and the oven line 2 : 1, so (20, 60) stays best while sourdough profit ÷ whole-wheat profit is between 1 and 2. With whole-wheat at $200, that is sourdough between **$200 and $400**.

```python
for p_s in (150, 199, 201, 300, 399, 401, 450):
    r = linprog([-p_s, -200], A_ub=[[2, 1], [1, 1], [1, 0]], b_ub=[100, 80, 40], method="highs")
    print(f"  ${p_s}: S = {r.x[0]:g}, W = {r.x[1]:g}, profit ${-r.fun:,.0f}")
```

```text
  $150: S = 0, W = 80, profit $16,000
  $199: S = 0, W = 80, profit $16,000
  $201: S = 20, W = 60, profit $16,020
  $300: S = 20, W = 60, profit $18,000
  $399: S = 20, W = 60, profit $19,980
  $401: S = 40, W = 20, profit $20,040
  $450: S = 40, W = 20, profit $22,000
```

Between $201 and $399 the **plan** does not change, only its profit. Below $200 the bakery should stop sourdough altogether; above $400 it should bake sourdough up to the starter limit. Planners love this: **a price can move a long way before the plan needs to change.** Module 2 gets these ranges straight from the solver (Day 12).

### Try it 3.1

Keep sourdough at $300. For what range of whole-wheat profit per batch does (20, 60) stay the best plan? Check three values with `linprog`.

<details><summary>Solution</summary>

We need 1 ≤ 300 ÷ $$p_W$$ ≤ 2, so $$p_W$$ between **$150 and $300**.

```python
for p_w in (149, 151, 200, 299, 301):
    r = linprog([-300, -p_w], A_ub=[[2, 1], [1, 1], [1, 0]], b_ub=[100, 80, 40], method="highs")
    print(p_w, r.x, -r.fun)
```

```text
149 [40. 20.] 14980.0
151 [20. 60.] 15060.0
200 [20. 60.] 18000.0
299 [20. 60.] 23940.0
301 [ 0. 80.] 24080.0
```

</details>

## 3.4 Four special cases

A model can fail to have one clean best corner in four ways. Each has a picture and a solver message you must recognise.

```python
cases = {
    "Infeasible: contract needs W >= 85": ({**BASE, "contract: W >= 85": (0, -1, -85)}, (300, 200)),
    "Unbounded: oven and bakers forgotten": ({k: v for k, v in BASE.items() if k.startswith(("starter", "S >=", "W >="))}, (300, 200)),
    "Many optima: sourdough profit $400": (BASE, (400, 200)),
    "Degenerate: three lines meet at one corner": ({**BASE, "demand: W <= 60": (0, 1, 60)}, (300, 200)),
}
fig, axes = plt.subplots(2, 2, figsize=(11, 11))
for ax, (title, (cons, prof)) in zip(axes.ravel(), cases.items()):
    draw(ax, cons, prof, title, unbounded=title.startswith("Unbounded"))
    A = [[a, b] for a, b, c in cons.values()]
    b = [c for a, b, c in cons.values()]
    r = linprog([-prof[0], -prof[1]], A_ub=A, b_ub=b, bounds=[(None, None)] * 2, method="highs")
    answer = f"S = {r.x[0]:g}, W = {r.x[1]:g}, profit ${-r.fun:,.0f}" if r.status == 0 else ""
    print(f"{title}\n  linprog status {r.status}: {r.message}  {answer}")
fig.tight_layout()
fig.savefig("figures/day03_special_cases.png", dpi=110)
```

```text
Infeasible: contract needs W >= 85
  linprog status 2: The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)  
Unbounded: oven and bakers forgotten
  linprog status 3: The problem is unbounded. (HiGHS Status 10: model_status is Unbounded; primal_status is Feasible)  
Many optima: sourdough profit $400
  linprog status 0: Optimization terminated successfully. (HiGHS Status 7: Optimal)  S = 40, W = 20, profit $20,000
Degenerate: three lines meet at one corner
  linprog status 0: Optimization terminated successfully. (HiGHS Status 7: Optimal)  S = 20, W = 60, profit $18,000
```

![Special cases](figures/day03_special_cases.png)

The non-negativity rules are in the constraint list here, so we pass `bounds=[(None, None)] * 2` to stop `linprog` adding its own.

| Case | What you see | What it means in the business | What to do |
|---|---|---|---|
| **Infeasible** | No green region: no plan satisfies every rule | A sales contract promised 85 whole-wheat batches; the bakers can make at most 80 | Find the conflicting rules (Day 7); relax one (overtime) or renegotiate |
| **Unbounded** | The region runs off the chart; profit grows forever | Almost always a **missing constraint**, here the oven and bakers | Look for the forgotten limit; real businesses are never unbounded |
| **Many optima** | The iso-profit line lies along an edge | At $400, sourdough and whole-wheat earn the same per oven hour, so every plan from (20, 60) to (40, 20) earns $20,000 | The solver returns one; pick among them on a second goal (Module 7) |
| **Degenerate** | Three lines pass through the best corner | Whole-wheat cap of 60 is exactly met by the plan that fills oven and bakers | Harmless for the answer, but shadow prices become ambiguous (Day 11) |

> **Decision it supports:** telling a manager *why* there is no plan, or several, instead of reporting a solver error.
>
> **Other decisions you can support the same way:** spotting an impossible promise in a service-level agreement · finding a missing rule in a model a colleague wrote.

### Try it 3.2

Yesterday's three-product answer baked 10 croissants. Fix croissants at 10: they use 15 oven hours and 15 baker hours, leaving $$2S + W \le 85$$ and $$S + W \le 65$$. Add the demand caps $$S \le 35$$, $$W \le 45$$. Draw it with `draw` and solve with `linprog`. What special case do you see?

<details><summary>Solution</summary>

```python
FIXED = {
    "oven: 2S + W <= 85": (2, 1, 85),
    "bakers: S + W <= 65": (1, 1, 65),
    "demand: S <= 35": (1, 0, 35),
    "demand: W <= 45": (0, 1, 45),
    "S >= 0": (-1, 0, 0),
    "W >= 0": (0, -1, 0),
}
fig, ax = plt.subplots(figsize=(6, 6))
draw(ax, FIXED, (300, 200), "Croissants fixed at 10")
fig.savefig("figures/day03_try_fixed_croissants.png", dpi=120)
r = linprog([-300, -200], A_ub=[[2, 1], [1, 1]], b_ub=[85, 65], bounds=[(0, 35), (0, 45)], method="highs")
print(r.x, -r.fun)
```

```text
[20. 45.] 15000.0
```

The best corner (20, 45) lies on three lines at once: oven, bakers and the whole-wheat cap. In two dimensions a corner needs only two, so it is **degenerate**. Add the croissants' $2,500 and you get yesterday's $17,500.

Yesterday's three-product corner had the same three tight constraints, but with three variables a corner *needs* three, so it was not degenerate. Fixing croissants removed a variable without removing a constraint.

</details>

## Mini-project: a December week with overtime

In December, demand for whole-wheat rises to 60 batches (sourdough stays capped at 35, croissants are still down for maintenance). The bakery can run the ovens 20 extra hours, for 120 in total, at a cost of $60 per extra oven hour. Bakers stay at 80 hours, starter at 40.

1. Draw the region with 100 oven hours and with 120, and find the best corner of each.
2. Is the overtime worth it? How many hours should the bakery book?
3. Which constraints bind with overtime? What would you look at next?

<details><summary>Solution</summary>

```python
for oven_h in (100, 120):
    r = linprog([-300, -200], A_ub=[[2, 1], [1, 1], [1, 0]], b_ub=[oven_h, 80, 40],
                bounds=[(0, 35), (0, 60)], method="highs")
    print(oven_h, r.x, -r.fun)
```

```text
100 [20. 60.] 18000.0
120 [35. 45.] 19500.0
```

**1.** With 100 oven hours the best corner is (20, 60), $18,000: oven and bakers full, whole-wheat at its new cap of 60 (a degenerate corner, like the fourth special case). With 120 hours it moves to (35, 45), $19,500: sourdough at its cap, bakers full.

**2.** All 20 hours add $1,500 for 20 × $60 = $1,200: worth it, by $300 a week. But look closer. Each extra oven hour lets the bakery swap one whole-wheat batch for one sourdough batch, worth $100, **until sourdough reaches its cap of 35 at 115 hours**:

```python
for oven_h in (105, 114, 115, 116):
    r = linprog([-300, -200], A_ub=[[2, 1], [1, 1], [1, 0]], b_ub=[oven_h, 80, 40],
                bounds=[(0, 35), (0, 60)], method="highs")
    print(oven_h, r.x, -r.fun)
```

```text
105 [25. 55.] 18500.0
114 [34. 46.] 19400.0
115 [35. 45.] 19500.0
116 [35. 45.] 19500.0
```

The last 5 overtime hours earn nothing. **Book 15 hours, not 20:** +$1,500 − $900 = **+$600 a week**.
**3.** At (35, 45) the bakers and the sourdough cap bind, and oven hours beyond 115 go unused. The next things to price are an extra baker hour, or a promotion that lifts sourdough demand.

</details>

## Commit

```bash
git add .
git commit -m "Day 3: LP geometry, corners, price ranges, special cases"
git push
```

### Day 3 checklist

- [ ] I can draw the feasible region of a two-variable LP and list its corners.
- [ ] I can explain with the iso-profit line why the optimum is at a corner.
- [ ] I found the price range ($200 to $400 for sourdough) over which the best plan does not change.
- [ ] I can recognise infeasible, unbounded, many-optima and degenerate models from a picture and from the solver status.
- [ ] I know that an "Optimal" status means the model was solved, not that the model is right.
- [ ] I priced overtime with the model before recommending it.
