# DAY 44 — Correlated Risks and Disruptions

Day 43 drew each product's forecast error on its own. But in Christmas week every product sells more than forecast, and in the first week of January every product sells less. When risks **move together**, the average hardly changes but the bad tail gets much worse. Today you measure how the Northbound risks move together, simulate them that way, and add **rare disruptions**, events that almost never happen but cost a lot when they do.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 44.1 Do the errors move together? |
| 0:45 – 1:45 | 44.2 The production plan, errors apart or together |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 44.3 Mornings with one weather |
| 2:45 – 3:45 | 44.4 Rare disruptions over a year; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: an oven maintenance contract |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Days 44–47: `bakery_disruptions` (an oven breakdown, a packing line failure), `receiving_log` (four weeks of trucks arriving at DC-MTL at night) and `receiving_settings` (doors, drivers, detention charges). Install SimPy for Day 46: `pip install simpy==4.1.2` (it's in `requirements.txt`).

## 44.1 Do the errors move together?

Create `day44/correlated.py`:

```python
"""Day 44: risks that move together. Whole historical weeks as demand scenarios, mornings with one weather,
and rare disruptions.

Import it:  from correlated import error_table, demand_weeks, pooled_morning, disrupted_profit
"""
import numpy as np
import pyomo.environ as pyo

from bakery import build_model, solve
from inputs import weekly_errors


def error_table(con):
    """Forecast errors (actual / forecast), one row per week, one column per history product."""
    return weekly_errors(con).pivot(index="week_start", columns="sku", values="ratio")


def demand_weeks(con, rng, together=True):
    """A demand sampler. together=True: draw one historical week and use all its errors at once (bread from the
    mean of BK-SD and BK-WW, everything else from BK-CR), so products move together as they did.
    together=False: every product draws its own week, as on Day 43."""
    e = error_table(con)
    bread = ((e["BK-SD"] + e["BK-WW"]) / 2).to_numpy()
    other = e["BK-CR"].to_numpy()
    n = len(e)

    def week(cap, category):
        k = rng.integers(n)
        out = {}
        for p, c in cap.items():
            j = k if together else rng.integers(n)
            out[p] = c * (bread[j] if category[p] == "bread" else other[j])
        return out
    return week


def pooled_morning(trips, rng):
    """Travel times with the weather ignored: every trip draws from all trips of all days, independently."""
    pool = trips.ratio.to_numpy()

    def morning():
        return "any", (lambda: pool[rng.integers(len(pool))])
    return morning


def disrupted_profit(d, resource, capacity_left):
    """The best profit in a week when a resource loses part of its capacity (the plan is re-made for that week)."""
    week = {**d, "capacity": {**d["capacity"], resource: d["capacity"][resource] * capacity_left}}
    m = build_model(week)
    solve(m)
    return pyo.value(m.total_profit)


def simulate_years(weekly_profits, events, loss, rng, years=2_000, chance=None):
    """Years of 52 weeks: weekly profits (from the demand simulation) minus each disruption's loss in the weeks
    it strikes. chance overrides an event's weekly chance, e.g. {"oven_breakdown": 0.015}.
    Returns (annual profit per year, number of disruptions per year)."""
    profits = np.asarray(weekly_profits)[: years * 52].reshape(years, 52).copy()
    count = np.zeros(years, dtype=int)
    for r in events.itertuples():
        p = (chance or {}).get(r.event, r.chance_per_week)
        hit = rng.random((years, 52)) < p
        profits -= hit * loss[r.event]
        count += hit.sum(axis=1)
    return profits.sum(axis=1), count
```

`demand_weeks(together=True)` draws **one historical week** and applies all its errors at once: the products keep the relationships they really had. Then `day44/run_correlated.py`:

```python
# ---------- 44.1 Do the products' errors move together? ----------
e = error_table(con)
print("Pearson correlation of forecast errors:")
print(e.corr().round(2).to_string())
print("Spearman (rank) correlation:")
print(e.corr(method="spearman").round(2).to_string())
worst = e.assign(spread=(e - 1).abs().max(axis=1)).sort_values("spread", ascending=False).head(5)
print("the five weeks with the biggest errors:")
print(worst.drop(columns="spread").round(3).to_string())
```

```text
Pearson correlation of forecast errors:
sku    BK-CR  BK-SD  BK-WW
sku
BK-CR   1.00   0.62   0.69
BK-SD   0.62   1.00   0.62
BK-WW   0.69   0.62   1.00
Spearman (rank) correlation:
sku    BK-CR  BK-SD  BK-WW
sku
BK-CR   1.00   0.17   0.27
BK-SD   0.17   1.00   0.21
BK-WW   0.27   0.21   1.00
the five weeks with the biggest errors:
sku         BK-CR  BK-SD  BK-WW
week_start
2024-12-16  1.492  1.127  1.140
2025-12-15  1.395  1.189  1.126
2025-01-06  0.624  0.853  0.857
2026-01-05  0.658  0.892  0.870
2025-09-01  1.125  1.205  1.143
                            errors      mean      sd        p5  mean_of_worst_5pct
```

Pearson correlations of 0.6 to 0.7, but **rank correlations of only 0.2 to 0.3**. The table of extreme weeks explains it: in an ordinary week the errors are nearly independent, but at Christmas they all jump up together and in early January they all fall together. The dependence lives in the **tails**. A single correlation number hides this; resampling whole weeks keeps it.

## 44.2 The production plan, errors apart or together

```python
# ---------- 44.2 The production plan, errors apart or together ----------
d = load_data(con)
m = build_model(d)
solve(m)
make = {p: pyo.value(m.x[p]) for p in m.P if pyo.value(m.x[p]) > 1e-6}
econ = bakery_economics(con)
rows = []
for name, together in [("each product its own week (Day 43)", False), ("whole weeks: products together", True)]:
    sims = simulate_bakery(make, d["cap"], d["category"], econ, demand_weeks(con, np.random.default_rng(44), together), 10_000)
    r = risk_summary(sims.profit)
    rows.append((name, round(r["mean"], 2), round(r["sd"], 2), round(r["p5"], 2), round(r["mean_of_worst_5pct"], 2)))
print(pd.DataFrame(rows, columns=["errors", "mean", "sd", "p5", "mean_of_worst_5pct"]).to_string(index=False))
```

```text
each product its own week (Day 43) 106024.21 1511.51 103020.32           101747.92
    whole weeks: products together 106020.37 5914.40 103010.74            86389.85
```

**Same average, four times the spread, and a much worse bad tail**: the worst 5% of weeks average $86,390 instead of $101,748. With independent errors, a product that sells too little is offset by one that sells too much. In a holiday week nothing offsets anything. Ignoring correlation makes a plan look safer than it is.

## 44.3 Mornings with one weather

```python
# ---------- 44.3 Mornings: one weather for every trip ----------
stops, ids, km, minutes = load_run(con)
plan = solve_vrptw(stops, ids, km, minutes, load_fleet(con))
trips, _ = clean_trips(load_trips(con))
rows = []
for name, sampler in [("trips independent, weather ignored", pooled_morning(trips, np.random.default_rng(1))),
                      ("one weather a morning", travel_sampler(trips, np.random.default_rng(1)))]:
    sims = simulate_routes(plan, stops, ids, minutes, sampler, 5_000)
    rows.append((name, round(sims.late.mean(), 3), round((sims.late >= 3).mean(), 4), round((sims.late >= 5).mean(), 4),
                 int(sims.late.max())))
print(pd.DataFrame(rows, columns=["travel times", "late_per_morning", "P(3+ late)", "P(5+ late)", "worst_morning"])
      .to_string(index=False))
```

```text
                      travel times  late_per_morning  P(3+ late)  P(5+ late)  worst_morning
trips independent, weather ignored             0.934      0.0398      0.0076              6
             one weather a morning             0.919      0.0514      0.0084              7
```

Here the effect is smaller: drawing one weather for the whole morning raises the chance of 3 or more late deliveries from 4.0% to 5.1%. Rain and snow slow every trip at once, so bad mornings come in clusters, but the trip-to-trip variation still dominates. Measure correlation; don't assume it's large or small.

## 44.4 Rare disruptions over a year

Two events from `bakery_disruptions`: an oven breakdown (3% chance a week; 30% of oven hours lost that week) and a packing line failure (2%; 25% of packing hours). When one strikes, the bakery re-plans that week (Day 6's model with less capacity).

```python
# ---------- 44.4 Rare disruptions over a year ----------
events = con.sql("SELECT * FROM bakery_disruptions").df()
normal = pyo.value(m.total_profit)
loss = {}
for r in events.itertuples():
    loss[r.event] = normal - disrupted_profit(d, r.resource, r.capacity_left)
    print(f"{r.event}: {r.chance_per_week:.0%} a week; that week the best plan earns ${loss[r.event]:,.2f} less")
weekly = simulate_bakery(make, d["cap"], d["category"], econ, demand_weeks(con, np.random.default_rng(4), True),
                         52 * 2_000).profit
annual, count = simulate_years(weekly, events, loss, np.random.default_rng(5))
print(f"a year: average ${annual.mean():,.0f}; 5th percentile ${np.quantile(annual, 0.05):,.0f}; "
      f"years with no disruption {np.mean(count == 0):.0%}; with 3 or more {np.mean(count >= 3):.1%}")
print(f"disruptions cost ${sum(52 * r.chance_per_week * loss[r.event] for r in events.itertuples()):,.0f} a year on average")
```

```text
oven_breakdown: 3% a week; that week the best plan earns $25,230.35 less
packing_line_failure: 2% a week; that week the best plan earns $15,797.68 less
a year: average $5,454,979; 5th percentile $5,354,489; years with no disruption 8%; with 3 or more 48.3%
disruptions cost $55,789 a year on average
```

- An oven breakdown costs about **$25,000 in the week** it happens; a packing failure about $16,000.
- Each is rare, but over a year they're likely: 9 years in 10 have at least one, and almost half have three or more.
- On average they cost about $56,000 a year: **simulate a year, not a week**, when rare events matter.

> **Decision it supports:** how much risk a plan really carries when bad things happen together, and what rare events cost.
>
> **Other decisions you can support the same way:** portfolio risk (assets falling together) · supplier risk (one region, one storm) · insurance and spare-parts decisions.

### The planted bug

`day44/bug_sales_correlation.py` measures how the products move together by correlating their weekly sales. Run it.

```python
sales = con.sql("""SELECT week_start, sku, sum(units_sold) AS units FROM bakery_sales_weekly
                   GROUP BY week_start, sku""").df().pivot(index="week_start", columns="sku", values="units")
print(sales.corr().round(2).to_string())
print("so demand errors are about 0.9 correlated: use that in the simulation")
```

<details><summary>What happens, and why</summary>

```text
sku    BK-CR  BK-SD  BK-WW
sku
BK-CR   1.00   0.84   0.86
BK-SD   0.84   1.00   0.95
BK-WW   0.86   0.95   1.00
so demand errors are about 0.9 correlated: use that in the simulation
```

Correlations of 0.84 to 0.95, because all three products follow the same **season** (more in December, less in summer). But the forecast already knows the season. The simulation needs the correlation of the **errors**, what the forecast gets wrong, which is much weaker except at holidays. Correlating the wrong quantity would wildly overstate the risk. Always simulate what is uncertain, not what is predictable.

</details>

## Mini-project: an oven maintenance contract

A maintenance contract costs $16,000 a year and halves the chance of an oven breakdown (3% to 1.5% a week). Simulate 2,000 years with and without it, on the same simulated weeks. Is it worth it on average? In a bad year?

<details><summary>Solution</summary>

```python
"""Day 44 mini-project: a maintenance contract for the ovens costs $16,000 a year and halves the chance of a
breakdown (3% to 1.5% a week). Is it worth it, on average and in a bad year?

Run from the repository root (after day06/load_tables.py):  python day44/mini_project_maintenance.py   (about 1 minute)
"""
import sys

import duckdb
import numpy as np
import pandas as pd
import pyomo.environ as pyo

for folder in ("day06", "day41", "day42", "day43", "day44"):
    sys.path.insert(0, folder)
from bakery import build_model, load_data, solve
from correlated import demand_weeks, disrupted_profit, simulate_years
from montecarlo import bakery_economics
from plan_risk import simulate_bakery

con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_data(con)
m = build_model(d)
solve(m)
make = {p: pyo.value(m.x[p]) for p in m.P if pyo.value(m.x[p]) > 1e-6}
events = con.sql("SELECT * FROM bakery_disruptions").df()
loss = {r.event: pyo.value(m.total_profit) - disrupted_profit(d, r.resource, r.capacity_left) for r in events.itertuples()}
weekly = simulate_bakery(make, d["cap"], d["category"], bakery_economics(con),
                         demand_weeks(con, np.random.default_rng(4), True), 52 * 2_000).profit
rows = []
for name, chance, fee in [("no contract", None, 0), ("contract", {"oven_breakdown": 0.015}, 16_000)]:
    annual, count = simulate_years(weekly, events, loss, np.random.default_rng(5), chance=chance)   # same years
    annual = annual - fee
    rows.append((name, round(annual.mean()), round(np.quantile(annual, 0.05)), round(np.quantile(annual, 0.01)),
                 round(count.mean(), 2)))
t = pd.DataFrame(rows, columns=["option", "average_year", "p5_year", "p1_year", "disruptions_a_year"])
print(t.to_string(index=False))
print(f"contract: {t.average_year[1] - t.average_year[0]:+,} on average, {t.p5_year[1] - t.p5_year[0]:+,} in a 1-in-20 "
      f"bad year, {t.p1_year[1] - t.p1_year[0]:+,} in a 1-in-100 year")
```

```text
     option  average_year  p5_year  p1_year  disruptions_a_year
no contract       5454979  5354489  5308464                2.62
   contract       5458899  5361683  5314526                1.83
contract: +3,920 on average, +7,194 in a 1-in-20 bad year, +6,062 in a 1-in-100 year
```

- On average the contract **pays for itself and a little more** (about +$3,900 a year): it avoids 0.78 breakdowns a year at about $25,000 each.
- In bad years it helps more (about +$7,000 in a 1-in-20 year): insurance is worth more than its average when it protects the tail.
- Both options use the same simulated weeks and the same random numbers for the events (`np.random.default_rng(5)`), so the difference is the contract, not luck.

*Memo:* "Sign the oven maintenance contract: it is slightly cheaper than the breakdowns it prevents on average ($3,900 a year), and it reduces the loss in a bad year."

</details>

## Commit

```bash
git add .
git commit -m "Day 44: correlated risks and disruptions"
git push
```

### Day 44 checklist

- [ ] I can measure dependence with Pearson and rank correlations, and look at the extremes.
- [ ] I can keep real dependence by resampling whole periods.
- [ ] I showed that correlation leaves the average and worsens the tail.
- [ ] I can simulate rare disruptions over a year, with re-planning when they strike.
- [ ] I correlate forecast errors, not sales.

---

# DAY 45 — Mini-project: How Often Do We Run Out?

Fresh croissants arrive at every store each morning; what isn't sold by closing is thrown away. Store managers complain about empty shelves, the finance team about waste. Today's mini-project answers with simulation, checked against formulas: how often does each store run out today, and how many croissants should each get?

| Time | Block |
|---|---|
| 0:00 – 0:30 | 45.1 The brief and the stores' demand |
| 0:30 – 1:30 | 45.2 Today's rule: send the average |
| 1:30 – 1:45 | Break |
| 1:45 – 2:45 | 45.3 A service target; 45.4 the most profitable quantity |
| 2:45 – 3:30 | 45.5 When every store has a bad day together |
| 3:30 – 3:45 | Break |
| 3:45 – 4:45 | 45.6 The memo |
| 4:45 – 5:00 | Commit with Git + checklist |

## 45.1 The brief and the stores' demand

- **Demand**: each store's croissant sales over the last 13 weeks, divided over 6 selling days with Day 40's day-of-week factors. Day 42 found a store's sales are **Poisson**, so daily demand is Poisson with that mean.
- **Today's rule**: each store gets its average, rounded.
- **Economics** (Day 41): a croissant sells for $7.00 and costs $2.00; unsold ones are thrown away.
- **Questions**: how often do stores run out? What's lost to empty shelves and to waste? What should each store receive?

Create `day45/stockouts.py`:

```python
"""Day 45 mini-project: how often do stores run out of croissants? Fresh croissants arrive every morning and
what isn't sold is thrown away.

Import it:  from stockouts import store_day_means, simulate_store_days, poisson_check
"""
import numpy as np
import pandas as pd
from scipy import stats


def store_day_means(con, weeks=13):
    """Each store's average croissants a day over the last `weeks` weeks (6 selling days), by day of week:
    the weekly average / 6 x the day's factor (Day 40's delivery_days)."""
    weekly = con.sql(f"""SELECT store_id, avg(units_sold) AS week_mean FROM bakery_sales_weekly
                         WHERE sku = 'BK-CR' AND week_start > (SELECT max(week_start) FROM bakery_sales_weekly)
                               - INTERVAL {weeks} WEEK
                         GROUP BY store_id ORDER BY store_id""").df().set_index("store_id").week_mean
    days = con.sql("SELECT day, rack_factor FROM delivery_days").df()
    factor = days.rack_factor / days.rack_factor.mean()
    rows = [(s, day, w / 6 * f) for s, w in weekly.items() for day, f in zip(days.day, factor)]
    return pd.DataFrame(rows, columns=["store_id", "day", "mean"])


def simulate_store_days(means, quantity, rng, weeks=52, common_sd=0.0):
    """Poisson demand for every store and day over `weeks` weeks. common_sd: a chain-wide factor for each date
    (weather, a holiday) that raises or lowers every store's demand together."""
    lam = np.tile(means["mean"].to_numpy(), (weeks, 1))                     # weeks x (stores x days)
    if common_sd > 0:
        n_days = means.day.nunique()
        day_index = pd.factorize(means.day)[0]
        shock = rng.lognormal(-common_sd ** 2 / 2, common_sd, (weeks, n_days))
        lam = lam * shock[:, day_index]
    demand = rng.poisson(lam)
    q = np.tile(np.asarray(quantity), (weeks, 1))
    sold = np.minimum(demand, q)
    return demand, q, sold


def poisson_check(means, quantity):
    """The same numbers without simulation: P(demand > q) and expected units short, from the Poisson formulas."""
    lam, q = means["mean"].to_numpy(), np.asarray(quantity)
    p_out = stats.poisson.sf(q, lam)
    k = np.arange(0, 200)[:, None]
    short = (np.maximum(k - q, 0) * stats.poisson.pmf(k, lam)).sum(axis=0)
    return p_out.mean(), short.sum()
```

Then `day45/run_stockouts.py`:

```python
# ---------- 45.1 The stores' daily demand ----------
means = store_day_means(con)
print(f"{means.store_id.nunique()} stores x {means.day.nunique()} days; croissants a store a day: "
      f"{means['mean'].min():.1f} to {means['mean'].max():.1f} (average {means['mean'].mean():.1f})")
print(means.pivot(index="store_id", columns="day", values="mean")[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]]
      .head(4).round(2).to_string())
```

```text
40 stores x 6 days; croissants a store a day: 2.8 to 10.3 (average 6.1)
day        Mon   Tue   Wed   Thu   Fri   Sat
store_id
S01       3.66  3.86  4.06  4.26  4.87  4.67
S02       3.41  3.60  3.79  3.98  4.55  4.36
S03       5.55  5.86  6.17  6.47  7.40  7.09
S04       6.42  6.78  7.14  7.50  8.57  8.21
```

## 45.2 Today's rule: send the average

```python
def evaluate(name, quantity, common_sd=0.0, seed=45):
    demand, q, sold = simulate_store_days(means, quantity, np.random.default_rng(seed), common_sd=common_sd)
    waste = q - sold
    out = demand > q
    per_day = out.reshape(out.shape[0], -1)
    return {"rule": name, "croissants_sent_a_week": int(np.asarray(quantity).sum()),
            "days_out": round(out.mean(), 3), "fill_rate": round(sold.sum() / demand.sum(), 4),
            "short_a_week": round((demand - sold).sum() / demand.shape[0], 1),
            "wasted_a_week": round(waste.sum() / demand.shape[0], 1),
            "profit_a_week": round((price * sold - cost * q).sum() / demand.shape[0], 2)}


# ---------- 45.2 Today's rule: send the average ----------
today = np.round(means["mean"]).astype(int)
rows = [evaluate("today: the average, rounded", today)]
p_out, short = poisson_check(means, today)
print(f"Poisson formulas for today's rule: out of stock on {p_out:.3f} of store-days; {short:.1f} croissants short a week "
      f"(simulation: see the table)")
```

```text
Poisson formulas for today's rule: out of stock on 0.392 of store-days; 232.1 croissants short a week (simulation: see the table)
```

The formula and the simulation agree: **stores run out on 39% of days**. Sending the average means demand exceeds it almost half the time. A check like this (a formula you trust against your simulation) is how you know the simulation is right before you use it for anything harder.

## 45.3 A service target, and 45.4 the most profitable quantity

```python
# ---------- 45.3 A service target, 45.4 the most profitable quantity ----------
ratio = (price - cost) / price                                  # Day 41: (lost margin) / (lost margin + waste)
for name, level in [("in stock 95% of days", 0.95), (f"most profitable (critical ratio {ratio:.0%})", ratio)]:
    rows.append(evaluate(name, stats.poisson.ppf(level, means["mean"]).astype(int)))
print(pd.DataFrame(rows).to_string(index=False))
```

```text
Poisson formulas for today's rule: out of stock on 0.392 of store-days; 232.1 croissants short a week (simulation: see the table)
                                rule  croissants_sent_a_week  days_out  fill_rate  short_a_week  wasted_a_week  profit_a_week
         today: the average, rounded                    1463     0.392     0.8409         232.8          232.1        5690.60
                in stock 95% of days                    2488     0.035     0.9901          14.5         1038.7        5168.88
most profitable (critical ratio 71%)                    1774     0.220     0.9206         116.2          426.4        5884.90
```

| Rule | Days out | Waste a week | Profit a week |
|---|---|---|---|
| Today: the average | 39% | 232 | $5,690.60 |
| In stock 95% of days | 3.5% | 1,039 | $5,168.88 |
| Most profitable (critical ratio 71%) | 22% | 426 | **$5,884.90** |

- A **95% in-stock target** almost ends empty shelves, but more than quadruples waste and earns $522 a week less than today.
- The **most profitable** quantity (Day 41's newsvendor rule: the 71st percentile of each store's demand) earns **$194 a week more** than today, with stores out on 22% of days.
- The decision is a trade-off between empty shelves and waste, and the right point depends on what an empty shelf really costs (a customer who doesn't come back costs more than one lost croissant).

## 45.5 When every store has a bad day together

A cold, rainy morning or a holiday raises or lowers demand at every store at once. Add a chain-wide factor for each date (a lognormal with a 10% spread):

```python
# ---------- 45.5 When every store has a bad day together ----------
best = stats.poisson.ppf(ratio, means["mean"]).astype(int)
rows = []
for name, sd in [("stores independent", 0.0), ("a chain-wide factor each day, sd 10%", 0.10)]:
    demand, q, sold = simulate_store_days(means, best, np.random.default_rng(7), common_sd=sd)
    out_stores = (demand > q).reshape(demand.shape[0], 40, 6).sum(axis=1)          # stores out, per date
    waste_day = (q - sold).reshape(demand.shape[0], 40, 6).sum(axis=1)
    rows.append((name, round(out_stores.mean(), 2), round((out_stores >= 20).mean(), 4), int(out_stores.max()),
                 round(np.quantile(waste_day, 0.95), 1)))
print(pd.DataFrame(rows, columns=["demand", "stores_out_a_day", "P(20+ stores out)", "most_stores_out",
                                  "waste_on_a_bad_day_p95"]).to_string(index=False))
```

```text
                              demand  stores_out_a_day  P(20+ stores out)  most_stores_out  waste_on_a_bad_day_p95
                  stores independent              9.03             0.0000               17                    91.4
a chain-wide factor each day, sd 10%              8.54             0.0128               24                   104.0
```

On average the same (about 9 stores out a day), but with the shared factor, **20 or more stores run out on 1.3% of days** (never with independent stores), and waste on a bad day is higher. Head office sees days like that as "a croissant shortage", though each store's rule is the same: correlation again.

## 45.6 The memo

<details><summary>A model memo</summary>

*To: Store Operations. Re: croissant deliveries.*

"Today each store receives its average daily croissant sales. Because daily sales vary, **stores run out on 39% of days**, and about 230 croissants a week are thrown away across the chain.

We recommend sending each store the quantity it sells out of only 29% of the time (the **71st percentile** of its daily demand), about 1,770 croissants a week instead of 1,460. Stores would run out on 22% of days instead of 39%, waste would rise to about 430 a week, and **profit would rise by about $190 a week** ($10,000 a year).

A 95% in-stock target is possible but costs about $520 a week against today: only worth it if an empty shelf loses customers for good. On a few days a year (holidays, storms) many stores will run out together; a small reserve at the bakery would help on those days."

</details>

## Commit

```bash
git add .
git commit -m "Day 45: stock-out mini-project"
git push
```

### Day 45 checklist

- [ ] I modelled daily store demand as Poisson from history.
- [ ] I checked a simulation against an exact formula before trusting it.
- [ ] I compared rules by days out, fill rate, waste and profit.
- [ ] I can find the most profitable quantity with the critical ratio.
- [ ] I showed how a shared factor makes many stores run out together.

---

# DAY 46 — Discrete-Event Simulation with SimPy

Monte Carlo draws numbers and adds them up. Some systems are built of **events in time**: a truck arrives, waits for a door, waits for a forklift driver, unloads, leaves, and every truck's wait depends on the trucks before it. **Discrete-event simulation** follows such a system event by event. Today you build one with **SimPy** for the night receiving dock at DC-MTL, where carriers charge **detention** when their trucks wait more than an hour.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 46.1 What the log says |
| 0:45 – 1:45 | 46.2 SimPy: processes, resources, events |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 46.3 Five hundred nights |
| 2:45 – 3:45 | Reading the results; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: a second night driver? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 46.1 What the log says

`receiving_log`: 306 trucks over 24 nights (22:00 to 06:00), with the arrival time and how long each took to unload. `receiving_settings`: 3 doors, **1 night forklift driver** (Day 32's coverage), detention of $85 an hour after the first hour. Create `day46/dock.py`:

```python
"""Day 46: the night receiving dock at DC-MTL as a discrete-event simulation with SimPy.
A truck needs a free door and a free forklift driver to unload.

Import it:  from dock import load_receiving, hourly_rates, simulate_night, simulate_nights
"""
import numpy as np
import pandas as pd
import simpy

START = 22 * 60          # 22:00, in minutes after midnight


def load_receiving(con):
    log = con.sql("SELECT * FROM receiving_log ORDER BY night_of, arrival_minute").df()
    log["minute_of_night"] = (log.arrival_minute - START) % 1440       # minutes after 22:00
    settings = dict(con.sql("SELECT * FROM receiving_settings").fetchall())
    return log, settings


def hourly_rates(log):
    """Trucks an hour, by hour of the night, averaged over the nights in the log."""
    nights = log.night_of.nunique()
    hour = (log.minute_of_night // 60).astype(int)
    return (hour.value_counts().sort_index() / nights).reindex(range(8), fill_value=0.0).to_numpy()


def simulate_night(rates, unload, doors, drivers, rng, trace=None):
    """One night. rates: trucks an hour for each hour from 22:00; unload(): one unloading time in minutes.
    Returns one row per truck: arrival, start of unloading, end (minutes after 22:00)."""
    env = simpy.Environment()
    dock_doors = simpy.Resource(env, capacity=doors)
    forklifts = simpy.Resource(env, capacity=drivers)
    trucks = []

    def truck(name, minutes):
        arrived = env.now
        with dock_doors.request() as door:
            yield door                                        # wait for a door
            with forklifts.request() as driver:
                yield driver                                  # then for a driver
                start = env.now
                if trace is not None:
                    trace.append(f"{env.now:6.1f}  {name} starts unloading ({minutes:.0f} min)")
                yield env.timeout(minutes)
        trucks.append((name, arrived, start, env.now))

    def arrivals():
        k = 0
        for hour, rate in enumerate(rates):                   # a Poisson process, rate changing each hour
            end = 60 * (hour + 1)
            while rate > 0:
                gap = rng.exponential(60 / rate)
                if env.now + gap >= end:
                    yield env.timeout(end - env.now)
                    break
                yield env.timeout(gap)
                k += 1
                if trace is not None:
                    trace.append(f"{env.now:6.1f}  T{k} arrives")
                env.process(truck(f"T{k}", unload()))
            else:
                yield env.timeout(end - env.now)

    env.process(arrivals())
    env.run()                                                 # until every truck is unloaded
    return pd.DataFrame(trucks, columns=["truck", "arrived", "start", "end"])


def simulate_nights(rates, unload_sampler, doors, drivers, nights, seed, free=60, per_hour=85.0):
    """Many independent nights. Detention: carriers charge per_hour for waiting beyond `free` minutes."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(nights):
        t = simulate_night(rates, lambda: unload_sampler(rng), doors, drivers, rng)
        wait = t.start - t.arrived
        rows.append({"trucks": len(t), "mean_wait": wait.mean() if len(t) else 0.0, "max_wait": wait.max() if len(t) else 0.0,
                     "waited_over_free": int((wait > free).sum()),
                     "detention": float(np.maximum(wait - free, 0).sum() / 60 * per_hour),
                     "last_done": t.end.max() if len(t) else 0.0})
    return pd.DataFrame(rows)
```

Then `day46/run_dock.py`:

```python
# ---------- 46.1 What the log says ----------
per_night = log.groupby("night_of").size()
print(f"{len(log)} trucks over {log.night_of.nunique()} nights: {per_night.mean():.2f} a night "
      f"(variance {per_night.var():.2f}; Poisson would give variance = mean)")
rates = hourly_rates(log)
print("trucks an hour:", " ".join(f"{22 + h if 22 + h < 24 else h - 2:02d}h {r:.2f}" for h, r in enumerate(rates)))
u = log.unload_minutes.to_numpy()
shape, _, scale = stats.gamma.fit(u, floc=0)
print(f"unloading: mean {u.mean():.2f} min, sd {u.std(ddof=1):.2f} (coefficient of variation {u.std(ddof=1) / u.mean():.2f}); "
      f"gamma fit shape {shape:.2f}, scale {scale:.2f}")
print(f"settings: {settings}")
```

```text
306 trucks over 24 nights: 12.75 a night (variance 9.67; Poisson would give variance = mean)
trucks an hour: 22h 0.92 23h 1.75 00h 2.67 01h 2.92 02h 0.92 03h 1.12 04h 1.58 05h 0.88
unloading: mean 29.75 min, sd 15.03 (coefficient of variation 0.51); gamma fit shape 4.01, scale 7.42
settings: {'doors': 3, 'night_forklift_drivers': 1, 'free_minutes_before_detention': 60, 'detention_per_hour': 85, 'extra_driver_cost_per_night': 245}
```

- About 13 trucks a night, **Poisson-like** (the variance is close to the mean).
- But not evenly: **almost 3 an hour between midnight and 02:00**, under 1 an hour at either end of the night. Each truck takes about 30 minutes to unload (variable: a coefficient of variation of 0.5), so one driver can handle 2 an hour.

## 46.2 SimPy: processes, resources, events

Four ideas are all of SimPy:

- An `Environment` keeps the **simulation clock**; nothing happens between events.
- A **process** is a Python generator: `truck()` is one truck's life, and `arrivals()` creates trucks.
- `yield env.timeout(minutes)` lets time pass for that process (unloading, the gap to the next arrival).
- A `Resource` (doors, drivers) has a capacity; `with resource.request() as req: yield req` **waits in line** until one is free, and releases it at the end of the `with` block.

```python
# ---------- 46.2 SimPy in one night ----------
trace = []
night = simulate_night(rates, lambda: 30.0, int(settings["doors"]), int(settings["night_forklift_drivers"]),
                       np.random.default_rng(3), trace=trace)
print("the first events of one night (minutes after 22:00):")
for line in trace[:8]:
    print("   ", line)
print(night.head(5).round(1).to_string(index=False))
```

```text
the first events of one night (minutes after 22:00):
       7.2  T1 arrives
       7.2  T1 starts unloading (30 min)
      32.7  T2 arrives
      37.2  T2 starts unloading (30 min)
     127.7  T3 arrives
     127.7  T3 starts unloading (30 min)
     133.5  T4 arrives
     143.3  T5 arrives
truck  arrived  start   end
   T1      7.2    7.2  37.2
   T2     32.7   37.2  67.2
   T3    127.7  127.7 157.7
   T4    133.5  157.7 187.7
   T5    143.3  187.7 217.7
```

Read the trace: T2 arrives at 32.7 but starts at 37.2, when T1 finishes; T4 and T5 queue behind T3. The table is the same night, one row per truck.

## 46.3 Five hundred nights

```python
# ---------- 46.3 Five hundred nights ----------
unload = lambda rng: u[rng.integers(len(u))]                  # resample the logged unloading times
nights = simulate_nights(rates, unload, int(settings["doors"]), int(settings["night_forklift_drivers"]), 500, seed=46,
                         free=settings["free_minutes_before_detention"], per_hour=settings["detention_per_hour"])
print(nights.describe().loc[["mean", "50%", "max"]].round(2).to_string())
print(f"nights with more than $800 of detention: {(nights.detention > 800).mean():.1%}; "
      f"last truck done after 06:00 (480 min) on {(nights.last_done > 480).mean():.1%} of nights")
```

```text
      trucks  mean_wait  max_wait  waited_over_free  detention  last_done
mean   12.97      47.95    108.99              4.64     417.05     510.96
50%    13.00      36.20     92.27              3.00      83.85     500.56
max    24.00     225.84    400.84             21.00    4313.81     817.55
nights with more than $800 of detention: 17.2%; last truck done after 06:00 (480 min) on 65.8% of nights
```

With one driver, trucks wait **48 minutes on average**, and the dock pays **$417 of detention a night** on average (but $84 on a typical night: a few bad nights, with a peak-hour pile-up, cost thousands). On two nights in three the last truck isn't unloaded until after 06:00. Average figures are dominated by the bad nights: report the distribution.

> **Decision it supports:** how many doors, people or machines a process needs, when arrivals and work times vary.
>
> **Other decisions you can support the same way:** checkout lanes · hospital beds and nurses · call-centre agents · machines in a factory.

### The planted bug

`day46/bug_driver.py` is a colleague's version of the truck: "request a door and a driver, then unload". Run it.

```python
    def truck(minutes):
        arrived = env.now
        with doors.request() as door:
            yield door
            driver = drivers.request()                         # ask for the forklift driver
            waits.append(env.now - arrived)
            yield env.timeout(minutes)
            drivers.release(driver)
```

<details><summary>What happens, and why</summary>

```text
average wait 0.98 min; trucks waiting over an hour: 0.0%
```

An average wait of one minute, and no truck ever waits an hour: the dock looks fine. `drivers.request()` is created but never **yielded**, so the truck doesn't wait for the driver: it starts unloading at once, and up to three trucks "share" the one driver. In SimPy, a request only holds the process back when you `yield` it; use `with ... as req: yield req`, which also releases it. Check a simulation against a case you can work out by hand (Day 47 does it with formulas).

</details>

## Mini-project: a second night driver?

A second night forklift driver costs $245 a night. Would that pay for itself in detention? Would a fourth door help instead? Compare 3 doors with 1, 2 and 3 drivers, and 4 doors with 1 driver, on the same 500 nights.

<details><summary>Solution</summary>

```python
"""Day 46 mini-project: hire a second night forklift driver ($245 a night) to cut detention charges?
And would a fourth door help instead?

Run from the repository root (after day06/load_tables.py):  python day46/mini_project_driver.py
"""
import sys

import duckdb
import pandas as pd

sys.path.insert(0, "day46")
from dock import hourly_rates, load_receiving, simulate_nights

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
log, settings = load_receiving(con)
rates, u = hourly_rates(log), log.unload_minutes.to_numpy()
rows = []
for doors, drivers in [(3, 1), (4, 1), (2, 2), (3, 2), (3, 3)]:
    n = simulate_nights(rates, lambda rng: u[rng.integers(len(u))], doors, drivers, 500, seed=46,   # same 500 nights
                        free=settings["free_minutes_before_detention"], per_hour=settings["detention_per_hour"])
    extra = (drivers - 1) * settings["extra_driver_cost_per_night"]
    rows.append((doors, drivers, round(n.mean_wait.mean(), 2), round(n.max_wait.mean(), 1), round(n.waited_over_free.mean(), 2),
                 round(n.detention.mean(), 2), extra, round(n.detention.mean() + extra, 2)))
t = pd.DataFrame(rows, columns=["doors", "drivers", "mean_wait", "longest_wait", "trucks_over_an_hour", "detention",
                                "extra_drivers", "cost_a_night"])
print(t.to_string(index=False))
```

```text
 doors  drivers  mean_wait  longest_wait  trucks_over_an_hour  detention  extra_drivers  cost_a_night
     3        1      47.95         109.0                 4.64     417.05              0        417.05
     4        1      47.95         109.0                 4.64     417.05              0        417.05
     2        2       5.22          23.1                 0.18       3.09            245        248.09
     3        2       5.22          23.1                 0.18       3.09            245        248.09
     3        3       0.90           6.0                 0.00       0.00            490        490.00
```

- **A second driver pays**: detention falls from $417 to $3 a night, so the dock costs $248 a night instead of $417: **about $169 a night saved** ($53,000 a year over 312 nights), and trucks wait 5 minutes instead of 48.
- **A fourth door changes nothing**: the driver is the bottleneck, not the doors. Two doors would be enough with two drivers.
- A third driver isn't worth it: it saves the last $3 of detention for $245.

*Memo to the DC manager:* "Add a second forklift driver on the night shift ($245 a night). It cuts carriers' detention charges by about $414 a night and gets every truck unloaded by 06:00 on most nights. Don't build a fourth door. The roster change goes into the Day 32 coverage table (night forklift: 2)."

</details>

## Commit

```bash
git add .
git commit -m "Day 46: discrete-event simulation of the receiving dock"
git push
```

### Day 46 checklist

- [ ] I can write a SimPy model with processes, timeouts and resources.
- [ ] I can model a task that needs two resources at once.
- [ ] I can replicate a simulation over many nights and report the distribution.
- [ ] I `yield` every resource request (or use `with`).
- [ ] I found the bottleneck and priced the fix.

---

# DAY 47 — Queueing Theory as a Check

A hundred years of **queueing theory** gives formulas for waiting lines. They need simple assumptions (steady arrivals at a constant rate, a long run) that real systems rarely meet, but they are perfect **checks**: run the simulation under the formula's assumptions, and if they disagree, the simulation is wrong. Then use the simulation for the cases the formulas can't handle. Today you check Day 46's model and learn where the formulas stop working.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 47.1 M/M/1 and M/M/c |
| 0:45 – 1:45 | 47.2 Real unloading times; 47.3 Little's law |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 47.4 The real night: where formulas stop |
| 2:45 – 3:45 | Units; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: a promise to the carriers |
| 4:45 – 5:00 | Commit with Git + checklist |

## 47.1 M/M/1 and M/M/c

**Kendall's notation**: arrivals / service / servers. **M** means Markovian: Poisson arrivals or exponential service times; **G** means any distribution. With arrival rate $$\lambda$$, mean service time $$s$$, $$c$$ servers and utilization $$\rho = \lambda s / c < 1$$:

| Queue | Average wait before service $$W_q$$ |
|---|---|
| M/M/1 | $$\frac{\rho}{1-\rho}\, s$$ |
| M/M/c | $$C(c, \lambda s) \cdot \frac{s}{c(1-\rho)}$$, with $$C$$ Erlang's formula for the chance of waiting |
| M/G/1 (Pollaczek–Khinchine) | $$\frac{\rho}{1-\rho} \cdot \frac{1 + cv_s^2}{2}\, s$$ |
| G/G/c (Allen–Cunneen, approximate) | $$W_q(\text{M/M/c}) \cdot \frac{cv_a^2 + cv_s^2}{2}$$ |

The $$\frac{1}{1-\rho}$$ is Day 41's lesson as a formula: at 80% busy, waits are 4 times the service time; at 90%, 9 times; at 100%, infinite. Create `day47/queueing.py`:

```python
"""Day 47: queueing formulas, to check a simulation (and to size a resource in a minute).
Rates are per minute, times in minutes: keep one unit of time everywhere.

Import it:  from queueing import mm1_wait, erlang_c, mmc_wait, mmc_wait_over, pk_wait, allen_cunneen
"""
import math


def utilization(lam, mean_service, servers=1):
    rho = lam * mean_service / servers
    if rho >= 1:
        raise ValueError(f"utilization {rho:.2f} >= 1: the line grows without limit, no steady state")
    return rho


def mm1_wait(lam, mean_service):
    """M/M/1: Poisson arrivals, exponential service, one server. Average wait before service."""
    rho = utilization(lam, mean_service)
    return rho / (1 - rho) * mean_service


def erlang_c(lam, mean_service, servers):
    """M/M/c: the chance that an arriving customer has to wait (Erlang's C formula)."""
    a = lam * mean_service                                   # offered load, in servers
    rho = utilization(lam, mean_service, servers)
    top = a ** servers / math.factorial(servers) / (1 - rho)
    return top / (sum(a ** k / math.factorial(k) for k in range(servers)) + top)


def mmc_wait(lam, mean_service, servers):
    """M/M/c: average wait before service."""
    return erlang_c(lam, mean_service, servers) * mean_service / (servers * (1 - utilization(lam, mean_service, servers)))


def mmc_wait_over(lam, mean_service, servers, t):
    """M/M/c: the chance that a customer waits more than t."""
    mu = 1 / mean_service
    return erlang_c(lam, mean_service, servers) * math.exp(-(servers * mu - lam) * t)


def pk_wait(lam, mean_service, cv_service):
    """M/G/1 (Pollaczek-Khinchine): any service distribution, through its coefficient of variation."""
    rho = utilization(lam, mean_service)
    return rho / (1 - rho) * (1 + cv_service ** 2) / 2 * mean_service


def allen_cunneen(lam, mean_service, servers, cv_arrival, cv_service):
    """G/G/c approximation: the M/M/c wait scaled by the variability of arrivals and service."""
    return mmc_wait(lam, mean_service, servers) * (cv_arrival ** 2 + cv_service ** 2) / 2
```

Then `day47/run_queueing.py` runs Day 46's SimPy model with a **constant** arrival rate (the night's average) for 20,000 hours, five times:

```python
def steady(lam_per_hour, unload, drivers, hours=20_000, reps=5, seed=47):
    """Long runs at a constant rate: the steady state that the formulas describe. Near full utilization one long
    run is not enough (each wait depends on the one before), so: 5 independent runs, and a confidence interval."""
    waits = []
    for k in range(reps):
        t = simulate_night([lam_per_hour] * hours, unload, drivers, drivers, np.random.default_rng(seed + k))
        t = t[t.arrived > 600]                                  # drop the first 10 hours (warm-up)
        waits.append((t.start - t.arrived).mean())
    half = 1.96 * np.std(waits, ddof=1) / np.sqrt(reps)
    return np.mean(waits), half, t


# ---------- 47.1 Exponential unloading: M/M/1 and M/M/2 ----------
rng = np.random.default_rng(1)
rows = []
for c, formula in [(1, mm1_wait(lam, mean_s)), (2, mmc_wait(lam, mean_s, 2))]:
    w, half, _ = steady(lam_h, lambda: rng.exponential(mean_s), c)
    rows.append((f"M/M/{c}", round(formula, 2), round(w, 2), round(half, 2)))
# ---------- 47.2 Real unloading times: M/G/1 and G/G/2 ----------
for c, formula in [(1, pk_wait(lam, mean_s, cv_s)), (2, allen_cunneen(lam, mean_s, 2, 1.0, cv_s))]:
    w, half, t = steady(lam_h, lambda: u[rng.integers(len(u))], c)
    rows.append((f"M/G/{c}", round(formula, 2), round(w, 2), round(half, 2)))
print(pd.DataFrame(rows, columns=["model", "formula_wait", "simulated_wait", "plus_minus_95"]).to_string(index=False))
```

```text
average arrival rate 1.594 trucks an hour = 0.0266 a minute; unloading 29.75 min (cv 0.51)
model  formula_wait  simulated_wait  plus_minus_95
M/M/1        112.09          108.34           5.91
M/M/2          5.50            5.47           0.25
M/G/1         70.36           71.01           3.63
M/G/2          3.45            3.59           0.07
```

**The formulas and the simulation agree**, within the simulation's confidence interval (Allen–Cunneen, an approximation, within 4%). Day 46's model passes the check. Notice the intervals: at 79% busy with one driver, even 20,000 hours give ±6 minutes, because each truck's wait depends on the one before. Near full utilization, simulations need long runs and replications.

## 47.3 Little's law

$$L = \lambda W$$: the average number in a system equals the arrival rate times the average time spent there. It holds for any queue, any distribution, any discipline, in the long run. It's the simplest check on any simulation's bookkeeping.

```python
# ---------- 47.3 Little's law ----------
span = t.end.max() - t.arrived.min()
in_line = (t.start - t.arrived).sum() / span                    # time-average number of trucks waiting
print(f"Little's law, M/G/2 run: trucks waiting on average {in_line:.4f}; "
      f"arrival rate x average wait = {len(t) / span:.4f} x {(t.start - t.arrived).mean():.2f} = "
      f"{len(t) / span * (t.start - t.arrived).mean():.4f}")
```

```text
Little's law, M/G/2 run: trucks waiting on average 0.0989; arrival rate x average wait = 0.0266 x 3.71 = 0.0989
```

## 47.4 The real night: where formulas stop

```python
# ---------- 47.4 The real night: formulas against the simulation ----------
rates = hourly_rates(log)
night = simulate_nights(rates, lambda r: u[r.integers(len(u))], 3, 1, 500, seed=46)
print(f"Day 46 simulation, 1 driver, real nights: average wait {night.mean_wait.mean():.2f} min")
print(f"formula at the night's average rate ({lam_h:.2f}/h): {pk_wait(lam, mean_s, cv_s):.2f} min")
for name, peak in [("the busiest hour", rates.max())]:
    rho = peak * mean_s / 60
    print(f"formula at {name} ({peak:.2f}/h): utilization {rho:.2f}: "
          + ("no steady state, the line grows all hour" if rho >= 1 else f"{pk_wait(peak / 60, mean_s, cv_s):.2f} min"))
print(f"with 2 drivers: simulation {simulate_nights(rates, lambda r: u[r.integers(len(u))], 3, 2, 500, seed=46).mean_wait.mean():.2f} min, "
      f"formula (average rate) {allen_cunneen(lam, mean_s, 2, 1.0, cv_s):.2f} min, "
      f"formula (busiest hour) {allen_cunneen(rates.max() / 60, mean_s, 2, 1.0, cv_s):.2f} min")
```

```text
Day 46 simulation, 1 driver, real nights: average wait 47.95 min
formula at the night's average rate (1.59/h): 70.36 min
formula at the busiest hour (2.92/h): utilization 1.45: no steady state, the line grows all hour
with 2 drivers: simulation 5.22 min, formula (average rate) 3.45 min, formula (busiest hour) 20.47 min
```

- At the night's **average** rate, the formula says 70 minutes, more than the simulated 48: real nights start with an empty dock and last only 8 hours, so the line never reaches the steady state the formula describes.
- At the **busiest hour** (2.92 trucks an hour), one driver is 145% busy: the formula has no answer, because no steady state exists.
- With two drivers, the formula at the average rate says 3.5 minutes, at the busiest hour 20.5; the simulation's 5.2 lies between.

Formulas answer quickly and check simulations; **time-varying arrivals, short horizons and starting conditions** need the simulation. A useful habit: the formula at the average and at the peak rate bracket the truth.

> **Decision it supports:** quick sizing of servers, and checks that a simulation is built right.
>
> **Other decisions you can support the same way:** call-centre staffing (Erlang C is the industry's standard) · server capacity for a website · hospital emergency rooms.

### The planted bug

`day47/bug_units.py` checks the dock with the M/M/1 formula. Run it.

```python
log = con.sql("SELECT * FROM receiving_log").df()
lam = len(log) / log.night_of.nunique() / 8                    # trucks an hour
service = log.unload_minutes.mean()                            # minutes a truck
rho = lam * service
wait = rho / (1 - rho) * service
print(f"utilization {rho:.2f}; average wait {wait:.1f} minutes: the dock is fine")
```

<details><summary>What happens, and why</summary>

```text
utilization 47.42; average wait -30.4 minutes: the dock is fine
```

A utilization of 47 and a **negative** wait, reported as "fine". The arrival rate is per **hour**, the service time in **minutes**: their product is 60 times too big. Two checks would have caught it: put every rate and time in one unit (Day 47's `queueing.py` uses minutes throughout), and never use a queueing formula without checking $$\rho < 1$$ (`utilization` raises an error).

</details>

## Mini-project: a promise to the carriers

Carriers ask for a promise: "90% of trucks start unloading within 30 minutes". How many night drivers does that take? Answer first with the M/M/c formula, $$P(W_q > t) = C(c, \lambda s)\, e^{-(c/s - \lambda) t}$$, at the average and at the busiest hour; then check with Day 46's simulation of real nights.

<details><summary>Solution</summary>

```python
"""Day 47 mini-project: carriers ask for a promise: "90% of trucks start unloading within 30 minutes".
How many night drivers would that take? Answer in a minute with the M/M/c formula, then check with Day 46's simulation.

Run from the repository root (after day06/load_tables.py):  python day47/mini_project_promise.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd

for folder in ("day46", "day47"):
    sys.path.insert(0, folder)
from dock import hourly_rates, load_receiving, simulate_night
from queueing import mmc_wait_over

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
log, settings = load_receiving(con)
u = log.unload_minutes.to_numpy()
rates = hourly_rates(log)
avg, peak = len(log) / log.night_of.nunique() / 8 / 60, rates.max() / 60
rows = []
for c in (1, 2, 3):
    def formula(lam):
        try:
            return round(1 - mmc_wait_over(lam, u.mean(), c, 30), 3)
        except ValueError:
            return "overloaded"
    rng = np.random.default_rng(470)
    waits = np.concatenate([(lambda t: (t.start - t.arrived).to_numpy())(
        simulate_night(rates, lambda: u[rng.integers(len(u))], 3, c, rng)) for _ in range(500)])
    rows.append((c, formula(avg), formula(peak), round(float((waits <= 30).mean()), 3)))
print(pd.DataFrame(rows, columns=["drivers", "formula_average_rate", "formula_busiest_hour", "simulated_real_nights"])
      .to_string(index=False))
```

```text
 drivers  formula_average_rate formula_busiest_hour  simulated_real_nights
       1                 0.360           overloaded                  0.475
       2                 0.934                0.653                  0.927
       3                 0.995                0.955                  0.994
```

- One driver can't promise anything: half the trucks wait more than 30 minutes.
- **Two drivers**: the formula at the average rate says 93%, at the busiest hour 65%; the simulation of real nights says **92.7%**. The promise can be made, just.
- Three drivers would give 99%.

*Note to the DC manager:* "With two night drivers (Day 46) we can promise carriers that 90% of trucks start unloading within 30 minutes; we'd meet it at about 93%. The promise fails during the 00:00–02:00 peak on busy nights: offer appointment slots outside the peak to the carriers that arrive then."

</details>

## Commit

```bash
git add .
git commit -m "Day 47: queueing formulas as a check"
git push
```

Then run the tests: `tests/test_risk_and_queues.py` checks Days 44–47 (whole-week errors, disruption-free years, the Poisson formula against simulation, known queueing answers, one driver at a time in SimPy). Add `"day44"` to `"day47"` to the folders in `conftest.py`.

```python
"""Module 5, Days 44-47: correlated risks, stock-outs, the SimPy dock and queueing formulas."""
import numpy as np
import pandas as pd
import pytest
from correlated import demand_weeks, simulate_years
from dock import simulate_night
from queueing import erlang_c, mm1_wait, mmc_wait, utilization
from stockouts import poisson_check, simulate_store_days, store_day_means


def test_whole_weeks_move_products_together(con):
    week = demand_weeks(con, np.random.default_rng(0), together=True)
    cap = {"A": 10.0, "B": 20.0, "C": 5.0}
    d = week(cap, {"A": "bread", "B": "bread", "C": "cake"})
    assert d["A"] / 10 == pytest.approx(d["B"] / 20)


def test_years_without_disruptions_are_just_the_weeks():
    events = pd.DataFrame({"event": ["x"], "chance_per_week": [0.0]})
    annual, count = simulate_years(np.ones(52 * 3), events, {"x": 100.0}, np.random.default_rng(0), years=3)
    assert list(annual) == [52, 52, 52] and count.sum() == 0


def test_poisson_formula_matches_simulation(con):
    means = store_day_means(con)
    q = np.round(means["mean"]).astype(int)
    p_out, _ = poisson_check(means, q)
    demand, qq, sold = simulate_store_days(means, q, np.random.default_rng(1), weeks=400)
    assert (demand > qq).mean() == pytest.approx(p_out, abs=0.01)


def test_queueing_formulas():
    assert mm1_wait(0.5, 1.0) == pytest.approx(1.0)
    assert erlang_c(0.5, 1.0, 1) == pytest.approx(0.5)                   # one server: P(wait) = utilization
    assert mmc_wait(1.0, 1.0, 2) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        utilization(2.0, 1.0)


def test_simpy_night_respects_one_driver():
    t = simulate_night([2.0] * 4, lambda: 25.0, 3, 1, np.random.default_rng(2)).sort_values("start")
    assert (t.start >= t.arrived).all()
    assert (t.start.to_numpy()[1:] >= t.end.to_numpy()[:-1] - 1e-9).all()   # one truck unloads at a time
```

```text
70 passed
```

### Day 47 checklist

- [ ] I know M/M/1, M/M/c, M/G/1 and an approximation for G/G/c, and when they apply.
- [ ] I checked a simulation against formulas under the formulas' assumptions.
- [ ] I used Little's law to check a simulation's bookkeeping.
- [ ] I can explain why formulas fail for time-varying, short, empty-start systems.
- [ ] I keep one unit of time, and check utilization before using a formula.
