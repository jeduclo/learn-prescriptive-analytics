# DAY 41 — The Flaw of Averages

Every model in Modules 1–4 used single numbers: the forecast, the planned travel time, the average unloading time. Module 5 asks what happens when those numbers turn out differently, as they always do. Today's lesson has a name, **the flaw of averages** (Sam Savage): *plans based on average conditions are wrong on average*. You'll see it in three Northbound plans, with nothing more than NumPy and a loop.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 41.1 A production plan made for average demand |
| 0:45 – 1:45 | 41.2 Routes planned with average travel times |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 41.3 A dock that is busy exactly as often as it is free |
| 2:45 – 3:45 | Why it happens; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: how many croissants? |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Days 41–43: `trip_log`, six months of van trips with planned and actual minutes and the morning's weather (Day 42).

## 41.1 A production plan made for average demand

**Monte Carlo simulation**: replace an uncertain number with random draws, run the plan thousands of times, and look at the results. The tools are a random number generator (`np.random.default_rng(seed)`: the seed makes runs repeatable) and a loop. Create `day41/montecarlo.py`:

```python
"""Days 41-43: Monte Carlo helpers. A plan made with average numbers, played out against random ones.

Import it:  from montecarlo import bakery_economics, bakery_profit, drive_routes, wait_in_line
"""
import numpy as np


def bakery_economics(con):
    """Per product: units per batch, price and ingredient cost per unit."""
    df = con.sql("SELECT sku, units_per_batch, transfer_price, ingredient_cost FROM bakery_range ORDER BY sku").df()
    return df.set_index("sku")


def bakery_profit(make, demand, econ):
    """One week's profit for batches made (dict) against batches demanded (dict): sell what is both made and
    wanted; unsold bread is thrown away, so its ingredients are lost."""
    total = 0.0
    for p, x in make.items():
        e = econ.loc[p]
        sold = min(x, demand[p])
        total += e.units_per_batch * (e.transfer_price * sold - e.ingredient_cost * x)
    return total


def drive_routes(plan, stops, ids, minutes, trip_factor):
    """Drive the routes with random travel times: trip_factor() gives actual / planned minutes for one trip.
    Leave when the dock opens, wait if early, unload, drive on. Returns {stop: minutes late (0 if on time)}."""
    late = {}
    for r in plan:
        t, here = stops.window_open[ids[0]], 0
        for node, _ in r["stops"]:
            s = ids[node]
            t += minutes[here][node] * trip_factor()
            late[s] = max(0.0, t - stops.window_close[s])
            t = max(t, stops.window_open[s]) + stops.service_minutes[s]
            here = node
    return late


def wait_in_line(arrival_gaps, service_times):
    """Waiting time of each truck at a single dock, first come first served (Lindley's recursion)."""
    wait = np.zeros(len(service_times))
    for k in range(1, len(service_times)):
        wait[k] = max(0.0, wait[k - 1] + service_times[k - 1] - arrival_gaps[k])
    return wait
```

Then `day41/flaw.py`. Day 6's plan makes 22 products at the forecast. Real demand is the forecast times an error, here a normal one with a standard deviation of 9% (BK-CR's history; Day 42 does better).

```python
# ---------- 41.1 A production plan made for average demand ----------
d = load_data(con)
m = build_model(d)
solve(m)
make = {p: pyo.value(m.x[p]) for p in m.P if pyo.value(m.x[p]) > 1e-6}
econ = bakery_economics(con)
planned = bakery_profit(make, d["cap"], econ)                  # demand exactly as forecast
print(f"Day 6 plan: {len(make)} products, profit if demand is exactly the forecast: ${planned:,.2f}")
profits = []
for _ in range(RUNS):                                          # demand = forecast x a random error, sd 9% (BK-CR's)
    demand = {p: d["cap"][p] * rng.normal(1.0, 0.09) for p in make}
    profits.append(bakery_profit(make, demand, econ))
profits = np.array(profits)
print(f"same plan, demand off by a typical forecast error: average ${profits.mean():,.2f} "
      f"({profits.mean() - planned:+,.2f}); worst 5% of weeks below ${np.percentile(profits, 5):,.2f}")
```

```text
Day 6 plan: 22 products, profit if demand is exactly the forecast: $108,351.28
same plan, demand off by a typical forecast error: average $102,741.89 (-5,609.39); worst 5% of weeks below $99,170.72
```

The model promised $108,351.28 a week. **On average the same plan earns $102,741.89: $5,609 less.** It's not bad luck. When demand is above the forecast, the bakery can't sell bread it didn't make; when demand is below, it throws bread away. Both errors cost money; neither earns any. Profit at average demand is not average profit.

## 41.2 Routes planned with average travel times

```python
# ---------- 41.2 Routes planned with average travel times ----------
stops, ids, km, minutes = load_run(con)
fleet = load_fleet(con)
plan = solve_vrptw(stops, ids, km, minutes, fleet)
on_plan = drive_routes(plan, stops, ids, minutes, lambda: 1.0)
print(f"Day 37 routes at the planned travel times: {sum(v > 0 for v in on_plan.values())} late deliveries")
late_counts, any_late = [], 0
for _ in range(2_000):                                         # each trip 0.85x to 1.4x the plan, typically 1.09x
    late = drive_routes(plan, stops, ids, minutes, lambda: rng.lognormal(0.08, 0.12))
    n = sum(v > 0 for v in late.values())
    late_counts.append(n)
    any_late += n > 0
print(f"with real travel times: {np.mean(late_counts):.2f} late deliveries a morning on average; "
      f"at least one late on {any_late / 20:.1f}% of mornings")
```

```text
Day 37 routes at the planned travel times: 0 late deliveries
with real travel times: 0.61 late deliveries a morning on average; at least one late on 60.8% of mornings
```

On paper, Day 37's routes are never late. With travel times that vary (each trip 0.85 to 1.4 times the plan), **6 mornings out of 10 have at least one late delivery**. A plan that just meets a deadline at average speed misses it whenever traffic is slower than average, which is about half the time on each trip.

## 41.3 A dock that is busy exactly as often as it is free

Trucks arrive at the bakery's dock at random, and unloading takes 10 minutes on average, sometimes more, sometimes less. `wait_in_line` follows each truck through the line.

```python
# ---------- 41.3 A dock that is busy exactly as often as it is free ----------
print("bakery dock, one truck unloaded at a time, average unloading 10 minutes, 200 arrivals:")
for every in (20.0, 12.5, 11.1, 10.5, 10.0):
    waits = [wait_in_line(rng.exponential(every, 200), rng.exponential(10.0, 200)).mean() for _ in range(500)]
    print(f"    a truck every {every:4.1f} min on average (dock busy {10 / every:.0%}): average wait {np.mean(waits):5.1f} min")
```

```text
bakery dock, one truck unloaded at a time, average unloading 10 minutes, 200 arrivals:
    a truck every 20.0 min on average (dock busy 50%): average wait   9.8 min
    a truck every 12.5 min on average (dock busy 80%): average wait  34.2 min
    a truck every 11.1 min on average (dock busy 90%): average wait  61.1 min
    a truck every 10.5 min on average (dock busy 95%): average wait  75.4 min
    a truck every 10.0 min on average (dock busy 100%): average wait  93.5 min
```

"A truck every 10 minutes, 10 minutes each: the dock is exactly full" sounds fine with averages. In reality the line never clears: over 200 arrivals trucks wait **93 minutes** on average, and the wait would keep growing on a longer day. Even at 80% busy, the wait is 34 minutes. Waiting grows much faster than the load (Day 47 gives the formula).

> **Decision it supports:** whether a plan built on averages will actually work, and what it really earns.
>
> **Other decisions you can support the same way:** project deadlines (the critical path at average task times is too short) · budgets built from average prices · staffing from average call volumes.

### Why it happens (math corner, optional)

For a function $$f$$ and a random input $$X$$, the average result $$E[f(X)]$$ equals the result at the average input $$f(E[X])$$ only when $$f$$ is a straight line. Sales are $$\min(\text{made}, \text{demand})$$: a bend, a **concave** function, so $$E[\min(x, D)] \le \min(x, E[D])$$ (**Jensen's inequality**). Lateness is $$\max(0, \text{arrival} - \text{deadline})$$, a **convex** function, so average lateness is more than the lateness at average times. Every plan has bends like these: capacity limits, deadlines, minimums. The flaw of averages is Jensen's inequality at work.

### The planted bug

`day41/bug_seed.py` is a colleague's Monte Carlo of the Day 6 plan, "seeded so the results can be repeated". Run it.

```python
profits = []
for run in range(10_000):
    rng = np.random.default_rng(2041)                          # a fixed seed, so anyone gets the same answer
    demand = {p: d["cap"][p] * rng.normal(1.0, 0.09) for p in make}
    profits.append(bakery_profit(make, demand, econ))
```

<details><summary>What happens, and why</summary>

```text
average $104,966.11, standard deviation $0.00, worst $104,966.11, best $104,966.11
```

10,000 runs, a standard deviation of zero, and an average that is simply one random week. The generator is created again inside the loop with the same seed, so every run draws the same numbers. **Seed once, before the loop.** A spread of exactly zero is a red flag in any simulation.

</details>

## Mini-project: how many croissants?

The forecast for croissants (BK-CR) is 31 batches a week, and demand is off by about 9% either way. A batch of 50 sells for $7.00 a croissant and costs $2.00 in ingredients; unsold croissants are thrown away. Simulate the expected profit of making 26 to 36 batches. Is the forecast the best quantity?

<details><summary>Solution</summary>

```python
"""Day 41 mini-project: how many batches of croissants (BK-CR) should the bakery make? The forecast is 31 batches,
and real demand is off by about 9% either way. Compare the expected profit of making 26 to 36 batches.

Run from the repository root (after day06/load_tables.py):  python day41/mini_project_croissants.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, "day41")
from montecarlo import bakery_economics, bakery_profit

con = duckdb.connect("northbound.duckdb", read_only=True)
econ = bakery_economics(con)
forecast = con.sql("SELECT demand_cap_batches FROM bakery_range_forecast WHERE sku = 'BK-CR'").fetchall()[0][0]
e = econ.loc["BK-CR"]
print(f"BK-CR: {e.units_per_batch} croissants a batch, sold at ${e.transfer_price:.2f}, ingredients ${e.ingredient_cost:.2f}")
rng = np.random.default_rng(41)
demand = forecast * rng.normal(1.0, 0.09, 20_000)              # the same 20,000 weeks for every choice
rows = []
for batches in range(26, 37):
    profit = np.array([bakery_profit({"BK-CR": batches}, {"BK-CR": x}, econ) for x in demand])
    unsold = np.maximum(batches - demand, 0).mean()
    short = np.maximum(demand - batches, 0).mean()
    rows.append((batches, round(profit.mean(), 2), round(unsold, 2), round(short, 2), round((demand > batches).mean(), 3)))
t = pd.DataFrame(rows, columns=["batches", "expected_profit", "batches_unsold", "batches_short", "chance_of_running_out"])
print(t.to_string(index=False))
best = t.loc[t.expected_profit.idxmax()]
print(f"best: {best.batches:.0f} batches, ${best.expected_profit:,.2f} a week "
      f"(+${best.expected_profit - t.expected_profit[t.batches == forecast].iloc[0]:,.2f} against the forecast)")
```

```text
BK-CR: 50.0 croissants a batch, sold at $7.00, ingredients $2.00
 batches  expected_profit  batches_unsold  batches_short  chance_of_running_out
      26          6486.17            0.04           5.04                  0.965
      27          6717.51            0.09           4.10                  0.925
      28          6930.53            0.20           3.20                  0.858
      29          7115.29            0.38           2.39                  0.764
      30          7260.96            0.68           1.69                  0.637
      31          7359.86            1.11           1.12                  0.499
      32          7409.98            1.69           0.69                  0.361
      33          7413.67            2.39           0.39                  0.238
      34          7379.39            3.20           0.20                  0.142
      35          7317.02            4.09           0.10                  0.078
      36          7236.65            5.04           0.04                  0.038
best: 33 batches, $7,413.67 a week (+$53.81 against the forecast)
```

**33 batches**, two more than the forecast: $54 a week more. A batch short costs $250 of profit (a lost sale); a batch too many costs $100 of ingredients. Since running out costs more, it pays to bake a little extra. Every simulated quantity uses the same 20,000 weeks of demand, so the differences come from the choice, not from luck (Day 48).

This is the **newsvendor problem**, and Day 51 solves it with a formula: make the quantity that demand stays below with probability $$\frac{250}{250 + 100} = 71\%$$. Here that is $$31 + 0.57 \times 2.8 \approx 33$$ batches.

</details>

## Commit

```bash
git add .
git commit -m "Day 41: the flaw of averages"
git push
```

### Day 41 checklist

- [ ] I can run a Monte Carlo simulation with a seeded generator and a loop.
- [ ] I showed that a plan's profit at average demand is not its average profit.
- [ ] I showed that deadlines met at average speed are missed often.
- [ ] I showed that a resource busy 100% of the time on average builds an endless line.
- [ ] I can explain the flaw with Jensen's inequality and the bends in a plan.
- [ ] I seed a generator once, outside the loop.

---

# DAY 42 — Input Distributions and Forecast Errors

A simulation is only as good as its random numbers. Today you build them from data: you clean the van trip log, compare actual and planned minutes, fit distributions and check whether they match the tails that matter, and turn the forecast errors from Day 19's history into demand inputs.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 42.1 Look at the data first |
| 0:45 – 1:45 | 42.2 Actual against planned; 42.3 which distribution? |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 42.4 Demand: forecast errors and counts |
| 2:45 – 3:45 | Fitted or empirical?; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: buffers by weather |
| 4:45 – 5:00 | Commit with Git + checklist |

## 42.1 Look at the data first

`trip_log` holds 1,872 van trips: date, weather, from and to, planned minutes (from `delivery_matrix`) and actual minutes. Create `day42/inputs.py`:

```python
"""Day 42: where the random numbers come from. Clean the trip log, fit travel-time and demand distributions.

Import it:  from inputs import load_trips, clean_trips, travel_sampler, demand_error_sd
"""
import numpy as np
import pandas as pd


def load_trips(con):
    t = con.sql("SELECT * FROM trip_log ORDER BY trip_id").df()
    t["ratio"] = t.actual_minutes / t.planned_minutes
    return t


def clean_trips(trips):
    """Drop impossible records: no time, negative time, or more than three times the plan plus an hour
    (a tracker left running). A short trip that took 30 minutes more is a real incident: keep it."""
    bad = (trips.actual_minutes <= 0) | (trips.actual_minutes > 3 * trips.planned_minutes + 60)
    return trips[~bad].copy(), trips[bad].copy()


def weekly_errors(con):
    """Chain-wide weekly units of the three history products, and the 'same as last week' forecast error
    as a ratio: actual / forecast."""
    s = con.sql("""SELECT week_start, sku, sum(units_sold) AS units FROM bakery_sales_weekly
                   GROUP BY week_start, sku ORDER BY sku, week_start""").df()
    s["forecast"] = s.groupby("sku").units.shift(1)
    s = s.dropna()
    s["ratio"] = s.units / s.forecast
    return s


def travel_sampler(trips, rng, weather_mix=None):
    """A function that draws one morning of trip factors: pick the morning's weather (from the log's mix,
    or weather_mix), then for each trip a ratio drawn from that weather's cleaned history (a bootstrap)."""
    by_weather = {w: g.ratio.to_numpy() for w, g in trips.groupby("weather")}
    days = trips.groupby("trip_date").weather.first()
    mix = weather_mix or days.value_counts(normalize=True).to_dict()
    names, probs = list(mix), np.array(list(mix.values()))

    def morning():
        w = names[rng.choice(len(names), p=probs / probs.sum())]
        pool = by_weather[w]
        return w, (lambda: pool[rng.integers(len(pool))])
    return morning


def demand_error_sd(con):
    """Forecast error (standard deviation of actual / forecast) by category: bread from BK-SD and BK-WW,
    viennoiserie from BK-CR; cakes and cookies have no history, so they borrow viennoiserie's (an assumption)."""
    e = weekly_errors(con).groupby("sku").ratio.std()
    bread, vien = float(e[["BK-SD", "BK-WW"]].mean()), float(e["BK-CR"])
    return {"bread": bread, "viennoiserie": vien, "cake": vien, "cookie": vien}
```

Then `day42/fit_inputs.py`:

```python
# ---------- 42.1 Look at the data first ----------
trips = load_trips(con)
print(f"{len(trips)} trips from {trips.trip_date.min():%Y-%m-%d} to {trips.trip_date.max():%Y-%m-%d}")
print(trips.ratio.describe().round(3).to_string())
clean, bad = clean_trips(trips)
print(f"dropped {len(bad)} impossible records:")
print(bad[["trip_id", "trip_date", "planned_minutes", "actual_minutes"]].to_string(index=False))
```

```text
1872 trips from 2026-05-25 to 2026-11-21
count    1872.000
mean        1.157
std         0.630
min        -0.107
25%         0.991
50%         1.091
75%         1.207
max        23.538
dropped 4 impossible records:
trip_id  trip_date  planned_minutes  actual_minutes
 T00041 2026-05-28               20             0.0
 T00312 2026-06-23               55             0.0
 T00903 2026-08-20               26           612.0
 T01456 2026-10-13               28            -3.0
```

A negative trip, two trips of zero minutes and one of 612 minutes for a 26-minute run: tracker glitches. They go. But a short trip that took 30 minutes more than planned stays: that's a real incident (a crash, a closed street), and exactly the kind of tail a simulation must keep. Write the cleaning rule down, and print what it drops.

## 42.2 Actual against planned

```python
# ---------- 42.2 Actual against planned ----------
print("--- actual / planned minutes, by weather")
print(clean.groupby("weather").ratio.agg(trips="size", mean="mean", median="median", p95=lambda r: r.quantile(0.95))
      .round(3).to_string())
print(f"the planning matrix is optimistic: on clear days trips take {clean[clean.weather == 'clear'].ratio.median():.3f} "
      f"times the plan (median)")
```

```text
--- actual / planned minutes, by weather
         trips   mean  median    p95
weather
clear     1353  1.116   1.064  1.391
rain       467  1.200   1.157  1.492
snow        48  1.495   1.390  1.795
the planning matrix is optimistic: on clear days trips take 1.064 times the plan (median)
```

**The planning matrix is optimistic**: even on clear days, the median trip takes 6% longer than planned; in rain 16%, in snow 39%. Every Module 4 route was planned with these minutes. This is the first finding of any input study: compare the model's numbers with what really happens.

## 42.3 Which distribution?

Fit candidate distributions to the clear-day ratios (actual / planned) with `scipy.stats` and compare them: the **log-likelihood** (higher is better), **AIC** (lower is better: it charges for parameters), the **Kolmogorov–Smirnov** statistic (the largest gap between the fitted and the data's cumulative distribution), and above all the **tail** that matters for lateness.

```python
# ---------- 42.3 Which distribution? ----------
x = clean[clean.weather == "clear"].ratio.to_numpy()
rows = []
for name, dist in [("normal", stats.norm), ("lognormal", stats.lognorm), ("gamma", stats.gamma)]:
    params = dist.fit(x) if name == "normal" else dist.fit(x, floc=0)
    ll = dist.logpdf(x, *params).sum()
    ks = stats.kstest(x, dist.cdf, args=params).statistic
    rows.append((name, round(ll, 1), round(2 * len(params) - 2 * ll, 1), round(ks, 4),
                 round(float(dist.sf(1.5, *params)), 4), round(float(dist.ppf(0.99, *params)), 3)))
print("--- distributions fitted to the clear-day ratios")
emp = [("data", None, None, None, round(float((x > 1.5).mean()), 4), round(float(np.quantile(x, 0.99)), 3))]
print(pd.DataFrame(emp + rows, columns=["model", "log_likelihood", "AIC", "KS", "P(ratio > 1.5)", "99th percentile"])
      .to_string(index=False))
```

```text
--- distributions fitted to the clear-day ratios
    model  log_likelihood    AIC     KS  P(ratio > 1.5)  99th percentile
     data             NaN    NaN    NaN          0.0362            2.607
   normal          -424.0  851.9 0.2128          0.1229            1.886
lognormal           137.0 -267.9 0.1179          0.0551            1.737
    gamma           -15.4   36.8 0.1488          0.0711            1.775
```

- The **lognormal** fits best of the three (AIC), as expected for a quantity that is a product of many small delays, and is never negative.
- The **normal** is the worst: it puts weight on impossible values and too much in the wrong tail.
- **None of them has the incidents**: the data's 99th percentile is 2.6 times the plan, the fits say 1.7 to 1.9. The incidents are a separate mechanism: about 3% of trips, plus 10 to 35 minutes.

So the simulation on Day 43 **resamples the cleaned trips themselves** (an empirical bootstrap), by weather: it keeps the incidents and the weather effect, with no distribution to choose. Fitted distributions are better when data are scarce, when you need to extrapolate (a worse winter than the log has seen), or when you want parameters to vary in experiments.

## 42.4 Demand: forecast errors and counts

```python
# ---------- 42.4 Demand: forecast errors and counts ----------
errors = weekly_errors(con)
print("--- forecast errors of 'same as last week', actual / forecast")
print(errors.groupby("sku").ratio.agg(weeks="size", mean="mean", sd="std").round(4).to_string())
for sku, g in errors.groupby("sku"):
    r = g.ratio.to_numpy()
    print(f"{sku}: lag-1 autocorrelation of errors {np.corrcoef(r[:-1], r[1:])[0, 1]:+.2f}, "
          f"Shapiro-Wilk p = {stats.shapiro(r).pvalue:.3f}")
print("error sd used for every category (Day 43):", {k: round(v, 4) for k, v in demand_error_sd(con).items()})
s01 = con.sql("SELECT units_sold FROM bakery_sales_weekly WHERE store_id = 'S01' AND sku = 'BK-CR'").df().units_sold
print(f"S01 croissants a week: mean {s01.mean():.2f}, variance {s01.var():.2f} "
      f"(Poisson has variance = mean: dispersion {s01.var() / s01.mean():.2f})")
```

```text
--- forecast errors of 'same as last week', actual / forecast
       weeks    mean      sd
sku
BK-CR    103  1.0048  0.0873
BK-SD    103  1.0021  0.0483
BK-WW    103  1.0025  0.0440
BK-CR: lag-1 autocorrelation of errors +0.13, Shapiro-Wilk p = 0.000
BK-SD: lag-1 autocorrelation of errors -0.19, Shapiro-Wilk p = 0.000
BK-WW: lag-1 autocorrelation of errors -0.02, Shapiro-Wilk p = 0.000
error sd used for every category (Day 43): {'bread': 0.0461, 'viennoiserie': 0.0873, 'cake': 0.0873, 'cookie': 0.0873}
S01 croissants a week: mean 25.55, variance 25.16 (Poisson has variance = mean: dispersion 0.98)
```

- The "same as last week" forecast is unbiased (mean ratio 1.00) with an error of **4–5% for bread and 9% for croissants**.
- The errors are **not normal** (Shapiro–Wilk p ≈ 0): holiday weeks give fat tails. Again, resampling the actual errors keeps them.
- Errors are only weakly autocorrelated: a bad week isn't usually followed by another. (Day 44 deals with errors that move together.)
- Cakes and cookies have no history; they borrow the croissant errors. That is an **assumption**, and it goes in the memo.
- A single store's weekly croissant sales look **Poisson**: the variance (25.16) equals the mean (25.55). At store level, Poisson or negative binomial counts are the natural model; chain-wide, a ratio error on the forecast is.

> **Decision it supports:** which random numbers to feed a simulation, so its risk estimates can be trusted.
>
> **Other decisions you can support the same way:** service times in a call centre · machine failure times · lead times from suppliers · patient arrivals.

### The planted bug

`day42/bug_dirty_fit.py` fits a normal distribution to the raw trip log. Run it.

```python
trips = load_trips(con)
mean, sd = stats.norm.fit(trips.ratio)
print(f"fitted: actual / planned ~ normal(mean {mean:.3f}, sd {sd:.3f})")
draws = np.random.default_rng(42).normal(mean, sd, 100_000)
print(f"simulated trips faster than half the plan: {(draws < 0.5).mean():.1%}; taking negative time: {(draws < 0).mean():.1%}")
print(f"real trips faster than half the plan: {(trips.ratio[trips.actual_minutes > 0] < 0.5).mean():.1%}")
```

<details><summary>What happens, and why</summary>

```text
fitted: actual / planned ~ normal(mean 1.157, sd 0.630)
simulated trips faster than half the plan: 15.0%; taking negative time: 3.4%
real trips faster than half the plan: 0.0%
```

The glitches (612 minutes, −3 minutes) inflate the standard deviation from 0.36 to 0.63, and the normal distribution then makes 3.4% of simulated trips take **negative time**, and 15% faster than half the plan, which never happens. Two mistakes compound: dirty data and a distribution that allows impossible values. Clean first, check the fitted distribution's range, and compare its tails with the data.

</details>

## Mini-project: buffers by weather

Dispatchers want a rule of thumb: "multiply the planned minutes by … in this weather". Compute the multiplier that covers 50%, 80%, 90% and 95% of trips in each weather, and the share of trips with an incident.

<details><summary>Solution</summary>

```python
"""Day 42 mini-project: dispatchers want a rule of thumb: "multiply the planned minutes by ... in this weather".
What multiplier covers half, 80%, 90% and 95% of trips in each weather, and how often is there an incident?

Run from the repository root (after day06/load_tables.py):  python day42/mini_project_buffers.py
"""
import sys

import duckdb
import pandas as pd

sys.path.insert(0, "day42")
from inputs import clean_trips, load_trips

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
trips, _ = clean_trips(load_trips(con))
trips["incident"] = trips.actual_minutes > 1.6 * trips.planned_minutes + 8      # far beyond the usual spread
table = trips.groupby("weather").agg(
    trips=("ratio", "size"), p50=("ratio", lambda r: r.quantile(0.50)), p80=("ratio", lambda r: r.quantile(0.80)),
    p90=("ratio", lambda r: r.quantile(0.90)), p95=("ratio", lambda r: r.quantile(0.95)), incidents=("incident", "mean"))
print(table.round(3).to_string())
days = trips.groupby("trip_date").weather.first().value_counts()
print("mornings in the log:", days.to_dict())
```

```text
         trips    p50    p80    p90    p95  incidents
weather
clear     1353  1.064  1.200  1.287  1.391      0.022
rain       467  1.157  1.293  1.391  1.492      0.017
snow        48  1.390  1.615  1.724  1.795      0.042
mornings in the log: {'clear': 113, 'rain': 39, 'snow': 4}
```

*Note to the dispatchers:* "Planned minutes are short. To arrive on time for 9 trips in 10, allow **1.3 times** the planned minutes in clear weather, **1.4** in rain and **1.7** in snow. About 2–4% of trips meet an incident that no multiplier covers; keep one van on standby on snow mornings. We'll update the planning matrix (Day 43 tests what a buffer costs)."

One caution: a route is several trips, and a buffer per trip is not the same as a buffer per route (errors partly cancel along a route, and the weather moves them all together). Day 43 simulates whole routes.

</details>

## Commit

```bash
git add .
git commit -m "Day 42: input distributions and forecast errors"
git push
```

### Day 42 checklist

- [ ] I look at and clean data before fitting, with a written rule, and keep real extremes.
- [ ] I compare the model's input numbers with what actually happens.
- [ ] I can fit and compare distributions with log-likelihood, AIC, KS and tail probabilities.
- [ ] I know when to resample the data (bootstrap) instead of fitting a distribution.
- [ ] I can turn a forecast history into forecast errors for a simulation, and state the assumptions.

---

# DAY 43 — Monte Carlo of a Plan

With Day 42's inputs, today you simulate two whole plans: the bakery's week (Day 6) and the morning routes (Day 37). You report risk, not just averages (percentiles, the bad tail, the chance of a bad outcome), and learn how many runs a simulation needs before its numbers can be trusted.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 43.1 The production plan |
| 0:45 – 1:45 | 43.2 How many runs? |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 43.3 The morning routes |
| 2:45 – 3:45 | Risk measures; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: plan with a buffer |
| 4:45 – 5:00 | Commit with Git + checklist |

## 43.1 The production plan

Create `day43/plan_risk.py`:

```python
"""Day 43: Monte Carlo of two plans, the Day 6 production plan and the Day 37 morning routes.

Import it:  from plan_risk import demand_sampler, simulate_bakery, simulate_routes, risk_summary
"""
import numpy as np
import pandas as pd

from inputs import weekly_errors
from montecarlo import bakery_profit, drive_routes


def demand_sampler(con, rng):
    """One week of demand: each product's forecast times a forecast error drawn from history (a bootstrap):
    bread from BK-SD and BK-WW's errors, everything else from BK-CR's (no history of its own)."""
    e = weekly_errors(con)
    pools = {"bread": e[e.sku.isin(["BK-SD", "BK-WW"])].ratio.to_numpy(), "other": e[e.sku == "BK-CR"].ratio.to_numpy()}

    def week(cap, category):
        return {p: c * rng.choice(pools["bread" if category[p] == "bread" else "other"]) for p, c in cap.items()}
    return week


def simulate_bakery(make, cap, category, econ, sampler, runs):
    rows = []
    for _ in range(runs):
        demand = sampler(cap, category)
        profit = bakery_profit(make, demand, econ)
        unsold = sum(max(0.0, x - demand[p]) for p, x in make.items())
        short = sum(max(0.0, demand[p] - x) for p, x in make.items())     # products the plan makes
        rows.append((profit, unsold, short))
    return pd.DataFrame(rows, columns=["profit", "batches_unsold", "batches_short"])


def simulate_routes(plan, stops, ids, minutes, morning, runs):
    """morning() picks the weather and returns (weather, trip_factor). One row per simulated morning."""
    rows = []
    for _ in range(runs):
        weather, factor = morning()
        late = drive_routes(plan, stops, ids, minutes, factor)
        rows.append({"weather": weather, "late": sum(v > 0 for v in late.values()),
                     "late_minutes": sum(late.values()), **{s: v > 0 for s, v in late.items()}})
    return pd.DataFrame(rows)


def risk_summary(x, worst=0.05):
    """Mean, spread and the bad tail of a simulated result (bigger is better)."""
    x = np.sort(np.asarray(x))
    k = max(1, int(round(worst * len(x))))
    return {"mean": x.mean(), "sd": x.std(ddof=1), "p5": np.quantile(x, 0.05), "median": np.median(x),
            "p95": np.quantile(x, 0.95), "mean_of_worst_5pct": x[:k].mean()}
```

`demand_sampler` resamples the real forecast errors (Day 42): bread from sourdough and whole-wheat history, everything else from croissants. Then `day43/run_monte_carlo.py`:

```python
# ---------- 43.1 The production plan ----------
d = load_data(con)
m = build_model(d)
solve(m)
make = {p: pyo.value(m.x[p]) for p in m.P if pyo.value(m.x[p]) > 1e-6}
econ = bakery_economics(con)
sims = simulate_bakery(make, d["cap"], d["category"], econ, demand_sampler(con, rng), 10_000)
print(f"Day 6 plan, 10,000 simulated weeks (the model promised $108,351.28):")
print(pd.Series(risk_summary(sims.profit)).map("${:,.2f}".format).to_string())
print(f"weeks below $100,000: {(sims.profit < 100_000).mean():.1%}; batches thrown away {sims.batches_unsold.mean():.1f} "
      f"and short {sims.batches_short.mean():.1f} a week on average")
```

```text
Day 6 plan, 10,000 simulated weeks (the model promised $108,351.28):
mean                  $105,821.95
sd                      $1,510.53
p5                    $102,804.44
median                $106,275.20
p95                   $107,478.84
mean_of_worst_5pct    $101,648.78
weeks below $100,000: 0.4%; batches thrown away 6.6 and short 20.9 a week on average
```

With the errors the bakery really has, the plan earns **$105,822 a week on average**, $2,529 less than promised (less than Day 41's guess, because bread errors are only 4–5%). One week in twenty earns less than $102,804, and those bad weeks average $101,649. **Report a plan with its spread and its bad tail**, not with one number:

| Measure | What it answers |
|---|---|
| Mean | What we earn over many weeks |
| Standard deviation | How much one week moves around the mean |
| 5th and 95th percentiles | A bad week and a good week |
| Mean of the worst 5% (CVaR) | How bad the bad weeks are on average (Day 58) |
| P(profit < target) | How often we miss a target |

## 43.2 How many runs?

A simulated average is itself random. Its **standard error** is the standard deviation of one run divided by $$\sqrt{n}$$, and a 95% confidence interval is the average ± 1.96 standard errors.

```python
# ---------- 43.2 How many runs? ----------
print("one simulated week's profit varies by about ${:,.0f} (sd); the average of n weeks by sd / sqrt(n):".format(sims.profit.std()))
for runs in (100, 1_000, 10_000, 100_000):
    x = simulate_bakery(make, d["cap"], d["category"], econ, demand_sampler(con, np.random.default_rng(runs)), runs).profit
    half = 1.96 * x.std(ddof=1) / np.sqrt(runs)
    print(f"{runs:>7,} runs: average ${x.mean():,.2f} +/- ${half:,.2f} (95% confidence)")
```

```text
one simulated week's profit varies by about $1,511 (sd); the average of n weeks by sd / sqrt(n):
    100 runs: average $105,794.26 +/- $302.70 (95% confidence)
  1,000 runs: average $105,724.40 +/- $98.45 (95% confidence)
 10,000 runs: average $105,800.05 +/- $30.46 (95% confidence)
100,000 runs: average $105,789.19 +/- $9.58 (95% confidence)
```

Ten times more runs buys only $$\sqrt{10} \approx 3$$ times more precision. 10,000 runs pin the average to ±$30, plenty to tell this plan from one that differs by $100; 100 runs (±$300) don't. Choose the number of runs from the precision the decision needs, and always print the interval.

## 43.3 The morning routes

Each simulated morning draws the weather from the log's mix, then every trip's actual / planned ratio from that weather's trips.

```python
# ---------- 43.3 The morning routes ----------
stops, ids, km, minutes = load_run(con)
plan = solve_vrptw(stops, ids, km, minutes, load_fleet(con))
trips, _ = clean_trips(load_trips(con))
mornings = simulate_routes(plan, stops, ids, minutes, travel_sampler(trips, rng), 5_000)
print(f"Day 37 routes, 5,000 simulated mornings: {mornings.late.mean():.2f} late deliveries on average; "
      f"at least one late on {(mornings.late > 0).mean():.1%} of mornings")
print(mornings.groupby("weather").agg(mornings=("late", "size"), late_per_morning=("late", "mean"),
                                      any_late=("late", lambda x: (x > 0).mean()),
                                      late_minutes=("late_minutes", "mean")).round(3).to_string())
by_stop = mornings[[s for s in ids[1:]]].mean().sort_values(ascending=False)
print("most often late:", ", ".join(f"{s} {p:.1%} ({stops.name[s]}, closes {int(stops.window_close[s]) // 60:02d}:"
                                    f"{int(stops.window_close[s]) % 60:02d})" for s, p in by_stop.head(5).items()))
```

```text
Day 37 routes, 5,000 simulated mornings: 0.91 late deliveries on average; at least one late on 69.4% of mornings
         mornings  late_per_morning  any_late  late_minutes
weather
clear        3636             0.741     0.602         8.246
rain         1243             1.208     0.932        16.365
snow          121             2.909     1.000        58.637
most often late: C18 64.7% (Cafe partner 18, closes 06:45), S34 7.6% (Northbound Laval 1, closes 07:00), C20 5.8% (Cafe partner 20, closes 06:15), C24 2.0% (Cafe partner 24, closes 06:45), C19 1.7% (Cafe partner 19, closes 07:15)
```

- **0.91 late deliveries a morning; at least one on 69% of mornings.** In snow, every morning has late deliveries.
- One café, **C18, is late on 65% of mornings**. The route planner fitted it in just before its window closes, at the planned speed; at real speeds it's usually late. Simulation finds the fragile part of a plan, not just its average.

> **Decision it supports:** how risky a plan is, and which parts of it fail first.
>
> **Other decisions you can support the same way:** a budget's chance of overrunning · a project's chance of finishing on time · an investment's downside.

### The planted bug

`day43/bug_confidence.py` compares the Day 6 plan with a plan that makes 5% more of everything, with 200 simulated weeks each. Run it.

```python
for name, make, seed in [("Day 6 plan", base, 1), ("5% more of everything", {p: 1.05 * x for p, x in base.items()}, 2)]:
    x = simulate_bakery(make, d["cap"], d["category"], econ, demand_sampler(con, np.random.default_rng(seed)), 200).profit
    half = 1.96 * x.std() / len(x)                            # the 95% margin of the average
    print(f"{name:<22} ${x.mean():,.2f} +/- ${half:,.2f}")
```

<details><summary>What happens, and why</summary>

```text
Day 6 plan             $105,642.61 +/- $16.44
5% more of everything  $105,287.28 +/- $18.49
```

The margins (±$16 and ±$18) are far too small: the code divides by $$n$$ instead of $$\sqrt{n}$$. With $$\sqrt{200} \approx 14$$ the margins are about ±$210: the two intervals overlap, and 200 runs can't tell these plans apart reliably. An overconfident simulation is worse than none: it settles decisions on noise. (Day 48 shows how to compare two plans with far fewer runs.)

</details>

## Mini-project: plan with a buffer

Re-plan the morning routes (Day 37's model) with the matrix minutes multiplied by 1.00, 1.05, 1.10, 1.15 and 1.20. Simulate each plan on the same 2,000 mornings (same seed). What does each buffer cost a day, and what does it save in late deliveries?

<details><summary>Solution</summary>

```python
"""Day 43 mini-project: plan the morning routes with padded travel times. Re-plan with the matrix minutes x 1.00,
1.05, 1.10, 1.15, 1.20, then simulate each plan on the same 2,000 mornings. What does each buffer cost a day,
and how many late deliveries does it save?

Run from the repository root (after day06/load_tables.py):  python day43/mini_project_buffer.py   (about 2 minutes)
"""
import sys

import duckdb
import numpy as np
import pandas as pd

for folder in ("day36", "day37", "day41", "day42", "day43"):
    sys.path.insert(0, folder)
from inputs import clean_trips, load_trips, travel_sampler
from plan_risk import simulate_routes
from routing import load_run
from vrptw import load_fleet, simulate, solve_vrptw

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
stops, ids, km, minutes = load_run(con)
fleet = load_fleet(con)
trips, _ = clean_trips(load_trips(con))
rows = []
for buffer in (1.00, 1.05, 1.10, 1.15, 1.20):
    padded = np.ceil(minutes * buffer)
    plan = solve_vrptw(stops, ids, km, padded, fleet)
    cost, total_km, _ = simulate(plan, stops, ids, km, padded, fleet)
    # the same 2,000 mornings for every plan: same seed, same weather and trip draws in the same order
    sims = simulate_routes(plan, stops, ids, minutes, travel_sampler(trips, np.random.default_rng(7)), 2_000)
    types = pd.Series([r["type"] for r in plan]).value_counts().to_dict()
    rows.append((f"x{buffer:.2f}", f"{types.get('van', 0)} vans, {types.get('truck', 0)} trucks", round(cost, 2),
                 round(sims.late.mean(), 3), round((sims.late > 0).mean(), 3), round(sims.late_minutes.mean(), 1)))
t = pd.DataFrame(rows, columns=["buffer", "vehicles", "cost_per_day", "late_per_morning", "mornings_with_late",
                                "late_minutes"])
print(t.to_string(index=False))
```

```text
buffer         vehicles  cost_per_day  late_per_morning  mornings_with_late  late_minutes
 x1.00 5 vans, 2 trucks        864.50             0.960               0.717          12.1
 x1.05 5 vans, 2 trucks        860.22             0.250               0.172           4.4
 x1.10 3 vans, 3 trucks        839.09             0.392               0.248           6.0
 x1.15 4 vans, 3 trucks        926.10             0.266               0.199           6.3
 x1.20 5 vans, 2 trucks        876.53             0.200               0.150           3.6
```

- **Any buffer helps a lot, and costs little**: with ×1.05 the late deliveries fall from 0.96 to 0.25 a morning, and mornings with a late delivery from 72% to 17%, at about the same cost.
- The numbers don't fall smoothly. Each buffer gives a different heuristic plan (Day 37's noise): ×1.10 is cheaper but later than ×1.05, ×1.15 uses an extra vehicle. Read the trend, not each row.
- Every plan was simulated on the same mornings, so the differences come from the plans, not from different random weather (Day 48 makes this precise).

*Memo to transport:* "Plan the morning routes with travel times 5% above the matrix. It costs nothing measurable and cuts late café deliveries by about three quarters. Update the matrix itself from the trip log (Day 42) so that planned times match real ones."

</details>

## Commit

```bash
git add .
git commit -m "Day 43: Monte Carlo of the production plan and the routes"
git push
```

Then run the tests: `tests/test_simulation.py` checks Days 41–43 (the plan's profit at the forecast, known queue and route answers, the cleaning rule, the error sizes, the risk summary). Add `"day41"`, `"day42"`, `"day43"` to the folders in `conftest.py`.

```python
"""Module 5, Days 41-43: Monte Carlo building blocks."""
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest
from bakery import build_model, load_data, solve
from inputs import clean_trips, demand_error_sd, load_trips
from montecarlo import bakery_economics, bakery_profit, drive_routes, wait_in_line
from plan_risk import risk_summary


def test_profit_at_the_forecast_is_the_model_profit(con):
    d = load_data(con)
    m = build_model(d)
    solve(m)
    make = {p: pyo.value(m.x[p]) for p in m.P}
    assert bakery_profit(make, d["cap"], bakery_economics(con)) == pytest.approx(108_351.28, abs=0.01)


def test_queue_and_routes_known_answers():
    # trucks every 5 minutes, 10 minutes each: every truck waits 5 minutes longer than the one before
    assert list(wait_in_line(np.full(4, 5.0), np.full(4, 10.0))) == [0, 5, 10, 15]
    stops = pd.DataFrame({"window_open": [0, 0], "window_close": [100, 10], "service_minutes": [0, 2]}, index=["D", "A"])
    plan = [{"stops": [(1, 0)]}]
    assert drive_routes(plan, stops, ["D", "A"], [[0, 8], [8, 0]], lambda: 1.0) == {"A": 0.0}
    assert drive_routes(plan, stops, ["D", "A"], [[0, 8], [8, 0]], lambda: 1.5) == {"A": 2.0}


def test_cleaning_and_errors(con):
    clean, bad = clean_trips(load_trips(con))
    assert sorted(bad.trip_id) == ["T00041", "T00312", "T00903", "T01456"]
    sd = demand_error_sd(con)
    assert 0.03 < sd["bread"] < sd["viennoiserie"] < 0.12


def test_risk_summary():
    r = risk_summary(np.arange(1, 101))
    assert r["mean"] == 50.5 and r["mean_of_worst_5pct"] == 3.0
```

```text
65 passed
```

### Day 43 checklist

- [ ] I can simulate a plan thousands of times with inputs resampled from data.
- [ ] I report mean, spread, percentiles, the bad tail and the chance of missing a target.
- [ ] I compute a confidence interval with the standard error (divide by $$\sqrt{n}$$).
- [ ] I choose the number of runs from the precision the decision needs.
- [ ] I use simulation to find the fragile parts of a plan.
- [ ] I compare plans on the same random numbers.
