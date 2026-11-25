# DAY 48 — Designing Simulation Experiments

A simulation is an experiment run on a computer, and its answers are estimates with noise. Day 46 said one driver
costs $417 of detention a night. That is an average over 500 random nights. Is it $417 ± $5 or ± $60? When you compare two options, is the difference real or luck? Today you learn to design simulation experiments: when to throw away a warm-up, how to compare options with **common random numbers**, how many replications a question needs, and why the best of many noisy options always looks better than it is.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 48.1 Terminating or steady state? The warm-up |
| 0:45 – 1:45 | 48.2 Comparing two options: common random numbers |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 48.3 How many replications? |
| 2:45 – 3:45 | 48.4 The optimizer's curse; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: how sure are we about the croissants? |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Days 48–50: `dock_shift_options` (shifts a second night driver could work, with their cost) and `stress_scenarios` (six bad weeks for the Day 50 stress test). No new packages.

## 48.1 Terminating or steady state? The warm-up

There are two kinds of simulation:

- A **terminating** simulation has a natural start and end. The night dock opens empty at 22:00 and closes when the last truck is unloaded. Each night is one replication, and the start is part of the question. There is no warm-up to throw away.
- A **steady-state** simulation asks about the long run. Day 47's formulas describe a dock that has been running forever at a constant rate. A run that starts empty is **biased** at first, so the first part (the warm-up) must be thrown away.

To find the warm-up, use **Welch's method**: run many replications from an empty start, average the $$k$$-th truck's wait across them, and smooth the result. The warm-up ends where the curve stops climbing. Create `day48/experiments.py`:

```python
"""Day 48: designing simulation experiments. Warm-up for steady-state runs, confidence intervals for a difference,
common random numbers, and how many replications a question needs.

Import it:  from experiments import lindley_runs, mean_ci, paired_ci, independent_ci, replications_needed
"""
import numpy as np
from scipy import stats


def lindley_runs(rate, service, trucks, reps, rng):
    """reps independent runs of a one-server queue that starts empty (Lindley's recursion, Day 41).
    rate: arrivals a minute; service: the pool of unloading times to resample. Returns waits, shape (reps, trucks)."""
    gaps = rng.exponential(1 / rate, (reps, trucks))
    s = service[rng.integers(len(service), size=(reps, trucks))]
    w = np.zeros((reps, trucks))
    for k in range(1, trucks):
        w[:, k] = np.maximum(0.0, w[:, k - 1] + s[:, k - 1] - gaps[:, k])
    return w


def mean_ci(x, level=0.95):
    """Mean and the half-width of its confidence interval (t distribution: replications are few)."""
    x = np.asarray(x, dtype=float)
    t = stats.t.ppf(0.5 + level / 2, len(x) - 1)
    return x.mean(), t * x.std(ddof=1) / np.sqrt(len(x))


def paired_ci(a, b, level=0.95):
    """The difference a - b when replication i of a and of b used the same random numbers (paired)."""
    return mean_ci(np.asarray(a) - np.asarray(b), level)


def independent_ci(a, b, level=0.95):
    """The difference a - b when the two options were simulated with separate random numbers (Welch)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return a.mean() - b.mean(), stats.t.ppf(0.5 + level / 2, df) * np.sqrt(va + vb)


def replications_needed(sd, half_width, level=0.95):
    """Replications for a confidence interval of +/- half_width, from a pilot run's standard deviation."""
    z = stats.norm.ppf(0.5 + level / 2)
    return int(np.ceil((z * sd / half_width) ** 2))
```

`lindley_runs` is Day 41's Lindley recursion, run for 400 replications at once with NumPy. Then `day48/run_experiments.py`:

```python
# ---------- 48.1 Terminating or steady state? The warm-up ----------
print("--- warm-up")
w = lindley_runs(lam, u, 3_000, 400, np.random.default_rng(48))
cv = u.std(ddof=1) / u.mean()
print(f"M/G/1 at {lam * 60:.3f} trucks an hour, formula (Pollaczek-Khinchine): {pk_wait(lam, u.mean(), cv):.2f} min")
for a, b in [(0, 20), (20, 100), (100, 500), (500, 1_500), (1_500, 3_000)]:
    print(f"trucks {a + 1:>5}-{b:<5}: average wait {w[:, a:b].mean():6.2f} min")
curve = w.mean(axis=0)
smooth = pd.Series(curve).rolling(25, center=True, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(curve, lw=0.6, color="#9aa4b2", label="average over 400 runs")
ax.plot(smooth, lw=2, color="#3b82f6", label="moving average (25 trucks)")
ax.axhline(pk_wait(lam, u.mean(), cv), ls="--", color="#e5484d", label="steady-state formula")
ax.set(xlabel="truck number (the dock starts empty)", ylabel="wait (min)", xlim=(0, 1_500),
       title="Welch's plot: how long until the queue forgets its empty start")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/day48_warmup.png", dpi=150)
print("saved figures/day48_warmup.png")
```

```text
--- warm-up
M/G/1 at 1.594 trucks an hour, formula (Pollaczek-Khinchine): 70.36 min
trucks     1-20   : average wait  38.66 min
trucks    21-100  : average wait  61.22 min
trucks   101-500  : average wait  72.28 min
trucks   501-1500 : average wait  70.31 min
trucks  1501-3000 : average wait  69.58 min
saved figures/day48_warmup.png
```

![Welch's plot for the dock](figures/day48_warmup.png)

The first 20 trucks wait about half the steady-state time. The curve reaches the formula's 70 minutes after about 150 trucks. Even after that it wanders, because each truck's wait depends on the one before. To estimate the steady state, discard the first 200 or so trucks of every run. Day 47 discarded 600 minutes, which is about 16 trucks: too few for this queue, but the 20,000-hour runs made the bias negligible.

## 48.2 Comparing two options: common random numbers

A dealer offers a faster forklift for $90 a night. It unloads in 85% of today's time. With one driver, does it cut detention by more than $90?

```python
# ---------- 48.2 Comparing two options: common random numbers ----------
# A dealer offers a faster forklift for $90 a night: it unloads in 85% of today's time. Worth it, with one driver?
print("--- two options")
FASTER, RENT = 0.85, 90.0
today = lambda rng: u[rng.integers(len(u))]
faster = lambda rng: FASTER * u[rng.integers(len(u))]          # the same draw, scaled: same truck, faster forklift
kw = dict(free=settings["free_minutes_before_detention"], per_hour=settings["detention_per_hour"])
n = 50
a = simulate_nights(rates, today, 3, 1, n, seed=480, **kw).detention
b_same = simulate_nights(rates, faster, 3, 1, n, seed=480, **kw).detention        # the same 50 nights
b_new = simulate_nights(rates, faster, 3, 1, n, seed=481, **kw).detention         # 50 other nights
for name, (diff, half) in [("separate random numbers", independent_ci(a, b_new)),
                           ("common random numbers", paired_ci(a, b_same))]:
    print(f"{name:>24}: the faster forklift cuts detention by ${diff:,.2f} a night +/- ${half:,.2f} ({n} nights each)")
print(f"correlation of the two options' nights: separate {np.corrcoef(a, b_new)[0, 1]:.2f}, "
      f"common {np.corrcoef(a, b_same)[0, 1]:.2f}")
```

```text
--- two options
 separate random numbers: the faster forklift cuts detention by $119.92 a night +/- $190.34 (50 nights each)
   common random numbers: the faster forklift cuts detention by $182.76 a night +/- $64.96 (50 nights each)
correlation of the two options' nights: separate 0.09, common 0.97
```

The estimates are the same size, but the **confidence intervals are not**. With separate random numbers, each option meets different nights: one option may get a quiet week and the other a busy one, and that noise swamps the difference. With **common random numbers** (CRN), both options see the same trucks arriving at the same times, and each truck takes 85% of the same unloading time. The two options' nights are 97% correlated, so most of the noise cancels in the difference:

$$\text{Var}(A - B) = \text{Var}(A) + \text{Var}(B) - 2\,\text{Cov}(A, B)$$

CRN works only when the random numbers are **synchronized**: the same draw must mean the same thing in both options. Day 46's model draws each truck's unloading time when the truck arrives, not when it starts unloading. So truck 7 gets the same draw however long it waited. If the draw happened at the start of unloading, a different number of draws would come before it in each option and the streams would drift apart.

## 48.3 How many replications?

A confidence interval's half-width is about $$1.96\,s/\sqrt{n}$$. Run a **pilot**, estimate the standard deviation $$s$$, then solve for $$n$$:

$$n = \left(\frac{1.96\, s}{h}\right)^2$$

```python
# ---------- 48.3 How many replications? ----------
print("--- how many")
d46 = simulate_nights(rates, today, 3, 1, 500, seed=46, **kw).detention
m46, h46 = mean_ci(d46)
print(f"Day 46's one-driver detention, 500 nights: ${m46:,.2f} +/- ${h46:,.2f} (a night's sd is ${d46.std(ddof=1):,.0f})")
target = 10.0
sd_sep, sd_crn = np.sqrt(a.var(ddof=1) + b_new.var(ddof=1)), (a - b_same).std(ddof=1)
for name, sd in [("separate", sd_sep), ("common", sd_crn)]:
    print(f"{name:>8}: sd of a night's difference ${sd:,.2f} -> {replications_needed(sd, target):,} nights for +/- ${target:.0f}")
need = replications_needed(sd_crn, target)
a = simulate_nights(rates, today, 3, 1, need, seed=482, **kw).detention
b = simulate_nights(rates, faster, 3, 1, need, seed=482, **kw).detention
diff, half = paired_ci(a, b)
print(f"{need:,} nights, common random numbers: saves ${diff:,.2f} +/- ${half:,.2f} a night, "
      f"against ${RENT:.0f} rent: {'worth it' if diff - half > RENT else 'not worth it' if diff + half < RENT else 'too close to call'}")
```

```text
--- how many
Day 46's one-driver detention, 500 nights: $417.05 +/- $61.55 (a night's sd is $701)
separate: sd of a night's difference $678.19 -> 17,669 nights for +/- $10
  common: sd of a night's difference $228.57 -> 2,007 nights for +/- $10
2,007 nights, common random numbers: saves $193.78 +/- $12.88 a night, against $90 rent: worth it
```

- Day 46's $417.05 was **±$62**: a night's detention varies by $700, so 500 nights pin the mean down only roughly. Report the interval, not just the number.
- To know the forklift's saving within ±$10, separate random numbers need about **17,700 nights**; CRN needs about **2,000**. The same answer for a ninth of the work.
- The final interval (±$12.88) is wider than planned, because the pilot's 50 nights underestimated the spread of a heavy-tailed quantity. If precision matters, check the interval you got and add replications.
- The forklift saves about $194 a night against $90 rent: **worth it**, with the whole interval above $90.

## 48.4 Many options: the optimizer's curse

```python
# ---------- 48.4 Many options: the optimizer's curse ----------
print("--- the curse")
cost = lambda seed, nights: simulate_nights(rates, today, 3, 1, nights, seed=seed, **kw).detention
estimates = [cost(1_000 + k, 30).mean() for k in range(10)]
best = int(np.argmin(estimates))
truth = cost(9_999, 3_000).mean()
print("ten estimates of the SAME option (today's dock), 30 nights each: " + ", ".join(f"{x:.0f}" for x in estimates))
print(f"the lowest looks like ${estimates[best]:,.2f} a night; 3,000 fresh nights say ${truth:,.2f}: "
      f"the winner of a noisy contest looks ${truth - estimates[best]:,.2f} better than it is")
```

```text
--- the curse
ten estimates of the SAME option (today's dock), 30 nights each: 286, 356, 341, 132, 400, 340, 196, 377, 406, 441
the lowest looks like $132.42 a night; 3,000 fresh nights say $361.20: the winner of a noisy contest looks $228.79 better than it is
```

The ten options are identical (today's dock, ten different sets of 30 nights), yet the "best" one looks $229 cheaper than the truth. Pick the best of many noisy estimates and you pick the one that was **luckiest**. This is the **optimizer's curse**, and it is why Day 49 always re-checks a winner on fresh runs.

> **Decision it supports:** whether a difference between options is real, and how much simulation it takes to know.
>
> **Other decisions you can support the same way:** A/B tests on a website (the same statistics) · comparing staffing plans in a call centre · trying a new layout in a factory simulation.

### The planted bug

`day48/bug_warmup.py` estimates the dock's steady-state wait (one driver, arrivals at the night's average rate all day long) from 40 runs of 60 trucks each. Run it.

```python
w = lindley_runs(lam, u, 60, 40, np.random.default_rng(1))       # 40 runs of 60 trucks, each from an empty dock
mean, half = mean_ci(w.mean(axis=1))
print(f"steady-state wait: {mean:.2f} +/- {half:.2f} min (95% confidence, 40 runs)")
```

<details><summary>What happens, and why</summary>

```text
steady-state wait: 56.33 +/- 11.12 min (95% confidence, 40 runs)
```

The answer is 56 minutes ± 11, and the interval doesn't even contain the true 70. Every run starts with an empty dock, and 60 trucks is less than the warm-up, so every run is biased low in the same direction. More replications only make the interval narrower around the **wrong** number: a confidence interval measures noise, not bias. Fixes: discard the warm-up (Welch's plot says about 200 trucks) and make each run much longer than that.

</details>

## Mini-project: how sure are we about the croissants?

Day 45 found the critical-ratio rule earns about $194 a week more than sending each store its average. Put a 95% confidence interval on that difference, with separate and with common random numbers, and work out how many simulated weeks each needs for ±$10.

<details><summary>Solution</summary>

```python
"""Day 48 mini-project: Day 45 said the critical-ratio rule earns about $194 a week more than sending each store its
average. How sure is that? Build a 95% confidence interval for the difference, with and without common random
numbers, and work out how many simulated weeks each needs for +/- $10.

Run from the repository root (after day06/load_tables.py):  python day48/mini_project_croissants.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

for folder in ("day45", "day48"):
    sys.path.insert(0, folder)
from experiments import independent_ci, paired_ci, replications_needed
from stockouts import simulate_store_days, store_day_means

con = duckdb.connect("northbound.duckdb", read_only=True)
price, cost = con.sql("SELECT transfer_price, ingredient_cost FROM bakery_range WHERE sku = 'BK-CR'").fetchall()[0]
means = store_day_means(con)
today = np.round(means["mean"]).astype(int).to_numpy()
best = stats.poisson.ppf((price - cost) / price, means["mean"]).astype(int)


def weekly_profit(quantity, seed, weeks=52):
    demand, q, sold = simulate_store_days(means, quantity, np.random.default_rng(seed), weeks=weeks)
    return (price * sold - cost * q).sum(axis=1)                # one profit per simulated week


rows = []
for name, a, b in [("separate random numbers", weekly_profit(best, 1), weekly_profit(today, 2)),
                   ("common random numbers", weekly_profit(best, 1), weekly_profit(today, 1))]:
    diff, half = (independent_ci if name.startswith("separate") else paired_ci)(a, b)
    sd = np.sqrt(a.var(ddof=1) + b.var(ddof=1)) if name.startswith("separate") else (a - b).std(ddof=1)
    rows.append((name, round(diff, 2), round(half, 2), round(sd, 2), replications_needed(sd, 10.0)))
print(pd.DataFrame(rows, columns=["comparison", "gain_a_week", "plus_minus", "sd_of_difference",
                                  "weeks_for_10_dollars"]).to_string(index=False))
```

```text
             comparison  gain_a_week  plus_minus  sd_of_difference  weeks_for_10_dollars
separate random numbers       177.35       65.79            238.80                  2191
  common random numbers       187.58       20.75             74.54                   214
```

- With CRN, the gain is **$188 ± $21 a week** from 52 weeks, consistent with Day 45's $194.
- For ±$10, CRN needs about 210 weeks and separate random numbers about 2,200, ten times as many.
- CRN works here because both rules face the same demand at every store on every day: only the quantities differ.

*Note for Store Operations:* "The new croissant rule earns $190 a week more (95% confidence: $167 to $208). The gain is real, not luck in the simulation."

</details>

## Commit

```bash
git add .
git commit -m "Day 48: designing simulation experiments"
git push
```

### Day 48 checklist

- [ ] I know when a simulation is terminating and when it needs a warm-up, and I can find the warm-up with Welch's plot.
- [ ] I compare options with common random numbers, and I keep the streams synchronized.
- [ ] I report a confidence interval for a difference, paired when the runs are paired.
- [ ] I can work out how many replications a question needs from a pilot run.
- [ ] I know the optimizer's curse, and that more replications don't fix bias.

---

# DAY 49 — Simulation-Optimization

Day 48 compared two options. Usually there are many: quantities for 240 store-days, shifts for a second driver, a forklift or not. **Simulation-optimization** searches over decisions with the simulator as the objective. It works when every option is run on the same random numbers, when the promise you must keep is a constraint the simulation estimates, and when the winner is checked on fresh runs. Today you search for croissant quantities and for the cheapest night-dock set-up that keeps the carriers' 30-minute promise (Day 47).

| Time | Block |
|---|---|
| 0:00 – 0:45 | 49.1 One decision: croissants for one store-day |
| 0:45 – 1:45 | 49.2 A decision with a promise: a second driver for the busy hours |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 49.3 Check the winner on fresh nights |
| 2:45 – 3:45 | A planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: empty shelves on at most 10% of days |
| 4:45 – 5:00 | Commit with Git + checklist |

## 49.1 One decision: croissants for one store-day

Day 45 used a formula, the critical ratio, for the best quantity. A simulator can find the same answer by trying every quantity, without the formula:

```python
# ---------- 49.1 One decision: how many croissants for one store-day? ----------
print("--- one store")
price, cost = con.sql("SELECT transfer_price, ingredient_cost FROM bakery_range WHERE sku = 'BK-CR'").fetchall()[0]
means = store_day_means(con)
lam = means.query("store_id == 'S04' and day == 'Fri'")["mean"].item()
qs = np.arange(0, 21)
demand = np.random.default_rng(49).poisson(lam, 2_000)                    # 2,000 Fridays, the same for every q
crn = [(price * np.minimum(demand, q) - cost * q).mean() for q in qs]
fresh = [(lambda dd: (price * np.minimum(dd, q) - cost * q).mean())(np.random.default_rng(100 + q).poisson(lam, 2_000))
         for q in qs]
print(f"S04 on a Friday: Poisson demand, mean {lam:.2f}; formula (critical ratio {(price - cost) / price:.3f}): "
      f"{int(stats.poisson.ppf((price - cost) / price, lam))}")
print(pd.DataFrame({"croissants": qs, "same_days": np.round(crn, 2), "new_days_each_q": np.round(fresh, 2)})
      .iloc[6:16].to_string(index=False))
print(f"best by simulation: same days {qs[int(np.argmax(crn))]}, new days for each q {qs[int(np.argmax(fresh))]}")

# every store-day, the same way
best_sim, best_formula = [], []
rng = np.random.default_rng(491)
for mu in means["mean"]:
    dd = rng.poisson(mu, 2_000)
    profit = [(price * np.minimum(dd, q) - cost * q).mean() for q in range(0, 30)]
    best_sim.append(int(np.argmax(profit)))
    best_formula.append(int(stats.poisson.ppf((price - cost) / price, mu)))
gap = np.array(best_sim) - np.array(best_formula)
print(f"all 240 store-days: simulation = formula on {np.mean(gap == 0):.0%}, one croissant off on {np.mean(abs(gap) == 1):.0%}")
```

```text
--- one store
S04 on a Friday: Poisson demand, mean 8.57; formula (critical ratio 0.714): 10
 croissants  same_days  new_days_each_q
          6      28.08            28.05
          7      31.30            31.57
          8      33.64            33.74
          9      34.98            35.53
         10      35.42            35.71
         11      35.07            35.14
         12      34.18            34.68
         13      32.89            32.63
         14      31.27            31.59
         15      29.45            29.43
best by simulation: same days 10, new days for each q 10
all 240 store-days: simulation = formula on 95%, one croissant off on 5%
```

- With the **same 2,000 Fridays** for every quantity (CRN), the profit curve is smooth and its top is the formula's 10.
- With new Fridays for each quantity, the curve is bumpy. Here it still peaks at 10, but the bumps could have moved the peak.
- Over all 240 store-days the search matches the formula on 95% and is one croissant off on the rest. Those are near-ties, where the formula's own answer sits on a flat top.

The lesson: once a formula exists, use it. The simulator is for decisions where no formula exists, like the next one.

## 49.2 A decision with a promise: a second driver for the busy hours

Day 46 found a second night driver saves money. But the trucks come mostly between 23:00 and 02:00. Could a **shorter shift** do as well for less? `dock_shift_options` lists a second driver all night ($245), 4-hour shifts ($150) or 6-hour shifts ($200) starting at different times. Each can be combined with Day 48's faster forklift ($90). The carriers' promise: **90% of trucks start unloading within 30 minutes**.

Drivers on shifts need a new SimPy trick. A driver who isn't on duty is held by a **blocker** process that has priority over the trucks. Create `day49/shifts.py`:

```python
"""Day 49: the night dock with drivers on shifts. Each driver is available from `start` to `end` (minutes after 22:00;
end=None: until the last truck is unloaded), so a second driver can cover just the busy hours.

Import it:  from shifts import simulate_night_shifts, evaluate_option
"""
import numpy as np
import pandas as pd
import simpy


def simulate_night_shifts(rates, unload, doors, shifts, rng):
    """Like Day 46's simulate_night, but `shifts` lists (start, end) for each driver. A driver who isn't on shift is
    held by a 'blocker' with priority over the trucks; a driver whose shift ends finishes the truck in hand first."""
    env = simpy.Environment()
    dock_doors = simpy.Resource(env, capacity=doors)
    forklifts = simpy.PriorityResource(env, capacity=len(shifts))
    trucks = []

    def off_duty(start, end):
        if start > 0:
            with forklifts.request(priority=-1) as req:          # takes a driver before any truck can
                yield req
                yield env.timeout(start)                          # released when the shift starts
        if end is not None:
            yield env.timeout(end - env.now)
            req = forklifts.request(priority=-1)                  # the next free driver goes home
            yield req                                             # ... and is never released

    def truck(name, minutes):
        arrived = env.now
        with dock_doors.request() as door:
            yield door
            with forklifts.request(priority=0) as driver:
                yield driver
                start = env.now
                yield env.timeout(minutes)
        trucks.append((name, arrived, start, env.now))

    def arrivals():
        k = 0
        for hour, rate in enumerate(rates):                       # the same stream of draws as Day 46
            end = 60 * (hour + 1)
            while rate > 0:
                gap = rng.exponential(60 / rate)
                if env.now + gap >= end:
                    yield env.timeout(end - env.now)
                    break
                yield env.timeout(gap)
                k += 1
                env.process(truck(f"T{k}", unload()))
            else:
                yield env.timeout(end - env.now)

    for start, end in shifts:
        env.process(off_duty(start, end))
    env.process(arrivals())
    env.run()
    return pd.DataFrame(trucks, columns=["truck", "arrived", "start", "end"])


def evaluate_option(rates, unload_sampler, doors, shifts, nights, seed, free=60, per_hour=85.0, promise=30):
    """Simulate `nights` nights (the same nights for every option with the same seed: common random numbers).
    Returns one row per night: detention ($), the share of trucks that started within `promise` minutes, trucks."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(nights):
        t = simulate_night_shifts(rates, lambda: unload_sampler(rng), doors, shifts, rng)
        wait = (t.start - t.arrived).to_numpy()
        rows.append((float(np.maximum(wait - free, 0).sum() / 60 * per_hour), int((wait <= promise).sum()), len(t)))
    return pd.DataFrame(rows, columns=["detention", "on_time", "trucks"])
```

Then search all 20 options on the same 500 nights:

```python
# ---------- 49.2 A decision with a promise: a second driver for the busy hours ----------
print("--- driver shifts")
log, settings = load_receiving(con)
rates, u = hourly_rates(log), log.unload_minutes.to_numpy()
options = con.sql("SELECT * FROM dock_shift_options").df()
FASTER, RENT = 0.85, 90.0                                                 # Day 48's faster forklift
kw = dict(free=settings["free_minutes_before_detention"], per_hour=settings["detention_per_hour"])


opt = options.set_index("option")


def shifts_for(name):
    """Driver 1 all night; driver 2 (if any) on the option's shift. A shift that runs to 06:00 stays until done."""
    o = opt.loc[name]
    if o.hours == 0:
        return [(0, None)]
    start, end = int(o.start_after_22h_minutes), int(o.start_after_22h_minutes + 60 * o.hours)
    return [(0, None), (start, None if end >= 480 else end)]


def run(name, forklift, nights, seed):
    scale = FASTER if forklift else 1.0
    n = evaluate_option(rates, lambda g: scale * u[g.integers(len(u))], 3, shifts_for(name), nights, seed, **kw)
    n["cost"] = n.detention + opt.cost_per_night[name] + (RENT if forklift else 0.0)
    return n


rows = []
for name in opt.index:
    for forklift in (False, True):
        n = run(name, forklift, 500, seed=49)                             # the same 500 nights for all 20 options
        rows.append((name, "faster" if forklift else "today", opt.cost_per_night[name] + (RENT if forklift else 0),
                     round(n.detention.mean(), 2), round(n.cost.mean(), 2), round(n.on_time.sum() / n.trucks.sum(), 3)))
table = pd.DataFrame(rows, columns=["second_driver", "forklift", "fixed_cost", "detention", "cost_a_night", "within_30_min"])
table["promise_kept"] = table.within_30_min >= 0.90
print(table.sort_values("cost_a_night").to_string(index=False))
```

```text
--- driver shifts
second_driver forklift  fixed_cost  detention  cost_a_night  within_30_min  promise_kept
   4h_from_00    today       150.0      18.53        168.53          0.850         False
   4h_from_23    today       150.0      31.02        181.02          0.864         False
   4h_from_01    today       150.0      34.11        184.11          0.774         False
   6h_from_23    today       200.0       7.55        207.55          0.918          True
   6h_from_00    today       200.0       9.22        209.22          0.893         False
   6h_from_22    today       200.0      13.67        213.67          0.891         False
   4h_from_22    today       150.0      67.89        217.89          0.820         False
   4h_from_02    today       150.0      84.37        234.37          0.655         False
   4h_from_00   faster       240.0       5.01        245.01          0.911          True
   4h_from_23   faster       240.0       7.43        247.43          0.917          True
    all_night    today       245.0       5.64        250.64          0.934          True
   4h_from_01   faster       240.0      12.14        252.14          0.851         False
   4h_from_22   faster       240.0      21.67        261.67          0.889         False
   4h_from_02   faster       240.0      40.17        280.17          0.740         False
         none   faster        90.0     200.28        290.28          0.600         False
   6h_from_23   faster       290.0       1.60        291.60          0.955          True
   6h_from_00   faster       290.0       2.18        292.18          0.940          True
   6h_from_22   faster       290.0       4.04        294.04          0.936          True
    all_night   faster       335.0       1.21        336.21          0.965          True
         none    today         0.0     405.43        405.43          0.481         False
```

- The cheapest options (4-hour shifts, $169–$184 a night) **break the promise**: only 85–86% of trucks start within 30 minutes.
- The cheapest option that keeps it is a **second driver from 23:00 to 05:00** ($200), at **$207.55 a night** with 91.8% within 30 minutes. That is $43 a night cheaper than a second driver all night.
- The faster forklift doesn't earn its place once there are two drivers: the second driver already removes most of the waiting.
- Without the promise, the 4-hour shift from 00:00 would win at $169. **The promise costs about $39 a night.** That is worth knowing before agreeing to it.

## 49.3 Check the winner on fresh nights

```python
# ---------- 49.3 Check the winners on fresh nights ----------
print("--- fresh nights")
top = table[table.promise_kept].nsmallest(3, "cost_a_night")
fresh = {}
for r in top.itertuples():
    n = fresh[(r.second_driver, r.forklift)] = run(r.second_driver, r.forklift == "faster", 2_000, seed=4_900)
    m, h = mean_ci(n.cost)
    print(f"{r.second_driver:>11} + {r.forklift:<6} forklift: search said ${r.cost_a_night:,.2f}, "
          f"2,000 fresh nights ${m:,.2f} +/- ${h:,.2f}; within 30 min {n.on_time.sum() / n.trucks.sum():.3f}")
(k1, n1), (k2, n2) = list(fresh.items())[:2]
d, h = paired_ci(n2.cost, n1.cost)
print(f"{k2[0]} + {k2[1]} forklift costs ${d:,.2f} +/- ${h:,.2f} a night more than {k1[0]} (same fresh nights)")
```

```text
--- fresh nights
 6h_from_23 + today  forklift: search said $207.55, 2,000 fresh nights $205.48 +/- $1.66; within 30 min 0.917
 4h_from_00 + faster forklift: search said $245.01, 2,000 fresh nights $245.38 +/- $1.54; within 30 min 0.910
 4h_from_23 + faster forklift: search said $247.43, 2,000 fresh nights $248.38 +/- $2.12; within 30 min 0.919
4h_from_00 + faster forklift costs $39.91 +/- $1.07 a night more than 6h_from_23 (same fresh nights)
```

The winner holds up on 2,000 new nights: $205.48 ± $1.66 a night, with 91.7% of trucks within 30 minutes. The runner-up costs $40 a night more, with an interval of ±$1. Here the optimizer's curse is mild, because 500 nights of CRN separate the options well. It is never zero, so always re-check.

> **Decision it supports:** choosing among many options when each can only be evaluated by simulation, including under a service promise.
>
> **Other decisions you can support the same way:** staffing levels by hour in a call centre · reorder points and quantities (Module 6) · the number of checkouts open by hour · buffer sizes on a production line.

### The planted bug

`day49/bug_new_seeds.py` is a colleague's version of the search: a new seed for every option ("more randomness is more realistic") and 100 nights each. Run it.

```python
def search(base, common, nights=100):
    rows = []
    for k, o in enumerate(options.itertuples()):
        start, end = o.start_after_22h_minutes, o.start_after_22h_minutes + 60 * o.hours
        shifts = [(0, None)] + ([] if o.hours == 0 else [(start, None if end >= 480 else end)])
        seed = base if common else base + k                                    # the bug: a new seed for each option
        n = evaluate_option(rates, lambda g: u[g.integers(len(u))], 3, shifts, nights, seed=seed)
        rows.append((o.option, round(n.detention.mean() + o.cost_per_night, 2), round(n.on_time.sum() / n.trucks.sum(), 3)))
    t = pd.DataFrame(rows, columns=["second_driver", "cost_a_night", "within_30_min"])
    kept = t[t.within_30_min >= 0.90]
    return t, (kept.nsmallest(1, "cost_a_night").iloc[0] if len(kept) else pd.Series({"second_driver": "no option"}))


t, best = search(490, common=False)
print(t.sort_values("cost_a_night").to_string(index=False))
print(f"choose {best.second_driver}: ${best.cost_a_night:,.2f} a night, promise kept ({best.within_30_min:.3f})")
print("--- the same search, 20 starting seeds")
for label, common, nights in [("new seed for each option, 100 nights", False, 100),
                              ("same seed for every option, 100 nights", True, 100),
                              ("same seed for every option, 500 nights", True, 500)]:
    picks = pd.Series([search(1_000 * b, common, nights)[1].second_driver for b in range(1, 21)]).value_counts()
    print(f"{label:>39}: " + ", ".join(f"{name} {k}x" for name, k in picks.items()))
```

<details><summary>What happens, and why</summary>

```text
second_driver  cost_a_night  within_30_min
   4h_from_00        161.24          0.848
   4h_from_23        175.02          0.846
   4h_from_01        183.83          0.794
   4h_from_22        196.73          0.843
   6h_from_23        206.80          0.925
   6h_from_00        208.82          0.903
   4h_from_02        209.06          0.662
   6h_from_22        209.86          0.904
    all_night        247.91          0.944
         none        339.67          0.508
choose 6h_from_23: $206.80 a night, promise kept (0.925)
--- the same search, 20 starting seeds
   new seed for each option, 100 nights: 6h_from_23 10x, 6h_from_00 7x, 6h_from_22 2x, all_night 1x
 same seed for every option, 100 nights: 6h_from_23 13x, 6h_from_00 3x, all_night 3x, no option 1x
 same seed for every option, 500 nights: 6h_from_23 16x, 6h_from_00 4x
```

This time it happens to pick the right option. But run the same search from 20 starting seeds and it picks the 23:00 shift only **half the time**. The rest of the time it picks the 00:00 shift, which breaks the promise in the 500-night search (89.3%), or a dearer option. With a new seed for each option, the comparison mixes the options' differences with the nights' differences.

Common random numbers help (13 of 20), and CRN with 500 nights helps more (16 of 20). The remaining picks of the 00:00 shift happen because it sits right at the 90% line: no amount of simulation makes a near-tie decisive. Fixes: use the same random numbers for every option; run enough nights for the promise, not just the cost; re-check the top few on fresh runs; and when two options are this close, choose the one with a safety margin.

</details>

## Mini-project: empty shelves on at most 10% of days

Store Operations wants empty croissant shelves on **at most 10% of store-days**, chain-wide, while making as much profit as possible. Day 45's rules give 22% (critical ratio) or 3.5% (95% in stock at every store-day). Search between them.

*Hint:* start from the critical-ratio quantities. Repeatedly add one croissant where it buys the most in-stock days per dollar of profit lost, until the chain reaches 10%. Do the search on 1,000 simulated weeks, then check it on 1,000 fresh ones.

<details><summary>Solution</summary>

```python
"""Day 49 mini-project: Store Operations wants empty croissant shelves on at most 10% of store-days (chain-wide).
Find the most profitable quantities that keep that promise. Search on 1,000 simulated weeks: start from the
critical-ratio quantities (Day 45) and add one croissant at a time where it buys the most in-stock days per dollar
of profit lost. Then check the plan on fresh weeks.

Run from the repository root (after day06/load_tables.py):  python day49/mini_project_shelves.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "day45")
from stockouts import simulate_store_days, store_day_means

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
price, cost = con.sql("SELECT transfer_price, ingredient_cost FROM bakery_range WHERE sku = 'BK-CR'").fetchall()[0]
means = store_day_means(con)
demand, _, _ = simulate_store_days(means, np.zeros(len(means), dtype=int), np.random.default_rng(49), weeks=1_000)


def score(q, demand):
    sold = np.minimum(demand, q)
    return (price * sold - cost * q).sum() / len(demand), (demand > q).mean()     # profit a week, share of days out


q = stats.poisson.ppf((price - cost) / price, means["mean"]).astype(int)
profit, out = score(q, demand)
steps = 0
while out > 0.10:
    # for each store-day: one more croissant gains in-stock days (P(demand = q)) and costs profit
    gain = (demand == q).mean(axis=0)
    lost = cost - price * (demand > q).mean(axis=0)
    k = int(np.argmax(gain / np.maximum(lost, 1e-9)))
    q[k] += 1
    profit, out = score(q, demand)
    steps += 1
print(f"{steps} croissants added to the critical-ratio plan; search weeks: profit ${profit:,.2f} a week, days out {out:.3f}")

fresh, _, _ = simulate_store_days(means, np.zeros(len(means), dtype=int), np.random.default_rng(4_900), weeks=1_000)
rows = []
for name, quantity in [("today: the average", np.round(means["mean"]).astype(int).to_numpy()),
                       ("critical ratio (Day 45)", stats.poisson.ppf((price - cost) / price, means["mean"]).astype(int)),
                       ("at most 10% of days out", q),
                       ("each store-day 90% in stock", stats.poisson.ppf(0.90, means["mean"]).astype(int))]:
    p, o = score(quantity, fresh)
    rows.append((name, int(quantity.sum()), round(o, 3), round(p, 2)))
print("fresh weeks:")
print(pd.DataFrame(rows, columns=["rule", "croissants_a_week", "days_out", "profit_a_week"]).to_string(index=False))
```

```text
342 croissants added to the critical-ratio plan; search weeks: profit $5,678.10 a week, days out 0.100
fresh weeks:
                       rule  croissants_a_week  days_out  profit_a_week
         today: the average               1463     0.392        5690.99
    critical ratio (Day 45)               1774     0.222        5884.21
    at most 10% of days out               2116     0.100        5679.57
each store-day 90% in stock               2234     0.074        5542.87
```

- The search adds 342 croissants a week. On fresh weeks the chain is out on 10.0% of store-days, and profit is $5,680 a week: about **$205 a week less** than the critical-ratio rule, and the same as today's profit with a quarter of the empty shelves.
- The obvious alternative, 90% in stock at every store-day, sends 118 more croissants and earns $137 a week less than the search. The search puts the extra croissants where they are cheapest: at stores where one more croissant often prevents an empty shelf.

*Memo:* "A 10% limit on empty shelves costs about $205 a week against the most profitable rule ($10,700 a year). The cheapest way to meet it is to send extra croissants to the stores where they do the most good, not the same service level everywhere."

</details>

## Commit

```bash
git add .
git commit -m "Day 49: simulation-optimization"
git push
```

### Day 49 checklist

- [ ] I can search a decision with a simulator as the objective, using the same random numbers for every option.
- [ ] I can model resources that change over time in SimPy (blockers with priority).
- [ ] I handle a promise as a constraint the simulation estimates, and I know what it costs.
- [ ] I re-check the winner on fresh runs.
- [ ] I can run a greedy search on sample paths when there are too many options to try them all.

---

# DAY 50 — Capstone: Stress-Testing the Network

Module 5 measured risks one at a time: bakery demand, morning routes, store shelves, the night dock. Today you put them together in **one simulated week of the network**, under six scenarios from `stress_scenarios`. You compare **today's way of working** with the module's **recommendations**:

- the Day 43 routes planned with a 5% time buffer;
- the Day 45 critical-ratio croissants;
- the Day 49 second driver from 23:00 to 05:00.

The question for management: does the network hold up when bad things happen, and do the recommendations still pay when they do?

| Time | Block |
|---|---|
| 0:00 – 0:30 | 50.1 The brief, the scenarios and the two policies |
| 0:30 – 1:30 | 50.2 One week of the network |
| 1:30 – 1:45 | Break |
| 1:45 – 2:45 | 50.3 Six scenarios; 50.4 breaking points |
| 2:45 – 3:30 | 50.5 The memo |
| 3:30 – 3:45 | Break |
| 3:45 – 4:30 | Module 5 wrap-up |
| 4:30 – 5:00 | Commit, tag, tests + checklist |

## 50.1 The brief, the scenarios and the two policies

| Scenario | What happens |
|---|---|
| normal | weeks drawn from history |
| holiday_rush | December demand weeks (Day 44's whole weeks), stores +15%, trucks +25% |
| snowstorm_week | every morning snows (Day 42's snow trips), stores −10% |
| oven_breakdown | ovens at 70% all week: the bakery re-plans (Day 44) |
| dock_surge | 40% more trucks at night |
| perfect_storm | all of it at once |

Create `day50/stress.py`. It reuses a module's worth of code: Day 6's bakery model, Day 37's routes, Day 42's trip sampler, Day 44's whole weeks, Day 45's stores and Day 49's driver shifts:

```python
"""Day 50 capstone: one simulated week of the Northbound network (bakery, morning routes, store croissants, the
night dock), under a stress scenario and an operating policy.

Import it:  from stress import load_network, stress_test
"""
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from scipy import stats

from bakery import build_model, load_data, solve
from correlated import error_table
from dock import hourly_rates, load_receiving
from inputs import clean_trips, load_trips, travel_sampler
from montecarlo import bakery_economics, bakery_profit, drive_routes
from routing import load_run
from shifts import evaluate_option
from stockouts import simulate_store_days, store_day_means
from vrptw import load_fleet, simulate, solve_vrptw

POLICIES = {
    "today": {"routes": "day37", "croissants": "average", "second_driver": "none"},
    "recommended": {"routes": "buffer_1.05", "croissants": "critical_ratio", "second_driver": "6h_from_23"},
}


def load_network(con):
    """Everything the weekly simulation needs, prepared once: plans, histories and settings."""
    net = {"con": con}
    d = load_data(con)
    net["bakery"], net["econ"] = d, bakery_economics(con)
    net["make"] = {}
    for left in sorted(set(con.sql("SELECT oven_capacity_left FROM stress_scenarios").df().oven_capacity_left)):
        week = {**d, "capacity": {**d["capacity"], "oven_hours": d["capacity"]["oven_hours"] * left}}
        m = build_model(week)
        solve(m)                                              # the plan the bakery makes with the ovens it has
        net["make"][left] = {p: pyo.value(m.x[p]) for p in m.P if pyo.value(m.x[p]) > 1e-6}
    e = error_table(con)
    net["errors"] = {"bread": ((e["BK-SD"] + e["BK-WW"]) / 2).to_numpy(), "other": e["BK-CR"].to_numpy(),
                     "december": np.array([w.month == 12 for w in pd.to_datetime(e.index)])}
    stops, ids, km, minutes = load_run(con)
    fleet = load_fleet(con)
    net["routes"] = {"stops": stops, "ids": ids, "minutes": minutes, "trips": clean_trips(load_trips(con))[0]}
    net["route_plans"], net["fleet_cost"] = {}, {}
    for name, buffer in [("day37", 1.00), ("buffer_1.05", 1.05), ("buffer_1.30", 1.30)]:
        padded = np.ceil(minutes * buffer)
        plan = solve_vrptw(stops, ids, km, padded, fleet)
        net["route_plans"][name], net["fleet_cost"][name] = plan, simulate(plan, stops, ids, km, padded, fleet)[0]
    price, cost = con.sql("SELECT transfer_price, ingredient_cost FROM bakery_range WHERE sku = 'BK-CR'").fetchall()[0]
    means = store_day_means(con)
    net["croissant"] = {"means": means, "price": price, "cost": cost,
                        "average": np.round(means["mean"]).astype(int).to_numpy(),
                        "critical_ratio": stats.poisson.ppf((price - cost) / price, means["mean"]).astype(int)}
    log, settings = load_receiving(con)
    net["dock"] = {"rates": hourly_rates(log), "unload": log.unload_minutes.to_numpy(), "settings": settings,
                   "options": con.sql("SELECT * FROM dock_shift_options").df().set_index("option")}
    net["scenarios"] = con.sql("SELECT * FROM stress_scenarios").df().set_index("scenario")
    return net


def driver_shifts(options, name):
    o = options.loc[name]
    if o.hours == 0:
        return [(0, None)]
    start, end = int(o.start_after_22h_minutes), int(o.start_after_22h_minutes + 60 * o.hours)
    return [(0, None), (start, None if end >= 480 else end)]


def stress_test(net, scenario, policy, weeks, seed, arrivals_factor=None):
    """`weeks` simulated weeks of the network. Each part draws from its own random stream (seeded from `seed`),
    so two policies run with the same seed face the same weeks: common random numbers."""
    s = net["scenarios"].loc[scenario]
    pol = POLICIES[policy] if isinstance(policy, str) else policy
    r_bak, r_rt, r_cr, r_dk = (np.random.default_rng([seed, k]) for k in range(4))

    # bakery: whole historical weeks of forecast errors (Day 44); December only in a holiday scenario
    e, d = net["errors"], net["bakery"]
    pool = np.flatnonzero(e["december"]) if s.demand_weeks == "december" else np.arange(len(e["bread"]))
    make = net["make"][s.oven_capacity_left]
    bakery = []
    for _ in range(weeks):
        k = pool[r_bak.integers(len(pool))]
        demand = {p: c * (e["bread"][k] if d["category"][p] == "bread" else e["other"][k]) for p, c in d["cap"].items()}
        bakery.append(bakery_profit(make, demand, net["econ"]))

    # morning routes: six mornings a week, one weather a morning (all snow in a snowstorm week)
    rt = net["routes"]
    plan = net["route_plans"][pol["routes"]]
    morning = travel_sampler(rt["trips"], r_rt, {"snow": 1.0} if s.weather == "snow" else None)
    late = []
    for _ in range(weeks):
        n = 0
        for _ in range(6):
            _, factor = morning()
            n += sum(v > 0 for v in drive_routes(plan, rt["stops"], rt["ids"], rt["minutes"], factor).values())
        late.append(n)

    # store croissants: Poisson demand with a chain-wide factor each day (Day 45.5)
    c = net["croissant"]
    means = c["means"].assign(mean=c["means"]["mean"] * s.store_demand_factor)
    dem, q, sold = simulate_store_days(means, c[pol["croissants"]], r_cr, weeks=weeks, common_sd=0.10)
    margin = (c["price"] * sold - c["cost"] * q).sum(axis=1)
    days_out = (dem > q).mean(axis=1)

    # the night dock: six nights a week, more trucks in a surge
    k = net["dock"]
    st = k["settings"]
    factor = s.truck_arrivals_factor if arrivals_factor is None else arrivals_factor
    nights = evaluate_option(k["rates"] * factor, lambda g: k["unload"][g.integers(len(k["unload"]))], int(st["doors"]),
                             driver_shifts(k["options"], pol["second_driver"]), 6 * weeks, seed=int(r_dk.integers(1 << 31)),
                             free=st["free_minutes_before_detention"], per_hour=st["detention_per_hour"])
    nights["week"] = np.repeat(np.arange(weeks), 6)
    w = nights.groupby("week").sum()
    driver = 6 * k["options"].cost_per_night[pol["second_driver"]]

    return pd.DataFrame({"bakery_profit": bakery, "late_deliveries": late,
                         "fleet_cost": 6 * net["fleet_cost"][pol["routes"]],
                         "croissant_margin": margin, "croissant_days_out": days_out,
                         "dock_cost": w.detention.to_numpy() + driver,
                         "trucks_within_30": (w.on_time / w.trucks).to_numpy()})
```

Each part of the network draws from its own random stream, seeded from the same number. The two policies therefore face exactly the same weeks (Day 48's common random numbers).

```python
# ---------- 50.1 The scenarios and the two policies ----------
print("--- scenarios")
print(net["scenarios"].to_string())
for name, pol in POLICIES.items():
    print(f"{name:>12}: " + "; ".join(f"{k} {v}" for k, v in pol.items()))
print("fleet cost a day: " + ", ".join(f"{k} ${v:,.2f}" for k, v in net["fleet_cost"].items()))
```

```text
--- scenarios
               demand_weeks  store_demand_factor weather  oven_capacity_left  truck_arrivals_factor
scenario
normal                  all                 1.00    draw                 1.0                   1.00
holiday_rush       december                 1.15    draw                 1.0                   1.25
snowstorm_week          all                 0.90    snow                 1.0                   1.00
oven_breakdown          all                 1.00    draw                 0.7                   1.00
dock_surge              all                 1.00    draw                 1.0                   1.40
perfect_storm      december                 1.15    snow                 0.7                   1.40
       today: routes day37; croissants average; second_driver none
 recommended: routes buffer_1.05; croissants critical_ratio; second_driver 6h_from_23
fleet cost a day: day37 $864.50, buffer_1.05 $860.22, buffer_1.30 $934.23
```

## 50.2 One week of the network

```python
# ---------- 50.2 One scenario, both policies ----------
print("--- normal week")
runs = {(s, p): stress_test(net, s, p, WEEKS, seed=50) for s in net["scenarios"].index for p in POLICIES}
summary = {p: runs[("normal", p)].agg(["mean", lambda x: x.quantile(0.05), "median", lambda x: x.quantile(0.95)])
           .set_axis(["mean", "p5", "median", "p95"]) for p in POLICIES}
print(pd.concat(summary).round(2).to_string())
```

```text
--- normal week
                    bakery_profit  late_deliveries  fleet_cost  croissant_margin  croissant_days_out  dock_cost  trucks_within_30
today       mean        105747.96             5.50     5187.00           5637.36                0.39    2189.57              0.49
            p5          102289.04             3.00     5187.00           5256.65                0.32     333.95              0.30
            median      107210.87             5.00     5187.00           5638.50                0.39    1891.61              0.48
            p95         108351.28             9.00     5187.00           5978.00                0.47    5577.10              0.67
recommended mean        105747.96             1.42     5161.32           5826.87                0.23    1228.20              0.92
            p5          102289.04             0.00     5161.32           5305.95                0.17    1200.00              0.82
            median      107210.87             1.00     5161.32           5825.00                0.22    1200.00              0.93
            p95         108351.28             4.05     5161.32           6308.00                0.29    1343.05              0.99
```

In a normal week the bakery's result is the same under both policies: the recommendations don't touch the production plan. The other three parts all improve:

- **Late deliveries**: 1.4 a week instead of 5.5.
- **Croissants**: shelves empty on 23% of store-days instead of 39%, with a higher margin.
- **The dock**: $1,228 a week instead of $2,190, and 92% of trucks within 30 minutes instead of 49%.

The recommended dock cost is also **predictable**: its 95th percentile is $1,343 a week, against $5,577 today.

## 50.3 Six scenarios

```python
# ---------- 50.3 All six scenarios ----------
print("--- stress table")


def kpis(r):
    return {"bakery_mean": r.bakery_profit.mean(), "bakery_p5": np.quantile(r.bakery_profit, 0.05),
            "late_a_week": r.late_deliveries.mean(), "late_p95": np.quantile(r.late_deliveries, 0.95),
            "days_out": r.croissant_days_out.mean(), "croissant_margin": r.croissant_margin.mean(),
            "dock_cost": r.dock_cost.mean(), "dock_p95": np.quantile(r.dock_cost, 0.95),
            "within_30": r.trucks_within_30.mean()}


table = pd.DataFrame([{"scenario": s, "policy": p, **kpis(r)} for (s, p), r in runs.items()])
print(table.round({"bakery_mean": 0, "bakery_p5": 0, "late_a_week": 2, "late_p95": 0, "days_out": 3,
                   "croissant_margin": 0, "dock_cost": 0, "dock_p95": 0, "within_30": 3})
      .to_string(index=False))

print("--- recommended minus today: croissant margin - dock cost - fleet cost, a week")
for s in net["scenarios"].index:
    a, b = runs[(s, "recommended")], runs[(s, "today")]
    gain = (a.croissant_margin - a.dock_cost - a.fleet_cost) - (b.croissant_margin - b.dock_cost - b.fleet_cost)
    m, h = paired_ci(gain, np.zeros(len(gain)))
    print(f"{s:>15}: ${m:,.2f} +/- ${h:,.2f} a week; late deliveries {a.late_deliveries.mean() - b.late_deliveries.mean():+.2f}")
```

```text
--- stress table
      scenario      policy  bakery_mean  bakery_p5  late_a_week  late_p95  days_out  croissant_margin  dock_cost  dock_p95  within_30
        normal       today     105748.0   102289.0         5.50       9.0     0.391            5637.0     2190.0    5577.0      0.487
        normal recommended     105748.0   102289.0         1.42       4.0     0.227            5827.0     1228.0    1343.0      0.921
  holiday_rush       today     107717.0   105783.0         5.50       9.0     0.526            6204.0     5546.0   11425.0      0.344
  holiday_rush recommended     107717.0   105783.0         1.42       4.0     0.346            6694.0     1330.0    1820.0      0.865
snowstorm_week       today     105748.0   102289.0        16.47      22.0     0.299            5134.0     2190.0    5577.0      0.487
snowstorm_week recommended     105748.0   102289.0        10.43      16.0     0.156            5124.0     1228.0    1343.0      0.921
oven_breakdown       today      81121.0    78411.0         5.50       9.0     0.391            5637.0     2190.0    5577.0      0.487
oven_breakdown recommended      81121.0    78411.0         1.42       4.0     0.227            5827.0     1228.0    1343.0      0.921
    dock_surge       today     105748.0   102289.0         5.50       9.0     0.391            5637.0     9090.0   17930.0      0.278
    dock_surge recommended     105748.0   102289.0         1.42       4.0     0.227            5827.0     1480.0    2393.0      0.807
 perfect_storm       today      82623.0    81176.0        16.47      22.0     0.526            6204.0     9090.0   17930.0      0.278
 perfect_storm recommended      82623.0    81176.0        10.43      16.0     0.346            6694.0     1480.0    2393.0      0.807
--- recommended minus today: croissant margin - dock cost - fleet cost, a week
         normal: $1,176.55 +/- $177.62 a week; late deliveries -4.07
   holiday_rush: $4,732.44 +/- $320.25 a week; late deliveries -4.07
 snowstorm_week: $976.94 +/- $177.34 a week; late deliveries -6.05
 oven_breakdown: $1,176.55 +/- $177.62 a week; late deliveries -4.07
     dock_surge: $7,824.45 +/- $469.86 a week; late deliveries -4.07
  perfect_storm: $8,125.52 +/- $469.89 a week; late deliveries -6.05
```

![The stress test](figures/day50_stress.png)

- **The recommendations pay in every scenario**, and the worse the week, the more they pay: $1,177 ± $178 a week in a normal week, $4,732 in a holiday rush, and **$7,800–$8,100 in a dock surge or the perfect storm**. That is a sign of a robust policy: it helps most when help is most needed.
- **The bakery's risk is the ovens**: a breakdown week costs about $24,600 whatever the policy. Day 44's maintenance contract is the answer there, not the weekly plan.
- **Snow beats the route buffer**: 10.4 late deliveries a week instead of 16.5. Better, but still twice a normal week under today's plan (5.5).
- **A dock surge breaks the 30-minute promise** (81% within 30 minutes), even though detention stays low.

## 50.4 Breaking points

A stress test should also find **where the system breaks**:

```python
# ---------- 50.4 Breaking points ----------
print("--- breaking point")
rows = []
for factor in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6):
    row = {"truck_arrivals": f"x{factor:.1f}"}
    for driver in ("6h_from_23", "all_night"):
        pol = {**POLICIES["recommended"], "second_driver": driver}
        r = stress_test(net, "normal", pol, 100, seed=51, arrivals_factor=factor)
        row[f"{driver}_within_30"] = round(r.trucks_within_30.mean(), 3)
        row[f"{driver}_cost_a_night"] = round(r.dock_cost.mean() / 6, 2)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

print("--- a snow-day plan")
for routes in ("day37", "buffer_1.05", "buffer_1.30"):
    r = stress_test(net, "snowstorm_week", {**POLICIES["recommended"], "routes": routes}, WEEKS, seed=50)
    print(f"{routes:>12}: fleet ${net['fleet_cost'][routes]:,.2f} a day; snowstorm week {r.late_deliveries.mean():.2f} late "
          f"(p95 {np.quantile(r.late_deliveries, 0.95):.0f})")
```

```text
--- breaking point
truck_arrivals  6h_from_23_within_30  6h_from_23_cost_a_night  all_night_within_30  all_night_cost_a_night
          x1.0                 0.907                   209.27                0.922                  251.77
          x1.1                 0.886                   211.59                0.907                  252.96
          x1.2                 0.871                   214.61                0.894                  255.74
          x1.3                 0.821                   237.34                0.847                  272.81
          x1.4                 0.802                   243.98                0.833                  275.94
          x1.5                 0.774                   280.94                0.808                  292.36
          x1.6                 0.704                   340.69                0.742                  337.32
--- a snow-day plan
       day37: fleet $864.50 a day; snowstorm week 16.47 late (p95 22)
 buffer_1.05: fleet $860.22 a day; snowstorm week 10.43 late (p95 16)
 buffer_1.30: fleet $934.23 a day; snowstorm week 8.50 late (p95 14)
```

- The 23:00–05:00 driver keeps the promise only at today's volume (90.7%). **10% more trucks** breaks it. A second driver all night lasts to about +10%. Beyond that, the promise needs a third driver at the peak, or **appointment slots** that move trucks out of 00:00–02:00 (Day 47's note).
- A snow-day route plan (travel times × 1.30, re-planned the evening before) cuts snow-week late deliveries only from 10.4 to 8.5, for $74 a day more. Snow is mostly beyond what a buffer can fix. Ask the cafés to widen their windows on snow days, or dispatch earlier (Day 37).

## 50.5 The memo

<details><summary>A model memo</summary>

*To: the Northbound operations committee. Re: how the network holds up in a bad week.*

"We simulated 300 weeks of the whole network (bakery, morning deliveries, store croissants, the Montreal night dock) under six scenarios, from a normal week to a holiday rush with snow, an oven breakdown and 40% more trucks at once.

**Adopt three changes now.** They cost nothing to try and pay in every scenario:

1. Plan delivery routes with 5% extra travel time: 4 fewer late deliveries a week, at the same fleet cost.
2. Send each store the croissants it sells out of only 29% of the time: 16 fewer empty-shelf days in every 100, and higher margin.
3. Add a second night forklift driver from 23:00 to 05:00 ($200 a night): it keeps the carriers' 30-minute promise at today's volume.

Together they save about **$1,200 in a normal week and $8,000 in a bad one**.

**Three risks remain:**

- **Oven breakdowns** cost about $25,000 a week each time. Sign the maintenance contract (Day 44).
- **Snow**: late deliveries rise three- to sevenfold however we plan. Agree wider windows with the cafés for snow days.
- **Growth at the dock**: 10% more night trucks breaks the 30-minute promise. Before volumes grow, introduce appointment slots for carriers arriving between midnight and 02:00, or plan a second driver all night."

</details>

## Module 5 wrap-up

| Day | Idea | Northbound decision |
|---|---|---|
| 41 | The flaw of averages | Plans look better at the forecast than they are |
| 42 | Input distributions | Trip times by weather; forecast errors |
| 43 | Monte Carlo | The production plan's and the routes' risk |
| 44 | Correlated risks, disruptions | Holiday weeks; an oven maintenance contract |
| 45 | Service vs waste | Croissants for 40 stores |
| 46 | Discrete-event simulation | A second night driver |
| 47 | Queueing formulas | Checking the dock model; a promise to carriers |
| 48 | Experiment design, CRN | A faster forklift? How sure are we? |
| 49 | Simulation-optimization | The cheapest dock that keeps the promise |
| 50 | Stress testing | The network in a bad week |

Module 6 turns these simulations into decisions made **under** uncertainty: the newsvendor in general (Day 51), stochastic programs that plan for many scenarios at once, and robust plans.

## Commit and tag

```bash
git add .
git commit -m "Day 50: capstone, stress-testing the network"
git tag v5.0-module5
git push
git push --tags
```

Then run the tests: `tests/test_experiments_and_search.py` checks Days 48–50. It covers a paired difference with no noise, the replication formula, Lindley's recursion against M/M/1 after the warm-up, two all-night drivers against Day 46, one driver before a shift starts, and the shift options. Add `"day48"`, `"day49"` and `"day50"` to the folders in `conftest.py`.

```python
"""Known answers for Days 48-50: confidence intervals, the warm-up queue, driver shifts."""
import numpy as np
import pandas as pd
import pytest

from dock import simulate_night
from experiments import lindley_runs, paired_ci, replications_needed
from queueing import mm1_wait
from shifts import simulate_night_shifts
from stress import driver_shifts


def test_paired_and_replications():
    a = np.array([10.0, 12.0, 9.0, 15.0])
    assert paired_ci(a, a - 2.0) == pytest.approx((2.0, 0.0))     # a constant difference has no noise at all
    assert replications_needed(100.0, 10.0) == 385                 # (1.96 x 100 / 10)^2 = 384.1 -> 385


def test_lindley_matches_mm1_after_warm_up():
    rng = np.random.default_rng(0)
    service = rng.exponential(10.0, 200_000)                        # exponential service, mean 10 min
    w = lindley_runs(0.05, service, 3_000, 200, np.random.default_rng(1))   # utilization 0.5
    assert w[:, 500:].mean() == pytest.approx(mm1_wait(0.05, 10.0), rel=0.05)


def test_two_drivers_all_night_is_day_46():
    rates = [1.5, 2.5, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    a = simulate_night(rates, lambda: 25.0, 3, 2, np.random.default_rng(5))
    b = simulate_night_shifts(rates, lambda: 25.0, 3, [(0, None), (0, None)], np.random.default_rng(5))
    pd.testing.assert_frame_equal(a, b)


def test_one_driver_before_the_shift():
    t = simulate_night_shifts([3.0] * 8, lambda: 30.0, 3, [(0, None), (120, 240)], np.random.default_rng(3))
    early = t[t.start < 120].sort_values("start")
    assert (early.start.to_numpy()[1:] >= early.end.to_numpy()[:-1] - 1e-9).all()   # one at a time before 00:00


def test_shift_options():
    options = pd.DataFrame({"start_after_22h_minutes": [0, 60, 0], "hours": [0, 6, 8], "cost_per_night": [0, 200, 245]},
                           index=["none", "6h_from_23", "all_night"])
    assert driver_shifts(options, "none") == [(0, None)]
    assert driver_shifts(options, "6h_from_23") == [(0, None), (60, 420)]
    assert driver_shifts(options, "all_night") == [(0, None), (0, None)]
```

```text
75 passed
```

### Day 50 checklist

- [ ] I can combine several simulations into one model of a week, with common random numbers across policies.
- [ ] I can define stress scenarios and report averages, bad tails and promises for each.
- [ ] I can find a breaking point by sweeping one stress.
- [ ] I can tell which risks a policy fixes and which need another decision.
- [ ] I can write a memo that a committee can act on.
