# DAY 64 — Bayesian Decision Analysis

Day 63 treated the winter range as one of three levels and the pilot as a "fine/short" report. Real evidence is richer. Last February a vendor lent DC-MTL a demo e-truck for 30 days, and the log says on which days it finished its route without a midday charge.

**Bayesian decision analysis** uses evidence like that directly:

- describe the uncertain quantity with a **prior distribution**;
- update it with the data by Bayes' rule, to a **posterior**;
- decide with the posterior.

With the right family of distributions, the update is one line.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 64.1 From three levels to a continuous belief |
| 0:45 – 1:45 | 64.2 Updating with the demo |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 64.3 Deciding with the posterior |
| 2:45 – 3:45 | 64.4 Learning day by day; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: was the demo winter typical? |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Days 64–67 (course assumptions, like Days 61–63):

| Table | What it holds |
|---|---|
| `ev_range_prior` | what each range level means, the winter penalty, pilot costs |
| `ev_demo_log` | the demo truck's 30 days |
| `fleet_vehicle_types` | diesel, HVO renewable diesel and electric trucks for store deliveries |
| `delivery_frequency` | stock-outs by deliveries a week |
| `dc_chargers` | how many electric trips each DC can charge |
| `network_carbon_settings` | today's plan, the 2030 target and the HVO supply limit |

## 64.1 From three levels to a continuous belief

Let $$\theta$$ be the share of winter days an e-truck finishes its route without a midday charge. The three levels mean $$\theta$$ = 0.95 (good), 0.85 (fair) and 0.65 (poor). The winter penalty grows as $$\theta$$ falls below 0.95: $75,000 per e-truck-year for each full point of $$\theta$$, so 0.10 below costs $7,500.

A **Beta distribution** is the natural belief about a share. Choose the Beta with the same mean and spread as the three-level prior (**moment matching**). Create `day64/bayes.py`:

```python
"""Day 64: Bayesian decision analysis for the e-truck winter range. theta = the share of winter days an e-truck
completes its route without a midday charge. A Beta prior, binomial data from a demo truck, a Beta posterior,
and decisions made with the posterior.

Import it:  from bayes import load_range, beta_from_levels, update, expected_penalty, electric_cost
"""
import numpy as np
from scipy import stats

from ev import pv_cost


def load_range(con):
    st = dict(con.sql("SELECT * FROM ev_range_prior").fetchall())
    levels = con.sql("SELECT level, probability, value FROM ev_uncertainties WHERE factor = 'winter_range'").fetchall()
    demo = con.sql("SELECT * FROM ev_demo_log ORDER BY demo_date").df()
    return st, levels, demo


def beta_from_levels(st, levels):
    """A Beta prior with the same mean and spread as the three-level prior (good / fair / poor -> theta values)."""
    theta = np.array([st[f"theta_{l}"] for l, _, _ in levels])
    p = np.array([q for _, q, _ in levels])
    mean, var = p @ theta, p @ (theta - p @ theta) ** 2
    total = mean * (1 - mean) / var - 1
    return mean * total, (1 - mean) * total


def update(a, b, successes, days):
    """Beta(a, b) prior + binomial data -> Beta(a + successes, b + failures) posterior."""
    return a + successes, b + (days - successes)


def penalty(theta, st):
    """Winter penalty per e-truck-year: extra rented trucks and drivers, linear below the no-penalty share."""
    return st["penalty_per_point"] * np.maximum(0.0, st["theta_no_penalty"] - theta)


def expected_penalty(a, b, st):
    """E[penalty] under Beta(a, b), in closed form: E[max(0, c - theta)] = c F(c; a, b) - a / (a + b) F(c; a + 1, b)."""
    c = st["theta_no_penalty"]
    return st["penalty_per_point"] * (c * stats.beta.cdf(c, a, b) - a / (a + b) * stats.beta.cdf(c, a + 1, b))


def electric_cost(s, options, exp_pen):
    """Expected PV cost of the all-electric fleet: linear in the penalty, so the expected penalty is enough."""
    return pv_cost(s, options.loc["electric_all"], exp_pen, 0.0)
```

Then `day64/run_bayes.py`:

```python
# ---------- 64.1 From three levels to a continuous belief ----------
print("--- prior")
a0, b0 = beta_from_levels(st, range_levels)
print(f"three levels: theta good {st['theta_good']}, fair {st['theta_fair']}, poor {st['theta_poor']} "
      f"(p = {', '.join(f'{p:.2f}' for _, p, _ in range_levels)})")
print(f"Beta prior with the same mean and spread: a = {a0:.2f}, b = {b0:.2f}; mean {a0 / (a0 + b0):.3f}, "
      f"90% interval {stats.beta.ppf(0.05, a0, b0):.3f} to {stats.beta.ppf(0.95, a0, b0):.3f}")
pen0 = expected_penalty(a0, b0, st)
print(f"expected winter penalty ${pen0:,.0f} a truck a year (three levels: "
      f"${sum(p * v for _, p, v in range_levels):,.0f})")
```

```text
--- prior
three levels: theta good 0.95, fair 0.85, poor 0.65 (p = 0.45, 0.35, 0.20)
Beta prior with the same mean and spread: a = 7.64, b = 1.30; mean 0.855, 90% interval 0.634 to 0.985
expected winter penalty $7,488 a truck a year (three levels: $7,025)
```

The Beta prior has the same mean (0.855) and spread, but it is smooth: every $$\theta$$ is possible, not just three. Its expected penalty ($7,488) is close to the three levels' ($7,025). The small gap comes from the shape between and beyond the three points.

## 64.2 Updating with the demo

The Beta is **conjugate** to binomial data. With a Beta($$a$$, $$b$$) prior and $$k$$ successes in $$n$$ days, the posterior is Beta($$a + k$$, $$b + n - k$$). The prior acts like $$a$$ successes and $$b$$ failures already seen.

```python
# ---------- 64.2 Updating with the demo ----------
print("--- posterior")
k, n = int(demo.completed_without_charge.sum()), len(demo)
a1, b1 = update(a0, b0, k, n)
print(f"demo, {demo.demo_date.iloc[0].date()} to {demo.demo_date.iloc[-1].date()}: {k} of {n} days completed without a midday charge")
for name, (a, b) in [("prior", (a0, b0)), ("posterior", (a1, b1))]:
    print(f"{name:>9}: Beta({a:.2f}, {b:.2f}), mean {a / (a + b):.3f}, 90% interval "
          f"{stats.beta.ppf(0.05, a, b):.3f}-{stats.beta.ppf(0.95, a, b):.3f}, P(theta < {st['theta_fair']}) "
          f"{stats.beta.cdf(st['theta_fair'], a, b):.3f}")
x = np.linspace(0.3, 1, 400)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(x, stats.beta.pdf(x, a0, b0), color="#9aa4b2", label="prior (from the three levels)")
ax.plot(x, stats.beta.pdf(x, a1, b1), color="#3b82f6", label=f"posterior (after {k} of {n} demo days)")
ax.axvline(k / n, ls="--", color="#e5484d", lw=1)
ax.text(k / n + 0.005, ax.get_ylim()[1] * 0.9, "demo rate", color="#e5484d", fontsize=8)
ax.set(xlabel="theta: share of winter days without a midday charge", ylabel="density",
       title="What the demo taught us about winter range")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("figures/day64_beta.png", dpi=150)
print("saved figures/day64_beta.png")
```

```text
--- posterior
demo, 2026-02-02 to 2026-03-03: 25 of 30 days completed without a midday charge
    prior: Beta(7.64, 1.30), mean 0.855, 90% interval 0.634-0.985, P(theta < 0.85) 0.394
posterior: Beta(32.64, 6.30), mean 0.838, 90% interval 0.733-0.923, P(theta < 0.85) 0.541
saved figures/day64_beta.png
```

![Prior and posterior](figures/day64_beta.png)

The demo finished 25 of 30 days (83%), slightly below the prior's mean. The posterior moves down a little, to 0.838, and becomes **much narrower**: its 90% interval shrinks from 0.63–0.99 to 0.73–0.92. The prior was worth about 9 days of evidence ($$a + b$$ = 8.9). The demo's 30 days now dominate.

## 64.3 Deciding with the posterior

The all-electric cost is linear in the penalty, so its expected value needs only the expected penalty. For a Beta, that has a closed form (in `expected_penalty`).

```python
# ---------- 64.3 Deciding with the posterior ----------
print("--- decision")
for name, (a, b) in [("prior", (a0, b0)), ("posterior", (a1, b1))]:
    pen = expected_penalty(a, b, st)
    e = electric_cost(s, options, pen)
    print(f"{name:>9}: expected penalty ${pen:,.0f}; all-electric ${e:,.0f} vs diesel ${diesel:,.0f} -> "
          f"{'electric' if e < diesel else 'diesel'} (by ${abs(e - diesel):,.0f})")
draws = stats.beta.rvs(a1, b1, size=200_000, random_state=np.random.default_rng(64))
cost_draws = np.array([electric_cost(s, options, 0.0)]) + (electric_cost(s, options, 1.0) - electric_cost(s, options, 0.0)) * penalty(draws, st)
print(f"posterior: P(all-electric costs less than diesel's expected cost) {np.mean(cost_draws < diesel):.3f}")
```

```text
--- decision
    prior: expected penalty $7,488; all-electric $3,699,228 vs diesel $3,607,813 -> diesel (by $91,415)
posterior: expected penalty $8,381; all-electric $3,734,891 vs diesel $3,607,813 -> diesel (by $127,079)
posterior: P(all-electric costs less than diesel's expected cost) 0.250
```

- The demo makes electric look **worse**, by $127,000 instead of $91,000. Diesel stays the choice.
- There is still a **25% chance** that the all-electric fleet would cost less than diesel's expected cost: the posterior still allows a good range. Uncertainty remains, but on the evidence so far it doesn't favour electric.

## 64.4 Learning day by day

Bayesian updating is **sequential**: today's posterior is tomorrow's prior, and updating day by day ends where updating all at once does.

```python
# ---------- 64.4 Learning day by day ----------
print("--- day by day")
rows = []
a, b = a0, b0
for i, r in demo.iterrows():
    a, b = update(a, b, r.completed_without_charge, 1)
    gap = electric_cost(s, options, expected_penalty(a, b, st)) - diesel
    if (i + 1) % 5 == 0 or i == 0:
        rows.append((i + 1, int(demo.completed_without_charge.iloc[: i + 1].sum()), round(a / (a + b), 3), round(gap)))
print(pd.DataFrame(rows, columns=["days", "successes", "posterior_mean", "electric_minus_diesel"]).to_string(index=False))
```

```text
--- day by day
 days  successes  posterior_mean  electric_minus_diesel
    1          1           0.870                  49864
    5          4           0.835                 140180
   10          9           0.879                  13829
   15         11           0.779                 305292
   20         15           0.782                 294258
   25         20           0.814                 198320
   30         25           0.838                 127079
```

After 10 days (9 successes) electric was within $14,000 of diesel. A cold spell in the next five days pushed it $305,000 behind, and the later days brought it back to $127,000. **Early evidence swings decisions.** A decision rule that stopped the demo after day 10 would have read too much into a mild start. This is the risk of peeking at an experiment, and Day 65 plans the pilot's length in advance.

> **Decision it supports:** decisions that should move with the evidence: how much a test, a demo or a trial should change the plan.
>
> **Other decisions you can support the same way:** A/B tests with prior knowledge · equipment reliability from failure logs · a new supplier's on-time rate after a few deliveries · clinical trials.

### The planted bug

`day64/bug_update.py` updates the range belief with the demo. Run it.

```python
k, n = int(demo.completed_without_charge.sum()), len(demo)
a, b = a + k, b + n                                              # "add the successes and the days"
print(f"posterior Beta({a:.2f}, {b:.2f}): mean {a / (a + b):.3f} (the demo's own rate: {k / n:.3f})")
```

<details><summary>What happens, and why</summary>

```text
posterior Beta(32.64, 31.30): mean 0.511 (the demo's own rate: 0.833)
```

The posterior mean is 0.51, far below both the prior (0.855) and the demo's own rate (0.833). That is impossible for an honest update: a posterior mean always lies between the prior mean and the data's rate. The code adds all 30 **days** to $$b$$, counting the 25 successes as failures too. $$b$$ must grow by the failures only, $$n - k$$. Check every update against that rule.

</details>

## Mini-project: was the demo winter typical?

Half the demo's days were below −10 °C; in a typical Montreal winter 55% are (`ev_range_prior`). Do the following:

- Split the demo by temperature.
- Update a belief for cold days and one for mild days, each from the same prior.
- Combine them by simulation for the demo's mix and for a typical winter.

Does the decision change?

<details><summary>Solution</summary>

```python
"""Day 64 mini-project: is the demo representative? Half its days were below -10 C; in a typical Montreal winter
55% are. Split the demo by temperature, update a belief for cold days and for mild days separately (each starting
from the same Beta prior), and combine them for a typical winter by simulation. Does the decision change?

Run from the repository root (after day06/load_tables.py):  python day64/mini_project_cold.py
"""
import sys

import duckdb
import numpy as np
from scipy import stats

for folder in ("day61", "day64"):
    sys.path.insert(0, folder)
from bayes import beta_from_levels, electric_cost, load_range, penalty, update
from ev import ev_tree, load_ev
from trees import rollback

con = duckdb.connect("northbound.duckdb", read_only=True)
s, options, levels = load_ev(con)
st, range_levels, demo = load_range(con)
diesel = -rollback(ev_tree(s, options, levels)["options"]["diesel_renew"])
a0, b0 = beta_from_levels(st, range_levels)
cold = demo.min_temp_c < st["cold_day_below_c"]
rng = np.random.default_rng(640)
theta = {}
for name, days in (("cold", demo[cold]), ("mild", demo[~cold])):
    a, b = update(a0, b0, int(days.completed_without_charge.sum()), len(days))
    theta[name] = stats.beta.rvs(a, b, size=200_000, random_state=rng)
    print(f"{name} days: {int(days.completed_without_charge.sum())} of {len(days)} -> Beta({a:.2f}, {b:.2f}), mean {a / (a + b):.3f}")
for label, share in (("the demo's mix", cold.mean()), ("a typical winter", st["typical_cold_share"])):
    mix = share * theta["cold"] + (1 - share) * theta["mild"]
    pen = penalty(mix, st).mean()
    e = electric_cost(s, options, pen)
    print(f"{label} ({share:.0%} cold days): theta {mix.mean():.3f}, expected penalty ${pen:,.0f}, "
          f"all-electric ${e:,.0f} vs diesel ${diesel:,.0f}")
```

```text
cold days: 11 of 15 -> Beta(18.64, 5.30), mean 0.779
mild days: 14 of 15 -> Beta(21.64, 2.30), mean 0.904
the demo's mix (50% cold days): theta 0.841, expected penalty $8,141, all-electric $3,725,298 vs diesel $3,607,813
a typical winter (55% cold days): theta 0.835, expected penalty $8,610, all-electric $3,744,023 vs diesel $3,607,813
```

- Cold days are clearly worse: about 78% without a charge, against 90% on mild days.
- Reweighted to a typical winter, $$\theta$$ falls slightly (0.835) and electric falls further behind diesel ($136,000).
- The decision holds, and **the demo was if anything flattering**.
- One caution: starting both groups from the same prior counts the prior twice. With 15 days each the data dominate, so it matters little here; with few days it would.

</details>

## Commit

```bash
git add .
git commit -m "Day 64: Bayesian decision analysis"
git push
```

### Day 64 checklist

- [ ] I can turn a discrete prior into a Beta prior by matching its mean and spread.
- [ ] I can update a Beta prior with binomial data, and check the posterior lies between prior and data.
- [ ] I can decide with a posterior, and state the probability that the other option was better.
- [ ] I understand sequential updating and why early evidence swings decisions.
- [ ] I check whether the data's conditions (here, temperature) match the decision's.

---

# DAY 65 — Mini-project: Piloting Electric Trucks

The fleet decision has come back three times. Diesel is cheaper on average, but not by much, and a good winter range would reverse it. The obvious next step is a **winter pilot** before the 2027 lease. Today's mini-project designs one:

- How many trucks?
- What is it worth?
- Is there a better way to spend the money?

| Time | Block |
|---|---|
| 0:00 – 0:30 | 65.1 The brief: where we start |
| 0:30 – 1:30 | 65.2 The value of a pilot, by size |
| 1:30 – 1:45 | Break |
| 1:45 – 2:45 | 65.3 Days of one winter are not independent |
| 2:45 – 3:30 | 65.4 Information or negotiation? |
| 3:30 – 3:45 | Break |
| 3:45 – 4:45 | 65.5 The memo |
| 4:45 – 5:00 | Commit with Git + checklist |

## 65.1 The brief: where we start

A pilot runs $$n$$ e-trucks through a 90-day winter and costs $30,000 a truck (`ev_range_prior`). Before it runs, we can predict what it will show: the number of good days follows a **Beta-binomial** distribution, the binomial averaged over our current Beta belief. Then:

- For each possible result, update the belief and decide.
- Weight the decisions by how likely each result is.
- The improvement over deciding now is the EVSI (Day 63), now for a continuous belief.

Create `day65/pilot.py`:

```python
"""Day 65 mini-project: designing the e-truck winter pilot. The value of a pilot of n truck-days (EVSI), from a
Beta belief about theta, with Beta-binomial predictions of what the pilot will show.

Import it:  from pilot import pilot_value
"""
import numpy as np
from scipy import stats

from bayes import electric_cost, expected_penalty


def pilot_value(a, b, truck_days, st, s, options, diesel, rho=0.0):
    """Expected value of a pilot's information, in dollars of expected PV cost saved.
    rho: correlation between days of the same winter; it shrinks the pilot to an effective sample of
    n / (1 + (n - 1) rho) independent days (a design effect). Returns EVSI and the choice after each result."""
    n_eff = int(round(truck_days / (1 + (truck_days - 1) * rho)))
    now = min(diesel, electric_cost(s, options, expected_penalty(a, b, st)))
    k = np.arange(n_eff + 1)
    prob = stats.betabinom.pmf(k, n_eff, a, b)                  # what the pilot may show, before it runs
    after = np.array([min(diesel, electric_cost(s, options, expected_penalty(a + j, b + n_eff - j, st))) for j in k])
    go_electric = [j for j in k if electric_cost(s, options, expected_penalty(a + j, b + n_eff - j, st)) < diesel]
    return now - prob @ after, n_eff, (min(go_electric) if go_electric else None), prob[go_electric].sum()
```

Then `day65/run_pilot.py`:

```python
# ---------- 65.1 Where we start ----------
print("--- start")
a, b = update(*beta_from_levels(st, range_levels), int(demo.completed_without_charge.sum()), len(demo))
e_now = electric_cost(s, options, expected_penalty(a, b, st))
print(f"belief after the demo: Beta({a:.2f}, {b:.2f}), mean {a / (a + b):.3f}")
print(f"decide now: diesel ${diesel:,.0f}, all-electric ${e_now:,.0f} -> diesel")
theta = np.random.default_rng(65).beta(a, b, 200_000)               # if we knew theta exactly ...
perfect = np.mean(np.minimum(diesel, electric_cost(s, options, penalty(theta, st))))
print(f"value of knowing theta exactly (EVPI): ${diesel - perfect:,.0f}")
```

```text
--- start
belief after the demo: Beta(32.64, 6.30), mean 0.838
decide now: diesel $3,607,813, all-electric $3,734,891 -> diesel
value of knowing theta exactly (EVPI): $19,509
```

After the demo, the value of knowing $$\theta$$ exactly is only **$19,509**. This is the ceiling for any pilot. It is much lower than Day 63's $93,404, because the demo already told us a lot and what it told us leans towards diesel.

## 65.2 The value of a pilot, by size

```python
# ---------- 65.2 The value of a pilot, by size ----------
print("--- pilot size")
days, per_truck = int(st["pilot_days"]), st["pilot_cost_per_truck"]
rows = []
for trucks in (1, 2, 3, 4, 6, 8):
    v, n_eff, threshold, p_go = pilot_value(a, b, trucks * days, st, s, options, diesel)
    rows.append((trucks, trucks * days, round(v), trucks * per_truck, round(v - trucks * per_truck),
                 f"{threshold}/{n_eff}" if threshold is not None else "-", round(p_go, 3)))
size = pd.DataFrame(rows, columns=["trucks", "truck_days", "evsi", "cost", "net", "go_electric_if_at_least", "p_go_electric"])
print(size.to_string(index=False))
```

```text
--- pilot size
 trucks  truck_days  evsi     cost     net go_electric_if_at_least  p_go_electric
      1          90 11372  30000.0  -18628                   81/90          0.222
      2         180 14747  60000.0  -45253                 161/180          0.213
      3         270 16130  90000.0  -73870                 240/270          0.230
      4         360 16871 120000.0 -103129                 319/360          0.239
      6         540 17660 180000.0 -162340                 478/540          0.238
      8         720 18075 240000.0 -221925                 636/720          0.245
```

- A one-truck pilot is worth $11,400 and costs $30,000. Every larger pilot is worth a little more (up to $18,000) and costs much more. **No pilot size pays.**
- The decision rule is clear in advance: with two trucks, go electric only if at least 161 of 180 truck-days (89%) need no midday charge. That happens with probability 21%.

## 65.3 Days of one winter are not independent

The pilot's days share one winter. If that winter is mild, every truck does well. Treating 180 truck-days as 180 independent observations overstates what the pilot can teach. A simple correction is the **design effect**: with correlation $$\rho$$ between days, $$n$$ days are worth $$n / (1 + (n-1)\rho)$$ independent ones.

```python
# ---------- 65.3 Days of one winter are not independent ----------
print("--- correlation")
rows = []
for rho in (0.0, 0.005, 0.02, 0.05):
    vals = [round(pilot_value(a, b, t * days, st, s, options, diesel, rho=rho)[0]) for t in (1, 2, 4, 8)]
    n_eff = [pilot_value(a, b, t * days, st, s, options, diesel, rho=rho)[1] for t in (1, 2, 4, 8)]
    rows.append((rho, *vals, n_eff[-1]))
print(pd.DataFrame(rows, columns=["rho", "evsi_1_truck", "evsi_2", "evsi_4", "evsi_8", "effective_days_8"]).to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 3.6))
trucks = np.arange(1, 9)
for rho, colour in [(0.0, "#3b82f6"), (0.02, "#22a06b"), (0.05, "#f59e0b")]:
    ax.plot(trucks, [pilot_value(a, b, t * days, st, s, options, diesel, rho=rho)[0] / 1000 for t in trucks], "o-",
            color=colour, label=f"value of the pilot, day correlation {rho}")
ax.plot(trucks, trucks * per_truck / 1000, "--", color="#e5484d", label="cost of the pilot")
ax.axhline((diesel - perfect) / 1000, ls=":", color="#6b7280", label="value of perfect information")
ax.set(xlabel="e-trucks in the winter pilot", ylabel="$ thousands", title="How big a pilot is worth running?")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("figures/day65_pilot.png", dpi=150)
print("saved figures/day65_pilot.png")
```

```text
--- correlation
  rho  evsi_1_truck  evsi_2  evsi_4  evsi_8  effective_days_8
0.000         11372   14747   16871   18075               720
0.005          9179   11757   13320   14173               157
0.020          4769    5806    7018    7158                47
0.050          1602    1854    2020    2020                19
saved figures/day65_pilot.png
```

![The value of a pilot](figures/day65_pilot.png)

With even a small correlation (0.02), eight trucks for a whole winter are worth only 47 independent days, and the pilot's value falls to $7,000. **More trucks in the same winter add little.** Only more winters (or more places) would. A pilot's design matters as much as its size.

## 65.4 Information or negotiation?

Day 61's tornado said the electric **lease price** matters most. Information about the range is worth little; what about a better price?

```python
# ---------- 65.4 Information or negotiation? ----------
print("--- lease")
rows = []
for lease in (58_000, 57_000, 56_000, 55_000, 54_000, 52_000):
    s2 = {**s, "electric_lease_per_year": lease}
    d2 = -rollback(ev_tree(s2, options, levels)["options"]["diesel_renew"])
    e2 = electric_cost(s2, options, expected_penalty(a, b, st))
    v2 = pilot_value(a, b, 2 * days, st, s2, options, d2, rho=0.02)[0]
    rows.append((lease, round(e2 - d2), "electric" if e2 < d2 else "diesel", round(v2), round(v2 - 2 * per_truck)))
print(pd.DataFrame(rows, columns=["electric_lease", "electric_minus_diesel", "decide_now", "evsi_2_trucks_rho_0.02",
                                  "net_of_pilot"]).to_string(index=False))
```

```text
--- lease
 electric_lease  electric_minus_diesel decide_now  evsi_2_trucks_rho_0.02  net_of_pilot
          58000                 127079     diesel                    5806        -54194
          57000                  87151     diesel                   13919        -46081
          56000                  47224     diesel                   26917        -33083
          55000                   7297     diesel                   45138        -14862
          54000                 -32630   electric                   35756        -24244
          52000                -112484   electric                   14922        -45078
```

- At $54,000 a truck a year, electric is the better choice now, without any pilot.
- At $55,000 the decision is close (a $7,000 gap) and information is worth most ($45,000), but a two-truck pilot still doesn't pay for itself.
- **Negotiating the lease is worth far more than piloting.** Each $1,000 off the lease moves electric about $40,000 closer in PV.

## 65.5 The memo

<details><summary>A model memo</summary>

*To: the VP Logistics. Re: electric trucks for DC-MTL in 2027.*

"**Recommendation: renew diesel for 2027, don't run a winter pilot, and negotiate the electric lease.**

1. On today's quote ($58,000 a truck a year), ten electric trucks would cost about $127,000 more than diesel over the five-year lease (expected present value). The vendor's February demo (25 of 30 days without a midday charge, and worse on cold days) made electric look slightly worse, not better.
2. **A pilot isn't worth it.** Even perfect knowledge of winter range would be worth only about $20,000 now, and a two-truck winter pilot about $6,000–15,000, against a cost of $60,000. Trucks piloted in one winter share its weather, so adding trucks adds little.
3. **The lease price decides it.** At $54,000 or less, electric is the better choice now. Ask the vendor for $54,000. If they agree, switch to electric in 2027 and put the pilot money towards chargers.
4. Revisit in 2029, when the lease and range data from other fleets will have moved."

</details>

## Commit

```bash
git add .
git commit -m "Day 65: e-truck pilot mini-project"
git push
```

### Day 65 checklist

- [ ] I can predict a test's results with a Beta-binomial and compute its EVSI.
- [ ] I can find a pilot's decision rule in advance (go only if at least k of n).
- [ ] I correct for correlated observations with a design effect.
- [ ] I compare buying information with other levers (negotiation, flexibility).
- [ ] I can write a recommendation that states what would change it.

---

# DAY 66 — Multi-Objective Optimization

Northbound's board has committed to cutting delivery CO2 by 40% by 2030. Cost still matters, and so does service: stores want frequent deliveries. **These goals conflict**: the cheapest plan isn't the greenest, and the greenest isn't the best for stores.

**Multi-objective optimization** doesn't pretend there is one answer. It finds the **Pareto-optimal** plans, those that can't improve one goal without hurting another, and lets people choose among them. Today you build the delivery model with three objectives and trace the cost–CO2 trade-off with three methods: weighted sums, ε-constraints and lexicographic optimization.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 66.1 Three objectives, one at a time |
| 0:45 – 1:45 | 66.2 Weighted sums: an internal carbon price |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 66.3 Epsilon-constraint: the whole frontier |
| 2:45 – 3:45 | 66.4 Lexicographic; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: a cap or a carbon price? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 66.1 Three objectives, one at a time

For each of the 40 stores, choose:

- a **vehicle type**:
  - diesel;
  - **HVO** (renewable diesel: 80% less CO2, 20% dearer per km; its supply covers at most 35% of the km);
  - **electric** (within a 240 km winter round trip, limited by each DC's chargers).
- **deliveries a week**: 3 to 6. Fewer deliveries mean fuller trucks and fewer trips, but more stock-outs.

The three objectives:

- **cost**: dollars a week;
- **CO2**: tonnes a week;
- **service**: the volume-weighted stock-out rate.

Create `day66/delivery_mo.py`:

```python
"""Days 66-67: a multi-objective plan for the store deliveries. For each store, choose a vehicle type (diesel,
HVO renewable diesel (its supply covers at most a share of the km), electric (within range, limited by each DC's
chargers)) and a number of deliveries a week. Three objectives: weekly cost,
weekly CO2 and service (the volume-weighted stock-out rate, which falls with more frequent deliveries).

Import it:  from delivery_mo import load_delivery, build_delivery, evaluate, today_plan
"""
import math

import pandas as pd
import pyomo.environ as pyo


def load_delivery(con):
    stores = con.sql("""SELECT s.store_id, s.dc_id, l.pallets_per_week, d.km
                        FROM stores s JOIN store_logistics l USING (store_id)
                        JOIN dc_store_lanes d ON d.dc_id = s.dc_id AND d.store_id = s.store_id
                        ORDER BY s.store_id""").df().set_index("store_id")
    vehicles = con.sql("SELECT * FROM fleet_vehicle_types").df().set_index("vehicle")
    freq = dict(con.sql("SELECT * FROM delivery_frequency ORDER BY deliveries_per_week").fetchall())
    chargers = dict(con.sql("SELECT * FROM dc_chargers").fetchall())
    st = dict(con.sql("SELECT * FROM network_carbon_settings").fetchall())
    cap = int(st["pallets_per_truck"])
    trips = {(s, f): f * math.ceil(r.pallets_per_week / (cap * f)) for s, r in stores.iterrows() for f in freq}
    return {"stores": stores, "vehicles": vehicles, "freq": freq, "chargers": chargers, "settings": st, "trips": trips}


def options(d):
    """(store, vehicle, frequency) choices allowed: electric only if a round trip fits its winter range."""
    return [(s, v, f) for s, r in d["stores"].iterrows() for v, veh in d["vehicles"].iterrows() for f in d["freq"]
            if 2 * r.km <= veh.max_round_trip_km]


def build_delivery(d):
    """The model with three objective expressions (cost $ a week, co2 tonnes a week, service: stock-out %) and
    no objective yet: the caller chooses one, or a weighted sum, and adds caps (epsilon constraints)."""
    m = pyo.ConcreteModel(name="delivery_plan")
    st, veh = d["stores"], d["vehicles"]
    m.K = pyo.Set(initialize=options(d), dimen=3)
    m.y = pyo.Var(m.K, within=pyo.Binary)
    m.one_choice = pyo.Constraint(st.index.tolist(), rule=lambda m, s: sum(m.y[k] for k in m.K if k[0] == s) == 1)
    m.chargers = pyo.Constraint(list(d["chargers"]), rule=lambda m, dc: sum(
        d["trips"][s, f] * m.y[s, v, f] for s, v, f in m.K if v == "electric" and st.dc_id[s] == dc) <= d["chargers"][dc])
    round_km = lambda s: 2 * st.km[s]
    km = lambda pick: sum(d["trips"][s, f] * round_km(s) * m.y[s, v, f] for s, v, f in m.K if pick(v))
    m.hvo_supply = pyo.Constraint(expr=km(lambda v: v == "hvo") <= float(d["settings"]["hvo_max_km_share"]) * km(lambda v: True))
    m.cost = pyo.Expression(expr=sum(d["trips"][s, f] * (veh.fixed_per_trip[v] + round_km(s) * veh.cost_per_km[v]) * m.y[s, v, f]
                                     for s, v, f in m.K))
    m.co2 = pyo.Expression(expr=sum(d["trips"][s, f] * round_km(s) * veh.co2_kg_per_km[v] / 1000 * m.y[s, v, f]
                                    for s, v, f in m.K))
    m.service = pyo.Expression(expr=100 * sum(st.pallets_per_week[s] * d["freq"][f] * m.y[s, v, f] for s, v, f in m.K)
                               / st.pallets_per_week.sum())
    return m


def chosen(m):
    return [k for k in m.K if pyo.value(m.y[k]) > 0.5]


def evaluate(d, plan):
    """Cost, CO2 and service of a plan given as (store, vehicle, frequency) choices."""
    st, veh = d["stores"], d["vehicles"]
    cost = sum(d["trips"][s, f] * (veh.fixed_per_trip[v] + 2 * st.km[s] * veh.cost_per_km[v]) for s, v, f in plan)
    co2 = sum(d["trips"][s, f] * 2 * st.km[s] * veh.co2_kg_per_km[v] / 1000 for s, v, f in plan)
    service = 100 * sum(st.pallets_per_week[s] * d["freq"][f] for s, v, f in plan) / st.pallets_per_week.sum()
    return {"cost": cost, "co2": co2, "service": service}


def today_plan(d):
    f, v = int(d["settings"]["current_frequency"]), d["settings"]["current_vehicle"]
    return [(s, v, f) for s in d["stores"].index]


def mix(m):
    """How many stores use each vehicle, and the deliveries a week they get."""
    c = pd.DataFrame(chosen(m), columns=["store", "vehicle", "freq"])
    return ", ".join(f"{v} {n}" for v, n in c.vehicle.value_counts().sort_index().items()) + \
        "; deliveries " + ", ".join(f"{f}x {n}" for f, n in c.freq.value_counts().sort_index().items())
```

Then `day66/run_multi.py`:

```python
# ---------- 66.1 Three objectives, one at a time ----------
print("--- objectives")
today = evaluate(d, today_plan(d))
print(f"today (diesel, 5 deliveries a week): cost ${today['cost']:,.0f} a week, CO2 {today['co2']:.2f} t, "
      f"stock-outs {today['service']:.2f}%")
in_range = sum(2 * r.km <= d["vehicles"].max_round_trip_km["electric"] for _, r in d["stores"].iterrows())
print(f"stores within an e-truck's winter range: {in_range} of {len(d['stores'])}")


def solve_with(objective, service_cap=None, co2_cap=None, sense=pyo.minimize):
    m = build_delivery(d)
    if service_cap is not None:
        m.keep_service = pyo.Constraint(expr=m.service <= service_cap + 1e-9)
    if co2_cap is not None:
        m.co2_cap = pyo.Constraint(expr=m.co2 <= co2_cap)
    m.obj = pyo.Objective(expr=objective(m), sense=sense)
    solve(m)
    return m


rows = []
for name, obj, cap in [("min cost", lambda m: m.cost, None), ("min cost, today's service", lambda m: m.cost, today["service"]),
                       ("min CO2", lambda m: m.co2, None), ("min CO2, today's service", lambda m: m.co2, today["service"]),
                       ("min stock-outs", lambda m: m.service, None)]:
    m = solve_with(obj, service_cap=cap)
    rows.append((name, round(pyo.value(m.cost)), round(pyo.value(m.co2), 2), round(pyo.value(m.service), 2), mix(m)))
print(pd.DataFrame(rows, columns=["objective", "cost", "co2_t", "stockout_pct", "plan"]).to_string(index=False))
```

```text
--- objectives
today (diesel, 5 deliveries a week): cost $59,914 a week, CO2 20.65 t, stock-outs 2.60%
stores within an e-truck's winter range: 33 of 40
                objective  cost  co2_t  stockout_pct                                                              plan
                 min cost 47722  16.50          3.40                          diesel 40; deliveries 3x 17, 4x 7, 5x 16
min cost, today's service 57401  18.17          2.60                    diesel 40; deliveries 3x 2, 4x 6, 5x 20, 6x 12
                  min CO2 68440   7.55          3.32 diesel 5, electric 28, hvo 7; deliveries 3x 14, 4x 8, 5x 15, 6x 3
 min CO2, today's service 73061   7.71          2.60 diesel 6, electric 25, hvo 9; deliveries 3x 3, 4x 6, 5x 14, 6x 17
           min stock-outs 71897  24.78          2.20                                       diesel 40; deliveries 6x 40
```

Each objective alone gives a different plan:

- **min cost** delivers 3 times a week to many stores (stock-outs 3.4%);
- **min CO2** goes electric and HVO and also delivers less often;
- **min stock-outs** delivers 6 times a week to every store, at the highest CO2.

Two findings before any trade-off:

- Today's plan (diesel, 5 a week) is **not efficient**: at the same service, a mix of frequencies costs $2,500 a week less and emits less.
- Optimizing CO2 without a service limit quietly cuts deliveries. Day 67 brings service in as a full objective; today it stays at today's level.

## 66.2 Weighted sums: an internal carbon price

The simplest combination is a weighted sum. With cost in dollars and CO2 in tonnes, the weight is a **carbon price**: minimize $$\text{cost} + p \times \text{CO2}$$.

```python
# ---------- 66.2 Weighted sums: an internal carbon price ----------
print("--- weighted sum")
S = today["service"]
rows, weighted = [], []
for price in (0, 50, 100, 200, 300, 500, 1_000, 3_000):
    m = solve_with(lambda m: m.cost + price * m.co2, service_cap=S)
    c, e = pyo.value(m.cost), pyo.value(m.co2)
    weighted.append((c, e))
    rows.append((price, round(c), round(e, 2), mix(m)))
print(pd.DataFrame(rows, columns=["carbon_price_per_t", "cost", "co2_t", "plan"]).to_string(index=False))
```

```text
--- weighted sum
 carbon_price_per_t  cost  co2_t                                                               plan
                  0 57401  18.17                     diesel 40; deliveries 3x 2, 4x 6, 5x 20, 6x 12
                 50 57406  18.18                     diesel 40; deliveries 3x 2, 4x 5, 5x 22, 6x 11
                100 57402  18.17                     diesel 40; deliveries 3x 2, 4x 7, 5x 18, 6x 13
                200 57401  18.17                     diesel 40; deliveries 3x 2, 4x 6, 5x 20, 6x 12
                300 58875  13.08             diesel 24, hvo 16; deliveries 3x 2, 4x 7, 5x 18, 6x 13
                500 58927  12.92             diesel 26, hvo 14; deliveries 3x 3, 4x 4, 5x 19, 6x 14
               1000 59549  11.93 diesel 27, electric 2, hvo 11; deliveries 3x 3, 4x 4, 5x 19, 6x 14
               3000 65272   8.74 diesel 7, electric 14, hvo 19; deliveries 3x 3, 4x 5, 5x 17, 6x 15
```

Below about $289 a tonne nothing changes. Above it, HVO takes over up to its supply limit. Electric needs over $1,000 a tonne, and the last tonnes $3,000. A weighted sum has a clear interpretation, but it **jumps**: prices between $200 and $300 move straight from 18.2 to 13.1 tonnes, and you can't aim at a CO2 level in between.

## 66.3 Epsilon-constraint: the whole frontier

The **ε-constraint** method minimizes one objective while capping the others: minimize cost subject to CO2 ≤ ε, for a sequence of caps. It reaches every Pareto point, including those no weight can find.

```python
# ---------- 66.3 Epsilon-constraint: the whole frontier ----------
print("--- epsilon")
lo = pyo.value(solve_with(lambda m: m.co2, service_cap=S).co2)
hi = pyo.value(solve_with(lambda m: m.cost, service_cap=S).co2)
front = []
for cap in np.linspace(hi, lo, 13):
    m = solve_with(lambda m: m.cost, service_cap=S, co2_cap=cap + 1e-6)
    front.append((cap, pyo.value(m.cost), pyo.value(m.co2)))
f = pd.DataFrame(front, columns=["co2_cap", "cost", "co2"]).round(2)


def supported(i, pts):
    """Is there a carbon price at which point i is the cheapest of all the points? (cost + price x CO2 smallest)
    Each other point j gives a condition on the price; i is supported if all conditions can hold together."""
    lo_p, hi_p = 0.0, np.inf
    ci, ei = pts[i]
    for j, (cj, ej) in enumerate(pts):
        if j == i:
            continue
        if ej < ei:                                                   # j greener: price must stay low enough
            hi_p = min(hi_p, (cj - ci) / (ei - ej))
        elif ej > ei:                                                 # j dirtier: price must be high enough
            lo_p = max(lo_p, (ci - cj) / (ej - ei))
        elif cj < ci:
            return False, None
    return lo_p <= hi_p + 1.0, (lo_p, hi_p)                          # $1 a tonne: points on a straight stretch tie


pts = list(zip(f.cost, f.co2))
f["weight_can_find_it"] = [supported(i, pts)[0] for i in range(len(pts))]
f["price_range"] = [("-" if not ok else f"{r[0]:,.0f}+" if not np.isfinite(r[1]) else f"about {r[1]:,.0f}"
                     if r[0] > r[1] else f"{r[0]:,.0f}-{r[1]:,.0f}") for ok, r in (supported(i, pts) for i in range(len(pts)))]
print(f.to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(f.co2, f.cost, "o-", color="#3b82f6", label="epsilon-constraint frontier")
w = np.array(weighted)
ax.scatter(w[:, 1], w[:, 0], s=80, facecolors="none", edgecolors="#e5484d", label="weighted sums (carbon prices)")
ax.scatter([today["co2"]], [today["cost"]], marker="x", color="#111827", label="today")
ax.set(xlabel="CO2, tonnes a week", ylabel="cost, $ a week", title="Cost against CO2, at today's service")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("figures/day66_frontier.png", dpi=150)
print("saved figures/day66_frontier.png")
```

```text
--- epsilon
 co2_cap     cost   co2  weight_can_find_it price_range
   18.17 57401.46 18.17                True       0-289
   17.30 57653.80 17.30                True   about 289
   16.43 57906.40 16.43               False           -
   15.56 58158.44 15.56               False           -
   14.69 58411.13 14.69               False           -
   13.81 58663.20 13.81                True     289-296
   12.94 58920.64 12.94                True     296-674
   12.07 59506.91 12.07                True   674-1,312
   11.20 60648.27 11.20                True 1,312-1,617
   10.33 62055.48 10.33                True 1,617-1,865
    9.46 63678.10  9.46                True 1,865-2,451
    8.58 65835.42  8.58                True 2,451-8,305
    7.71 73061.10  7.71                True      8,305+
saved figures/day66_frontier.png
```

![Cost against CO2](figures/day66_frontier.png)

- The frontier's **slope is the marginal cost of abatement**. The first five tonnes cost $289 each (HVO), then $674–1,312 (the last HVO and the first electric trucks), then up to $8,300 a tonne for the last ones (electric on short routes).
- Three points (16.4–14.7 t) are **unsupported**: they lie a hair above the straight line between their neighbours, because whole stores switch vehicle (integer choices). No carbon price finds them; a cap does.
- In integer models, weighted sums miss points. Use ε-constraints to trace frontiers, and use the slopes to report carbon prices.

## 66.4 Lexicographic: CO2 first, then cost

When one objective clearly comes first: optimize it, then optimize the next one among the plans that are (nearly) as good on the first.

```python
# ---------- 66.4 Lexicographic: CO2 first, then cost ----------
print("--- lexicographic")
m = solve_with(lambda m: m.co2, service_cap=S)
best_co2 = pyo.value(m.co2)
for slack in (0.0, 0.02, 0.10):
    m = solve_with(lambda m: m.cost, service_cap=S, co2_cap=best_co2 * (1 + slack) + 1e-6)
    print(f"CO2 within {slack:.0%} of its minimum ({best_co2:.2f} t): cost ${pyo.value(m.cost):,.0f}, "
          f"CO2 {pyo.value(m.co2):.2f} t; {mix(m)}")
```

```text
--- lexicographic
CO2 within 0% of its minimum (7.71 t): cost $73,061, CO2 7.71 t; diesel 5, electric 25, hvo 10; deliveries 3x 3, 4x 6, 5x 14, 6x 17
CO2 within 2% of its minimum (7.71 t): cost $70,922, CO2 7.87 t; diesel 10, electric 23, hvo 7; deliveries 3x 3, 4x 4, 5x 20, 6x 13
CO2 within 10% of its minimum (7.71 t): cost $66,338, CO2 8.48 t; diesel 8, electric 16, hvo 16; deliveries 3x 3, 4x 5, 5x 17, 6x 15
```

The absolute minimum CO2 (7.71 t) costs $73,061 a week. **Accepting 10% more CO2 saves $6,700 a week.** Strict lexicographic optimization is rarely wise: always show what a little slack buys.

> **Decision it supports:** plans with several goals that can't be priced against each other honestly, and the trade-offs management must choose between.
>
> **Other decisions you can support the same way:** portfolio risk against return · hospital waiting times against cost · product design (weight, cost, strength) · energy mix (cost, emissions, reliability).

### The planted bug

`day66/bug_service_cap.py` traces the cost–CO2 frontier with ε-constraints. Run it.

```python
for cut in (0.2, 0.4, 0.6):
    m = build_delivery(d)
    m.co2_cap = pyo.Constraint(expr=m.co2 <= (1 - cut) * today["co2"])
    m.obj = pyo.Objective(expr=m.cost)
    solve(m)
    print(f"CO2 -{cut:.0%}: cost ${pyo.value(m.cost):,.0f} a week (today ${today['cost']:,.0f}); {mix(m)}")
```

<details><summary>What happens, and why</summary>

```text
CO2 -20%: cost $47,722 a week (today $59,914); diesel 40; deliveries 3x 17, 4x 7, 5x 16
CO2 -40%: cost $48,910 a week (today $59,914); diesel 23, hvo 17; deliveries 3x 17, 4x 7, 5x 16
CO2 -60%: cost $55,983 a week (today $59,914); diesel 11, electric 18, hvo 11; deliveries 3x 17, 4x 7, 5x 16
```

A 40% cut for $11,000 a week **less** than today, and a 20% cut with all diesel trucks. Every plan delivers 3 times a week to 17 stores: the model cut CO2 by cutting deliveries, raising stock-outs from 2.6% to 3.4%. The colleague left out the service constraint, so the objective that wasn't modelled was quietly sacrificed. An optimizer exploits whatever you leave out. Constrain (or optimize) every objective that matters, and report all of them for every plan.

</details>

## Mini-project: a cap or a carbon price?

The board wants delivery CO2 40% below today's, at today's service. There are two ways to get there:

- a **cap** (an ε-constraint);
- an **internal carbon price** (a weighted sum).

Find the cheapest plan under the cap. Then find the lowest carbon price whose plan meets the target. Do they agree? What does the next tonne cost?

<details><summary>Solution</summary>

```python
"""Day 66 mini-project: the board wants delivery CO2 40% below today's, at today's service. Two ways to get there:
a cap (epsilon constraint) or an internal carbon price (a weighted sum). Find the cheapest plan under the cap,
then the lowest carbon price whose plan meets the target. Do they agree? What does each tonne cost at the margin?

Run from the repository root (after day06/load_tables.py):  python day66/mini_project_carbon_price.py
"""
import sys

import duckdb
import pyomo.environ as pyo

for folder in ("day06", "day66"):
    sys.path.insert(0, folder)
from bakery import solve
from delivery_mo import build_delivery, evaluate, load_delivery, mix, today_plan

con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_delivery(con)
today = evaluate(d, today_plan(d))
target = (1 - float(d["settings"]["target_2030_cut"])) * today["co2"]


def plan(price=0.0, cap=None):
    m = build_delivery(d)
    m.keep_service = pyo.Constraint(expr=m.service <= today["service"] + 1e-9)
    if cap is not None:
        m.cap = pyo.Constraint(expr=m.co2 <= cap)
    m.obj = pyo.Objective(expr=m.cost + price * m.co2)
    solve(m)
    return m


m = plan(cap=target)
print(f"target: CO2 <= {target:.2f} t a week (today {today['co2']:.2f})")
print(f"cap: cost ${pyo.value(m.cost):,.0f} a week ({pyo.value(m.cost) - today['cost']:+,.0f} vs today), "
      f"CO2 {pyo.value(m.co2):.2f} t; {mix(m)}")
lo, hi = 0.0, 10_000.0
for _ in range(30):                                                 # the lowest price whose plan meets the target
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if pyo.value(plan(price=mid).co2) > target + 1e-6 else (lo, mid)
p = plan(price=hi)
print(f"price: ${hi:,.0f} a tonne -> cost ${pyo.value(p.cost):,.0f}, CO2 {pyo.value(p.co2):.2f} t; {mix(p)}")
step = plan(cap=target - 0.5)
print(f"the next half tonne below the target costs ${pyo.value(step.cost) - pyo.value(m.cost):,.0f} a week "
      f"(${(pyo.value(step.cost) - pyo.value(m.cost)) / 0.5:,.0f} a tonne)")
```

```text
target: CO2 <= 12.39 t a week (today 20.65)
cap: cost $59,273 a week (-641 vs today), CO2 12.39 t; diesel 16, electric 1, hvo 23; deliveries 3x 3, 4x 4, 5x 19, 6x 14
price: $650 a tonne -> cost $59,549, CO2 11.93 t; diesel 26, electric 2, hvo 12; deliveries 3x 3, 4x 4, 5x 19, 6x 14
the next half tonne below the target costs $354 a week ($709 a tonne)
```

- The **cap** meets the target exactly (12.39 t) at $59,273 a week, **$641 less than today**: making today's plan efficient pays for the first 40% of the cut.
- The **price** needs $650 a tonne, and then overshoots to 11.93 t at $276 a week more. A carbon price can't aim at a target in an integer model: it jumps between plans.
- The next half tonne below the target costs $709 a tonne: the marginal cost at the target.
- **Use a cap to plan to a target, and quote the marginal cost ($700 a tonne) as its implied carbon price.**

</details>

## Commit

```bash
git add .
git commit -m "Day 66: multi-objective optimization"
git push
```

### Day 66 checklist

- [ ] I can model several objectives as expressions and optimize each one alone.
- [ ] I can use a weighted sum and read the weight as a price.
- [ ] I can trace a frontier with ε-constraints, and know weighted sums miss unsupported points in integer models.
- [ ] I can apply a lexicographic order with slack.
- [ ] I keep every objective that matters in the model, as an objective or a constraint.

---

# DAY 67 — Pareto Frontiers: Cost, Carbon, Service

Day 66 held service at today's level. The board's real question has three dimensions:

- How much CO2 can we cut?
- At what cost?
- With better or worse service for the stores?

Three objectives make a **frontier surface**. Today you trace it with a grid of ε-constraints, filter it to the non-dominated plans, read the trade-off rates, and choose points to show the board.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 67.1 A grid of caps |
| 0:45 – 1:45 | 67.2 The non-dominated set |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 67.3 Reading the frontier |
| 2:45 – 3:45 | 67.4 Choosing a point; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: three options for the board |
| 4:45 – 5:00 | Commit with Git + checklist |

## 67.1 A grid of caps

With three objectives, minimize cost under **two** caps: one on CO2 and one on stock-outs. Each pair of caps gives one plan, or none if the caps can't both be met.

Create `day67/pareto.py`:

```python
"""Day 67: Pareto dominance. A plan dominates another if it is no worse in every objective and better in at least
one (all objectives here: smaller is better).

Import it:  from pareto import dominates, non_dominated
"""
import numpy as np


def dominates(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated(points):
    """A boolean mask: True for the points no other point dominates."""
    pts = np.asarray(points, dtype=float)
    return np.array([not any(dominates(q, p) for j, q in enumerate(pts) if j != i) for i, p in enumerate(pts)])
```

Then `day67/run_pareto.py`:

```python
# ---------- 67.1 A grid of caps ----------
print("--- grid")
services = [3.4, 3.0, 2.8, 2.6, 2.4, 2.2]
co2_caps = [20.0, 17.0, 14.0, 12.0, 10.0, 9.0, 8.0]
grid, points = {}, []
for sv in services:
    for cap in co2_caps:
        m = cheapest(sv, cap)
        grid[sv, cap] = None if m is None else round(pyo.value(m.cost))
        if m is not None:
            points.append((pyo.value(m.cost), pyo.value(m.co2), pyo.value(m.service), mix(m)))
table = pd.DataFrame([[grid[sv, c] for c in co2_caps] for sv in services],
                     index=[f"stock-outs <= {s}%" for s in services], columns=[f"CO2 <= {c} t" for c in co2_caps])
print("cheapest cost a week ($) for each pair of caps (blank: impossible)")
print(table.to_string(na_rep=""))
```

```text
--- grid
cheapest cost a week ($) for each pair of caps (blank: impossible)
                    CO2 <= 20.0 t  CO2 <= 17.0 t  CO2 <= 14.0 t  CO2 <= 12.0 t  CO2 <= 10.0 t  CO2 <= 9.0 t  CO2 <= 8.0 t
stock-outs <= 3.4%          47908          47908          48633        49212.0        51574.0       53393.0       58315.0
stock-outs <= 3.0%          51351          51351          52212        52896.0        55458.0       57473.0       62294.0
stock-outs <= 2.8%          53975          54091          54960        55698.0        58520.0       60524.0       65449.0
stock-outs <= 2.6%          57401          57741          58609        59527.0        62697.0       64677.0       69643.0
stock-outs <= 2.4%          62107          62768          63637        65464.0        68745.0       71662.0
stock-outs <= 2.2%          73282          74593          79162
```

Read the table across a row (CO2 at fixed service) or down a column (service at fixed CO2). Blank cells are impossible: 2.2% stock-outs (six deliveries a week everywhere) can't go below 14 tonnes, because six weekly trips to every store emit too much even with the available HVO and chargers.

## 67.2 The non-dominated set

A plan **dominates** another if it is no worse in every objective and better in at least one. The Pareto frontier is the set of plans nothing dominates.

```python
# ---------- 67.2 The non-dominated set ----------
print("--- non-dominated")
pts = pd.DataFrame(points, columns=["cost", "co2", "service", "plan"]).round({"cost": 0, "co2": 2, "service": 2})
pts = pts.drop_duplicates(subset=["cost", "co2", "service"])
keep = non_dominated(pts[["cost", "co2", "service"]].to_numpy())
front = pts[keep].sort_values(["service", "cost"]).reset_index(drop=True)
print(f"{len(pts)} distinct plans from the grid; {len(front)} are Pareto-optimal")
print(front[["cost", "co2", "service"]].to_string(index=False))
```

```text
--- non-dominated
35 distinct plans from the grid; 35 are Pareto-optimal
   cost   co2  service
73282.0 20.00     2.20
74593.0 17.00     2.20
79162.0 14.00     2.20
62107.0 19.28     2.40
62768.0 17.00     2.40
63637.0 14.00     2.40
65464.0 12.00     2.40
68745.0 10.00     2.40
71662.0  9.00     2.40
57401.0 18.17     2.60
57741.0 17.00     2.60
58609.0 14.00     2.60
59527.0 12.00     2.60
62697.0 10.00     2.60
64677.0  9.00     2.60
69643.0  8.00     2.60
53975.0 17.39     2.80
54091.0 16.99     2.80
54960.0 14.00     2.80
55698.0 12.00     2.80
58520.0 10.00     2.80
60524.0  9.00     2.80
65449.0  8.00     2.80
51351.0 16.99     3.00
52212.0 14.00     3.00
52896.0 12.00     3.00
55458.0 10.00     3.00
57473.0  9.00     3.00
62294.0  8.00     3.00
47908.0 16.50     3.38
48633.0 14.00     3.38
49212.0 12.00     3.38
51574.0 10.00     3.38
53393.0  9.00     3.38
58315.0  8.00     3.38
```

All 35 plans from the grid are non-dominated. That is no accident: each minimizes cost under its caps, so a cheaper plan meeting the same caps would have been found instead. Grids of ε-constraints produce Pareto points (at worst weakly dominated ones; filter anyway).

## 67.3 Reading the frontier

```python
# ---------- 67.3 Reading the frontier ----------
print("--- trade-offs")
for sv in (3.4, 2.6, 2.2):
    sub = front[np.isclose(front.service, sv, atol=0.06)].sort_values("co2", ascending=False)
    if len(sub) > 1:
        rate = (sub.cost.diff() / -sub.co2.diff()).dropna()
        print(f"stock-outs {sv}%: CO2 from {sub.co2.max():.2f} to {sub.co2.min():.2f} t; cost per tonne avoided "
              + ", ".join(f"${r:,.0f}" for r in rate))
for cap in (17.0, 12.0, 9.0):
    col = [(sv, grid[sv, cap]) for sv in services if grid[sv, cap] is not None]
    steps = ", ".join(f"{a}%->{b}%: ${cb - ca:,}" for (a, ca), (b, cb) in zip(col, col[1:]))
    print(f"CO2 <= {cap} t: each step in service costs {steps}")
fig, ax = plt.subplots(figsize=(7.5, 4))
for sv, colour in zip([3.4, 3.0, 2.6, 2.2], ["#9aa4b2", "#22a06b", "#3b82f6", "#e5484d"]):
    c = [(cap, grid[sv, cap]) for cap in co2_caps if grid[sv, cap] is not None]
    sub = [(pyo.value(m.co2), pyo.value(m.cost)) for cap, _ in c for m in [cheapest(sv, cap)]]
    sub = sorted(set((round(e, 2), round(k)) for e, k in sub))
    ax.plot([e for e, _ in sub], [k / 1000 for _, k in sub], "o-", color=colour, label=f"stock-outs <= {sv}%")
ax.scatter([today["co2"]], [today["cost"] / 1000], marker="x", s=60, color="#111827", label="today")
ax.set(xlabel="CO2, tonnes a week", ylabel="cost, $ thousands a week", title="Cost, carbon and service: one frontier per service level")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("figures/day67_pareto.png", dpi=150)
print("saved figures/day67_pareto.png")
```

```text
--- trade-offs
stock-outs 3.4%: CO2 from 16.50 to 8.00 t; cost per tonne avoided $290, $290, $1,181, $1,819, $4,922
stock-outs 2.6%: CO2 from 18.17 to 8.00 t; cost per tonne avoided $291, $289, $459, $1,585, $1,980, $4,966
stock-outs 2.2%: CO2 from 20.00 to 14.00 t; cost per tonne avoided $437, $1,523
CO2 <= 17.0 t: each step in service costs 3.4%->3.0%: $3,443, 3.0%->2.8%: $2,740, 2.8%->2.6%: $3,650, 2.6%->2.4%: $5,027, 2.4%->2.2%: $11,825
CO2 <= 12.0 t: each step in service costs 3.4%->3.0%: $3,684, 3.0%->2.8%: $2,802, 2.8%->2.6%: $3,829, 2.6%->2.4%: $5,937
CO2 <= 9.0 t: each step in service costs 3.4%->3.0%: $4,080, 3.0%->2.8%: $3,051, 2.8%->2.6%: $4,153, 2.6%->2.4%: $6,985
saved figures/day67_pareto.png
```

![The frontier, one curve per service level](figures/day67_pareto.png)

- **Along a curve** (fixed service), each tonne gets dearer: $290 (HVO), then $1,200–2,000 (electric), then about $5,000.
- **Between curves** (fixed CO2), each step in service gets dearer too: a few thousand dollars a week per step down to 2.6%, then $5,000–7,000 a week for 2.4%, and $12,000 for 2.2%.
- **The curves are roughly parallel**: cutting CO2 costs about the same whatever the service level, and better service costs about the same whatever the CO2 target. The two decisions are nearly independent, which makes the board's choice easier to explain.
- **Today's plan (×) is above every curve**: at today's 2.6% stock-outs and 20.65 t, a frontier plan is cheaper and greener at once.

## 67.4 Choosing a point

There is no "best" point without preferences. One neutral starting point is **compromise programming**:

- Scale each objective from its best value on the frontier (the **ideal**) to its worst (the **nadir**).
- Pick the plan closest to the ideal.

```python
# ---------- 67.4 Choosing a point ----------
print("--- choosing")
ideal = front[["cost", "co2", "service"]].min()
nadir = front[["cost", "co2", "service"]].max()
scaled = (front[["cost", "co2", "service"]] - ideal) / (nadir - ideal)
front["distance_to_ideal"] = np.sqrt((scaled ** 2).sum(axis=1)).round(3)
print(f"ideal (each objective at its best): cost ${ideal.cost:,.0f}, CO2 {ideal.co2:.2f} t, stock-outs {ideal.service:.2f}%")
print(front.nsmallest(3, "distance_to_ideal")[["cost", "co2", "service", "distance_to_ideal", "plan"]].to_string(index=False))
```

```text
--- choosing
ideal (each objective at its best): cost $47,908, CO2 8.00 t, stock-outs 2.20%
   cost  co2  service  distance_to_ideal                                                               plan
59527.0 12.0      2.6              0.604 diesel 23, electric 2, hvo 15; deliveries 3x 3, 4x 4, 5x 19, 6x 14
62697.0 10.0      2.6              0.605 diesel 11, electric 9, hvo 20; deliveries 3x 3, 4x 5, 5x 16, 6x 16
58520.0 10.0      2.8              0.634 diesel 16, electric 8, hvo 16; deliveries 3x 3, 4x 11, 5x 21, 6x 5
```

The closest plans keep **today's service (2.6%)** and cut CO2 to 10–12 tonnes, at about $59,500–62,700 a week, around today's cost. This is a reasonable default to put in front of the board, but the distance depends on the scaling and on which plans are in the set. Use it to start a discussion, never to end one.

> **Decision it supports:** choices between plans that are each best at something, with the price of every trade-off visible.
>
> **Other decisions you can support the same way:** sustainability targets · public budgets (cost, coverage, waiting times) · network design (cost, resilience, speed) · product portfolios.

### The planted bug

`day67/bug_two_objectives.py` filters a list of candidate plans down to the "efficient" ones before the board meeting. Run it.

```python
plans = pd.DataFrame(rows, columns=["cost", "co2", "service"])
efficient = plans[non_dominated(plans[["cost", "co2"]].to_numpy())]           # "cost and carbon are what matter"
print(f"{len(plans)} candidate plans; {len(efficient)} efficient:")
print(efficient.to_string(index=False))
```

<details><summary>What happens, and why</summary>

```text
12 candidate plans; 3 efficient:
 cost  co2  service
47908 16.5     3.38
49212 12.0     3.38
53393  9.0     3.38
```

Only the three plans with the **worst** service survive. The filter compared cost and CO2 only, so every plan with better service looks "dominated" by a cheaper one that delivers less often. With service included, all 12 are Pareto-optimal. **Filter on every objective**, or the screening makes the choice that was supposed to be the board's.

</details>

## Mini-project: three options for the board

The board will choose a delivery plan for the 40% CO2 cut. Prepare three options on the frontier, all meeting the target:

- **A**: the cheapest;
- **B**: the one with today's service;
- **C**: the best service the target allows.

For each, show the weekly cost against today, CO2, stock-outs and the plan's mix, and what moving to the next option buys and costs.

<details><summary>Solution</summary>

```python
"""Day 67 mini-project: the board will choose a delivery plan for the 40% CO2 cut. Prepare three options on the
frontier, all meeting the target: the cheapest, the one with today's service, and the best service the target
allows. For each: weekly cost against today, CO2, stock-outs, the plan's mix, and what moving to the next option
buys and costs.

Run from the repository root (after day06/load_tables.py):  python day67/mini_project_board.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd
import pyomo.environ as pyo

for folder in ("day06", "day66", "day67"):
    sys.path.insert(0, folder)
from bakery import solve
from delivery_mo import build_delivery, evaluate, load_delivery, mix, today_plan

pd.set_option("display.width", 200)
con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_delivery(con)
today = evaluate(d, today_plan(d))
target = (1 - float(d["settings"]["target_2030_cut"])) * today["co2"]


def plan(service_cap):
    m = build_delivery(d)
    m.target = pyo.Constraint(expr=m.co2 <= target)
    if service_cap is not None:
        m.service_cap = pyo.Constraint(expr=m.service <= service_cap + 1e-9)
    m.obj = pyo.Objective(expr=m.cost)
    solve(m)
    return m


best = build_delivery(d)
best.target = pyo.Constraint(expr=best.co2 <= target)
best.obj = pyo.Objective(expr=best.service)
solve(best)
best_service = pyo.value(best.service)
rows = []
for name, cap in [("A: cheapest", None), ("B: today's service", today["service"]), ("C: best service", best_service)]:
    m = plan(cap)
    rows.append((name, round(pyo.value(m.cost)), round(pyo.value(m.cost) - today["cost"]), round(pyo.value(m.co2), 2),
                 round(pyo.value(m.service), 2), mix(m)))
t = pd.DataFrame(rows, columns=["option", "cost_a_week", "vs_today", "co2_t", "stockout_pct", "plan"])
print(f"today: ${today['cost']:,.0f} a week, {today['co2']:.2f} t CO2, {today['service']:.2f}% stock-outs; "
      f"target {target:.2f} t")
print(t.to_string(index=False))
for a, b in zip(t.itertuples(), t.iloc[1:].itertuples()):
    print(f"{a.option[:1]} -> {b.option[:1]}: ${b.cost_a_week - a.cost_a_week:,} a week (${(b.cost_a_week - a.cost_a_week) * 52:,} a year) "
          f"for {a.stockout_pct - b.stockout_pct:.2f} points fewer stock-outs")
```

```text
today: $59,914 a week, 20.65 t CO2, 2.60% stock-outs; target 12.39 t
            option  cost_a_week  vs_today  co2_t  stockout_pct                                                               plan
       A: cheapest        48910    -11004  12.39           3.4                   diesel 23, hvo 17; deliveries 3x 17, 4x 7, 5x 16
B: today's service        59273      -641  12.39           2.6 diesel 18, electric 1, hvo 21; deliveries 3x 3, 4x 4, 5x 19, 6x 14
   C: best service        86432     26518  12.39           2.2                    diesel 14, electric 21, hvo 5; deliveries 6x 40
A -> B: $10,363 a week ($538,876 a year) for 0.80 points fewer stock-outs
B -> C: $27,159 a week ($1,412,268 a year) for 0.40 points fewer stock-outs
```

| Option | What it costs against today | What it does for service |
|---|---|---|
| **A** | saves $11,000 a week | delivers 3 times a week to 17 stores; stock-outs rise from 2.6% to 3.4% |
| **B** | saves $641 a week | keeps today's service, switches to HVO and a mix of 3–6 deliveries |
| **C** | costs $26,500 a week more | delivers 6 times a week everywhere, with 21 electric trucks to stay under the cap |

Each step up in service costs more than the last:

- **A → B**: about $540,000 a year for 0.8 points fewer stock-outs.
- **B → C**: about $1.4 million a year for another 0.4 points.

*Memo:* "All three options meet the 40% CO2 target. We recommend **B**: it keeps today's service and costs slightly less than today's deliveries. A saves $570,000 a year by letting stock-outs rise by a third. C improves service further at $1.4 million a year. If stock-outs cost the stores more in lost sales than that, choose C; Store Operations should estimate it."

</details>

## Commit

```bash
git add .
git commit -m "Day 67: Pareto frontiers for cost, carbon and service"
git push
```

Then run the tests: `tests/test_bayes_and_pareto.py` checks Days 64–67. It covers:

- the Beta prior's mean and variance, and the update rule;
- the closed-form expected penalty against integration;
- dominance on a small example;
- the delivery model: electric only within range, and the optimizer never worse than today's plan.

Add `"day64"` to `"day67"` to the folders in `conftest.py`.

```python
"""Known answers for Days 64-67: Beta updating, the value of a pilot, Pareto dominance, the delivery model."""
import numpy as np
import pyomo.environ as pyo
import pytest
from scipy import integrate, stats

from bakery import solve
from bayes import beta_from_levels, expected_penalty, penalty, update
from delivery_mo import build_delivery, evaluate, load_delivery, options, today_plan
from pareto import dominates, non_dominated


def test_beta_prior_and_update():
    st = {"theta_good": 0.9, "theta_poor": 0.5, "penalty_per_point": 100.0, "theta_no_penalty": 0.9}
    a, b = beta_from_levels(st, [("good", 0.5, 0), ("poor", 0.5, 0)])
    assert a / (a + b) == pytest.approx(0.7)
    assert stats.beta.var(a, b) == pytest.approx(0.04)                  # the two levels' variance: 0.2^2
    assert update(2, 3, 7, 10) == (9, 6)
    numeric = integrate.quad(lambda t: penalty(t, st) * stats.beta.pdf(t, a, b), 0, 1)[0]
    assert expected_penalty(a, b, st) == pytest.approx(numeric)


def test_dominance():
    assert dominates([1, 2, 3], [1, 2, 4]) and not dominates([1, 2, 3], [1, 2, 3])
    mask = non_dominated([[1, 5], [2, 2], [3, 3], [5, 1]])
    assert mask.tolist() == [True, True, False, True]


def test_delivery_model(con):
    d = load_delivery(con)
    assert all(2 * d["stores"].km[s] <= 240 for s, v, f in options(d) if v == "electric")
    today = evaluate(d, today_plan(d))
    m = build_delivery(d)
    m.keep = pyo.Constraint(expr=m.service <= today["service"] + 1e-9)
    m.obj = pyo.Objective(expr=m.cost)
    solve(m)
    assert pyo.value(m.cost) <= today["cost"] + 1e-6                     # today's plan is one of the options
```

```text
96 passed
```

### Day 67 checklist

- [ ] I can trace a three-objective frontier with a grid of ε-constraints.
- [ ] I can filter plans to the non-dominated set, using every objective.
- [ ] I can read trade-off rates along and between frontier curves.
- [ ] I can pick a compromise point, and know it depends on scaling.
- [ ] I can present three options with what each step buys and costs.
