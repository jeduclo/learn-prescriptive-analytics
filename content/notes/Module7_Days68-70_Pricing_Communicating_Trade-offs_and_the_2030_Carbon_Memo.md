# DAY 68 — Pricing and Markdown Optimization

Every model so far took prices as given. But price is a decision too, and often the most powerful one: it moves demand, which moves everything else. The Applied Causal Inference course estimated how Northbound's bakery demand responds to price, as **price elasticities** by category. Today you use them three ways:

- the classic markup rule, and why it misleads when capacity is tight;
- prices chosen **together** with the production plan;
- a **markdown** policy for the yule logs, by dynamic programming.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 68.1 The optimal markup, and why it misleads here |
| 0:45 – 1:45 | 68.2 Prices and the plan together |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 68.3 A markdown policy for the yule logs |
| 2:45 – 3:45 | 68.4 Elasticities are estimates; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: an evening markdown for croissants |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Days 68–70 (course assumptions):

| Table | What it holds |
|---|---|
| `price_elasticities` | elasticity and standard error by category |
| `price_ladder` | price changes from −10% to +10% |
| `yule_markdown_days`, `yule_markdown_prices` | the yule-log selling days and markdown prices |
| `evening_markdown` | the evening croissant markdown |
| `carbon_2030_settings`, `carbon_2030_stress` | the capstone's growth, charger banks and stress tests |

## 68.1 The optimal markup, and why it misleads here

With constant elasticity $$\varepsilon$$ (demand changes by $$\varepsilon$$% for each 1% change in price), profit per unit times demand is maximized at

$$p^* = c\,\frac{\varepsilon}{1 + \varepsilon} \qquad (\varepsilon < -1)$$

This is the **Lerner rule**: the less price-sensitive the demand, the higher the markup over the marginal cost $$c$$. Create `day68/pricing.py`:

```python
"""Day 68: pricing and markdowns. Constant-elasticity demand, the optimal markup, prices chosen together with the
bakery's production plan, and a markdown policy by dynamic programming.

Import it:  from pricing import load_pricing, markup_price, build_price_plan, markdown_policy
"""
import numpy as np
import pyomo.environ as pyo
from scipy import stats

from bakery import load_data


def load_pricing(con):
    d = load_data(con)
    e = con.sql("SELECT sku, units_per_batch, transfer_price, ingredient_cost FROM bakery_range").df().set_index("sku")
    el = con.sql("SELECT * FROM price_elasticities").df().set_index("category")
    ladder = [r[0] for r in con.sql("SELECT price_change FROM price_ladder ORDER BY price_change").fetchall()]
    return d, e, el, ladder


def demand_at(base, change, elasticity):
    """Constant elasticity: demand = base x (1 + change) ^ elasticity (change: -0.10 is 10% cheaper)."""
    return base * (1 + change) ** elasticity


def markup_price(cost, elasticity):
    """The profit-maximizing price for constant elasticity e < -1 (Lerner): p = cost x e / (1 + e)."""
    if elasticity >= -1:
        raise ValueError("with elasticity >= -1, raising the price always raises profit: no finite optimum")
    return cost * elasticity / (1 + elasticity)


def build_price_plan(d, e, elasticity, ladder, fixed_change=None):
    """Day 6's plan with prices: for each product choose one step on the price ladder (binary), then batches.
    Demand at each step follows the category's elasticity; sales can't exceed demand at the chosen price or the
    batches made. fixed_change: keep every price at this step (0.0: today's prices)."""
    m = pyo.ConcreteModel(name="price_and_plan")
    m.P, m.R = pyo.Set(initialize=d["products"]), pyo.Set(initialize=d["resources"])
    m.K = pyo.Set(initialize=range(len(ladder)))
    dem = {(p, k): demand_at(d["cap"][p], ladder[k], elasticity[d["category"][p]]) for p in m.P for k in m.K}
    m.z = pyo.Var(m.P, m.K, within=pyo.Binary)                     # price step chosen
    m.s = pyo.Var(m.P, m.K, within=pyo.NonNegativeReals)            # batches sold at that step
    m.x = pyo.Var(m.P, within=pyo.NonNegativeReals)                 # batches made
    m.one_price = pyo.Constraint(m.P, rule=lambda m, p: sum(m.z[p, k] for k in m.K) == 1)
    m.sell_at_price = pyo.Constraint(m.P, m.K, rule=lambda m, p, k: m.s[p, k] <= dem[p, k] * m.z[p, k])
    m.sell_made = pyo.Constraint(m.P, rule=lambda m, p: sum(m.s[p, k] for k in m.K) <= m.x[p])
    m.capacity_limit = pyo.Constraint(m.R, rule=lambda m, r: sum(d["use"].get((r, p), 0) * m.x[p] for p in m.P) <= d["capacity"][r])
    if fixed_change is not None:
        for p in m.P:
            for k in m.K:
                m.z[p, k].fix(1 if abs(ladder[k] - fixed_change) < 1e-9 else 0)
    price = lambda p, k: e.units_per_batch[p] * e.transfer_price[p] * (1 + ladder[k])
    cost = lambda p: e.units_per_batch[p] * e.ingredient_cost[p]
    m.profit = pyo.Objective(expr=sum(price(p, k) * m.s[p, k] for p in m.P for k in m.K) - sum(cost(p) * m.x[p] for p in m.P),
                             sense=pyo.maximize)
    m.dem = dem
    return m


def chosen_changes(m, ladder):
    return {p: ladder[k] for p in m.P for k in m.K if pyo.value(m.z[p, k]) > 0.5}


def markdown_policy(prices, shares, season_mean, elasticity, stock_max, salvage):
    """Dynamic programming over the selling days: with n logs left on a day, choose that day's price; demand is
    Poisson(share x season mean x (price / list price) ^ elasticity); leftovers after the last day earn salvage.
    Returns the value of starting with each stock level and the best price for each (day, stock)."""
    list_price = max(prices)
    days = len(shares)
    value = np.array([salvage * n for n in range(stock_max + 1)], dtype=float)
    policy = np.zeros((days, stock_max + 1))
    for t in reversed(range(days)):
        new = np.zeros(stock_max + 1)
        for n in range(stock_max + 1):
            best = -np.inf
            for p in prices:
                lam = shares[t] * season_mean * (p / list_price) ** elasticity
                k = np.arange(n + 1)
                pmf = stats.poisson.pmf(k, lam)
                pmf[-1] = 1 - pmf[:-1].sum()                       # demand of n or more sells all n
                v = pmf @ (p * k + value[n - k])
                if v > best + 1e-9:
                    best, policy[t, n] = v, p
            new[n] = best
        value = new
    return value, policy
```

Then `day68/run_pricing.py`:

```python
# ---------- 68.1 The optimal markup, and why it misleads here ----------
print("--- markup")
base = build_model(d)
base.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
solve(base)
oven_value = base.dual[base.capacity_limit["oven_hours"]]
rows = []
for sku in ("BK-SD", "BK-WW", "BK-CR", "BK-12", "BK-27"):
    cat, units = d["category"][sku], e.units_per_batch[sku]
    c = e.ingredient_cost[sku]
    capacity_cost = sum(base.dual[base.capacity_limit[r]] * d["use"].get((r, sku), 0) for r in d["resources"]) / units
    rows.append((sku, cat, elasticity[cat], e.transfer_price[sku], c, round(markup_price(c, elasticity[cat]), 2),
                 round(capacity_cost, 2), round(markup_price(c + capacity_cost, elasticity[cat]), 2)))
print(pd.DataFrame(rows, columns=["sku", "category", "elasticity", "price_today", "ingredients", "markup_on_ingredients",
                                  "capacity_cost_per_unit", "markup_on_full_cost"]).to_string(index=False))
```

```text
--- markup
  sku     category  elasticity  price_today  ingredients  markup_on_ingredients  capacity_cost_per_unit  markup_on_full_cost
BK-SD        bread        -1.6         4.50         1.50                   4.00                    3.00                12.00
BK-WW        bread        -1.6         3.20         1.20                   3.20                    1.47                 7.11
BK-CR viennoiserie        -2.1         7.00         2.00                   3.82                    4.11                11.67
BK-12 viennoiserie        -2.1         8.45         2.81                   5.36                    4.89                14.70
BK-27       cookie        -2.4         6.30         2.35                   4.03                    3.40                 9.85
```

On ingredient cost alone, the rule says **cut** prices: sourdough to $4.00 (from $4.50) and croissants to $3.82 (from $7.00). But the bakery is short of oven hours, and selling more croissants means baking fewer of something else. The true marginal cost includes the **shadow price of capacity** (Day 11): an oven hour is worth $99. Per croissant that adds $4.11. On **full** marginal cost, the rule says **raise** prices: croissants to $11.67. The markup rule is right; the cost you feed it must include what capacity is worth.

## 68.2 Prices and the plan together

Let the model choose both: for each product, one step on a price ladder (−10% to +10%), with demand at each step from the elasticity, and then the batches. One binary per product and step keeps it linear: a MIP.

```python
# ---------- 68.2 Prices and the plan together ----------
print("--- price and plan")
today = build_price_plan(d, e, elasticity, ladder, fixed_change=0.0)
solve(today)
both = build_price_plan(d, e, elasticity, ladder)
solve(both)
print(f"today's prices, best plan: ${pyo.value(today.profit):,.2f} (Day 6: ${pyo.value(base.total_profit):,.2f})")
print(f"prices chosen with the plan: ${pyo.value(both.profit):,.2f} (+${pyo.value(both.profit) - pyo.value(today.profit):,.2f} a week)")
ch = pd.Series(chosen_changes(both, ladder))
print("price changes: " + ", ".join(f"{v:+.0%} {n}" for v, n in ch.value_counts().sort_index().items()))
by_cat = pd.DataFrame({"category": pd.Series(d["category"]), "change": ch})
print(by_cat.groupby("category").change.agg(["mean", "min", "max"]).round(3).to_string())
made = sum(pyo.value(both.x[p]) for p in both.P)
print(f"batches made: {made:.1f} (today's prices: {sum(pyo.value(today.x[p]) for p in today.P):.1f}); "
      f"oven hours used {sum(d['use'].get(('oven_hours', p), 0) * pyo.value(both.x[p]) for p in both.P):.1f}")
```

```text
--- price and plan
today's prices, best plan: $108,351.28 (Day 6: $108,351.28)
prices chosen with the plan: $118,868.79 (+$10,517.51 a week)
price changes: -10% 1, +0% 1, +5% 1, +10% 27
               mean  min  max
category
bread         0.100  0.1  0.1
cake          0.067 -0.1  0.1
cookie        0.075  0.0  0.1
viennoiserie  0.100  0.1  0.1
batches made: 417.0 (today's prices: 429.6); oven hours used 600.0
```

- With today's prices, the model reproduces Day 6 exactly ($108,351.28), a good check.
- Choosing prices adds **$10,518 a week**. It raises 27 of 30 prices by the maximum 10%, bakes slightly fewer batches, and still uses every oven hour.
- With capacity binding, a higher price loses a few customers whose batches would have displaced others, and earns more on everyone else.
- The model sits at the **edge of the ladder** for most products. That says "go further", but the elasticities were estimated on price changes of a few percent: don't trust them for +25%. Test +10% first, then re-estimate.

## 68.3 A markdown policy for the yule logs

Yule logs sell for four days (December 21–24), then go to Boxing Day at half price. Each day the store can keep the list price ($32) or mark down to $28, $24 or $20. Demand responds with the yule-log elasticity. A **dynamic program** (Day 59) over days and logs left finds the best price for every situation.

```python
# ---------- 68.3 A markdown policy for the yule logs ----------
print("--- markdown")
yule = dict(con.sql("SELECT * FROM yule_log_economics").fetchall())
days = con.sql("SELECT * FROM yule_markdown_days ORDER BY sale_date").df()
prices = [r[0] for r in con.sql("SELECT price FROM yule_markdown_prices ORDER BY price DESC").fetchall()]
season = con.sql("SELECT avg(forecast_logs) FROM yule_log_forecast").fetchall()[0][0]
salvage = yule["boxing_day_price"] * yule["boxing_day_sell_share"]
value_md, policy = markdown_policy(prices, days.share_of_season_demand.tolist(), season, elasticity["yule_log"], 120, salvage)
value_fixed, _ = markdown_policy([prices[0]], days.share_of_season_demand.tolist(), season, elasticity["yule_log"], 120, salvage)
for name, v in (("list price only", value_fixed), ("markdowns allowed", value_md)):
    q = int(np.argmax(v - yule["cost"] * np.arange(len(v))))
    print(f"{name:>17}: bake {q} logs for an average store; expected profit ${v[q] - yule['cost'] * q:,.2f}")
q = int(np.argmax(value_md - yule["cost"] * np.arange(len(value_md))))
print("the policy: price by logs left (rows: day; columns: logs left)")
cols = [0, 5, 10, 15, 20, 30, 40, 60]
print(pd.DataFrame(policy[:, cols], index=days.sale_date.astype(str), columns=[f"{c} left" for c in cols]).to_string())
```

```text
--- markdown
  list price only: bake 58 logs for an average store; expected profit $959.63
markdowns allowed: bake 74 logs for an average store; expected profit $989.98
the policy: price by logs left (rows: day; columns: logs left)
            0 left  5 left  10 left  15 left  20 left  30 left  40 left  60 left
sale_date
2026-12-21    32.0    32.0     32.0     32.0     32.0     32.0     32.0     32.0
2026-12-22    32.0    32.0     32.0     32.0     32.0     32.0     32.0     28.0
2026-12-23    32.0    32.0     32.0     32.0     32.0     32.0     28.0     24.0
2026-12-24    32.0    32.0     32.0     32.0     28.0     24.0     20.0     20.0
```

- **The policy reads like a manager's rule of thumb, but with numbers**: keep the list price while stock is in line with the days left; mark down when stock is high for the time left. On the 24th, with 20 or more logs left, mark down.
- **Markdowns change the order too.** A store that can mark down should bake **74** logs instead of 58, and earns about $30 more a store each Christmas. The markdown is a safety valve that makes a bigger order safe. Day 51's newsvendor ordered 76 for an average store, close to this.

## 68.4 Elasticities are estimates

The elasticities come with standard errors. Draw 2,000 sets of elasticities, and compare the new prices with today's in each:

```python
# ---------- 68.4 Elasticities are estimates ----------
print("--- uncertainty")
rng = np.random.default_rng(68)
plan_change = chosen_changes(both, ladder)
x_new = {p: pyo.value(both.x[p]) for p in both.P}
x_old = {p: pyo.value(today.x[p]) for p in today.P}
gains = []
for _ in range(2_000):
    draw = {c: rng.normal(el.elasticity[c], el.std_error[c]) for c in el.index}

    def profit(change, x):
        tot = 0.0
        for p in d["products"]:
            ch = change.get(p, 0.0)
            sold = min(x[p], demand_at(d["cap"][p], ch, draw[d["category"][p]]) if ch else d["cap"][p])
            tot += e.units_per_batch[p] * (e.transfer_price[p] * (1 + ch) * sold - e.ingredient_cost[p] * x[p])
        return tot
    gains.append(profit(plan_change, x_new) - profit({}, x_old))
gains = np.array(gains)
print(f"gain of the new prices over today's, across 2,000 draws of the elasticities: mean ${gains.mean():,.2f}, "
      f"5th percentile ${np.quantile(gains, 0.05):,.2f}, P(gain > 0) {np.mean(gains > 0):.3f}")
```

```text
--- uncertainty
gain of the new prices over today's, across 2,000 draws of the elasticities: mean $8,775.79, 5th percentile $6,194.31, P(gain > 0) 1.000
```

The gain is $8,776 a week on average, lower than the model's $10,518, because the model assumed the point estimates. But it is positive in **every** draw, with a 5th percentile of $6,194. The decision (raise prices) is robust to the estimation error. The **amount** gained is not, so report the range.

> **Decision it supports:** setting prices and markdowns together with what capacity and stock allow.
>
> **Other decisions you can support the same way:** airline and hotel pricing · end-of-season fashion markdowns · ride-hailing surge prices · promotions in grocery.

### The planted bug

`day68/bug_markup.py` checks today's croissant price with the markup rule. Run it.

```python
best = cost * e / (e - 1)                                          # "price = cost x e / (e - 1)"
print(f"elasticity {e}: best price ${best:.2f} (today ${price:.2f}, cost ${cost:.2f})")
```

<details><summary>What happens, and why</summary>

```text
elasticity -2.1: best price $1.35 (today $7.00, cost $2.00)
```

A best price of $1.35, below the $2.00 cost. The formula has $$\varepsilon - 1$$ where it needs $$1 + \varepsilon$$. With $$\varepsilon = -2.1$$, $$\varepsilon/(1+\varepsilon) = 1.91$$ (a markup), but $$\varepsilon/(\varepsilon - 1) = 0.68$$ (a markdown below cost). Two sanity checks:

- an optimal price under constant elasticity is always **above** marginal cost;
- the markup factor $$\varepsilon/(1+\varepsilon)$$ falls towards 1 as demand gets more elastic.

</details>

## Mini-project: an evening markdown for croissants

At 6 pm an average store has about 4 croissants left, and evening customers are 15% of the day's demand. A markdown sells more of them, but some daytime customers learn to wait for it: 5% of daytime buyers for each 10% off. Leftovers are thrown away. Which markdown, if any: 0%, 20%, 30%, 40% or 50% off?

<details><summary>Solution</summary>

```python
"""Day 68 mini-project: at 6 pm an average store has about 4 croissants left, and evening customers are 15% of the
day's demand. A markdown sells more of them, but some daytime customers learn to wait for it (5% of daytime buyers
for each 10% off). Which markdown, if any: 0%, 20%, 30%, 40% or 50% off? Leftovers are thrown away.

Run from the repository root (after day06/load_tables.py):  python day68/mini_project_evening.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "day45")
from stockouts import store_day_means

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
ev = dict(con.sql("SELECT * FROM evening_markdown").fetchall())
price, cost = con.sql("SELECT transfer_price, ingredient_cost FROM bakery_range WHERE sku = 'BK-CR'").fetchall()[0]
e = con.sql("SELECT elasticity FROM price_elasticities WHERE category = 'viennoiserie'").fetchall()[0][0]
daily = store_day_means(con)["mean"].mean()                        # Day 45: croissants a store sells a day
day_buyers, evening_full = (1 - ev["evening_share_of_demand"]) * daily, ev["evening_share_of_demand"] * daily
left = ev["croissants_left_at_6pm"]
rows = []
for off in (0.0, 0.2, 0.3, 0.4, 0.5):
    waiters = ev["waiting_share_per_10pct"] * (off / 0.1) * day_buyers if off > 0 else 0.0
    lam = evening_full * (1 - off) ** e + waiters                  # evening demand at the markdown, plus the waiters
    stock = left + waiters                                          # the waiters didn't buy at full price earlier
    k = np.arange(0, 60)
    sold = stats.poisson.pmf(k, lam) @ np.minimum(k, stock)
    revenue = price * (1 - off) * sold
    lost_full_price = price * waiters                               # they would have paid full price by day
    rows.append((f"{off:.0%}", round(lam, 2), round(sold, 2), round(revenue - lost_full_price, 2),
                 round(stock - sold, 2)))
t = pd.DataFrame(rows, columns=["markdown", "evening_demand", "evening_sold", "net_revenue_vs_daytime", "thrown_away"])
print(f"an average store: {daily:.2f} croissants a day; {left:.0f} left at 6 pm; price ${price:.2f}, elasticity {e}")
print(t.to_string(index=False))
best = t.loc[t.net_revenue_vs_daytime.idxmax()]
print(f"best: {best.markdown} off, ${best.net_revenue_vs_daytime - t.net_revenue_vs_daytime[0]:.2f} a store-day more "
      f"than no markdown (${(best.net_revenue_vs_daytime - t.net_revenue_vs_daytime[0]) * 40 * 312:,.0f} a year, 40 stores)")
```

```text
an average store: 6.10 croissants a day; 4 left at 6 pm; price $7.00, elasticity -2.1
markdown  evening_demand  evening_sold  net_revenue_vs_daytime  thrown_away
      0%            0.91          0.91                    6.38         3.09
     20%            1.98          1.93                    7.20         2.58
     30%            2.71          2.59                    7.26         2.19
     40%            3.71          3.40                    7.05         1.63
     50%            5.22          4.34                    6.12         0.96
best: 30% off, $0.88 a store-day more than no markdown ($10,982 a year, 40 stores)
```

- **30% off** is best: it sells 2.6 of the 4 leftovers instead of 0.9, and earns $0.88 more a store a day, about **$11,000 a year** across the chain.
- **Deeper markdowns backfire**: at 50% off, more daytime customers wait and pay half price, and the extra evening sales don't make up for it.
- **Cannibalization** (customers trading down to the cheaper option) is the cost every markdown must beat. The 5% waiting rate is an assumption. Measure it in a test (a few stores, a few weeks) before rolling out, the Applied Causal Inference way.

</details>

## Commit

```bash
git add .
git commit -m "Day 68: pricing and markdown optimization"
git push
```

### Day 68 checklist

- [ ] I can use the Lerner rule, with marginal cost including the shadow price of capacity.
- [ ] I can choose prices and production together with a price ladder in a MIP.
- [ ] I can find a markdown policy by dynamic programming, and see how it changes the order.
- [ ] I can test a pricing decision against the uncertainty in the elasticities.
- [ ] I account for cannibalization when marking down.

---

# DAY 69 — Communicating Trade-offs

A frontier, a tornado, a markdown policy: none of it matters until someone decides. Decision-makers see a chart for a minute and a memo for five. Today is about getting the numbers across honestly:

- framing, and how the same number can tell different stories;
- **ranges** instead of false precision;
- charts that don't mislead;
- **memos whose numbers come straight from the model run**, checked automatically.

The material is Day 67's three options for the 40% CO2 cut.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 69.1 The same numbers, three stories |
| 0:45 – 1:45 | 69.2 Ranges, not points |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 69.3 Honest charts |
| 2:45 – 3:45 | 69.4 Memos whose numbers trace back; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: the e-truck exhibit |
| 4:45 – 5:00 | Commit with Git + checklist |

## 69.1 The same numbers, three stories

Create `day69/exhibits.py`:

```python
"""Day 69: communicating trade-offs. Ranges for each option, words for probabilities, and memos whose numbers are
filled in from the results and checked against them.

Import it:  from exhibits import option_ranges, probability_words, fill_memo, check_memo
"""
import itertools
import re

import numpy as np

from delivery_mo import evaluate


def option_ranges(d, plans, low_high):
    """Re-cost fixed plans under every combination of uncertain inputs (each at low, base or high).
    low_high: {(vehicle, column): (low, high)}. Returns {option: array of weekly costs}, the same combinations
    for every option (so options can be compared case by case)."""
    keys = list(low_high)
    base = d["vehicles"].copy()
    out = {name: [] for name in plans}
    for combo in itertools.product(*[(lo, base.loc[v, c], hi) for (v, c), (lo, hi) in low_high.items()]):
        veh = base.copy()
        for (v, c), value in zip(keys, combo):
            veh.loc[v, c] = value
        for name, plan in plans.items():
            out[name].append(evaluate({**d, "vehicles": veh}, plan)["cost"])
    return {k: np.array(v) for k, v in out.items()}


# The likelihood scale used by the IPCC: a word stands for a probability range.
SCALE = [(0.99, "virtually certain"), (0.90, "very likely"), (0.66, "likely"), (0.33, "about as likely as not"),
         (0.10, "unlikely"), (0.01, "very unlikely"), (0.0, "exceptionally unlikely")]


def probability_words(p):
    """A word for a probability, with the number kept beside it: words alone are read very differently."""
    for threshold, word in SCALE:
        if p >= threshold:
            return f"{word} ({p:.0%})"
    return f"exceptionally unlikely ({p:.0%})"


def fill_memo(template, numbers):
    """Fill {name} placeholders from the results dictionary: every number in the memo comes from the model run."""
    return template.format(**numbers)


NUMBER = re.compile(r"(?<![A-Za-z])\$?-?\d[\d,]*(?:\.\d+)?%?")          # not the 2 in CO2


def check_memo(text, numbers, allowed=()):
    """Every number written in the memo must be one of the results (as formatted) or explicitly allowed
    (years, counts in the prose). Returns the numbers that can't be traced."""
    known = {n for v in numbers.values() for n in NUMBER.findall(str(v))} | {str(a) for a in allowed}
    return [n for n in NUMBER.findall(text) if n.rstrip(".,") not in known]
```

Then `day69/run_exhibits.py`:

```python
# ---------- 69.1 The same numbers, three stories ----------
print("--- framing")
for k in ("A", "B", "C"):
    r = res[k]
    print(f"{k}: {r['cost'] - today['cost']:+,.0f} $ a week | {(r['cost'] / today['cost'] - 1):+.1%} of delivery cost | "
          f"{(r['cost'] - today['cost']) * 52 / 1e6:+.2f} $ million a year | "
          f"{(r['cost'] - today['cost']) * 52 / 40:+,.0f} $ a store a year; CO2 {r['co2'] / today['co2'] - 1:+.0%}; "
          f"stock-outs {r['service']:.1f}% (today {today['service']:.1f}%)")
```

```text
--- framing
A: -11,004 $ a week | -18.4% of delivery cost | -0.57 $ million a year | -14,305 $ a store a year; CO2 -40%; stock-outs 3.4% (today 2.6%)
B: -641 $ a week | -1.1% of delivery cost | -0.03 $ million a year | -833 $ a store a year; CO2 -40%; stock-outs 2.6% (today 2.6%)
C: +26,518 $ a week | +44.3% of delivery cost | +1.38 $ million a year | +34,473 $ a store a year; CO2 -40%; stock-outs 2.2% (today 2.6%)
```

Option C's extra cost is "+44% of delivery cost", "+$1.38 million a year" or "+$34,000 a store a year". Each is true. Percentages make small bases look dramatic; per-store figures make big totals look small. **Pick the unit the decision is made in** (the annual budget, here), and give the others only if they help. Never switch units between options.

## 69.2 Ranges, not points

The weekly costs depend on input prices nobody controls: diesel, renewable diesel (HVO) and the electric lease. Re-cost each **fixed** plan under every combination of low, base and high prices:

```python
# ---------- 69.2 Ranges, not points ----------
print("--- ranges")
uncertain = {("diesel", "cost_per_km"): (0.95, 1.25), ("hvo", "cost_per_km"): (1.20, 1.50),
             ("electric", "fixed_per_trip"): (270, 330)}
ranges = option_ranges(d, plans, uncertain)
rows = []
for k, c in ranges.items():
    rows.append((k, round(c.min()), round(np.median(c)), round(c.max()), round(np.mean(c < ranges["today"]), 3)))
print(pd.DataFrame(rows, columns=["option", "lowest", "middle", "highest", "share_cheaper_than_today"]).to_string(index=False))
b_vs_today = ranges["B"] - ranges["today"]
print(f"B against today, case by case: {b_vs_today.min():+,.0f} to {b_vs_today.max():+,.0f} $ a week; "
      f"B is cheaper in {np.mean(b_vs_today < 0):.0%} of the {len(b_vs_today)} cases")
```

```text
--- ranges
option  lowest  middle  highest  share_cheaper_than_today
 today   56653   59914    63175                     0.000
     A   46467   48910    51677                     1.000
     B   56584   59273    62356                     0.593
     C   79980   86485    93429                     0.000
B against today, case by case: -3,092 to +2,205 $ a week; B is cheaper in 59% of the 27 cases
```

- "B saves $641 a week" sounds certain. Across the 27 price combinations, B ranges from $3,092 cheaper than today to $2,205 dearer. It is cheaper in **59%** of the cases.
- The honest statement: B costs about the same as today, and cuts CO2 by 40%.
- A and C are clear: A is always cheaper than today, C always dearer.
- Ranges show which comparisons are decided and which are close calls. Close calls should be decided on other grounds (here, service and risk), not on a point estimate.

## 69.3 Honest charts

```python
# ---------- 69.3 Honest charts ----------
print("--- charts")
names = ["today", "A", "B", "C"]
mid = [res[k]["cost"] / 1000 for k in names]
lo = [mid[i] - ranges[k].min() / 1000 for i, k in enumerate(names)]
hi = [ranges[k].max() / 1000 - mid[i] for i, k in enumerate(names)]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].bar(names, mid, color=["#9aa4b2", "#3b82f6", "#3b82f6", "#3b82f6"])
axes[0].set_ylim(45, 90)
axes[0].set_title("Misleading: cut-off axis, no ranges", fontsize=10)
axes[1].bar(names, mid, color=["#9aa4b2", "#3b82f6", "#3b82f6", "#3b82f6"], yerr=[lo, hi], capsize=4, ecolor="#111827")
axes[1].set_ylim(0, 100)
axes[1].set_title("Honest: from zero, with the range of input prices", fontsize=10)
for ax in axes:
    ax.set_ylabel("delivery cost, $ thousands a week")
    ax.spines[["top", "right"]].set_visible(False)
for i, k in enumerate(names):
    axes[1].text(i, 3, f"{res[k]['co2']:.1f} t\n{res[k]['service']:.1f}%", ha="center", fontsize=7, color="white")
fig.tight_layout()
fig.savefig("figures/day69_charts.png", dpi=150)
print("saved figures/day69_charts.png")
```

```text
--- charts
saved figures/day69_charts.png
```

![Misleading and honest charts](figures/day69_charts.png)

The left chart's axis starts at $45,000, so C looks four times as expensive as A. The right chart starts from zero, shows each option's range, and labels CO2 and service on the bars, so all three objectives are visible.

Rules that cover most cases:

- bars start at zero;
- uncertainty is drawn;
- every objective appears;
- the same scale for every option;
- no more decimal places than the uncertainty allows.

## 69.4 Memos whose numbers trace back

Numbers typed into a memo by hand go stale when the model is re-run. Instead:

- write the memo as a **template**;
- fill it from the results;
- **check** that every number in the text appears in the results.

Also write probabilities as a word with the number beside it. Words like "likely" mean anything from 55% to 90% to different readers. The IPCC's scale pins each word to a range.

```python
# ---------- 69.4 Memos whose numbers trace back ----------
print("--- memo")
p_b = float(np.mean(b_vs_today < 0))
numbers = {"target": f"{target:.1f}", "today_co2": f"{today['co2']:.1f}", "cut": f"{float(d['settings']['target_2030_cut']):.0%}",
           "b_cost": f"${res['B']['cost']:,.0f}", "b_vs_today": f"${today['cost'] - res['B']['cost']:,.0f}",
           "b_service": f"{res['B']['service']:.1f}%", "b_low": f"${ranges['B'].min():,.0f}",
           "b_high": f"${ranges['B'].max():,.0f}", "b_likely": probability_words(p_b),
           "a_saving": f"${(today['cost'] - res['A']['cost']) * 52:,.0f}", "a_service": f"{res['A']['service']:.1f}%",
           "c_extra": f"${(res['C']['cost'] - res['B']['cost']) * 52:,.0f}", "c_service": f"{res['C']['service']:.1f}%"}
template = ("Recommendation: option B. It cuts delivery CO2 by {cut} (to {target} t a week from {today_co2} t) and costs "
            "{b_cost} a week, {b_vs_today} less than today, with stock-outs unchanged at {b_service}. Depending on fuel "
            "and lease prices it costs {b_low} to {b_high} a week; it is {b_likely} to cost less than today's plan. "
            "Option A would save {a_saving} a year against today but raise stock-outs to {a_service}; option C would cut "
            "stock-outs to {c_service} for {c_extra} a year more than B.")
memo = fill_memo(template, numbers)
print(memo)
print(f"numbers that can't be traced to the results: {check_memo(memo, numbers) or 'none'}")
```

```text
--- memo
Recommendation: option B. It cuts delivery CO2 by 40% (to 12.4 t a week from 20.7 t) and costs $59,273 a week, $641 less than today, with stock-outs unchanged at 2.6%. Depending on fuel and lease prices it costs $56,584 to $62,356 a week; it is about as likely as not (59%) to cost less than today's plan. Option A would save $572,204 a year against today but raise stock-outs to 3.4%; option C would cut stock-outs to 2.2% for $1,412,250 a year more than B.
numbers that can't be traced to the results: none
```

Every number in the memo comes from the run and passes the check. "About as likely as not (59%)" is more honest than "B saves money".

> **Decision it supports:** getting a trade-off understood and decided, with no more confidence than the analysis supports.
>
> **Other decisions you can support the same way:** every one in this course.

### The planted bug

`day69/bug_typed_number.py`: after the model was re-run with this year's lease quotes, a colleague edited last month's memo by hand. Run the check.

```python
numbers = {"cost": f"${pyo.value(m.cost):,.0f}", "saving": f"${today['cost'] - pyo.value(m.cost):,.0f}",
           "cut": "40%", "service": f"{today['service']:.1f}%"}
memo = ("Option B cuts delivery CO2 by 40% and costs $59,723 a week, $641 less than today, "
        "with stock-outs unchanged at 2.6%.")
print(memo)
print(f"numbers that can't be traced to the results: {check_memo(memo, numbers) or 'none'}")
print(f"the model says: {numbers}")
```

<details><summary>What happens, and why</summary>

```text
Option B cuts delivery CO2 by 40% and costs $59,723 a week, $641 less than today, with stock-outs unchanged at 2.6%.
numbers that can't be traced to the results: ['$59,723']
the model says: {'cost': '$59,273', 'saving': '$641', 'cut': '40%', 'service': '2.6%'}
```

The check flags **$59,723**: the model says $59,273. Two digits were swapped when the number was typed. Nobody would notice in a meeting; somebody would when the budget doesn't add up. Fill numbers from results, and run the check before sending. Allowed exceptions, such as years, are listed explicitly.

</details>

## Mini-project: the e-truck exhibit

Turn Day 65's electric-truck recommendation into a one-page exhibit:

- one chart: the cost gap against the lease price, with the decision threshold and today's quote marked;
- a short memo whose every number comes from the results and passes the check.

<details><summary>Solution</summary>

```python
"""Day 69 mini-project: turn Day 65's electric-truck recommendation into a one-page exhibit: one chart (the cost gap
against the lease price, with the decision threshold and today's quote marked) and a short memo whose every number
comes from the results and passes the check.

Run from the repository root (after day06/load_tables.py):  python day69/mini_project_ev_exhibit.py
"""
import sys

import duckdb
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

for folder in ("day61", "day64", "day66", "day69"):
    sys.path.insert(0, folder)
from bayes import beta_from_levels, electric_cost, expected_penalty, load_range, update
from ev import ev_tree, load_ev
from exhibits import check_memo, fill_memo, probability_words
from scipy import stats
from trees import rollback

con = duckdb.connect("northbound.duckdb", read_only=True)
s, options, levels = load_ev(con)
st, range_levels, demo = load_range(con)
a, b = update(*beta_from_levels(st, range_levels), int(demo.completed_without_charge.sum()), len(demo))
leases = np.arange(50_000, 62_001, 250)
gaps = []
for lease in leases:
    s2 = {**s, "electric_lease_per_year": float(lease)}
    diesel = -rollback(ev_tree(s2, options, levels)["options"]["diesel_renew"])
    gaps.append(electric_cost(s2, options, expected_penalty(a, b, st)) - diesel)
gaps = np.array(gaps)
threshold = float(np.interp(0, gaps, leases))                      # the gap rises with the lease
quote = s["electric_lease_per_year"]
gap_now = float(np.interp(quote, leases, gaps))
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(leases / 1000, gaps / 1000, color="#3b82f6")
ax.axhline(0, color="#111827", lw=1)
ax.axvline(threshold / 1000, ls="--", color="#22a06b")
ax.axvline(quote / 1000, ls=":", color="#e5484d")
ax.text(threshold / 1000 - 0.2, gaps.max() / 1000 * 0.8, f"electric wins\nbelow ${threshold / 1000:.1f}k", ha="right",
        fontsize=8, color="#22a06b")
ax.text(quote / 1000 + 0.2, gaps.min() / 1000 * 0.6, f"today's quote\n${quote / 1000:.0f}k", fontsize=8, color="#e5484d")
ax.set(xlabel="electric truck lease, $ thousands a truck a year", ylabel="all-electric minus diesel,\nexpected PV, $ thousands",
       title="Electric trucks pay off if the lease falls below the threshold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/day69_ev_exhibit.png", dpi=150)
draws = stats.beta.rvs(a, b, size=100_000, random_state=np.random.default_rng(69))
numbers = {"quote": f"${quote:,.0f}", "gap": f"${gap_now:,.0f}", "threshold": f"${threshold:,.0f}",
           "demo": f"{int(demo.completed_without_charge.sum())} of {len(demo)}",
           "range_ok": probability_words(float(np.mean(draws >= st["theta_fair"]))), "years": "five"}
memo = fill_memo("Renew diesel for 2027 unless the electric lease falls to {threshold} a truck a year or less. "
                 "At today's quote of {quote}, ten electric trucks would cost {gap} more over the {years}-year lease "
                 "(expected present value). The vendor's demo managed {demo} winter days without a midday charge; "
                 "it is {range_ok} that the fleet would do at least as well as a 'fair' winter range.", numbers)
print(memo)
print(f"numbers that can't be traced to the results: {check_memo(memo, numbers, allowed=('2027',)) or 'none'}")
print("saved figures/day69_ev_exhibit.png")
```

```text
Renew diesel for 2027 unless the electric lease falls to $54,817 a truck a year or less. At today's quote of $58,000, ten electric trucks would cost $127,079 more over the five-year lease (expected present value). The vendor's demo managed 25 of 30 winter days without a midday charge; it is about as likely as not (46%) that the fleet would do at least as well as a 'fair' winter range.
numbers that can't be traced to the results: none
saved figures/day69_ev_exhibit.png
```

![The e-truck exhibit](figures/day69_ev_exhibit.png)

- The chart answers the only question the VP needs: **at what lease price does electric win?** At **$54,817** a truck a year. Today's quote is $58,000.
- The memo leads with the action, gives the threshold, the size of today's gap and one piece of evidence, and states the uncertainty in words and numbers.
- The year (2027) is allowed explicitly; everything else traces back to the run.

</details>

## Commit

```bash
git add .
git commit -m "Day 69: communicating trade-offs"
git push
```

### Day 69 checklist

- [ ] I choose the unit a decision is made in and keep it for every option.
- [ ] I give ranges from re-costing fixed plans under uncertain inputs, and say which comparisons are close.
- [ ] My charts start bars at zero, draw uncertainty and show every objective.
- [ ] My memos are filled from results and checked, and say probabilities in words with numbers.
- [ ] I can build a one-page exhibit that leads with the decision.

---

# DAY 70 — Capstone: The 2030 Carbon Memo

The board committed to cutting delivery CO2 40% below today's by 2030. Volumes will grow 3% a year in the meantime, so standing still means emissions **rise**. Module 7's capstone produces the memo the board will decide on, using the whole module:

- the delivery model (Day 66);
- 2030 volumes and the e-truck lease Northbound should negotiate (Days 61–65);
- **charger banks** as an investment;
- three options on the frontier (Day 67);
- stress tests (Days 50 and 60);
- a memo whose numbers trace back (Day 69).

| Time | Block |
|---|---|
| 0:00 – 0:30 | 70.1 The brief: 2030 if nothing changes |
| 0:30 – 1:30 | 70.2 Three options for 2030 |
| 1:30 – 1:45 | Break |
| 1:45 – 2:45 | 70.3 Stress tests |
| 2:45 – 3:30 | 70.4 The memo |
| 3:30 – 3:45 | Break |
| 3:45 – 4:30 | Module 7 wrap-up |
| 4:30 – 5:00 | Commit, tag, tests + checklist |

## 70.1 The brief: 2030 if nothing changes

Create `day70/carbon2030.py`. It is Day 66's model at 2030 volumes, with e-trucks at a cheaper 2030 lease (`carbon_2030_settings`: the $54,000 Day 65 recommended negotiating). It can also add **charger banks** at the DCs: 20 more electric trips a week each, for $1,150 a week.

```python
"""Day 70 capstone: the 2030 delivery plan. Day 66's model at 2030 volumes, with the option to add charger banks at
the DCs, under the board's target (CO2 40% below today's, however much volume grows).

Import it:  from carbon2030 import load_2030, build_2030, banks_of
"""
import math

import pyomo.environ as pyo

from delivery_mo import build_delivery, load_delivery


def load_2030(con, growth=None, hvo_share=None, electric_fixed=None):
    """Day 66's data, grown to 2030; electric trucks at the 2030 lease (Day 65's negotiation) unless overridden."""
    d = load_delivery(con)
    s = dict(con.sql("SELECT * FROM carbon_2030_settings").fetchall())
    g = s["growth_per_year"] if growth is None else growth
    factor = (1 + g) ** int(s["years"])
    stores = d["stores"].assign(pallets_per_week=d["stores"].pallets_per_week * factor)
    cap = int(d["settings"]["pallets_per_truck"])
    trips = {(st, f): f * math.ceil(r.pallets_per_week / (cap * f)) for st, r in stores.iterrows() for f in d["freq"]}
    vehicles = d["vehicles"].copy()
    vehicles.loc["electric", "fixed_per_trip"] = s["electric_fixed_per_trip_2030"] if electric_fixed is None else electric_fixed
    settings = {**d["settings"], **({} if hvo_share is None else {"hvo_max_km_share": hvo_share})}
    return {**d, "stores": stores, "trips": trips, "vehicles": vehicles, "settings": settings, "plan_2030": s,
            "growth_factor": factor}


def build_2030(d, fixed_banks=None, max_banks=4):
    """Day 66's model plus charger banks: each bank adds charging for a number of electric trips a week, at a weekly
    cost. m.total = delivery cost + charger banks. fixed_banks: keep an investment already decided."""
    m = build_delivery(d)
    s = d["plan_2030"]
    m.banks = pyo.Var(list(d["chargers"]), within=pyo.NonNegativeIntegers, bounds=(0, max_banks))
    if fixed_banks is not None:
        for dc, n in fixed_banks.items():
            m.banks[dc].fix(n)
    m.chargers.deactivate()
    st = d["stores"]
    m.chargers_2030 = pyo.Constraint(list(d["chargers"]), rule=lambda m, dc: sum(
        d["trips"][x, f] * m.y[x, v, f] for x, v, f in m.K if v == "electric" and st.dc_id[x] == dc)
        <= d["chargers"][dc] + s["charger_trips_per_bank"] * m.banks[dc])
    m.total = pyo.Expression(expr=m.cost + s["charger_bank_cost_per_week"] * sum(m.banks[dc] for dc in d["chargers"]))
    return m


def banks_of(m):
    return {dc: int(round(pyo.value(m.banks[dc]))) for dc in m.banks}
```

Then `day70/run_capstone_2030.py`:

```python
# ---------- 70.1 The brief: 2030 if nothing changes ----------
print("--- 2030 baseline")
d = load_2030(con)
base = evaluate(d, today_plan(d))
print(f"2026: ${today['cost']:,.0f} a week, {today['co2']:.2f} t CO2, stock-outs {today['service']:.2f}%")
print(f"2030, same way of working (volume x{d['growth_factor']:.3f}): ${base['cost']:,.0f} a week, {base['co2']:.2f} t CO2")
print(f"target: {target:.2f} t a week ({base['co2'] / target - 1:.0%} above it if nothing changes)")


def solve_option(d, service_cap=None, best_service=False, fixed_banks=None):
    m = build_2030(d, fixed_banks=fixed_banks)
    m.target = pyo.Constraint(expr=m.co2 <= target)
    if service_cap is not None:
        m.cap = pyo.Constraint(expr=m.service <= service_cap + 1e-9)
    if best_service:
        m.obj = pyo.Objective(expr=m.service)
        solve(m)
        m.cap = pyo.Constraint(expr=m.service <= pyo.value(m.service) + 1e-9)
        m.obj.deactivate()
    m.obj_cost = pyo.Objective(expr=m.total)
    solve(m)
    return m
```

```text
--- 2030 baseline
2026: $59,914 a week, 20.65 t CO2, stock-outs 2.60%
2030, same way of working (volume x1.126): $72,985 a week, 25.72 t CO2
target: 12.39 t a week (108% above it if nothing changes)
```

Delivering 2030's volumes the way we do today emits 25.7 t a week, **more than double** the target of 12.4 t, and costs $73,000 a week.

## 70.2 Three options for 2030

```python
# ---------- 70.2 Three options for 2030 ----------
print("--- options")
specs = {"A: cheapest": {}, "B: today's service": {"service_cap": today["service"]}, "C: best service": {"best_service": True}}
opts, rows = {}, []
for name, kw in specs.items():
    m = solve_option(d, **kw)
    opts[name] = {"banks": banks_of(m), "service": pyo.value(m.service), "total": pyo.value(m.total)}
    rows.append((name, round(pyo.value(m.total)), round(pyo.value(m.total) - base["cost"]), round(pyo.value(m.co2), 2),
                 round(pyo.value(m.service), 2), ", ".join(f"{k[3:]} {v}" for k, v in banks_of(m).items() if v) or "none",
                 mix(m)))
table = pd.DataFrame(rows, columns=["option", "cost_a_week", "vs_2030_as_today", "co2_t", "stockout_pct", "charger_banks", "plan"])
print(table.to_string(index=False))
```

```text
--- options
            option  cost_a_week  vs_2030_as_today  co2_t  stockout_pct charger_banks                                                               plan
       A: cheapest        54097            -18887  12.39          3.78          none       diesel 24, electric 2, hvo 14; deliveries 3x 18, 4x 13, 5x 9
B: today's service        60302            -12682  12.39          2.60          none diesel 20, electric 3, hvo 17; deliveries 3x 2, 4x 8, 5x 16, 6x 14
   C: best service        81615              8630  12.39          2.20         MIS 1                   diesel 13, electric 17, hvo 10; deliveries 6x 40
```

All three meet the target:

| Option | Cost against delivering 2030 volumes as today | Stock-outs | How |
|---|---|---|---|
| **A** | $18,900 a week less | rise to 3.8% | fewer deliveries |
| **B** | $12,700 a week less | today's 2.6% | renewable diesel on 17 stores' routes, a few electric trucks, a mix of 3–6 deliveries; **no new chargers** |
| **C** | $8,600 a week more | 2.2% | six deliveries a week everywhere, 17 electric trucks and one new charger bank at Mississauga |

Meeting the target **saves money** against standing still: today's plan wasn't efficient, and the efficiency gain pays for the carbon cut.

## 70.3 Stress tests

Charger banks are built once; the rest of the plan can adapt. Stress each option with its chargers fixed and everything else re-planned, at the option's service level:

- renewable diesel scarce (20% of km instead of 35%);
- the lease negotiation fails;
- growth of 5% a year.

```python
# ---------- 70.3 Stress tests ----------
print("--- stress")
stress = con.sql("SELECT * FROM carbon_2030_stress").df()
rows = []
for s in stress.itertuples():
    ds = load_2030(con, growth=s.growth_per_year, hvo_share=s.hvo_max_km_share, electric_fixed=s.electric_fixed_per_trip)
    for name, o in opts.items():
        cap = o["service"]
        try:
            m = solve_option(ds, service_cap=cap, fixed_banks=o["banks"])           # the chargers are already built
            rows.append((s.scenario, name, round(pyo.value(m.total)), "met"))
        except RuntimeError:
            m = build_2030(ds, fixed_banks=o["banks"])
            m.cap = pyo.Constraint(expr=m.service <= cap + 1e-9)
            m.obj = pyo.Objective(expr=m.co2)
            solve(m)
            rows.append((s.scenario, name, None, f"missed: best {pyo.value(m.co2):.2f} t"))
st = pd.DataFrame(rows, columns=["scenario", "option", "cost_a_week", "target"])
print(st.pivot(index="option", columns="scenario", values="cost_a_week")[stress.scenario].to_string())
print(st.pivot(index="option", columns="scenario", values="target")[stress.scenario].to_string())
fig, ax = plt.subplots(figsize=(8, 3.8))
width = 0.2
for i, s in enumerate(stress.scenario):
    g = st[st.scenario == s].set_index("option").reindex(list(opts))
    ax.bar(np.arange(len(opts)) + (i - 1.5) * width, g.cost_a_week.fillna(0) / 1000, width, label=s.replace("_", " "))
    for j, (cost, tgt) in enumerate(zip(g.cost_a_week, g.target)):
        if tgt != "met":
            ax.text(j + (i - 1.5) * width, 2, "missed", rotation=90, ha="center", va="bottom", fontsize=7, color="#e5484d")
ax.set_xticks(range(len(opts)), list(opts))
ax.set(ylabel="cost, $ thousands a week (2030)", title="The 2030 options under stress (target met unless marked)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
fig.tight_layout()
fig.savefig("figures/day70_options.png", dpi=150)
print("saved figures/day70_options.png")
```

```text
--- stress
scenario            as_planned  hvo_scarce  lease_not_cut  fast_growth
option
A: cheapest            54097.0     55691.0        54337.0      57135.0
B: today's service     60303.0     62361.0        60692.0      61367.0
C: best service        81615.0         NaN        84675.0      81615.0
scenario           as_planned            hvo_scarce lease_not_cut fast_growth
option
A: cheapest               met                   met           met         met
B: today's service        met                   met           met         met
C: best service           met  missed: best 14.66 t           met         met
saved figures/day70_options.png
```

![The 2030 options under stress](figures/day70_options.png)

- **A and B stay on target in every stress**, for at most $3,000 and $2,100 a week more respectively.
- **C misses the target if renewable diesel is scarce** (14.7 t at best). Six deliveries a week everywhere leaves too little room, and one charger bank isn't enough to replace the missing HVO with electric trucks. C would need more chargers, built in advance, to be safe.
- B is the **robust** choice: today's service, on target in every stress, and no investment that could be stranded.

## 70.4 The memo

```python
# ---------- 70.4 The memo ----------
print("--- memo")
b, a, c = opts["B: today's service"], opts["A: cheapest"], opts["C: best service"]
b_stress = st[st.option == "B: today's service"].set_index("scenario")
numbers = {"base_co2": f"{base['co2']:.1f}", "target": f"{target:.1f}", "today_co2": f"{today['co2']:.1f}",
           "b_cost": f"${b['total']:,.0f}", "b_vs": f"${base['cost'] - b['total']:,.0f}", "service": f"{b['service']:.1f}%",
           "chargers": (f"{sum(b['banks'].values())} new charger banks" if sum(b["banks"].values())
                        else "the chargers we already have"), "b_worst": f"${b_stress.cost_a_week.max():,.0f}",
           "a_saving": f"${(b['total'] - a['total']) * 52:,.0f}", "a_service": f"{a['service']:.1f}%",
           "c_extra": f"${(c['total'] - b['total']) * 52:,.0f}", "c_service": f"{c['service']:.1f}%",
           "missed": ", ".join(sorted(set(st[st.target != "met"].scenario))) or "none"}
template = ("To: the Board. Re: the 2030 delivery carbon target.\n"
            "If we keep delivering as we do today, growth takes delivery CO2 to {base_co2} t a week by 2030; the target is "
            "{target} t ({today_co2} t today). We recommend option B: renewable diesel where supply allows, electric "
            "trucks on the shorter routes using {chargers}, and a mix of 3 to 6 deliveries a week. It "
            "meets the target with stock-outs unchanged at {service}, for {b_cost} a week, {b_vs} less than delivering "
            "2030 volumes the way we do today. In our stress tests (scarce renewable diesel, no cheaper electric lease, "
            "faster growth) it stays on target and costs at most {b_worst} a week. Option A would save {a_saving} a year "
            "but raise stock-outs to {a_service}; option C would cut them to {c_service} for {c_extra} a year more. "
            "Scenarios where an option misses the target: {missed}.")
memo = fill_memo(template, numbers)
print(memo)
print(f"numbers that can't be traced to the results: {check_memo(memo, numbers, allowed=('2030', '3', '6')) or 'none'}")
```

```text
--- memo
To: the Board. Re: the 2030 delivery carbon target.
If we keep delivering as we do today, growth takes delivery CO2 to 25.7 t a week by 2030; the target is 12.4 t (20.7 t today). We recommend option B: renewable diesel where supply allows, electric trucks on the shorter routes using the chargers we already have, and a mix of 3 to 6 deliveries a week. It meets the target with stock-outs unchanged at 2.6%, for $60,302 a week, $12,682 less than delivering 2030 volumes the way we do today. In our stress tests (scarce renewable diesel, no cheaper electric lease, faster growth) it stays on target and costs at most $62,361 a week. Option A would save $322,654 a year but raise stock-outs to 3.8%; option C would cut them to 2.2% for $1,108,239 a year more. Scenarios where an option misses the target: hvo_scarce.
numbers that can't be traced to the results: none
```

Before sending, add what the model can't say:

- **Who decides on stock-outs.** Store Operations should value an extra point of stock-outs. If it costs more than about $1.1 million a year in lost sales and goodwill, choose C, and add charger banks to make it robust.
- **What to do now.** Negotiate the electric lease (Day 65), secure renewable diesel supply contracts (B depends on them), and re-run this model each year as prices and volumes move.
- **Which numbers are assumptions.** HVO supply, the 2030 lease, growth, and the stock-out rates by delivery frequency. Each was stress-tested or should be.

## Module 7 wrap-up

| Day | Idea | Northbound decision |
|---|---|---|
| 61 | Decision trees, sensitivity | Diesel or electric trucks; wait on Ottawa |
| 62 | Utility, risk tolerance | Whose risk attitude? The diesel hedge |
| 63 | Value of information | Is a winter pilot worth it? |
| 64 | Bayesian updating | What the demo truck taught us |
| 65 | Pilot design | No pilot; negotiate the lease |
| 66 | Multi-objective optimization | Cost against CO2; carbon prices and caps |
| 67 | Pareto frontiers | Cost, carbon and service: three options |
| 68 | Pricing and markdowns | Prices with the plan; yule-log markdowns |
| 69 | Communicating trade-offs | Ranges, honest charts, checked memos |
| 70 | Capstone | The 2030 carbon memo |

Module 8 turns these models into tools people use every week: services, apps, tests, monitoring, and the human judgement around them.

## Commit and tag

```bash
git add .
git commit -m "Day 70: capstone, the 2030 carbon memo"
git tag v7.0-module7
git push
git push --tags
```

Then run the tests: `tests/test_pricing_and_memos.py` checks Days 68–70:

- the markup rule, and refusing inelastic demand;
- today's prices reproducing Day 6;
- a one-day markdown against the Poisson formula;
- the memo check catching a typed number;
- the 2030 model with no growth and no new chargers.

Add `"day68"`, `"day69"` and `"day70"` to the folders in `conftest.py`.

```python
"""Known answers for Days 68-70: markups, prices with the plan, markdowns, memo checks, the 2030 model."""
import numpy as np
import pyomo.environ as pyo
import pytest

from bakery import build_model, solve
from carbon2030 import build_2030, load_2030
from exhibits import check_memo, fill_memo, probability_words
from pricing import build_price_plan, demand_at, load_pricing, markdown_policy, markup_price


def test_markup_and_demand():
    assert markup_price(2.0, -2.0) == pytest.approx(4.0)                # e / (1 + e) = 2: double the cost
    with pytest.raises(ValueError):
        markup_price(2.0, -0.8)
    assert demand_at(100, -0.10, -2.0) == pytest.approx(100 * 0.9 ** -2)


def test_todays_prices_reproduce_day_6(con):
    d, e, el, ladder = load_pricing(con)
    m = build_price_plan(d, e, el.elasticity.to_dict(), ladder, fixed_change=0.0)
    solve(m)
    base = build_model(d)
    solve(base)
    assert pyo.value(m.profit) == pytest.approx(pyo.value(base.total_profit))


def test_markdown_one_price_no_salvage():
    """One day, one price: the value of n units is price x E[min(n, D)]."""
    value, _ = markdown_policy([10.0], [1.0], 3.0, -2.0, 5, 0.0)
    from scipy import stats
    k = np.arange(0, 60)
    assert value[2] == pytest.approx(10 * (stats.poisson.pmf(k, 3.0) @ np.minimum(k, 2)))


def test_memo_check():
    numbers = {"cost": "$59,273", "share": probability_words(0.59)}
    memo = fill_memo("It costs {cost} a week; it is {share} to save money.", numbers)
    assert check_memo(memo, numbers) == []
    assert check_memo(memo.replace("$59,273", "$59,723"), numbers) == ["$59,723"]
    assert probability_words(0.95).startswith("very likely")


def test_2030_growth_and_chargers(con):
    d = load_2030(con, growth=0.0)
    m = build_2030(d, fixed_banks={dc: 0 for dc in d["chargers"]})
    assert d["growth_factor"] == 1.0
    m.obj = pyo.Objective(expr=m.total)
    solve(m)
    assert pyo.value(m.total) == pytest.approx(pyo.value(m.cost))        # no banks, no charger cost
```

```text
101 passed
```

### Day 70 checklist

- [ ] I can project a model to a future year and see what standing still costs.
- [ ] I can include an investment (charger banks) as a first-stage decision and stress-test with it fixed.
- [ ] I can present three options that meet a target, with what each gives up.
- [ ] I can tell a robust option from one that is optimal only as planned.
- [ ] I can write a board memo whose numbers come from the model, and add what the model can't say.
