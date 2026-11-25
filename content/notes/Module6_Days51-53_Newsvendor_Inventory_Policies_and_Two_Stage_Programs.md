# DAY 51 — The Newsvendor

Module 5 measured how plans behave when the forecast is wrong. Module 6 builds that uncertainty **into the decision itself**. It starts with the simplest decision under uncertainty, and one of the most useful.

You order once, before demand is known. Order too much and the extra is wasted; order too little and the sales are lost. Newspapers, Christmas trees, fresh croissants and yule logs all work this way. The best order has a famous answer, the **critical ratio**, and you have used it twice already (Days 41 and 45). Today you derive it and choose the right distribution for it. You also see how leftovers that still sell and customers who don't come back change it.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 51.1 The critical ratio |
| 0:45 – 1:45 | 51.2 Which distribution? |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 51.3 Leftovers and lost customers |
| 2:45 – 3:45 | 51.4 How flat is the top?; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: yule logs for Christmas |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Days 51–53:

- `yule_log_history`, `yule_log_forecast` and `yule_log_economics`: three Christmases of yule log sales, and this year's forecast.
- `dc_items`, `dc_suppliers`, `dc_daily_demand`, `purchase_orders`, `inventory_settings` and `dc_supplier_offers`: six ambient items at DC-MTL.
- `bakery_two_stage`: the day-old shelf and a mid-week top-up bake.

## 51.1 The critical ratio

Think about the **last unit** you order. It sells if demand is at least that high, with chance $$P(D \ge Q)$$, and earns the margin you would otherwise lose, $$c_u$$ (cost of **under**-ordering). Otherwise it is left over and costs $$c_o$$ (cost of **over**-ordering). Keep ordering while the last unit is worth it:

$$P(D \ge Q)\, c_u \;\ge\; P(D < Q)\, c_o \quad\Longrightarrow\quad P(D < Q^*) = \frac{c_u}{c_u + c_o}$$

The best order is the **quantile** of demand at the critical ratio. If $$c_u$$ is large compared with $$c_o$$, order well above the forecast; if it is small, order below it. Create `day51/newsvendor.py`:

```python
"""Day 51: the newsvendor. One order before demand is known; too much is wasted, too little is lost.

Import it:  from newsvendor import critical_ratio, q_normal, q_empirical, expected_profit, normal_expected_profit
"""
import numpy as np
from scipy import stats


def critical_ratio(price, cost, salvage=0.0, penalty=0.0):
    """under = what one unit too few costs (lost margin + lost goodwill); over = what one unit too many costs
    (cost - what a leftover still brings in). Order up to the quantile under / (under + over)."""
    under, over = price - cost + penalty, cost - salvage
    if under <= 0 or over <= 0:
        raise ValueError("a unit short and a unit left over must both cost something")
    return under / (under + over)


def q_normal(mean, sd, ratio):
    return mean + stats.norm.ppf(ratio) * sd


def q_empirical(samples, ratio):
    """The ratio-quantile of demand seen in history (or simulated): no distribution assumed."""
    return float(np.quantile(np.asarray(samples, dtype=float), ratio, method="inverted_cdf"))


def expected_profit(q, demand, price, cost, salvage=0.0, penalty=0.0):
    """Average profit of ordering q against many demand values (history or simulation)."""
    d = np.asarray(demand, dtype=float)
    sold = np.minimum(d, q)
    return float(np.mean(price * sold - cost * q + salvage * (q - sold) - penalty * (d - sold)))


def normal_expected_profit(q, mean, sd, price, cost, salvage=0.0, penalty=0.0):
    """The same for normal demand, with the loss function: E[short] = sd * L(z), L(z) = pdf(z) - z (1 - cdf(z))."""
    z = (q - mean) / sd
    short = sd * (stats.norm.pdf(z) - z * stats.norm.sf(z))
    sold = mean - short
    return price * sold - cost * q + salvage * (q - sold) - penalty * short
```

Then `day51/run_newsvendor.py`:

```python
# ---------- 51.1 The critical ratio ----------
print("--- critical ratio")
e = econ.loc["BK-CR"]
price, cost = e.units_per_batch * e.transfer_price, e.units_per_batch * e.ingredient_cost      # per batch
ratio = critical_ratio(price, cost)
q = q_normal(e.forecast, 0.09 * e.forecast, ratio)
print(f"BK-CR: a batch short loses ${price - cost:,.0f}, a batch left over loses ${cost:,.0f}: "
      f"critical ratio {ratio:.3f}")
print(f"forecast {e.forecast:.0f} batches, sd 9% (Day 41): order up to {q:.2f} -> {int(np.ceil(q))} batches "
      f"(Day 41's simulation: 33)")
```

```text
--- critical ratio
BK-CR: a batch short loses $250, a batch left over loses $100: critical ratio 0.714
forecast 31 batches, sd 9% (Day 41): order up to 32.58 -> 33 batches (Day 41's simulation: 33)
```

The formula gives Day 41's simulation answer in one line: 33 batches.

## 51.2 Which distribution?

Day 41 assumed normal errors with a 9% spread. Day 42 found the real forecast errors aren't normal: mostly small, with a few huge holiday weeks. The newsvendor needs one number from the distribution, its quantile at the critical ratio, so get that number right:

```python
# ---------- 51.2 Which distribution? ----------
print("--- distribution")
errors = error_table(con)
rows = []
for sku in ("BK-SD", "BK-WW", "BK-CR"):
    e = econ.loc[sku]
    price, cost = e.units_per_batch * e.transfer_price, e.units_per_batch * e.ingredient_cost
    r = critical_ratio(price, cost)
    hist = e.forecast * errors[sku].to_numpy()                    # last 103 weeks' errors on this week's forecast
    qn = q_normal(hist.mean(), hist.std(ddof=1), r)
    qe = q_empirical(hist, r)
    rows.append((sku, round(r, 3), e.forecast, round(qn, 2), round(qe, 2),
                 round(expected_profit(e.forecast, hist, price, cost), 2), round(expected_profit(qn, hist, price, cost), 2),
                 round(expected_profit(qe, hist, price, cost), 2)))
print(pd.DataFrame(rows, columns=["sku", "ratio", "forecast", "q_normal", "q_history", "profit_forecast",
                                  "profit_normal", "profit_history"]).to_string(index=False))
print("(profits: average over the 103 historical weeks, per week; quantities in batches)")
```

```text
--- distribution
  sku  ratio  forecast  q_normal  q_history  profit_forecast  profit_normal  profit_history
BK-SD  0.667      35.0     35.80      35.47         10264.01       10271.16        10280.50
BK-WW  0.625      45.0     45.74      45.68          8805.39        8819.54         8820.05
BK-CR  0.714      31.0     32.68      31.57          7557.81        7523.74         7581.89
(profits: average over the 103 historical weeks, per week; quantities in batches)
```

- For bread, the two answers are close. For **croissants**, the normal distribution says 32.7 batches and the history says 31.6.
- The normal's spread is inflated by the Christmas and January weeks. That pushes its 71st percentile too high, and **ordering it earns less than simply ordering the forecast** ($7,524 against $7,558).
- The history's own quantile earns the most.
- When the critical ratio is in the middle of the distribution and the tails are odd, use the **empirical quantile**. Fit a distribution when you need the far tail and have too little data for it.

## 51.3 Leftovers and lost customers

Two things change the ratio:

- **Salvage**: leftovers that still bring in something. The bakery could open a **day-old shelf**: 30% of the price, and 40% of leftovers sell there.
- **Penalty**: a customer who finds an empty shelf may not come back. That costs more than today's margin.

$$c_u = \text{price} - \text{cost} + \text{penalty}, \qquad c_o = \text{cost} - \text{salvage}$$

```python
# ---------- 51.3 Leftovers and lost customers ----------
print("--- salvage and penalty")
salvage_share = shelf["day_old_price_share"] * shelf["day_old_sell_share"]
rows = []
for sku in ("BK-SD", "BK-WW", "BK-CR"):
    e = econ.loc[sku]
    hist = e.forecast * errors[sku].to_numpy()
    for label, salvage, penalty in [("thrown away", 0.0, 0.0), ("day-old shelf", salvage_share * e.transfer_price, 0.0),
                                    ("day-old shelf + lost goodwill $1", salvage_share * e.transfer_price, 1.0)]:
        r = critical_ratio(e.transfer_price, e.ingredient_cost, salvage, penalty)       # per unit
        rows.append((sku, label, round(r, 3), round(q_empirical(hist, r), 2)))
print(f"day-old shelf: {shelf['day_old_price_share']:.0%} of the price, {shelf['day_old_sell_share']:.0%} of leftovers "
      f"sell -> a leftover brings back {salvage_share:.0%} of the price")
print(pd.DataFrame(rows, columns=["sku", "leftovers", "ratio", "order_up_to"]).to_string(index=False))
```

```text
--- salvage and penalty
day-old shelf: 30% of the price, 40% of leftovers sell -> a leftover brings back 12% of the price
  sku                        leftovers  ratio  order_up_to
BK-SD                      thrown away  0.667        35.47
BK-SD                    day-old shelf  0.758        35.73
BK-SD day-old shelf + lost goodwill $1  0.806        35.82
BK-WW                      thrown away  0.625        45.68
BK-WW                    day-old shelf  0.710        45.83
BK-WW day-old shelf + lost goodwill $1  0.786        46.04
BK-CR                      thrown away  0.714        31.57
BK-CR                    day-old shelf  0.812        31.81
BK-CR day-old shelf + lost goodwill $1  0.838        31.89
```

The day-old shelf raises every ratio by about 0.09, because a leftover now loses less. Lost goodwill raises it further. The quantities move only a little here, because the history's errors are tightly packed in the middle. The **ratio** is the policy: Store Operations can debate the penalty, and the quantity follows.

## 51.4 How flat is the top?

With normal demand there is a formula for expected profit, through the **loss function** $$L(z) = \varphi(z) - z(1 - \Phi(z))$$. Expected units short are $$\sigma L(z)$$. Expected sales are the mean minus that.

```python
# ---------- 51.4 How flat is the top? ----------
print("--- flat top")
e = econ.loc["BK-CR"]
price, cost = e.units_per_batch * e.transfer_price, e.units_per_batch * e.ingredient_cost
mean, sd = e.forecast, 0.09 * e.forecast
best = q_normal(mean, sd, critical_ratio(price, cost))
top = normal_expected_profit(best, mean, sd, price, cost)
for q in (27, 29, 31, 33, 35, 37):
    print(f"{q} batches: expected ${normal_expected_profit(q, mean, sd, price, cost):,.2f} "
          f"({normal_expected_profit(q, mean, sd, price, cost) - top:+,.2f} against the best, {best:.2f})")
qs = np.linspace(24, 40, 161)
fig, ax = plt.subplots(figsize=(7, 3.6))
for k, colour in [(0.09, "#3b82f6"), (0.18, "#e5484d")]:
    s = k * mean
    ax.plot(qs, [normal_expected_profit(x, mean, s, price, cost) for x in qs], color=colour, label=f"error sd {k:.0%}")
    qb = q_normal(mean, s, critical_ratio(price, cost))
    ax.plot([qb], [normal_expected_profit(qb, mean, s, price, cost)], "o", color=colour)
ax.axvline(mean, ls="--", color="#9aa4b2", lw=1)
ax.text(mean + 0.2, ax.get_ylim()[0] + 50, "forecast", color="#6b7280", fontsize=8)
ax.set(xlabel="croissant batches made", ylabel="expected profit a week ($)", title="The newsvendor's profit curve (BK-CR)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/day51_newsvendor.png", dpi=150)
print("saved figures/day51_newsvendor.png")
```

```text
--- flat top
27 batches: expected $6,716.77 (-701.31 against the best, 32.58)
29 batches: expected $7,114.42 (-303.67 against the best, 32.58)
31 batches: expected $7,360.43 (-57.65 against the best, 32.58)
33 batches: expected $7,414.42 (-3.67 against the best, 32.58)
35 batches: expected $7,316.77 (-101.31 against the best, 32.58)
37 batches: expected $7,144.51 (-273.57 against the best, 32.58)
saved figures/day51_newsvendor.png
```

![The newsvendor's profit curve](figures/day51_newsvendor.png)

- Near the top, the curve is **flat**: 33 batches loses $4 against the exact optimum.
- The forecast (31) loses $58 a week. Two batches below the forecast, 29, loses $304. Being **short** costs more than being over, because $$c_u > c_o$$.
- More uncertainty (the red curve) lowers the whole curve and moves the best order up.
- The cost of uncertainty is the gap between the curves at their tops, about $330 a week here. Better forecasts are worth money: Day 55 puts a number on that.

> **Decision it supports:** a one-shot order under uncertain demand: fresh food, seasonal goods, event capacity.
>
> **Other decisions you can support the same way:** how many seats to overbook (the same ratio, with no-shows) · how much cloud capacity to reserve · how many temporary staff to hire for a peak.

### The planted bug

`day51/bug_ratio.py` sets the croissant order with the critical ratio. Run it.

```python
under, over = price - cost, cost
ratio = over / (under + over)                     # "the chance of having too much"
q = forecast + stats.norm.ppf(ratio) * 0.09 * forecast
print(f"critical ratio {ratio:.3f}: make {q:.2f} batches (forecast {forecast:.0f})")
```

<details><summary>What happens, and why</summary>

```text
critical ratio 0.286: make 29.42 batches (forecast 31)
```

The answer is 29 batches, **below** the forecast for a product where running short costs 2.5 times as much as a leftover. The ratio is upside down: $$c_o/(c_u + c_o)$$ is the chance of being **short** at the optimum, not the quantile to order. Sanity check: when a shortage costs more than a leftover, the answer must be **above** the median forecast.

</details>

## Mini-project: yule logs for Christmas

Yule logs are baked once, on December 20, and sell until the 24th; on Boxing Day the leftovers go at half price and 60% of them sell. `yule_log_history` has three Christmases at every store (S40 only since it opened). Compare two options for 2026:

- **A:** every store orders its own newsvendor quantity.
- **B:** order for the chain, then move logs between stores on the 22nd and 23rd ($1.50 a log) to where they sell.

<details><summary>Solution</summary>

```python
"""Day 51 mini-project: how many yule logs for Christmas 2026? The bakery makes them once (December 20); what isn't
sold by the 24th goes on sale at half price on Boxing Day, where 60% of them sell. Option A: every store orders its
own newsvendor quantity. Option B: order for the chain and move logs between stores on the 22nd and 23rd
($1.50 a log) to where they sell. Which earns more, and why?

Run from the repository root (after day06/load_tables.py):  python day51/mini_project_yule_logs.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, "day51")
from newsvendor import critical_ratio, q_empirical

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
econ = dict(con.sql("SELECT * FROM yule_log_economics").fetchall())
hist = con.sql("SELECT *, demand_logs / forecast_logs AS ratio FROM yule_log_history").df()
fc = con.sql("SELECT store_id, forecast_logs FROM yule_log_forecast ORDER BY store_id").df()
price, cost, move = econ["price"], econ["cost"], econ["transfer_cost_per_log"]
salvage = econ["boxing_day_price"] * econ["boxing_day_sell_share"]
ratio = critical_ratio(price, cost, salvage)
season = hist.groupby("season").ratio.mean()
print(f"a log short loses ${price - cost:.2f}, a log left over ${cost - salvage:.2f}: critical ratio {ratio:.3f}")
print("average error by season (demand / forecast): " + ", ".join(f"{s} {v:.3f}" for s, v in season.items())
      + f"; store-level spread {hist.ratio.std(ddof=1):.3f}")

# scenarios: a season's chain-wide level, times a store's own error from the same history (a bootstrap)
rng = np.random.default_rng(51)
resid = (hist.ratio / hist.season.map(season)).to_numpy()
n, f = 20_000, fc.forecast_logs.to_numpy()
level = season.to_numpy()[rng.integers(len(season), size=n)]
demand = np.round(f * level[:, None] * resid[rng.integers(len(resid), size=(n, len(f)))])

# A: every store on its own
qa = np.array([np.ceil(q_empirical(fi * hist.ratio, ratio)) for fi in f])
sold_a = np.minimum(demand, qa)
profit_a = (price * sold_a - cost * qa + salvage * (qa - sold_a)).sum(axis=1)


# B: one chain order, shared out by forecast, then moved to the stores that run short
def chain(q_total):
    alloc = np.floor(q_total * f / f.sum())
    alloc[: int(q_total - alloc.sum())] += 1
    short, spare = np.maximum(demand - alloc, 0).sum(axis=1), np.maximum(alloc - demand, 0).sum(axis=1)
    moved = np.minimum(short, spare)
    sold = np.minimum(demand.sum(axis=1), q_total)
    return price * sold - cost * q_total + salvage * (q_total - sold) - move * moved, moved


results = {q: chain(q)[0].mean() for q in range(int(f.sum() * 0.9), int(f.sum() * 1.4), 5)}
qb = max(results, key=results.get)
profit_b, moved = chain(qb)
rows = [("A: each store its own newsvendor", int(qa.sum()), profit_a.mean(), np.quantile(profit_a, 0.05),
         (qa - sold_a).sum(axis=1).mean(), 0.0),
        ("B: chain order + moves", qb, profit_b.mean(), np.quantile(profit_b, 0.05),
         np.maximum(qb - demand.sum(axis=1), 0).mean(), moved.mean())]
print(pd.DataFrame(rows, columns=["option", "logs_made", "expected_profit", "p5_profit", "left_over", "logs_moved"])
      .round(1).to_string(index=False))
print(f"forecast total {f.sum()}; B gains ${profit_b.mean() - profit_a.mean():,.2f} on average")
```

```text
a log short loses $20.50, a log left over $1.90: critical ratio 0.915
average error by season (demand / forecast): 2023 1.050, 2024 1.016, 2025 1.022; store-level spread 0.321
                          option  logs_made  expected_profit  p5_profit  left_over  logs_moved
A: each store its own newsvendor       3055          37969.0    34358.7     1100.8         0.0
          B: chain order + moves       2143          39881.5    36168.7      169.3       171.3
forecast total 1921; B gains $1,912.56 on average
```

- The ratio is very high (0.915): a missed sale loses $20.50, and a leftover only $1.90 after Boxing Day.
- **Option A** has to cover each store's own 91st percentile, and a store's demand varies by about 32%. It bakes 3,055 logs and throws or discounts about 1,100.
- **Option B** pools the stores. The **chain's** demand varies much less than each store's, because the store errors partly cancel. It needs only 2,143 logs, moves about 170, and earns **about $1,900 more**.
- B doesn't pool perfectly: the scenarios keep each season's chain-wide level (all stores high in 2023), and that part doesn't cancel (Days 44 and 45).

*Memo:* "Bake 2,140 yule logs for the chain (forecast 1,921) and plan to move about 170 between stores on December 22–23. It earns about $1,900 more than letting each store order on its own, and leaves 900 fewer logs for Boxing Day."

</details>

## Commit

```bash
git add .
git commit -m "Day 51: the newsvendor"
git push
```

### Day 51 checklist

- [ ] I can derive the critical ratio from the cost of one unit too few and one too many.
- [ ] I use the empirical quantile when the distribution's shape is in doubt.
- [ ] I include salvage and lost goodwill in the ratio.
- [ ] I know the profit curve is flat at the top and steeper on the short side.
- [ ] I can explain why pooling demand needs less stock, and why correlation limits it.

---

# DAY 52 — Inventory Policies and Safety Stock

The newsvendor orders once. A distribution centre orders **again and again**, and each order arrives after a **lead time** that varies. DC-MTL keeps six ambient items (pasta, canned tomatoes, coffee, olive oil, rice, cereal) and orders them weekly today, by habit.

Three questions decide its stock:

- How much to order each time?
- When to order (the **reorder point**)?
- How much extra to keep for bad luck (**safety stock**)?

Today you answer them with the classic formulas, then check the formulas with a simulation built on the DC's own history.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 52.1 What the history says |
| 0:45 – 1:45 | 52.2 How much to order, and when |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 52.3 Check it by simulation |
| 2:45 – 3:45 | 52.4 Ordering once a week; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: a faster supplier? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 52.1 What the history says

Create `day52/inventory.py`:

```python
"""Day 52: inventory policies for the ambient items at DC-MTL. EOQ, safety stock, reorder points, and a simulation
to check them. Unmet demand is lost (the stores are short-shipped).

Import it:  from inventory import load_items, eoq, sigma_lead_time, k_for_fill_rate, simulate_policy
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats


def load_items(con):
    """One row per item: demand a day (mean, sd), lead time in days from the purchase orders (mean, sd), costs."""
    settings = dict(con.sql("SELECT * FROM inventory_settings").fetchall())
    items = con.sql("""
        SELECT i.*, q.quoted_lead_days, d.mean_day, d.sd_day, l.lead_mean, l.lead_sd
        FROM dc_items i
        JOIN dc_suppliers q USING (supplier_id)
        JOIN (SELECT item_id, avg(cases) AS mean_day, stddev_samp(cases) AS sd_day FROM dc_daily_demand GROUP BY ALL) d
             USING (item_id)
        JOIN (SELECT item_id, avg(received_on - ordered_on) AS lead_mean, stddev_samp(received_on - ordered_on) AS lead_sd
              FROM purchase_orders GROUP BY ALL) l USING (item_id)
        ORDER BY item_id""").df().set_index("item_id")
    items["hold_per_case_year"] = items.cost_per_case * settings["holding_rate_per_year"]
    return items, settings


def demand_history(con, item):
    return con.sql(f"SELECT cases FROM dc_daily_demand WHERE item_id = '{item}' ORDER BY demand_date").df().cases.to_numpy()


def lead_history(con, item):
    return con.sql(f"""SELECT received_on - ordered_on AS days FROM purchase_orders WHERE item_id = '{item}'""").df().days.to_numpy()


def eoq(annual_demand, order_cost, hold_per_year):
    """Economic order quantity: balances ordering (K D / Q) against holding (h Q / 2)."""
    return np.sqrt(2 * annual_demand * order_cost / hold_per_year)


def sigma_lead_time(mean_day, sd_day, lead_mean, lead_sd):
    """Spread of demand over a random lead time: sqrt(L sd_d^2 + d^2 sd_L^2)."""
    return np.sqrt(lead_mean * sd_day ** 2 + mean_day ** 2 * lead_sd ** 2)


def normal_loss(k):
    """Expected units short per unit of sd when stock covers mean + k sd: L(k) = pdf(k) - k (1 - cdf(k))."""
    return stats.norm.pdf(k) - k * stats.norm.sf(k)


def k_for_fill_rate(fill, q, sigma):
    """The safety factor k that gives fill rate `fill` with order quantity q: sigma L(k) = (1 - fill) q."""
    target = (1 - fill) * q / sigma
    return optimize.brentq(lambda k: normal_loss(k) - target, -4, 6)


def simulate_policy(demand_pool, lead_pool, rng, days, reorder_point=None, order_qty=None, review=None, order_up_to=None,
                    start=None):
    """Day by day: orders arrive, demand is met from stock (the rest is lost), then the policy looks at the
    inventory position (on hand + on order). (s, Q): when the position is at or below s, order Q.
    (R, S): every R days, order up to S. Demand and lead times are resampled from history."""
    on_hand = start if start is not None else (order_up_to or reorder_point + order_qty)
    pipeline = {}                                                   # arrival day -> cases
    demand_total = shipped = orders = 0.0
    stock_days = 0.0
    for day in range(days):
        on_hand += pipeline.pop(day, 0)
        d = demand_pool[rng.integers(len(demand_pool))]
        ship = min(d, on_hand)
        on_hand -= ship
        demand_total += d
        shipped += ship
        stock_days += on_hand
        position = on_hand + sum(pipeline.values())
        qty = 0
        if review is None and position <= reorder_point:
            qty = order_qty
        elif review is not None and day % review == 0:
            qty = max(0, order_up_to - position)
        if qty > 0:
            arrive = day + int(lead_pool[rng.integers(len(lead_pool))])
            pipeline[arrive] = pipeline.get(arrive, 0) + qty
            orders += 1
    return {"fill_rate": shipped / demand_total, "avg_on_hand": stock_days / days, "orders": orders,
            "lost": demand_total - shipped}
```

Then `day52/run_inventory.py`:

```python
# ---------- 52.1 What the history says ----------
print("--- history")
print(items[["item_name", "supplier_id", "mean_day", "sd_day", "quoted_lead_days", "lead_mean", "lead_sd"]]
      .round(2).to_string())
```

```text
--- history
                           item_name supplier_id  mean_day  sd_day  quoted_lead_days  lead_mean  lead_sd
item_id
AM-01               Pasta 500 g (12)       SUP-A    119.34   36.17                 5       5.96     1.11
AM-02    Canned tomatoes 796 ml (24)       SUP-A     87.04   26.30                 5       6.00     1.06
AM-03               Coffee 900 g (6)       SUP-B     26.37   11.40                10      10.65     2.77
AM-04              Olive oil 1 L (6)       SUP-B     17.74    8.72                10      11.77     3.50
AM-05          Basmati rice 2 kg (6)       SUP-C     42.62   13.57                 7       8.19     1.60
AM-06          Oat cereal 500 g (12)       SUP-C     61.93   17.59                 7       8.08     1.62
```

The suppliers are **late**: SUP-A quotes 5 days and takes 6, SUP-B quotes 10 and takes 11–12 with a spread of 3 days, SUP-C quotes 7 and takes 8. The lead time that matters is the one in the purchase orders, not the contract.

## 52.2 How much to order, and when

A **continuous-review (s, Q) policy**: whenever the inventory position (stock on hand plus stock on order) falls to $$s$$, order $$Q$$.

- **How much**, the **economic order quantity**, balances the cost of ordering ($$K$$ per order, $$D/Q$$ orders a year) against the cost of holding ($$h$$ per case per year, $$Q/2$$ cases on average): $$Q^* = \sqrt{2DK/h}$$.
- **When**: $$s$$ covers the demand expected during the lead time plus a **safety stock**. Demand over a random lead time has spread

$$\sigma_{LT} = \sqrt{\bar L\,\sigma_d^2 + \bar d^{\,2}\,\sigma_L^2}$$

  The second term is the lead time's own variation, which is large for SUP-B.
- **How much safety**: two service measures are common.
  - **Cycle service level**: the chance of no stock-out in an order cycle, $$\Phi(k)$$.
  - **Fill rate**: the share of demand shipped.

  The fill rate is what the stores feel. With $$k$$ the safety factor, each cycle is short by $$\sigma_{LT}L(k)$$ on average, so a fill rate $$\beta$$ needs $$\sigma_{LT}L(k) = (1-\beta)\,Q$$.

```python
# ---------- 52.2 How much to order, and when ----------
print("--- policy")
pol = pd.DataFrame(index=items.index)
pol["Q"] = np.round(eoq(items.mean_day * YEAR, K, items.hold_per_case_year))
pol["sigma_LT"] = sigma_lead_time(items.mean_day, items.sd_day, items.lead_mean, items.lead_sd)
pol["k"] = [k_for_fill_rate(fill, q, s) for q, s in zip(pol.Q, pol.sigma_LT)]
pol["safety_stock"] = np.round(pol.k * pol.sigma_LT)
pol["s"] = np.round(items.mean_day * items.lead_mean + pol.safety_stock)
pol["cycle_service"] = stats.norm.cdf(pol.k)
pol["orders_a_year"] = items.mean_day * YEAR / pol.Q
print(pol.round(2).to_string())
```

```text
--- policy
              Q  sigma_LT     k  safety_stock      s  cycle_service  orders_a_year
item_id
AM-01    1990.0    159.49  0.35          55.0  766.0           0.64          21.89
AM-02    1293.0    112.41  0.40          45.0  567.0           0.66          24.57
AM-03     443.0     81.98  0.86          70.0  351.0           0.81          21.73
AM-04     393.0     68.94  0.83          57.0  266.0           0.80          16.47
AM-05     861.0     78.49  0.43          34.0  383.0           0.67          18.07
AM-06     828.0    112.26  0.68          76.0  576.0           0.75          27.30
```

A 98% fill rate needs only a small safety factor for pasta ($$k$$ = 0.35, a cycle service level of 64%). Each pasta order is big (1,990 cases, 17 days of demand), so stock-outs are rare in relative terms. Coffee and olive oil need more ($$k \approx 0.85$$): smaller orders, and a late, erratic supplier. **Cycle service and fill rate are very different targets.** Say which one you mean.

## 52.3 Check it by simulation

The formulas assume normal demand during the lead time and orders that don't cross. The simulation assumes neither: it resamples the DC's own daily demand and its own purchase-order lead times, for two years, 20 times.

```python
# ---------- 52.3 Check it by simulation ----------
print("--- simulation (s, Q)")


def run(item, reps=20, seed=52, **policy):
    dem, lead = demand_history(con, item), lead_history(con, item)
    rng = np.random.default_rng(seed)
    sims = pd.DataFrame([simulate_policy(dem, lead, rng, DAYS, **policy) for _ in range(reps)]).mean()
    i = items.loc[item]
    sims["cost_a_year"] = (sims.avg_on_hand * i.hold_per_case_year + sims.orders / DAYS * YEAR * K
                           + sims.lost / DAYS * YEAR * i.margin_per_case)
    return sims


rows = []
for item in items.index:
    r = pol.loc[item]
    s = run(item, reorder_point=r.s, order_qty=r.Q)
    rows.append((item, r.s, r.Q, round(s.fill_rate, 4), round(s.avg_on_hand, 1), round(s.orders / DAYS * YEAR, 1),
                 round(s.cost_a_year, 0)))
sq = pd.DataFrame(rows, columns=["item", "s", "Q", "fill_rate", "avg_on_hand", "orders_a_year", "cost_a_year"])
print(sq.to_string(index=False))
```

```text
--- simulation (s, Q)
 item     s      Q  fill_rate  avg_on_hand  orders_a_year  cost_a_year
AM-01 766.0 1990.0     0.9883       1077.4           21.4       8957.0
AM-02 567.0 1293.0     0.9885        710.1           23.9      10146.0
AM-03 351.0  443.0     0.9866        301.2           21.2       9867.0
AM-04 266.0  393.0     0.9881        265.3           15.9       6885.0
AM-05 383.0  861.0     0.9852        477.4           17.5       7381.0
AM-06 576.0  828.0     0.9866        504.1           26.6      12602.0
```

Every item reaches 98.5–98.9% against the 98% target. The formula is slightly **conservative** here, because the DC loses unmet demand rather than backordering it, and the formula counts backorders. The policies are right; the simulation is the check that makes you trust them.

## 52.4 Ordering once a week: (R, S)

Purchasing orders every item weekly today. A **periodic-review (R, S) policy**: every $$R$$ days, order up to $$S$$. It must protect over the review period **plus** the lead time:

$$S = \bar d\,(R + \bar L) + k\,\sigma_{R+L}$$

```python
# ---------- 52.4 Ordering once a week: (R, S) ----------
print("--- simulation (R, S)")
rows = []
for item in items.index:
    i = items.loc[item]
    sigma = sigma_lead_time(i.mean_day, i.sd_day, i.lead_mean + R, i.lead_sd)      # protect over review + lead time
    q = i.mean_day * R                                                              # a week's demand per order
    S = np.round(i.mean_day * (R + i.lead_mean) + k_for_fill_rate(fill, q, sigma) * sigma)
    s = run(item, review=R, order_up_to=S)
    rows.append((item, S, round(s.fill_rate, 4), round(s.avg_on_hand, 1), round(s.orders / DAYS * YEAR, 1),
                 round(s.cost_a_year, 0)))
rs = pd.DataFrame(rows, columns=["item", "S", "fill_rate", "avg_on_hand", "orders_a_year", "cost_a_year"])
print(rs.to_string(index=False))
print(f"total cost a year: (s, Q) ${sq.cost_a_year.sum():,.0f}; weekly (R, S) ${rs.cost_a_year.sum():,.0f}; "
      f"stock (s, Q) {sq.avg_on_hand.sum():,.0f} cases, (R, S) {rs.avg_on_hand.sum():,.0f}")
```

```text
--- simulation (R, S)
 item      S  fill_rate  avg_on_hand  orders_a_year  cost_a_year
AM-01 1725.0     0.9945        665.9           52.5      11095.0
AM-02 1257.0     0.9959        486.5           52.5      11535.0
AM-03  582.0     0.9929        228.5           52.5      12434.0
AM-04  437.0     0.9957        181.7           52.5      10586.0
AM-05  742.0     0.9938        272.2           52.5      10325.0
AM-06 1065.0     0.9945        389.2           52.5      13212.0
total cost a year: (s, Q) $55,838; weekly (R, S) $69,187; stock (s, Q) 3,336 cases, (R, S) 2,224
```

The weekly policy holds **less** stock (2,224 cases against 3,336) but places **52 orders a year per item**, against 16–27. At $150 an order it costs about $13,000 a year more across the six items. Two practical notes:

- **Joint ordering**: pasta and tomatoes come from the same supplier. One weekly order for both halves the order cost, which narrows the gap. This is an exercise worth doing.
- Weekly ordering is simpler to run. Use the numbers to decide whether the simplicity is worth $13,000.

> **Decision it supports:** how much and when to reorder, and how much safety stock a service target needs.
>
> **Other decisions you can support the same way:** spare parts for the fleet · hospital supplies · raw materials at a plant · cash kept in ATMs.

### The planted bug

`day52/bug_quoted_lead.py` sets the reorder points from the suppliers' **quoted** lead times, with no lead-time variation ("that's what the contract says"). Run it.

```python
rows = []
for item, i in items.iterrows():
    q = np.round(eoq(i.mean_day * 365, settings["order_cost_per_po"], i.hold_per_case_year))
    sigma = sigma_lead_time(i.mean_day, i.sd_day, i.quoted_lead_days, 0.0)          # quoted, and never late
    s = np.round(i.mean_day * i.quoted_lead_days + k_for_fill_rate(settings["target_fill_rate"], q, sigma) * sigma)
    rng = np.random.default_rng(52)
    sims = [simulate_policy(demand_history(con, item), lead_history(con, item), rng, 730, reorder_point=s, order_qty=q)
            for _ in range(20)]
    rows.append((item, s, q, round(np.mean([x["fill_rate"] for x in sims]), 4)))
print(pd.DataFrame(rows, columns=["item", "s", "Q", "fill_rate"]).to_string(index=False))
```

<details><summary>What happens, and why</summary>

```text
 item     s      Q  fill_rate
AM-01 583.0 1990.0     0.9491
AM-02 431.0 1293.0     0.9460
AM-03 277.0  443.0     0.9500
AM-04 184.0  393.0     0.9228
AM-05 293.0  861.0     0.9461
AM-06 438.0  828.0     0.9318
```

The fill rate falls to 92–95%, against the 98% promised. Two errors compound: the suppliers are a day or more late on average, so even the expected lead-time demand is too low; and the lead time's spread, $$\bar d^{\,2}\sigma_L^2$$, is left out of $$\sigma_{LT}$$. Olive oil (SUP-B: 1.8 days late, with a 3.5-day spread) suffers most. Measure lead times from receipts; contracts describe what suppliers promise, not what they do.

</details>

## Mini-project: a faster supplier?

SUP-D offers coffee (AM-03) and olive oil (AM-04) with a lead time of 4 ± 1 days, instead of SUP-B's 11–12 ± 3, at **2% more** per case. Set a 98% fill-rate (s, Q) policy for each supplier and compare the full cost a year: purchases, holding, orders and lost margin.

<details><summary>Solution</summary>

```python
"""Day 52 mini-project: supplier SUP-D offers coffee (AM-03) and olive oil (AM-04) with a lead time of 4 +/- 1 days
instead of SUP-B's 10 (11-12 in practice, +/- 3), at 2% more per case. Set a 98% fill-rate (s, Q) policy for each
supplier and compare the full cost a year: purchases, holding, orders and lost margin. Check with the simulation.

Run from the repository root (after day06/load_tables.py):  python day52/mini_project_supplier.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, "day52")
from inventory import demand_history, eoq, k_for_fill_rate, lead_history, load_items, sigma_lead_time, simulate_policy

pd.set_option("display.width", 180)
con = duckdb.connect("northbound.duckdb", read_only=True)
items, settings = load_items(con)
K, fill = settings["order_cost_per_po"], settings["target_fill_rate"]
offers = con.sql("SELECT * FROM dc_supplier_offers").df().set_index("item_id")
rows = []
for item, o in offers.iterrows():
    i = items.loc[item]
    dem = demand_history(con, item)
    for supplier, lead_mean, lead_sd, premium, leads in [
            (i.supplier_id, i.lead_mean, i.lead_sd, 0.0, lead_history(con, item)),
            (o.supplier_id, o.lead_mean_days, o.lead_sd_days, o.price_premium,
             np.clip(np.round(np.random.default_rng(7).normal(o.lead_mean_days, o.lead_sd_days, 500)), 1, None))]:
        cost = i.cost_per_case * (1 + premium)
        hold = cost * settings["holding_rate_per_year"]
        q = np.round(eoq(i.mean_day * 365, K, hold))
        sigma = sigma_lead_time(i.mean_day, i.sd_day, lead_mean, lead_sd)
        s = np.round(i.mean_day * lead_mean + k_for_fill_rate(fill, q, sigma) * sigma)
        rng = np.random.default_rng(520)                          # the same demand days for both suppliers
        sims = pd.DataFrame([simulate_policy(dem, leads, rng, 730, reorder_point=s, order_qty=q) for _ in range(20)]).mean()
        year = 365 / 730
        purchases = (i.mean_day * 365 - sims.lost * year) * cost          # cases shipped a year, at the price
        rows.append((item, supplier, s, q, round(sims.fill_rate, 4), round(sims.avg_on_hand, 1),
                     round(purchases), round(sims.avg_on_hand * hold), round(sims.orders * year * K),
                     round(sims.lost * year * i.margin_per_case)))
t = pd.DataFrame(rows, columns=["item", "supplier", "s", "Q", "fill_rate", "avg_on_hand", "purchases", "holding",
                                "ordering", "lost_margin"])
t["total"] = t[["purchases", "holding", "ordering", "lost_margin"]].sum(axis=1)
print(t.to_string(index=False))
for item, g in t.groupby("item"):
    print(f"{item}: switching to {g.supplier.iloc[1]} changes the cost by ${g.total.iloc[1] - g.total.iloc[0]:+,.0f} a year "
          f"(stock {g.avg_on_hand.iloc[1] - g.avg_on_hand.iloc[0]:+,.0f} cases)")
    other = g[["holding", "ordering", "lost_margin"]].sum(axis=1)
    print(f"    break-even premium: {(other.iloc[0] - other.iloc[1]) / g.purchases.iloc[0]:.2%} "
          f"(the faster supplier saves ${other.iloc[0] - other.iloc[1]:,.0f} a year before the price)")
```

```text
 item supplier     s     Q  fill_rate  avg_on_hand  purchases  holding  ordering  lost_margin  total
AM-03    SUP-B 351.0 443.0     0.9853        297.9     557488     4379      3191         2522 567580
AM-03    SUP-D 117.0 439.0     0.9874        237.2     569863     3556      3229         2162 578810
AM-04    SUP-B 266.0 393.0     0.9874        259.8     322135     3273      2415         1241 329064
AM-04    SUP-D  76.0 389.0     0.9876        205.3     328651     2638      2434         1219 334942
AM-03: switching to SUP-D changes the cost by $+11,230 a year (stock -61 cases)
    break-even premium: 0.21% (the faster supplier saves $1,145 a year before the price)
AM-04: switching to SUP-D changes the cost by $+5,878 a year (stock -54 cases)
    break-even premium: 0.20% (the faster supplier saves $638 a year before the price)
```

- The faster supplier cuts the reorder point by two-thirds and the stock by 54–61 cases. But holding costs only 25% a year, so that saves about $1,100 (coffee) and $640 (olive oil) a year.
- The 2% premium adds about $12,400 and $6,500 a year to the purchases. **Decline the offer.**
- The break-even premium is about **0.2%**. A shorter lead time is worth exactly the stock it saves, and stock is cheap for items that don't spoil.

*Note to Purchasing:* "Stay with SUP-B for coffee and olive oil. SUP-D's faster delivery would save about $1,800 a year in stock and lost sales, but its 2% price premium adds about $18,900: $17,100 a year more in all. It would be worth it at a premium of 0.2% or less. Instead, raise SUP-B's late deliveries with them: they average 1–2 days beyond the quote."

</details>

## Commit

```bash
git add .
git commit -m "Day 52: inventory policies and safety stock"
git push
```

### Day 52 checklist

- [ ] I can compute an EOQ, a safety stock and a reorder point.
- [ ] I include lead-time variation, measured from receipts, in the safety stock.
- [ ] I know the difference between cycle service level and fill rate.
- [ ] I can set an (R, S) policy and compare it with (s, Q).
- [ ] I check inventory formulas with a simulation on the real history.

---

# DAY 53 — Two-Stage Stochastic Programming

The newsvendor has one product and no second chance. The bakery has 30 products sharing ovens, bakers, packing and starter. It also has a second chance: a **mid-week top-up bake** in 40 overtime oven hours ($55 an hour extra), once early-week sales show where demand is.

A **two-stage stochastic program** plans for that:

- **Stage 1** decides now (the week's plan).
- **Stage 2** decides later, differently in every scenario (the top-up, what sells, what goes to the day-old shelf).
- The objective is the **expected** profit over the scenarios.

Today you build it with 103 historical weeks as scenarios and compare its plan with Day 6's.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 53.1 One product: the newsvendor as a scenario LP |
| 0:45 – 1:45 | 53.2 The extensive form: 30 products, 103 weeks |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 53.3 The stochastic plan against the Day 6 plan |
| 2:45 – 3:45 | 53.4 What an oven hour is worth now; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: lease ovens for December? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 53.1 One product: the newsvendor as a scenario LP

Write the scenarios out. With $$x$$ the first-stage plan, and $$y_w, s_w$$ the top-up and sales in scenario $$w$$ (probability $$\pi_w$$):

$$\max\; \sum_w \pi_w \Big[\, p\,s_w - c\,(x + y_w) + v\,(x + y_w - s_w) - h\,y_w \Big]$$

$$\text{s.t.}\quad s_w \le x + y_w,\quad s_w \le D_w,\quad \text{capacity on } x \text{ and on } y_w$$

Here $$p$$ is the price, $$c$$ the ingredient cost, $$v$$ the salvage value on the day-old shelf, and $$h$$ the top-up's extra oven cost. Each scenario gets its own copy of the second-stage variables. This is the **extensive form**. Create `day53/two_stage.py`:

```python
"""Day 53: two-stage stochastic programming for the bakery's week.
Stage 1 (now): how many batches to plan for the week, within the ovens, bakers, packing and starter.
Stage 2 (once demand is known, a different answer in every scenario): top up with a mid-week bake in overtime
oven hours, sell what is wanted, put leftovers on the day-old shelf.

Import it:  from two_stage import load_two_stage, build_two_stage, scenario_profits
"""
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from bakery import load_data
from correlated import error_table


def load_two_stage(con, december_only=False):
    """The Day 6 data, per-batch economics, the day-old shelf and top-up settings, and the scenarios:
    every historical week's forecast errors applied to this week's forecast (Day 44: products move together)."""
    d = load_data(con)
    e = con.sql("SELECT sku, units_per_batch, transfer_price, ingredient_cost FROM bakery_range").df().set_index("sku")
    st = dict(con.sql("SELECT * FROM bakery_two_stage").fetchall())
    econ = pd.DataFrame({"price": e.units_per_batch * e.transfer_price, "cost": e.units_per_batch * e.ingredient_cost})
    econ["salvage"] = econ.price * st["day_old_price_share"] * st["day_old_sell_share"]
    err = error_table(con)
    if december_only:
        err = err[[w.month == 12 for w in pd.to_datetime(err.index)]]
    bread, other = ((err["BK-SD"] + err["BK-WW"]) / 2).to_numpy(), err["BK-CR"].to_numpy()
    demand = {(p, w): d["cap"][p] * (bread[w] if d["category"][p] == "bread" else other[w])
              for p in d["products"] for w in range(len(err))}
    return d, econ, st, demand, list(err.index.astype(str))


def build_two_stage(d, econ, st, demand, weeks, plan=None, anticipate=False, lease=None):
    """The extensive form: one copy of the second stage per scenario, all equally likely.
    plan: fix the first stage (to evaluate a given plan). anticipate=True: the planted bug (Day 53).
    lease: (extra oven hours, cost a week) to offer as a yes/no first-stage choice (the mini-project)."""
    m = pyo.ConcreteModel(name="two_stage_bakery")
    m.P, m.R, m.W = pyo.Set(initialize=d["products"]), pyo.Set(initialize=d["resources"]), pyo.Set(initialize=range(len(weeks)))
    use = lambda r, p: d["use"].get((r, p), 0.0)
    # stage 1
    if anticipate:
        m.x = pyo.Var(m.P, m.W, within=pyo.NonNegativeReals)              # a plan for each scenario: the bug
        x = lambda p, w: m.x[p, w]
    else:
        m.x = pyo.Var(m.P, within=pyo.NonNegativeReals)
        x = lambda p, w: m.x[p]
    m.lease = pyo.Var(within=pyo.Binary) if lease else pyo.Param(initialize=0, mutable=False)
    extra = lambda r: (lease[0] * m.lease if lease and r == "oven_hours" else 0)
    if anticipate:
        m.capacity = pyo.Constraint(m.R, m.W, rule=lambda m, r, w: sum(use(r, p) * m.x[p, w] for p in m.P) <= d["capacity"][r])
    else:
        m.capacity = pyo.Constraint(m.R, rule=lambda m, r: sum(use(r, p) * m.x[p] for p in m.P) <= d["capacity"][r] + extra(r))
    if plan is not None:
        for p in m.P:
            m.x[p].fix(plan.get(p, 0.0))
    # stage 2, scenario by scenario
    m.y = pyo.Var(m.P, m.W, within=pyo.NonNegativeReals)                  # top-up batches
    m.s = pyo.Var(m.P, m.W, within=pyo.NonNegativeReals)                  # batches sold
    m.topup = pyo.Constraint(m.W, rule=lambda m, w: sum(use("oven_hours", p) * m.y[p, w] for p in m.P) <= st["topup_oven_hours"])
    m.sell_made = pyo.Constraint(m.P, m.W, rule=lambda m, p, w: m.s[p, w] <= x(p, w) + m.y[p, w])
    m.sell_wanted = pyo.Constraint(m.P, m.W, rule=lambda m, p, w: m.s[p, w] <= demand[p, w])

    def week_profit(m, w):
        return sum(econ.price[p] * m.s[p, w] - econ.cost[p] * (x(p, w) + m.y[p, w])
                   + econ.salvage[p] * (x(p, w) + m.y[p, w] - m.s[p, w])
                   - st["topup_cost_per_oven_hour"] * use("oven_hours", p) * m.y[p, w] for p in m.P)
    m.week_profit = pyo.Expression(m.W, rule=week_profit)
    lease_cost = lease[1] * m.lease if lease else 0
    m.expected_profit = pyo.Objective(expr=sum(m.week_profit[w] for w in m.W) / len(weeks) - lease_cost,
                                      sense=pyo.maximize)
    return m


def scenario_profits(m):
    return np.array([pyo.value(m.week_profit[w]) for w in m.W])
```

The scenarios are Day 44's whole historical weeks: each week's forecast errors applied to this week's forecast, so products move together as they did. First the check, one product with no top-up and no capacity limit:

```python
# ---------- 53.1 One product: the newsvendor as a scenario LP ----------
print("--- one product")
cr = {"products": ["BK-CR"], "resources": ["oven_hours"], "use": {("oven_hours", "BK-CR"): d["use"]["oven_hours", "BK-CR"]},
      "capacity": {"oven_hours": 1e9}, "cap": {"BK-CR": d["cap"]["BK-CR"]}}             # no capacity limit, no top-up
m = build_two_stage(cr, econ, {**st, "topup_oven_hours": 0}, demand, weeks)
solve(m)
e = econ.loc["BK-CR"]
hist = [demand["BK-CR", w] for w in range(W)]
print(f"scenario LP: {pyo.value(m.x['BK-CR']):.2f} batches; Day 51's formula with the day-old shelf: "
      f"{q_empirical(hist, critical_ratio(e.price, e.cost, e.salvage)):.2f}")
```

```text
--- one product
scenario LP: 31.81 batches; Day 51's formula with the day-old shelf: 31.81
```

The LP finds Day 51's empirical newsvendor quantity exactly. A two-stage program **is** a newsvendor when there is one product and nothing else. Its power is everything else: shared capacity, a second chance, many products at once.

## 53.2 The extensive form: 30 products, 103 weeks

```python
# ---------- 53.2 The extensive form: 30 products, 103 weeks ----------
print("--- extensive form")
sp = build_two_stage(d, econ, st, demand, weeks)
solve(sp)
print(f"{sp.nvariables():,} variables, {sp.nconstraints():,} constraints")
print(f"stochastic plan: expected profit ${pyo.value(sp.expected_profit):,.2f} a week")
```

```text
--- extensive form
6,210 variables, 6,287 constraints
stochastic plan: expected profit $112,600.11 a week
```

The model has about 6,200 variables, 200 times Day 6's, and HiGHS solves it in about a second. The size grows with the number of scenarios. Day 54 asks how many scenarios are enough.

## 53.3 The stochastic plan against the Day 6 plan

Evaluate both plans the same way: fix the first stage, and let the second stage do its best in each of the 103 weeks.

```python
# ---------- 53.3 The stochastic plan against the Day 6 plan ----------
print("--- plans compared")
det = build_model(d)
solve(det)
day6 = {p: pyo.value(det.x[p]) for p in d["products"]}
stoch = {p: pyo.value(sp.x[p]) for p in d["products"]}
rows, topups = [], {}
for name, plan in [("Day 6 plan (the forecast)", day6), ("stochastic plan", stoch)]:
    ev = build_two_stage(d, econ, st, demand, weeks, plan=plan)
    solve(ev)
    prof = scenario_profits(ev)
    topups[name] = pd.Series({p: np.mean([pyo.value(ev.y[p, w]) for w in ev.W]) for p in ev.P})
    y = sum(pyo.value(ev.y[p, w]) for p in ev.P for w in ev.W) / W
    left = sum(pyo.value(plan[p] + ev.y[p, w] - ev.s[p, w]) for p in ev.P for w in ev.W) / W
    rows.append((name, round(sum(plan.values()), 1), round(prof.mean(), 2), round(np.quantile(prof, 0.05), 2),
                 round(prof.min(), 2), round(y, 1), round(left, 1)))
print(pd.DataFrame(rows, columns=["plan", "batches", "expected_profit", "p5", "worst_week", "topup_batches",
                                  "leftover_batches"]).to_string(index=False))
diff = pd.DataFrame({"category": pd.Series(d["category"]), "forecast": pd.Series(d["cap"]), "day6": pd.Series(day6),
                     "stochastic": pd.Series(stoch)}).round(2)
diff["change"] = (diff.stochastic - diff.day6).round(2)
diff["topup_day6"] = topups["Day 6 plan (the forecast)"].round(2)
diff["topup_stochastic"] = topups["stochastic plan"].round(2)
print("largest changes:")
print(diff[diff.change.abs() > 0.5].sort_values("change").to_string())
```

```text
--- plans compared
                     plan  batches  expected_profit        p5  worst_week  topup_batches  leftover_batches
Day 6 plan (the forecast)    429.6        111778.48 109095.30    73974.37           34.7               6.1
          stochastic plan    429.2        112600.11 111994.46    79042.69           34.6               2.5
largest changes:
           category  forecast  day6  stochastic  change  topup_day6  topup_stochastic
BK-30        cookie        20  20.0       13.15   -6.85        0.32              6.64
BK-SD         bread        35  35.0       30.84   -4.16        0.00              0.08
BK-03         bread        25  25.0       21.37   -3.63        0.33              3.69
BK-27        cookie        25  25.0       23.74   -1.26        0.56              1.43
BK-14  viennoiserie        26  26.0       24.86   -1.14        0.00              0.00
BK-28        cookie        20  20.0       18.99   -1.01        0.45              1.23
BK-25        cookie        16  16.0       14.99   -1.01        0.36              1.18
BK-29        cookie        19  19.0       18.08   -0.92        0.26              0.90
BK-WW         bread        45  45.0       44.11   -0.89        0.47              0.99
BK-CR  viennoiserie        31  31.0       30.11   -0.89        0.00              0.00
BK-23          cake        20  20.0       19.14   -0.86        0.00              0.00
BK-13  viennoiserie        27  27.0       26.40   -0.60        0.00              0.00
BK-18  viennoiserie        30  25.0       28.40    3.40        0.00              0.00
BK-26        cookie         9   1.6        7.72    6.12        7.16              1.22
BK-22          cake         9   0.0        7.40    7.40        0.08              0.02
BK-07         bread        29   0.0        7.79    7.79       24.04             15.83
```

- The stochastic plan earns **$822 a week more** on average ($42,700 a year), and its bad weeks are much better: the 5th percentile is $2,900 higher.
- It bakes almost the same total, but **a different mix**. The Day 6 plan leaves BK-07 out, then spends 24 batches of top-up on it in nearly every week: the top-up becomes a costly second shift for a product it could have planned.
- The stochastic plan puts BK-07, BK-22 and BK-26 into the plan. It **holds back** on products whose demand is uncertain and can be baked mid-week, such as cookies (BK-30) and rye (BK-03), and tops them up only in the weeks they sell. It plans **less** sourdough than the forecast (30.8 instead of 35), freeing starter for other products.
- The deterministic model can't see any of this, because it doesn't know there will be a second chance.

## 53.4 What an oven hour is worth now

```python
# ---------- 53.4 What an oven hour is worth now ----------
print("--- capacity values")
sp.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
solve(sp)
det.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
solve(det)
for r in d["resources"]:
    print(f"{r:>18}: Day 6 (forecast) ${det.dual[det.capacity_limit[r]]:7.2f}   stochastic (expected) ${sp.dual[sp.capacity[r]]:7.2f}")
```

```text
--- capacity values
       baker_hours: Day 6 (forecast) $   0.00   stochastic (expected) $   0.00
        oven_hours: Day 6 (forecast) $  99.34   stochastic (expected) $  94.77
     packing_hours: Day 6 (forecast) $  94.46   stochastic (expected) $  86.02
 sourdough_starter: Day 6 (forecast) $  54.09   stochastic (expected) $  56.34
```

In a two-stage program the first-stage duals are **expected** shadow prices, averaged over the scenarios. An oven hour is worth $94.77 rather than $99.34, and a packing hour $86 rather than $94. Part of their value at the forecast came from demand that may not be there, and the top-up can cover some of the rest. Capacity decisions taken on deterministic duals overvalue capacity a little here. The mini-project makes such a decision properly.

> **Decision it supports:** a plan committed before uncertainty resolves, when there are ways to adjust afterwards.
>
> **Other decisions you can support the same way:** staffing a week before its calls arrive (overtime as recourse) · buying energy a day ahead (the spot market as recourse) · pre-positioning stock before a storm · capacity bought now, used later.

### The planted bug

`day53/bug_anticipate.py` gives every scenario its own plan, "so the plan fits each week". Run it.

```python
m = build_two_stage(d, econ, st, demand, weeks, anticipate=True)
solve(m)
print(f"expected profit ${pyo.value(m.expected_profit):,.2f} a week")
for week in ("2024-12-16", "2025-01-06", "2025-06-30"):
    w = weeks.index(week)
    print(f"week like {week}: plan {sum(pyo.value(m.x[p, w]) for p in m.P):.1f} batches, "
          f"sourdough {pyo.value(m.x['BK-SD', w]):.2f}, croissants {pyo.value(m.x['BK-CR', w]):.2f}")
```

<details><summary>What happens, and why</summary>

```text
expected profit $113,648.14 a week
week like 2024-12-16: plan 435.7 batches, sourdough 23.70, croissants 46.24
week like 2025-01-06: plan 388.8 batches, sourdough 29.92, croissants 19.34
week like 2025-06-30: plan 427.3 batches, sourdough 35.46, croissants 29.10
```

The expected profit is $113,648, about $1,050 more than the stochastic plan, and the plan bakes 46 batches of croissants for a week like Christmas and 19 for a week like early January. Nice, if you know which week is coming. But the plan must be made **before** demand is known. Letting the first stage depend on the scenario breaks **non-anticipativity**: the model peeks at the future.

What it computes has a name, the **wait-and-see** value. The gap to the real stochastic plan is the value of perfect information, which Day 55 uses. In a stochastic program, every first-stage variable has **no scenario index**.

</details>

## Mini-project: lease ovens for December?

The lease from Day 21 adds 100 oven hours for $900 a week, and must be signed before the week's demand is known. Make it a yes/no first-stage variable. Decide it with all 103 historical weeks, then with December's weeks only, and compare with the decision on the forecast.

<details><summary>Solution</summary>

```python
"""Day 53 mini-project: lease extra ovens for a December week? The lease (Day 21) adds 100 oven hours for $900 and
must be signed before the week's demand is known. Decide it inside the two-stage model (a yes/no first-stage
variable), with December's historical weeks as scenarios, and with all weeks. Compare with the decision made on the
forecast alone.

Run from the repository root (after day06/load_tables.py):  python day53/mini_project_lease.py
"""
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

for folder in ("day06", "day42", "day44", "day53"):
    sys.path.insert(0, folder)
from bakery import build_model, solve
from two_stage import build_two_stage, load_two_stage

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
hours, cost = con.sql("SELECT extra_capacity, cost_per_week FROM bakery_investments WHERE option = 'lease_ovens'").fetchall()[0]

d, *_ = load_two_stage(con)
gain_forecast = []
for extra in (0, hours):
    m = build_model({**d, "capacity": {**d["capacity"], "oven_hours": d["capacity"]["oven_hours"] + extra}})
    solve(m)
    gain_forecast.append(pyo.value(m.total_profit))
print(f"on the forecast: the lease adds ${gain_forecast[1] - gain_forecast[0]:,.2f} a week for ${cost:,.0f}: "
      f"net ${gain_forecast[1] - gain_forecast[0] - cost:+,.2f}")

rows = []
for label, dec in [("all 103 weeks", False), ("December weeks", True)]:
    d, econ, st, demand, weeks = load_two_stage(con, december_only=dec)
    value = {}
    for lease in (0, 1):
        m = build_two_stage(d, econ, st, demand, weeks, lease=(hours, cost))
        m.lease.fix(lease)
        solve(m)
        value[lease] = pyo.value(m.expected_profit)
    m = build_two_stage(d, econ, st, demand, weeks, lease=(hours, cost))
    solve(m)
    rows.append((label, len(weeks), round(value[0], 2), round(value[1], 2), round(value[1] - value[0], 2),
                 "lease" if pyo.value(m.lease) > 0.5 else "don't lease"))
print(pd.DataFrame(rows, columns=["scenarios", "weeks", "no_lease", "lease", "net_gain_of_lease", "model_decides"])
      .to_string(index=False))
```

```text
on the forecast: the lease adds $1,083.62 a week for $900: net $+183.62
     scenarios  weeks  no_lease     lease  net_gain_of_lease model_decides
 all 103 weeks    103 112600.11 113084.92             484.82         lease
December weeks     10 113773.36 114314.08             540.72         lease
```

- On the forecast, the lease pays only $184 a week net, a close call.
- In the two-stage model it pays **$485 a week** (all weeks) and **$541** (December). Extra oven hours let the plan include more products **and** keep top-up room for the weeks demand runs high. The forecast can't see that second use.
- The model decides the lease and the plan together, and the lease is a binary, so this is a small stochastic **MIP**. HiGHS solves it in seconds.

*Memo:* "Lease the extra ovens for December: in December-like weeks they add about $540 a week after the $900 lease, three times what the forecast suggests, because they also give room to bake more when demand runs high."

</details>

## Commit

```bash
git add .
git commit -m "Day 53: two-stage stochastic programming"
git push
```

Then run the tests: `tests/test_uncertainty.py` checks Days 51–53. It covers the critical ratio, the normal expected-profit formula against simulation, EOQ and lead-time spread, a policy with no uncertainty that never runs out, the scenario LP equal to the newsvendor, and the anticipative model never below the stochastic one. Add `"day51"`, `"day52"` and `"day53"` to the folders in `conftest.py`.

```python
"""Known answers for Days 51-53: the newsvendor, inventory policies, the two-stage model."""
import numpy as np
import pyomo.environ as pyo
import pytest

from bakery import solve
from inventory import eoq, sigma_lead_time, simulate_policy
from newsvendor import critical_ratio, expected_profit, normal_expected_profit, q_empirical
from two_stage import build_two_stage, load_two_stage


def test_newsvendor_formulas():
    assert critical_ratio(7.0, 2.0) == pytest.approx(5 / 7)
    assert critical_ratio(7.0, 2.0, salvage=1.0, penalty=1.0) == pytest.approx(6 / 7)
    assert q_empirical(range(1, 11), 0.7) == 7
    demand = np.random.default_rng(0).normal(100, 20, 400_000)
    assert expected_profit(110, demand, 7, 2) == pytest.approx(normal_expected_profit(110, 100, 20, 7, 2), rel=1e-3)


def test_inventory_formulas():
    assert eoq(1_000, 50, 2.5) == pytest.approx(200)
    assert sigma_lead_time(10, 3, 4, 0) == pytest.approx(6)              # sqrt(4) x 3 when the lead time is fixed


def test_policy_without_uncertainty():
    """Constant demand of 10 a day, a 2-day lead time: a reorder point of 30 never runs out."""
    r = simulate_policy(np.array([10]), np.array([2]), np.random.default_rng(1), 200, reorder_point=30, order_qty=50)
    assert r["fill_rate"] == 1.0
    assert r["orders"] == pytest.approx(200 * 10 / 50, abs=1)


def test_scenario_lp_is_the_newsvendor(con):
    d, econ, st, demand, weeks = load_two_stage(con)
    one = {"products": ["BK-CR"], "resources": ["oven_hours"], "use": {("oven_hours", "BK-CR"): 1.5},
           "capacity": {"oven_hours": 1e9}, "cap": {"BK-CR": d["cap"]["BK-CR"]}}
    m = build_two_stage(one, econ, {**st, "topup_oven_hours": 0}, demand, weeks)
    solve(m)
    e = econ.loc["BK-CR"]
    q = q_empirical([demand["BK-CR", w] for w in range(len(weeks))], critical_ratio(e.price, e.cost, e.salvage))
    assert pyo.value(m.x["BK-CR"]) == pytest.approx(q, abs=1e-6)


def test_knowing_the_future_is_worth_something(con):
    d, econ, st, demand, weeks = load_two_stage(con)
    values = []
    for anticipate in (False, True):
        m = build_two_stage(d, econ, st, demand, weeks, anticipate=anticipate)
        solve(m)
        values.append(pyo.value(m.expected_profit))
    assert values[1] >= values[0] - 1e-6
```

```text
80 passed
```

### Day 53 checklist

- [ ] I can write a two-stage stochastic program in extensive form.
- [ ] I can build scenarios from history that keep products moving together.
- [ ] I evaluate any plan by fixing the first stage and solving the second.
- [ ] I can explain how a stochastic plan differs from a deterministic one, and why.
- [ ] My first-stage variables never depend on the scenario (non-anticipativity).
