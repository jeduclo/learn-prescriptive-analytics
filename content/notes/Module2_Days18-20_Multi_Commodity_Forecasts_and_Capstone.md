# DAY 18 — Multi-Commodity and Multi-Period Flows

Days 14–17 moved one kind of pallet in one week. December is harder: ambient and chilled pallets share each DC's docks, chilled pallets need a refrigerated zone and can't wait, and demand climbs 20% in the two weeks before Christmas while the DCs can receive no more than usual. Today's model plans **two commodities over four weeks**, and shows when it pays to receive goods early and hold them.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 18.1 The December data |
| 0:45 – 1:45 | 18.2 Two commodities: plan them together or apart? |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 18.3 Four weeks: receive early, hold, ship |
| 2:45 – 3:45 | 18.4 Which limits bind, and when; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: a storm in Montreal |
| 4:45 – 5:00 | Commit with Git + checklist |

## 18.1 The December data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). Five new tables:

| Table | What it holds |
|---|---|
| `dc_weeks` | Per DC and week: receiving capacity (pallets in from suppliers), outbound capacity (pallets out to stores; +15% in weeks 3 and 4 with extra shifts), chilled zone capacity, ambient storage, and the ambient stock on hand at the start |
| `store_week_forecast` | The planning team's forecast: ambient and chilled pallets per store for weeks 1–4 (30 November to 21 December) |
| `store_pallets_history` | 53 weeks of actual pallets per store, to the week of 23 November 2026 (Day 19) |
| `store_week_actual` | What each store actually needed in weeks 1–4 (Day 19: only known afterwards) |
| `distribution_costs` | Receiving $3 a pallet, holding $2.50 a pallet-week, chilled lanes 15% dearer, a lost pallet $120 of margin, a same-week rush order $25 extra |

Create `day18/run_weeks.py` and start by comparing December's forecast with what the DCs can handle:

```python
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day18")
from flows import build_weeks, load_weeks, report, v

pd.set_option("display.width", 140)
con = duckdb.connect("northbound.duckdb", read_only=True)
solver = pyo.SolverFactory("highs")

# ---------- 18.1 December's demand against the DCs' capacity ----------
print(con.sql("""
    WITH dem AS (
        SELECT week, sum(ambient_pallets)::INTEGER AS ambient, sum(chilled_pallets)::INTEGER AS chilled
        FROM store_week_forecast GROUP BY week
    ), cap AS (
        SELECT week, sum(receiving_pallets)::INTEGER AS can_receive, sum(outbound_pallets)::INTEGER AS can_ship,
               sum(chilled_zone_pallets)::INTEGER AS chilled_zones
        FROM dc_weeks GROUP BY week
    )
    SELECT week, ambient, chilled, ambient + chilled AS total, can_receive, can_ship, chilled_zones
    FROM dem JOIN cap USING (week) ORDER BY week"""))
```

```text
┌───────┬─────────┬─────────┬───────┬─────────────┬──────────┬───────────────┐
│ week  │ ambient │ chilled │ total │ can_receive │ can_ship │ chilled_zones │
│ int64 │  int32  │  int32  │ int32 │    int32    │  int32   │     int32     │
├───────┼─────────┼─────────┼───────┼─────────────┼──────────┼───────────────┤
│     1 │    2593 │     911 │  3504 │        3910 │     3800 │          1040 │
│     2 │    2828 │     991 │  3819 │        3910 │     3800 │          1040 │
│     3 │    3137 │    1098 │  4235 │        3910 │     4370 │          1040 │
│     4 │    3137 │    1108 │  4245 │        3910 │     4370 │          1040 │
└───────┴─────────┴─────────┴───────┴─────────────┴──────────┴───────────────┘

```

Weeks 3 and 4 need about 4,240 pallets, while the three DCs can **receive** only 3,910 a week between them. Some of the peak must come in earlier. Chilled demand at the peak (about 1,100 pallets) is also above the chilled zones' 1,040.

## 18.2 Two commodities: plan them together or apart?

A **multi-commodity flow** has a flow variable per commodity, per arc, per week, linked by limits the commodities share. Ours:

- **Separate:** each store's ambient and chilled demand; the chilled zone (chilled pallets only); a balance per commodity at each DC (ambient may be held, chilled may not).
- **Shared:** receiving (all pallets in) and outbound docks (all pallets out).

Create `day18/flows.py`, with the model in one function:

```python
"""Day 18: two commodities over four weeks. Ambient and chilled pallets share each DC's docks;
ambient pallets can wait at a DC, chilled pallets cannot.

Import it:  from flows import load_weeks, build_weeks, report
"""
import pandas as pd
import pyomo.environ as pyo

KINDS = ["ambient", "chilled"]


def load_weeks(con, demand_table="store_week_forecast", weeks=4):
    costs = dict(con.sql("SELECT item, value FROM distribution_costs").fetchall())
    dcw = con.sql(f"SELECT * FROM dc_weeks WHERE week <= {weeks}").df().set_index(["dc_id", "week"])
    dem = con.sql(f"SELECT * FROM {demand_table} WHERE week <= {weeks}").df()
    demand = {}
    for r in dem.itertuples():
        demand["ambient", r.store_id, r.week] = r.ambient_pallets
        demand["chilled", r.store_id, r.week] = r.chilled_pallets
    lanes = con.sql("SELECT dc_id, store_id, cost_per_pallet FROM dc_store_lanes").fetchall()
    lane = {}
    for d, s, c in lanes:
        lane["ambient", d, s] = c
        lane["chilled", d, s] = round(c * costs["chilled_lane_factor"], 4)
    return {
        "dcs": sorted(dcw.index.get_level_values(0).unique()),
        "stores": sorted(dem.store_id.unique()),
        "weeks": list(range(1, weeks + 1)),
        "dc_weeks": dcw,
        "demand": demand,
        "lane": lane,
        "costs": costs,
    }


def build_weeks(d, link_weeks=True):
    """ship[k, d, s, t]: pallets of kind k from DC d to store s in week t.
    receive[d, t]: pallets received at DC d (ambient + chilled); stock[d, t]: ambient pallets held at the end of week t.
    short[k, s, t]: pallets a store needed but did not get (lost sales)."""
    c, w = d["costs"], d["dc_weeks"]
    m = pyo.ConcreteModel(name="distribution_weeks")
    m.K = pyo.Set(initialize=KINDS)
    m.D = pyo.Set(initialize=d["dcs"])
    m.S = pyo.Set(initialize=d["stores"])
    m.T = pyo.Set(initialize=d["weeks"], ordered=True)

    m.ship = pyo.Var(m.K, m.D, m.S, m.T, within=pyo.NonNegativeReals)
    m.receive = pyo.Var(m.K, m.D, m.T, within=pyo.NonNegativeReals)
    m.stock = pyo.Var(m.D, m.T, within=pyo.NonNegativeReals)
    m.short = pyo.Var(m.K, m.S, m.T, within=pyo.NonNegativeReals)

    def out(m, k, dc, t):
        return sum(m.ship[k, dc, s, t] for s in m.S)

    # Stores: shipments plus shortfall cover demand, kind by kind
    m.demand = pyo.Constraint(m.K, m.S, m.T, rule=lambda m, k, s, t:
                              sum(m.ship[k, dc, s, t] for dc in m.D) + m.short[k, s, t] == d["demand"][k, s, t])

    # DCs: ambient stock carries over; chilled goes out the week it comes in
    def ambient_balance(m, dc, t):
        before = (m.stock[dc, m.T.prev(t)] if t != m.T.first() else w.opening_ambient_stock[dc, 1]) if link_weeks else 0
        return before + m.receive["ambient", dc, t] == out(m, "ambient", dc, t) + m.stock[dc, t]

    m.ambient_balance = pyo.Constraint(m.D, m.T, rule=ambient_balance)
    m.chilled_balance = pyo.Constraint(m.D, m.T, rule=lambda m, dc, t: m.receive["chilled", dc, t] == out(m, "chilled", dc, t))

    # Shared and separate limits
    m.receiving = pyo.Constraint(m.D, m.T, rule=lambda m, dc, t: sum(m.receive[k, dc, t] for k in m.K) <= w.receiving_pallets[dc, t])
    m.outbound = pyo.Constraint(m.D, m.T, rule=lambda m, dc, t: sum(out(m, k, dc, t) for k in m.K) <= w.outbound_pallets[dc, t])
    m.chilled_zone = pyo.Constraint(m.D, m.T, rule=lambda m, dc, t: out(m, "chilled", dc, t) <= w.chilled_zone_pallets[dc, t])
    m.storage = pyo.Constraint(m.D, m.T, rule=lambda m, dc, t: m.stock[dc, t] <= (w.ambient_storage_pallets[dc, t] if link_weeks else 0))

    m.transport = pyo.Expression(expr=sum(d["lane"][k, dc, s] * m.ship[k, dc, s, t] for k in m.K for dc in m.D for s in m.S for t in m.T))
    m.inbound = pyo.Expression(expr=c["inbound_per_pallet"] * sum(m.receive[k, dc, t] for k in m.K for dc in m.D for t in m.T))
    m.holding = pyo.Expression(expr=c["holding_per_pallet_week"] * sum(m.stock[dc, t] for dc in m.D for t in m.T))
    m.lost = pyo.Expression(expr=c["lost_sale_per_pallet"] * sum(m.short[k, s, t] for k in m.K for s in m.S for t in m.T))
    m.cost = pyo.Objective(expr=m.transport + m.inbound + m.holding + m.lost)
    return m


def v(x):
    return round(pyo.value(x), 2) + 0.0


def report(m):
    rows = []
    for t in m.T:
        rows.append((t,
                     v(sum(m.ship[k, dc, s, t] for k in m.K for dc in m.D for s in m.S)),
                     v(sum(m.receive[k, dc, t] for k in m.K for dc in m.D)),
                     v(sum(m.stock[dc, t] for dc in m.D)),
                     v(sum(m.short["ambient", s, t] for s in m.S)),
                     v(sum(m.short["chilled", s, t] for s in m.S))))
    return pd.DataFrame(rows, columns=["week", "shipped", "received", "dc_stock_end", "short_ambient", "short_chilled"])
```

Points to slow down for:

- **Lost sales as a variable.** `short` lets a store get less than it needs, at $120 a pallet. Without it, a week where demand exceeds capacity makes the model infeasible; with it, the model tells you how much demand can't be met, and where.
- **Ambient carries over, chilled doesn't.** `ambient_balance` links the weeks through `stock`, as on Day 8. `chilled_balance` has no stock: what comes in goes out the same week.
- `link_weeks=False` gives the week-by-week version: no stock carried over, not even the opening stock.

First, one week, to see why commodities must be planned together:

```python
# ---------- 18.2 One week, two commodities: plan them together or apart? ----------
d1 = load_weeks(con, weeks=1)
together = build_weeks(d1)
solver.solve(together)
apart = build_weeks(d1)
apart.outbound.deactivate()           # each commodity planned as if it had the docks to itself
solver.solve(apart)
for name, m in (("together", together), ("apart", apart)):
    use = {dc: v(sum(m.ship[k, dc, s, 1] for k in m.K for s in m.S)) for dc in m.D}
    print(f"{name:<9} cost ${pyo.value(m.cost):,.2f}; pallets out of each DC {use}")
print("outbound capacity:", {dc: int(d1["dc_weeks"].outbound_pallets[dc, 1]) for dc in d1["dcs"]})
```

```text
together  cost $48,029.08; pallets out of each DC {'DC-MIS': 1304.0, 'DC-MTL': 1500.0, 'DC-OTT': 700.0}
apart     cost $45,401.57; pallets out of each DC {'DC-MIS': 1190.0, 'DC-MTL': 1474.0, 'DC-OTT': 840.0}
outbound capacity: {'DC-MIS': 1600, 'DC-MTL': 1500, 'DC-OTT': 700}
```

Planning each commodity **apart** (here: without the shared outbound limit) is $2,627 cheaper, and impossible: it sends 840 pallets out of Ottawa, which has docks for 700. Each commodity's plan is fine on its own; together they overload the shared docks. Ambient and chilled must be planned in one model, or the docks must be split between them in advance (which wastes capacity whenever one commodity needs less than its share).

> **Decision it supports:** how to share docks, trucks and zones between product types that compete for them.
>
> **Other decisions you can support the same way:** passengers and cargo on the same flights · several products on one pipeline · data streams on one network link.

## 18.3 Four weeks: receive early, hold, ship

```python
# ---------- 18.3 Four weeks ----------
d = load_weeks(con)
m = build_weeks(d)
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
solver.solve(m)
print(report(m).to_string(index=False))
print(f"4-week cost ${pyo.value(m.cost):,.2f} = transport ${pyo.value(m.transport):,.2f} + inbound ${pyo.value(m.inbound):,.2f} "
      f"+ holding ${pyo.value(m.holding):,.2f} + lost sales ${pyo.value(m.lost):,.2f}")

stock = pd.DataFrame({dc: [v(m.stock[dc, t]) for t in m.T] for dc in m.D}, index=list(m.T))
stock.index.name = "week"
print("ambient stock at each DC at the end of each week:")
print(stock.to_string())
```

```text
 week  shipped  received  dc_stock_end  short_ambient  short_chilled
    1   3504.0    3398.0         424.0            0.0            0.0
    2   3800.0    3910.0         534.0            0.0           19.0
    3   4177.0    3910.0         267.0            0.0           58.0
    4   4177.0    3910.0           0.0            0.0           68.0
4-week cost $256,444.53 = transport $190,598.03 + inbound $45,384.00 + holding $3,062.50 + lost sales $17,400.00
ambient stock at each DC at the end of each week:
      DC-MIS  DC-MTL  DC-OTT
week
1       87.0   307.0    30.0
2      137.0   307.0    90.0
3       91.0   131.0    45.0
4        0.0     0.0     0.0
```

What the plan does:

- **Weeks 1 and 2 receive more than they ship** (or, in week 1, ship from the opening stock): ambient stock at the DCs rises to 534 pallets by the end of week 2, ready for the peak.
- **Weeks 3 and 4 ship 4,177 pallets** with only 3,910 received: the difference comes out of stock, which is used up by the end of week 4.
- **Ambient demand is always met. Chilled demand is not**: 145 chilled pallets are lost over the four weeks, all in weeks 2–4. Chilled pallets can't be received early, and the chilled zones are too small for the peak.
- The cost splits into transport ($190,598), receiving ($45,384), holding ($3,063) and lost sales ($17,400).

## 18.4 Which limits bind, and when

```python
# ---------- 18.4 Which limits bind, and when? ----------
def prices(con_):
    return pd.DataFrame({dc: [round(abs(m.dual[con_[dc, t]]), 2) + 0.0 for t in m.T] for dc in m.D}, index=list(m.T))


for name, con_ in (("receiving", m.receiving), ("outbound", m.outbound), ("chilled zone", m.chilled_zone), ("storage", m.storage)):
    p = prices(con_)
    p.index.name = "week"
    print(f"shadow price of {name} ($ a pallet):")
    print(p.to_string())

week_by_week = build_weeks(d, link_weeks=False)
solver.solve(week_by_week)
print(f"without carrying ambient stock between weeks: ${pyo.value(week_by_week.cost):,.2f} "
      f"(+${pyo.value(week_by_week.cost) - pyo.value(m.cost):,.2f})")
flows = [pyo.value(x) for x in m.ship.values()]
print("every shipment a whole number of pallets:", all(abs(f - round(f)) < 1e-6 for f in flows))
```

```text
shadow price of receiving ($ a pallet):
      DC-MIS  DC-MTL  DC-OTT
week
1        0.0   65.23     0.0
2        2.5   67.73     2.5
3        5.0   70.23     5.0
4        7.5   72.73     7.5
shadow price of outbound ($ a pallet):
      DC-MIS  DC-MTL  DC-OTT
week
1       0.00    0.00   48.18
2       9.99    9.99   58.17
3       0.00    0.00   48.18
4       0.00    0.00   48.18
shadow price of chilled zone ($ a pallet):
      DC-MIS  DC-MTL  DC-OTT
week
1       0.00    0.00    0.00
2       0.00    0.00    7.23
3      68.76    7.95   50.30
4      66.26    5.50   47.85
shadow price of storage ($ a pallet):
      DC-MIS  DC-MTL  DC-OTT
week
1        0.0     0.0     0.0
2        0.0     0.0     0.0
3        0.0     0.0     0.0
4        0.0     0.0     0.0
without carrying ambient stock between weeks: $294,124.38 (+$37,679.85)
every shipment a whole number of pallets: True
```

Read the four tables of shadow prices together:

- **Montreal's receiving is the most valuable limit**: $65 to $73 a pallet in every week. A pallet received there in week 1 can be held and shipped in week 3, so every week's receiving is worth about the same, rising by the $2.50 holding cost per week (a pallet received later saves holding).
- **Ottawa's outbound docks are full every week** ($48–58 a pallet). Ottawa is short of docks, not of goods.
- **Chilled zones bind at the peak** (weeks 3–4), most in Mississauga ($66–69) and Ottawa ($48–50): those are the chilled pallets lost.
- **Storage never binds**: the DCs have room for more stock than the plan wants to hold. Bigger warehouses wouldn't help this December.
- Carrying stock between weeks is worth **$37,679.85** over the four weeks (against planning each week alone).

Unlike Day 14, there is no guarantee of whole-number answers in a multi-commodity flow: the shared limits break the special structure. This plan happens to come out whole; bigger ones often don't, and then rounding (or integer variables) is needed.

### The planted bug

`day18/bug_opening.py` "simplifies" the ambient balance. Run it, then compare pallets in with pallets out.

```python
con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_weeks(con)
w = d["dc_weeks"]
m = build_weeks(d)
for dc in m.D:
    for t in m.T:
        m.ambient_balance[dc, t].set_value(
            w.opening_ambient_stock[dc, 1] + m.receive["ambient", dc, t]
            == sum(m.ship["ambient", dc, s, t] for s in m.S) + m.stock[dc, t])
pyo.SolverFactory("highs").solve(m)
received = sum(pyo.value(m.receive["ambient", dc, t]) for dc in m.D for t in m.T)
shipped = sum(pyo.value(m.ship["ambient", dc, s, t]) for dc in m.D for s in m.S for t in m.T)
opening = sum(w.opening_ambient_stock[dc, 1] for dc in m.D)
print(f"4-week cost ${pyo.value(m.cost):,.2f}")
print(f"ambient: opening {opening:,.0f} + received {received:,.0f} = {opening + received:,.0f}, but shipped {shipped:,.0f}")
```

<details><summary>What happens, and how to spot it</summary>

```text
4-week cost $242,891.39
ambient: opening 530 + received 9,575 = 10,105, but shipped 11,695
```

The model is $13,553 cheaper, and has shipped 1,590 more ambient pallets than ever existed. The balance adds the **opening stock in every week** instead of last week's stock: the same 530 pallets are counted four times, and stock disappears between weeks. Day 9's conservation check (opening + received = shipped + final stock) exposes it in one line. It is the kind of check that belongs in the tests (`tests/test_distribution.py`, Day 20).

</details>

## Mini-project: a storm in Montreal

The weather service forecasts a major snowstorm in week 3: the Montreal DC will receive only half its usual pallets that week. The warning arrives at the start of week 1.

1. Re-plan with Montreal's week 3 receiving halved. What does the storm cost?
2. How does the plan prepare (stock and receiving at Montreal, week by week)?
3. Suppose instead the storm arrives without warning: weeks 1 and 2 go as originally planned. What is the warning worth?

<details><summary>Solution</summary>

```python
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day18")
from flows import build_weeks, load_weeks, report, v

pd.set_option("display.width", 140)
con = duckdb.connect("northbound.duckdb", read_only=True)
solver = pyo.SolverFactory("highs")
d = load_weeks(con)
base = build_weeks(d)
solver.solve(base)

storm = build_weeks(d)
half = d["dc_weeks"].receiving_pallets["DC-MTL", 3] / 2
storm.receiving["DC-MTL", 3].set_value(sum(storm.receive[k, "DC-MTL", 3] for k in storm.K) <= half)
solver.solve(storm)
print(report(storm).to_string(index=False))
print(f"storm costs ${pyo.value(storm.cost) - pyo.value(base.cost):,.2f} over four weeks")
for name, m in (("normal", base), ("storm", storm)):
    print(f"{name:<7} Montreal stock at end of weeks 1-4:", [v(m.stock["DC-MTL", t]) for t in m.T],
          " received:", [v(sum(m.receive[k, "DC-MTL", t] for k in m.K)) for t in m.T])

# If the storm comes as a surprise: weeks 1-2 as planned, then the storm hits
surprise = build_weeks(d)
for k in surprise.K:
    for dc in surprise.D:
        for t in (1, 2):
            surprise.receive[k, dc, t].fix(pyo.value(base.receive[k, dc, t]))
surprise.receiving["DC-MTL", 3].set_value(sum(surprise.receive[k, "DC-MTL", 3] for k in surprise.K) <= half)
solver.solve(surprise)
print(f"surprise storm costs ${pyo.value(surprise.cost) - pyo.value(base.cost):,.2f}; "
      f"knowing two weeks ahead is worth ${pyo.value(surprise.cost) - pyo.value(storm.cost):,.2f}")
```

```text
 week  shipped  received  dc_stock_end  short_ambient  short_chilled
    1   3435.0    3800.0         895.0            0.0           69.0
    2   3740.0    3910.0        1065.0            0.0           79.0
    3   4020.0    3160.0         205.0          122.0           93.0
    4   4115.0    3910.0           0.0           31.0           99.0
storm costs $56,486.71 over four weeks
normal  Montreal stock at end of weeks 1-4: [307.0, 307.0, 131.0, 0.0]  received: [1500.0, 1500.0, 1500.0, 1500.0]
storm   Montreal stock at end of weeks 1-4: [565.0, 625.0, 0.0, 0.0]  received: [1500.0, 1500.0, 750.0, 1500.0]
surprise storm costs $62,464.49; knowing two weeks ahead is worth $5,977.78
```

1. The storm costs **$56,486.71** over four weeks, even with two weeks' warning.
2. Montreal builds stock to 625 pallets by the end of week 2 (against 307 normally), receives 750 in week 3, and empties its stock that week. Notice the chilled losses in weeks 1 and 2: Montreal's receiving is shared, and the model uses it for ambient stock, giving up chilled pallets now to save ambient pallets later. Both cost $120 when lost, so the model is indifferent; if chilled losses hurt more (customers leave), they need a higher penalty.
3. Without warning, the storm costs $62,464.49: the **warning is worth $5,977.78**. That's a first number for the value of information, which Module 7 studies properly.

`fix()` on the week 1 and 2 receiving variables freezes decisions already taken: the standard way to model "what's done is done" in a re-plan.

</details>

## Commit

```bash
git add .
git commit -m "Day 18: multi-commodity, multi-week distribution"
git push
```

### Day 18 checklist

- [ ] I can model several commodities that share some limits and not others.
- [ ] I showed why planning commodities apart can overload a shared limit.
- [ ] I modelled stock that carries over for one commodity and not the other.
- [ ] I used a lost-sales variable to keep the model feasible and to measure unmet demand.
- [ ] I read a table of weekly shadow prices to find the binding limits.
- [ ] I used `fix()` to model decisions already taken, and valued a warning.

---

# DAY 19 — Feeding the Model with Forecasts

Day 18's plan used the planning team's forecast. But a plan is only as good as its forecast, and "good" for a forecast doesn't mean what forecasters usually measure. Today you build four forecasts from the history, plan with each, and then **live through December as it actually happened**: the forecast whose plan costs least in reality wins, which is not always the most accurate one.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 19.1 The history |
| 0:45 – 1:45 | 19.2 Four forecasts and their accuracy |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 19.3 Plan, then live through the actual weeks |
| 2:45 – 3:45 | 19.4 Is it better to forecast high? |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: a fifth forecast |
| 4:45 – 5:00 | Commit with Git + checklist |

## 19.1 The history

`store_pallets_history` has 53 weeks of pallets per store, ending with the week of 23 November 2026. Create `day19/compare.py`:

```python
import sys

import duckdb
import pandas as pd

sys.path.insert(0, "day19")
from evaluate import plan_and_live
from forecasts import accuracy, make_forecasts

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)

# ---------- 19.1 The history ----------
print(con.sql("""
    SELECT week_start, sum(pallets)::INTEGER AS pallets FROM store_pallets_history
    GROUP BY ALL ORDER BY week_start""").df().iloc[[0, 1, 2, 3, 4, 5, 26, -4, -3, -2, -1]].to_string(index=False))
```

```text
week_start  pallets
2025-11-24     3394
2025-12-01     3405
2025-12-08     3707
2025-12-15     4110
2025-12-22     4117
2025-12-29     3033
2026-05-25     3521
2026-11-02     3596
2026-11-09     3461
2026-11-16     3491
2026-11-23     3574
```

Last December the chain went from about 3,400 pallets a week to 4,117 in the week before Christmas, then fell to about 3,000 in the first week of January. This year's recent weeks run about 3,500–3,575. Any forecast for December 2026 has to bring both facts together: this year's level and last year's December shape.

## 19.2 Four forecasts and their accuracy

`day19/forecasts.py` builds three forecasts with SQL, in the same shape as `store_week_forecast` so the Day 18 model can read them unchanged. Each splits a store's total into ambient and chilled using the store's usual chilled share.

| Forecast | Weeks 1–4 total per store |
|---|---|
| `fc_last_week` | The last week of history, repeated |
| `fc_avg_4_weeks` | The average of the last 4 weeks, repeated |
| `fc_last_year_plus_3pct` | The same week last year, plus 3% growth (the planning team's method) |

```python
"""Day 19: four ways to forecast December's pallets, store by store.

Import it:  from forecasts import make_forecasts
"""
import duckdb
import pandas as pd


def make_forecasts(con):
    """Write one forecast table per method, in the same shape as store_week_forecast
    (week, store_id, ambient_pallets, chilled_pallets). Forecasts are made at the end of the history."""
    con.sql("""
        CREATE OR REPLACE TEMP VIEW hist AS
        SELECT h.*, (date_diff('week', h.week_start, (SELECT max(week_start) FROM store_pallets_history))) AS weeks_ago
        FROM store_pallets_history h""")
    con.sql("""
        CREATE OR REPLACE TEMP VIEW chilled_share AS
        SELECT store_id, chilled_pallets_per_week / pallets_per_week AS share FROM store_logistics""")
    totals = {
        "last_week": "SELECT store_id, w.week, pallets AS total FROM hist, range(1, 5) w(week) WHERE weeks_ago = 0",
        "avg_4_weeks": """SELECT store_id, w.week, avg(pallets) AS total FROM hist, range(1, 5) w(week)
                          WHERE weeks_ago < 4 GROUP BY store_id, w.week""",
        "last_year_plus_3pct": """SELECT store_id, 52 - weeks_ago AS week, 1.03 * pallets AS total FROM hist
                                  WHERE weeks_ago BETWEEN 48 AND 51""",
    }
    for name, sql in totals.items():
        con.sql(f"""
            CREATE OR REPLACE TEMP TABLE fc_{name} AS
            SELECT week::INTEGER AS week, store_id,
                   round(total) - round(round(total) * share) AS ambient_pallets,
                   round(round(total) * share) AS chilled_pallets
            FROM ({sql}) JOIN chilled_share USING (store_id)""")
    return ["fc_" + n for n in totals]


def accuracy(con, table):
    """Mean absolute error and bias per store-week, against what actually happened."""
    return con.sql(f"""
        SELECT '{table}' AS forecast,
               round(avg(abs(f.ambient_pallets + f.chilled_pallets - a.ambient_pallets - a.chilled_pallets)), 2) AS mae_pallets,
               round(avg(f.ambient_pallets + f.chilled_pallets - a.ambient_pallets - a.chilled_pallets), 2) AS bias_pallets,
               sum(f.ambient_pallets + f.chilled_pallets)::INTEGER AS forecast_total,
               sum(a.ambient_pallets + a.chilled_pallets)::INTEGER AS actual_total
        FROM {table} f JOIN store_week_actual a USING (week, store_id)""").df()
```

`date_diff('week', …)` counts weeks back from the last week of history; `range(1, 5) w(week)` repeats a value for weeks 1–4. `accuracy` compares each forecast with `store_week_actual`: **MAE** is the average miss per store and week, in pallets; **bias** is the average signed miss (negative means the forecast is too low).

```python
# ---------- 19.2 Four forecasts and their accuracy ----------
tables = make_forecasts(con) + ["store_week_forecast", "store_week_actual"]
acc = pd.concat([accuracy(con, t) for t in tables])
print(acc.to_string(index=False))
```

```text
              forecast  mae_pallets  bias_pallets  forecast_total  actual_total
          fc_last_week        12.38        -10.07           14296         15908
        fc_avg_4_weeks        12.30        -11.05           14140         15908
fc_last_year_plus_3pct         7.33         -0.66           15803         15908
   store_week_forecast         7.33         -0.66           15803         15908
     store_week_actual         0.00          0.00           15908         15908
```

- **Last week** and **4-week average** miss by about 12 pallets per store-week and are **biased low by about 10–11 pallets**: they know nothing about December, so they forecast November into the peak.
- **Last year plus 3%** misses by 7.3 and is almost unbiased: it carries last December's shape into this one.
- `store_week_forecast` is the planning team's version of the same method. It differs only in rounding: Python rounds halves to the even number (`round(2.5) = 2`), DuckDB rounds them away from zero (2.5 becomes 3). Tiny, but a reminder that "the same method" in two tools can give different numbers.
- `store_week_actual` is the **perfect forecast**, available only in hindsight: the benchmark for what any forecast could achieve.

## 19.3 Plan, then live through the actual weeks

A forecast's real test is the cost of the decisions made with it. December has a natural split:

1. **Decide now:** the supplier orders, i.e. how many pallets each DC receives each week. They are placed at the start of the month and can't be cancelled.
2. **Adapt later:** once actual demand is known week by week, shipments to stores adapt. A DC can **rush** extra pallets in the same week at $25 a pallet more (within its receiving capacity). Chilled pallets that arrive but aren't needed are thrown away.

Create `day19/evaluate.py`:

```python
"""Day 19: plan with a forecast, then live through the actual weeks.

Import it:  from evaluate import plan_and_live
"""
import sys

import pyomo.environ as pyo

sys.path.insert(0, "day18")
from flows import build_weeks, load_weeks

solver = pyo.SolverFactory("highs")


def plan_and_live(con, forecast_table, adjust=None):
    """1. Plan the four weeks with the forecast. The supplier orders (pallets received at each DC each week)
          are placed now and cannot be cancelled.
       2. Live through the actual weeks: shipments adapt to actual demand; a DC can order extra pallets
          for the same week at a premium (rush); chilled pallets that arrive but are not needed are thrown away.
       adjust(d), if given, changes the data (for example a capacity) in both steps.
       Returns (planned cost, actual cost, pallets rushed, chilled pallets thrown away, lost pallets)."""
    adjust = adjust or (lambda d: d)
    plan = build_weeks(adjust(load_weeks(con, forecast_table)))
    solver.solve(plan)

    d = adjust(load_weeks(con, "store_week_actual"))
    live = build_weeks(d)
    premium = d["costs"]["emergency_premium_per_pallet"]
    live.rush = pyo.Var(live.K, live.D, live.T, within=pyo.NonNegativeReals)
    for k in live.K:
        for dc in live.D:
            for t in live.T:
                ordered = pyo.value(plan.receive[k, dc, t])
                live.receive[k, dc, t].setlb(ordered)            # what was ordered arrives
                live.receive[k, dc, t].setub(ordered + 10_000)
                live.add_component(f"rush_{k}_{dc}_{t}".replace("-", "_"),
                                   pyo.Constraint(expr=live.rush[k, dc, t] >= live.receive[k, dc, t] - ordered))
    # Unneeded chilled pallets may go to waste instead of being shipped
    for dc in live.D:
        for t in live.T:
            live.chilled_balance[dc, t].set_value(live.receive["chilled", dc, t] >= sum(live.ship["chilled", dc, s, t] for s in live.S))
    live.cost.deactivate()
    live.actual_cost = pyo.Objective(expr=live.cost.expr + premium * sum(live.rush[k, dc, t] for k in live.K for dc in live.D for t in live.T))
    result = solver.solve(live, load_solutions=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"{forecast_table}: {result.solver.termination_condition}")
    live.solutions.load_from(result)
    rushed = sum(pyo.value(x) for x in live.rush.values())
    wasted = sum(pyo.value(live.receive["chilled", dc, t]) - sum(pyo.value(live.ship["chilled", dc, s, t]) for s in live.S)
                 for dc in live.D for t in live.T)
    lost = sum(pyo.value(x) for x in live.short.values())
    return pyo.value(plan.cost), pyo.value(live.actual_cost), rushed, wasted, lost
```

How it works:

- The plan is solved with the forecast; its `receive` values are the orders.
- The "live" model is the Day 18 model with **actual** demand, where each `receive` variable must be at least what was ordered (`setlb`), and anything above it is a rush order, paid at the premium.
- The chilled balance becomes "received ≥ shipped": unneeded chilled pallets are lost, not stored.
- The live model's cost is what December would really cost with that forecast.

```python
# ---------- 19.3 Plan with each forecast, then live through the actual weeks ----------
rows = []
for t in tables:
    planned, actual, rushed, wasted, lost = plan_and_live(con, t)
    rows.append((t, round(planned, 2), round(actual, 2), round(rushed), round(wasted), round(lost)))
result = pd.DataFrame(rows, columns=["forecast", "planned_cost", "actual_cost", "rushed", "chilled_wasted", "lost"])
best = result.actual_cost[result.forecast == "store_week_actual"].iloc[0]
result["cost_of_forecast_error"] = (result.actual_cost - best).round(2)
print(result.to_string(index=False))
```

```text
              forecast  planned_cost  actual_cost  rushed  chilled_wasted  lost  cost_of_forecast_error
          fc_last_week     202164.26    302130.89    1271               0   341                33907.89
        fc_avg_4_weeks     196924.47    305767.87    1436               0   332                37544.87
fc_last_year_plus_3pct     256514.93    269372.85       0               0   251                 1149.85
   store_week_forecast     256444.53    269362.74       0               0   250                 1139.74
     store_week_actual     268223.00    268223.00       0               0   162                    0.00
```

The ranking changes the size of the gaps:

- The two naive forecasts cost **$34,000–$37,500 more** than perfect hindsight. They under-order for the peak, rush 1,300–1,400 pallets at a premium, and lose more pallets.
- Last year plus 3% costs only **$1,140 more** than perfect hindsight. Its MAE is half the naive forecasts' (7.3 against 12.3), but its cost of error is about a thirtieth.
- Even the perfect forecast loses 162 pallets: capacity, not forecasting, limits December.
- **Planned cost is not actual cost.** The naive forecasts *plan* to spend under $203,000 and *actually* spend over $300,000. A plan's own cost estimate is only as good as its forecast.

> **Decision it supports:** which forecast to plan with, judged by what its plans cost in practice.
>
> **Other decisions you can support the same way:** staffing from a call-volume forecast · energy purchases from a load forecast · inventory from a sales forecast.

## 19.4 Is it better to forecast high?

Running short costs $120 a pallet; ordering a pallet too many costs a few dollars of receiving and holding. So should planners inflate the forecast a little? `day19/bias_sweep.py` scales the planning team's forecast and lives through December each time:

```python
import sys

import duckdb

sys.path.insert(0, "day19")
from evaluate import plan_and_live

con = duckdb.connect("northbound.duckdb", read_only=True)
print(f"{'factor':>7}{'planned cost':>15}{'actual cost':>15}{'rushed':>8}{'wasted':>8}{'lost':>6}")
for factor in (0.90, 0.95, 1.00, 1.02, 1.05, 1.10, 1.20):
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE fc_scaled AS
        SELECT week, store_id, round(ambient_pallets * {factor}) AS ambient_pallets,
               round(chilled_pallets * {factor}) AS chilled_pallets
        FROM store_week_forecast""")
    planned, actual, rushed, wasted, lost = plan_and_live(con, "fc_scaled")
    print(f"{factor:>7.2f}{planned:>15,.2f}{actual:>15,.2f}{rushed:>8.0f}{wasted:>8.0f}{lost:>6.0f}")
```

```text
 factor   planned cost    actual cost  rushed  wasted  lost
   0.90     198,824.79     303,385.49    1330       0   338
   0.95     218,750.34     284,692.75     580       0   338
   1.00     256,444.53     269,362.74       0       0   250
   1.02     277,855.65     271,370.72       0       0   225
   1.05     312,579.24     272,986.69       0      26   190
   1.10     381,306.56     274,339.01       0       0   228
   1.20     536,182.67     287,035.74       0      19   408
```

- **Forecasting low is expensive**: 10% low costs $34,000 more than the unscaled forecast.
- **Forecasting high is not free either**: 5% high costs $3,600 more, 20% high $17,700 more, and at 20% high more pallets go short than with the unscaled forecast (408 against 250). With receiving full, every extra ambient pallet ordered takes a receiving slot a chilled pallet needed, and chilled pallets can't be received early. When capacity is shared, over-ordering one commodity starves the other.
- The best factor here is **1.00**: an unbiased forecast. Inflating a forecast to protect against shortages works when capacity is plentiful (Module 6's newsvendor), not when it is the bottleneck.

## Mini-project: a fifth forecast

Build a **seasonal index** forecast, store by store: this year's level (average of the last 2 weeks) times last year's December shape (each December week last year divided by last year's level at the same point). History starts 52 weeks ago, so last year's level can only use the 2 weeks around 1 December 2025. Measure its accuracy and its December cost, and recommend a forecast to the planning team.

<details><summary>Solution</summary>

```python
import sys

import duckdb

sys.path.insert(0, "day19")
from evaluate import plan_and_live
from forecasts import accuracy, make_forecasts

con = duckdb.connect("northbound.duckdb", read_only=True)
make_forecasts(con)                      # creates the hist and chilled_share views
con.sql("""
    CREATE OR REPLACE TEMP TABLE fc_seasonal_index AS
    WITH level_now AS (
        SELECT store_id, avg(pallets) AS level FROM hist WHERE weeks_ago < 2 GROUP BY store_id
    ), level_last_year AS (
        SELECT store_id, avg(pallets) AS level FROM hist WHERE weeks_ago BETWEEN 51 AND 52 GROUP BY store_id
    ), shape AS (
        SELECT h.store_id, 52 - h.weeks_ago AS week, h.pallets / l.level AS index
        FROM hist h JOIN level_last_year l USING (store_id) WHERE h.weeks_ago BETWEEN 48 AND 51
    )
    SELECT week::INTEGER AS week, store_id,
           round(n.level * s.index) - round(round(n.level * s.index) * c.share) AS ambient_pallets,
           round(round(n.level * s.index) * c.share) AS chilled_pallets
    FROM shape s JOIN level_now n USING (store_id) JOIN chilled_share c USING (store_id)""")
print(con.sql("SELECT min(weeks_ago) AS newest, max(weeks_ago) AS oldest FROM hist"))
print(accuracy(con, "fc_seasonal_index").to_string(index=False))
planned, actual, rushed, wasted, lost = plan_and_live(con, "fc_seasonal_index")
print(f"actual cost ${actual:,.2f}; rushed {rushed:.0f}, wasted {wasted:.0f}, lost {lost:.0f}")
```

```text
┌────────┬────────┐
│ newest │ oldest │
│ int64  │ int64  │
├────────┼────────┤
│      0 │     52 │
└────────┴────────┘

         forecast  mae_pallets  bias_pallets  forecast_total  actual_total
fc_seasonal_index         7.62          0.24           15947         15908
actual cost $269,644.29; rushed 0, wasted 0, lost 214
```

The seasonal index misses a little more often than last year plus 3% (MAE 7.62 against 7.33) and costs a little more ($269,644.29 against $269,362.74). Both are within 0.6% of perfect hindsight. The first query shows the limit of the data: with only 53 weeks of history, last year's level rests on 2 weeks.

*Memo to planning:* "Keep using last year's week plus 3%: it costs only $1,140 more over December than a perfect forecast would. Don't inflate it: with our receiving capacity full, a high forecast costs more than an accurate one. Don't use recent weeks alone for December: they under-order the peak and cost about $35,000. Collect a second year of history so a seasonal index can rest on more than two weeks."

</details>

## Commit

```bash
git add .
git commit -m "Day 19: forecasts, plan versus actual, bias sweep"
git push
```

### Day 19 checklist

- [ ] I built several forecasts in SQL, in the shape the model reads.
- [ ] I measured accuracy (MAE) and bias against actual demand.
- [ ] I separated decisions made now (orders) from decisions that adapt (shipments, rush orders).
- [ ] I evaluated each forecast by the cost of living through the actual weeks.
- [ ] I found that the most accurate forecast is not automatically the cheapest, and why the gap sizes differ.
- [ ] I tested whether biasing a forecast helps, and explained why it doesn't here.

---

# DAY 20 — Capstone: The Weekly Distribution Plan

Module 2 ends with the question every operations director asks in November: **"which capacity should we buy for December?"** You build the December distribution plan, live through the actual weeks, and evaluate four capacity options three ways: shadow prices, re-solving, and the actual December. The memo is the deliverable.

| Time | Block |
|---|---|
| 0:00 – 0:30 | 20.1 The brief |
| 0:30 – 1:45 | 20.2 The plan, stored |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 20.3 Live through December |
| 2:45 – 3:45 | 20.4 Which capacity should we buy? |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | 20.5 The chart and the memo |
| 4:45 – 5:00 | Commit, tag and checklist |

## 20.1 The brief

> **To:** Module 2 analysts **From:** Director of Operations
>
> I have budget for some temporary capacity for the four weeks from 30 November. The options on the table:
>
> | Option | Adds | Cost |
> |---|---|---|
> | A: Montreal second receiving shift | 200 pallets a week in, weeks 1–4 | $6,000 |
> | B: Mississauga reefer trailers | 60 chilled pallets a week, weeks 3–4 | $5,000 |
> | C: Ottawa reefer trailers | 40 chilled pallets a week, weeks 3–4 | $2,400 |
> | D: Ottawa extra outbound shift | 100 pallets a week out, weeks 1–4 | $12,000 |
>
> I need the December plan, what it will really cost, and which options to buy. Keep it to one page.

### Deliverables and marking

| Deliverable | Weight | What earns full marks |
|---|---|---|
| `day20/december_plan.py` | 25% | Reuses Days 18–19; options as data, applied by one function |
| Stored plan and chart | 15% | `december_shipments` and `december_dc_plan` tables; one clear chart |
| `tests/test_distribution.py` | 20% | A known answer, conservation, every limit, and the perfect-forecast check |
| Option analysis | 20% | Each option valued by duals, by re-solving and in the actual December, with an explanation of the differences |
| `day20/memo.md` | 20% | One page; a decision for each option; every number from a run |

Name the script `december_plan.py`, not `capstone.py`: Day 10 already has a `capstone.py`, and with both folders on the test path, Python would import whichever it finds first. (It happened while writing this lesson: 36 tests became "1 error".)

## 20.2 The plan, stored

<details><summary>Model solution: options as data</summary>

```python
import os
import sys

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day18")
sys.path.insert(0, "day19")
from evaluate import plan_and_live
from flows import build_weeks, load_weeks, report, v

pd.set_option("display.width", 160)
WEEKS = [1, 2, 3, 4]
OPTIONS = {
    # name: (limit, DC, weeks, extra pallets a week, total cost over the weeks)
    "A: Montreal second receiving shift": ("receiving_pallets", "DC-MTL", WEEKS, 200, 6000),
    "B: Mississauga reefer trailers": ("chilled_zone_pallets", "DC-MIS", [3, 4], 60, 5000),
    "C: Ottawa reefer trailers": ("chilled_zone_pallets", "DC-OTT", [3, 4], 40, 2400),
    "D: Ottawa extra outbound shift": ("outbound_pallets", "DC-OTT", WEEKS, 100, 12000),
}
CONSTRAINT = {"receiving_pallets": "receiving", "chilled_zone_pallets": "chilled_zone", "outbound_pallets": "outbound"}


def with_options(*names):
    """A function that adds the chosen options' capacity to the data."""
    def adjust(d):
        w = d["dc_weeks"].copy()
        for n in names:
            limit, dc, weeks, extra, _ = OPTIONS[n]
            for t in weeks:
                if (dc, t) in w.index:
                    w.loc[(dc, t), limit] += extra
        return dict(d, dc_weeks=w)
    return adjust


def plan(con, adjust=None):
    m = build_weeks((adjust or (lambda d: d))(load_weeks(con)))
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    pyo.SolverFactory("highs").solve(m)
    return m
```

Each option is a row of data: which limit, which DC, which weeks, how much, at what cost. `with_options` returns a function that adds the chosen options to a copy of the data: the same `adjust` hook `plan_and_live` now accepts (Day 19's `evaluate.py`, final version).

</details>

```python
if __name__ == "__main__":
    con = duckdb.connect("northbound.duckdb")

    # ---------- 20.2 The plan ----------
    m = plan(con)
    weekly = report(m)
    print(weekly.to_string(index=False))
    print(f"planned 4-week cost ${pyo.value(m.cost):,.2f}")
    dc_rows = []
    w = load_weeks(con)["dc_weeks"]
    for dc in m.D:
        for t in m.T:
            dc_rows.append((dc, t, v(sum(m.receive[k, dc, t] for k in m.K)), w.receiving_pallets[dc, t],
                            v(sum(m.ship[k, dc, s, t] for k in m.K for s in m.S)), w.outbound_pallets[dc, t],
                            v(sum(m.ship["chilled", dc, s, t] for s in m.S)), w.chilled_zone_pallets[dc, t], v(m.stock[dc, t])))
    dc_plan = pd.DataFrame(dc_rows, columns=["dc_id", "week", "received", "receiving_cap", "shipped", "outbound_cap",
                                             "chilled", "chilled_cap", "stock_end"])
    print(dc_plan.to_string(index=False))
    ships = pd.DataFrame([(k, dc, s, t, v(m.ship[k, dc, s, t])) for k in m.K for dc in m.D for s in m.S for t in m.T
                          if pyo.value(m.ship[k, dc, s, t]) > 1e-6], columns=["kind", "dc_id", "store_id", "week", "pallets"])
    con.sql("CREATE OR REPLACE TABLE december_shipments AS SELECT * FROM ships")
    con.sql("CREATE OR REPLACE TABLE december_dc_plan AS SELECT * FROM dc_plan")
    print(con.sql("""
        SELECT week, count(DISTINCT store_id) AS stores, count(*) AS lanes_used,
               count(*) FILTER (WHERE kind = 'chilled') AS chilled_lanes, sum(pallets)::INTEGER AS pallets
        FROM december_shipments GROUP BY week ORDER BY week"""))
```

```text
 week  shipped  received  dc_stock_end  short_ambient  short_chilled
    1   3504.0    3398.0         424.0            0.0            0.0
    2   3800.0    3910.0         534.0            0.0           19.0
    3   4177.0    3910.0         267.0            0.0           58.0
    4   4177.0    3910.0           0.0            0.0           68.0
planned 4-week cost $256,444.53
 dc_id  week  received  receiving_cap  shipped  outbound_cap  chilled  chilled_cap  stock_end
DC-MIS     1    1248.0           1650   1411.0          1600    342.0          400       87.0
DC-MIS     2    1650.0           1650   1600.0          1600    374.0          400      137.0
DC-MIS     3    1650.0           1650   1696.0          1840    400.0          400       91.0
DC-MIS     4    1650.0           1650   1741.0          1840    400.0          400        0.0
DC-MTL     1    1500.0           1500   1393.0          1500    358.0          420      307.0
DC-MTL     2    1500.0           1500   1500.0          1500    378.0          420      307.0
DC-MTL     3    1500.0           1500   1676.0          1725    420.0          420      131.0
DC-MTL     4    1500.0           1500   1631.0          1725    420.0          420        0.0
DC-OTT     1     650.0            760    700.0           700    211.0          220       30.0
DC-OTT     2     760.0            760    700.0           700    220.0          220       90.0
DC-OTT     3     760.0            760    805.0           805    220.0          220       45.0
DC-OTT     4     760.0            760    805.0           805    220.0          220        0.0
┌───────┬────────┬────────────┬───────────────┬─────────┐
│ week  │ stores │ lanes_used │ chilled_lanes │ pallets │
│ int64 │ int64  │   int64    │     int64     │  int32  │
├───────┼────────┼────────────┼───────────────┼─────────┤
│     1 │     40 │         81 │            40 │    3504 │
│     2 │     40 │         83 │            41 │    3800 │
│     3 │     40 │         81 │            40 │    4177 │
│     4 │     40 │         81 │            39 │    4177 │
└───────┴────────┴────────────┴───────────────┴─────────┘

```

The plan ships every store every week, through 81–83 lanes a week (most stores get ambient and chilled from the same DC). Every receiving dock is full except Mississauga's and Ottawa's in week 1; Ottawa's outbound docks are full every week; Mississauga's and Montreal's chilled zones are full at the peak.

## 20.3 Live through December

```python
    # ---------- 20.3 Live through the actual weeks ----------
    planned, actual, rushed, wasted, lost = plan_and_live(con, "store_week_forecast")
    print(f"lived through December: ${actual:,.2f} (planned ${planned:,.2f}); rushed {rushed:.0f}, lost {lost:.0f}")
```

```text
lived through December: $269,362.74 (planned $256,444.53); rushed 0, lost 250
```

December really costs **$269,362.74**, $12,918 more than planned: actual demand was 105 pallets higher than forecast, and 250 pallets go short instead of 145.

## 20.4 Which capacity should we buy?

Each option is valued three ways:

1. **Predicted by duals:** shadow price × extra capacity, summed over the weeks (Day 11). Instant.
2. **Re-solved:** the plan's cost with and without the option (Day 13).
3. **In the actual December:** plan with the option, then live through the actual weeks (Day 19). The number the director cares about.

```python
    # ---------- 20.4 Which capacity should we buy? ----------
    rows = []
    base_live = actual
    for name, (limit, dc, weeks, extra, cost) in OPTIONS.items():
        con_ = getattr(m, CONSTRAINT[limit])
        predicted = sum(abs(m.dual[con_[dc, t]]) * extra for t in weeks)
        resolved = pyo.value(m.cost) - pyo.value(plan(con, with_options(name)).cost)
        lived = base_live - plan_and_live(con, "store_week_forecast", with_options(name))[1]
        rows.append((name, cost, round(predicted, 2), round(resolved, 2), round(lived, 2), round(lived - cost, 2)))
    options = pd.DataFrame(rows, columns=["option", "cost", "saving_predicted_by_duals", "saving_re_solved",
                                          "saving_in_actual_december", "net"])
    print(options.to_string(index=False))
    good = [r.option for r in options.itertuples() if r.net > 0]
    both = base_live - plan_and_live(con, "store_week_forecast", with_options(*good))[1]
    cost_good = sum(OPTIONS[g][4] for g in good)
    print(f"buy {good}: saves ${both:,.2f} for ${cost_good:,}, net ${both - cost_good:,.2f}")
```

```text
                            option  cost  saving_predicted_by_duals  saving_re_solved  saving_in_actual_december      net
A: Montreal second receiving shift  6000                   55184.00           9492.14                   13993.77  7993.77
    B: Mississauga reefer trailers  5000                    8101.20           4687.12                    3576.60 -1423.40
         C: Ottawa reefer trailers  2400                    3925.98           3757.46                    3424.51  1024.51
    D: Ottawa extra outbound shift 12000                   20270.60           5489.53                    5489.99 -6510.01
buy ['A: Montreal second receiving shift', 'C: Ottawa reefer trailers']: saves $17,388.63 for $8,400, net $8,988.63
```

- **The duals overstate every option**, Montreal's receiving shift by a factor of six ($55,184 predicted, $9,492 re-solved). 200 extra pallets a week is far outside the range over which Montreal's $65–73 shadow price holds (Day 12): after the first few dozen pallets, other limits bind. Use duals to rank; never to price a big change.
- **A forecast-based plan can understate an option**: Montreal's shift saves $9,492 on the forecast but $13,994 in the actual December, because actual demand was higher than forecast. Capacity is worth most when demand surprises upward.
- **Buy A and C.** Montreal's receiving shift nets $7,994; Ottawa's reefer trailers $1,025. Together they save $17,388.63 for $8,400: **net $8,988.63**.
- **Don't buy B or D.** Mississauga's trailers save $3,577 for $5,000. Ottawa's outbound shift saves $5,490 for $12,000: Ottawa's docks are full, but each extra pallet out of Ottawa is worth little once nearby stores are covered from Montreal.

## 20.5 The chart and the memo

```python
    # ---------- 20.5 The chart ----------
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, (dc, g) in zip(axes, dc_plan.groupby("dc_id")):
        ax.bar(g.week - 0.2, g.received, width=0.4, color="#4f7bd9", label="received")
        ax.bar(g.week + 0.2, g.shipped, width=0.4, color="#e08a2c", label="shipped")
        ax.plot(g.week, g.receiving_cap, "b--", lw=1, label="receiving capacity")
        ax.plot(g.week, g.outbound_cap, "--", color="#e08a2c", lw=1, label="outbound capacity")
        ax.plot(g.week, g.stock_end, "k-o", label="ambient stock (end of week)")
        ax.set_title(dc)
        ax.set_xticks(WEEKS)
        ax.set_xlabel("December week")
    axes[0].set_ylabel("pallets")
    axes[0].legend(fontsize=7)
    fig.suptitle("December distribution plan: pallets received, shipped and stocked at each DC")
    fig.tight_layout()
    fig.savefig("figures/day20_december_plan.png", dpi=120)
    print("saved figures/day20_december_plan.png")
    con.close()
```

![December distribution plan](figures/day20_december_plan.png)

<details><summary>Model solution: the tests</summary>

```python
"""Module 2, Days 18-20: the multi-commodity, multi-week distribution plan."""
import pyomo.environ as pyo
import pytest
from evaluate import plan_and_live
from flows import build_weeks, load_weeks

TOL = 1e-6


@pytest.fixture(scope="module")
def december(con):
    d = load_weeks(con)
    m = build_weeks(d)
    pyo.SolverFactory("highs").solve(m)
    return d, m


def test_known_answer(december):
    _, m = december
    assert pyo.value(m.cost) == pytest.approx(256_444.53, abs=0.01)


def test_ambient_pallets_are_conserved(december):
    d, m = december
    w = d["dc_weeks"]
    for dc in m.D:
        opening = w.opening_ambient_stock[dc, 1]
        received = sum(pyo.value(m.receive["ambient", dc, t]) for t in m.T)
        shipped = sum(pyo.value(m.ship["ambient", dc, s, t]) for s in m.S for t in m.T)
        assert opening + received == pytest.approx(shipped + pyo.value(m.stock[dc, m.T.last()]), abs=TOL)


def test_limits_hold_and_stores_get_at_most_their_demand(december):
    d, m = december
    w = d["dc_weeks"]
    for dc in m.D:
        for t in m.T:
            assert sum(pyo.value(m.ship["chilled", dc, s, t]) for s in m.S) <= w.chilled_zone_pallets[dc, t] + TOL
            assert sum(pyo.value(m.receive[k, dc, t]) for k in m.K) <= w.receiving_pallets[dc, t] + TOL
    for (k, s, t), dem in d["demand"].items():
        assert sum(pyo.value(m.ship[k, dc, s, t]) for dc in m.D) <= dem + TOL


def test_perfect_forecast_costs_nothing_extra(con):
    planned, actual, rushed, wasted, lost = plan_and_live(con, "store_week_actual")
    assert actual == pytest.approx(planned, abs=0.01)
    assert rushed == pytest.approx(0, abs=TOL)


def test_forecast_error_never_helps(con):
    _, perfect, *_ = plan_and_live(con, "store_week_actual")
    _, official, *_ = plan_and_live(con, "store_week_forecast")
    assert official >= perfect - TOL
```

Add `"day18"`, `"day19"` and `"day20"` to the folders in `conftest.py`.

```text
36 passed
```

</details>

Write `day20/memo.md`.

<details><summary>Model answer</summary>

**To:** Director of Operations **Re:** December capacity

**Buy two options, for $8,400:** Montreal's second receiving shift ($6,000) and Ottawa's reefer trailers ($2,400). Against December's actual demand they save **$17,389**, a net gain of **$8,989**. Don't buy Mississauga's trailers (they save $3,577 for $5,000) or Ottawa's outbound shift (saves $5,490 for $12,000).

**Why Montreal's receiving.** It is our tightest limit in December: every pallet the DCs can receive in weeks 1–2 can be held for the Christmas peak. With the extra shift, Montreal receives more early and ships it at the peak.

**The plan.** Receive at full capacity from week 2, build 534 pallets of ambient stock by mid-December, ship 4,177 pallets a week in weeks 3–4. It meets every store's ambient demand. Chilled pallets can't be received early, so about 1 in 20 chilled pallets in the peak weeks can't be delivered.

**What December will cost.** About **$269,000** with our current forecast: $13,000 more than the plan says, because demand will run slightly above forecast. Our forecast (last year's week plus 3%) is within $1,200 of perfect hindsight; inflating it would make things worse, because receiving capacity is full.

**Caveat.** Shadow prices overstate the value of large additions by up to six times; every saving above comes from re-planning with the option, and from living through the actual weeks.

</details>

## Commit and tag

The end of Module 2:

```bash
git add .
git commit -m "Day 20: December distribution plan, capacity options, tests and memo"
git tag v2.0-module2
git push --tags
```

### Day 20 checklist

- [ ] My plan covers both commodities and four weeks, and is stored in two tables.
- [ ] I lived through the actual December and explained why it costs more than planned.
- [ ] I valued each option by duals, by re-solving and in the actual December.
- [ ] I can explain why duals overstate large changes and forecasts can understate them.
- [ ] My tests check a known answer, conservation, every limit and the perfect forecast.
- [ ] My memo gives the director a decision on each option, with its net value.
