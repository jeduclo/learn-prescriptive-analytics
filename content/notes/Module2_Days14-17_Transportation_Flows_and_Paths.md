# DAY 14 — Transportation and Assignment

The bakery is planned; now the pallets have to reach the stores. Northbound Fresh ships about 3,500 pallets a week from 3 distribution centres (DCs) to 40 stores, and the Ottawa DC is running over capacity. Today you solve the **transportation problem** (how many pallets go from each DC to each store) and its cousin, the **assignment problem** (who does which job), and meet a surprise: models with whole-number answers for free.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 14.1 The network data; today's cost |
| 0:45 – 1:45 | 14.2 The transportation problem |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 14.3 Shadow prices: the delivered cost of a pallet |
| 2:45 – 3:45 | 14.4 The assignment problem; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: drivers with rules |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page, add networkx to your packages, and rebuild:

```bash
echo "networkx==3.7" >> requirements.txt
pip install -r requirements.txt
python make_data.py
python day01/ladder.py
python day06/load_tables.py
```

The new tables:

| Table | What it holds |
|---|---|
| `places` | 17 cities and towns with approximate coordinates |
| `road_segments` | 28 road segments: km, driving minutes, and the trucks a week Northbound's carriers can run on each (Day 17) |
| `dc_logistics` | Each DC's weekly capacity in pallets, and whether it stocks French-labelled products (Mississauga doesn't) |
| `store_logistics` | Each store's location and weekly pallets (all products, and chilled only) |
| `dc_store_lanes` | Every DC–store pair: km, minutes and cost per pallet ($4 handling + $0.11 per pallet-km) |
| `driver_route_minutes` | 8 Montreal drivers × 8 routes: minutes each driver needs for each route |
| `chilled_nodes`, `chilled_arcs` | The chilled network: plants, DCs, cross-docks, stores, and the lanes between them (Day 16) |

## 14.1 Today's network and its cost

Create `day14/transport.py`. A `load_network` function reads DCs, stores and lanes; a `build_transport` function will hold the model. The script part, under `if __name__ == "__main__":`, starts with today's situation: every store served by its current DC.

```python
import duckdb
import pandas as pd
import pyomo.environ as pyo


def load_network(con):
    """DCs with capacity, stores with weekly pallets, and the cost of every DC-store lane."""
    dcs = con.sql("SELECT * FROM dc_logistics ORDER BY dc_id").df().set_index("dc_id")
    stores = con.sql("""
        SELECT s.store_id, s.city, s.province, s.dc_id AS current_dc, l.pallets_per_week
        FROM stores s JOIN store_logistics l USING (store_id) ORDER BY s.store_id""").df().set_index("store_id")
    lanes = con.sql("SELECT * FROM dc_store_lanes").df().set_index(["dc_id", "store_id"])
    return dcs, stores, lanes
```

The `if __name__ == "__main__":` line lets other files import `load_network` and `build_transport` (Day 15 does) without running the analysis below it.

```python
    # ---------- 14.1 Today: every store served by its current DC ----------
    current = stores.groupby("current_dc").pallets_per_week.sum()
    current_cost = sum(lanes.cost_per_pallet[stores.current_dc[s], s] * stores.pallets_per_week[s] for s in stores.index)
    print(pd.DataFrame({"pallets_today": current, "capacity": dcs.pallets_per_week_capacity}).to_string())
    print(f"total demand {stores.pallets_per_week.sum():,} pallets; today's weekly shipping cost ${current_cost:,.2f}")
```

```text
        pallets_today  capacity
DC-MIS           1195      1600
DC-MTL           1427      1500
DC-OTT            923       700
total demand 3,545 pallets; today's weekly shipping cost $35,194.71
```

Today's routing costs **$35,194.71 a week**, but it is not feasible: the Ottawa DC ships 923 pallets a week with room for 700. The difference is handled today by overtime and a rented trailer yard, which the lane costs don't show. Montreal is nearly full too. A plan that respects the capacities will cost more than today's on paper.

## 14.2 The transportation problem

Every DC $$d$$ has a capacity, every store $$s$$ a demand, every lane a cost per pallet. Choose $$x_{d,s}$$, the pallets a week from $$d$$ to $$s$$:

$$
\begin{aligned}
\min\quad & \sum_{d,s} \text{cost}_{d,s}\, x_{d,s} \\
\text{s.t.}\quad & \textstyle\sum_d x_{d,s} = \text{demand}_s && \forall s \text{ (every store gets its pallets)} \\
& \textstyle\sum_s x_{d,s} \le \text{capacity}_d && \forall d \text{ (no DC ships more than it can)} \\
& x \ge 0
\end{aligned}
$$

```python
def build_transport(dcs, stores, lanes, capacity=None):
    """Minimize the cost of shipping every store's pallets, within each DC's weekly capacity."""
    capacity = capacity or dcs.pallets_per_week_capacity.to_dict()
    m = pyo.ConcreteModel(name="transportation")
    m.D = pyo.Set(initialize=dcs.index.tolist())
    m.S = pyo.Set(initialize=stores.index.tolist())
    m.ship = pyo.Var(m.D, m.S, within=pyo.NonNegativeReals)          # pallets a week
    m.demand = pyo.Constraint(m.S, rule=lambda m, s: sum(m.ship[d, s] for d in m.D) == stores.pallets_per_week[s])
    m.supply = pyo.Constraint(m.D, rule=lambda m, d: sum(m.ship[d, s] for s in m.S) <= capacity[d])
    m.cost = pyo.Objective(expr=sum(lanes.cost_per_pallet[d, s] * m.ship[d, s] for d in m.D for s in m.S))
    return m


def loads(m):
    return {d: round(sum(pyo.value(m.ship[d, s]) for s in m.S), 2) for d in m.D}
```

```python
    # ---------- 14.2 The transportation problem ----------
    m = build_transport(dcs, stores, lanes)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    result = pyo.SolverFactory("highs").solve(m)
    print(result.solver.termination_condition, f"weekly cost ${pyo.value(m.cost):,.2f}", loads(m))
    split = {s: {d: pyo.value(m.ship[d, s]) for d in m.D if pyo.value(m.ship[d, s]) > 1e-6}
             for s in m.S if sum(pyo.value(m.ship[d, s]) > 1e-6 for d in m.D) > 1}
    print("stores served by two DCs:", split)
    moved = [s for s in m.S if pyo.value(m.ship[stores.current_dc[s], s]) < stores.pallets_per_week[s] - 1e-6]
    print("stores no longer (fully) served by their current DC:", moved)
    flows = [pyo.value(m.ship[d, s]) for d in m.D for s in m.S]
    print("every flow a whole number of pallets:", all(abs(f - round(f)) < 1e-6 for f in flows))
```

```text
optimal weekly cost $39,089.81 {'DC-MIS': 1345.0, 'DC-MTL': 1500.0, 'DC-OTT': 700.0}
stores served by two DCs: {'S16': {'DC-MIS': 16.0, 'DC-MTL': 69.0}, 'S21': {'DC-MTL': 4.0, 'DC-OTT': 108.0}}
stores no longer (fully) served by their current DC: ['S16', 'S21', 'S22', 'S23']
every flow a whole number of pallets: True
```

The cheapest plan that fits costs **$39,089.81 a week**. Ottawa and Montreal are exactly full, Mississauga has 255 pallets to spare. Four Ottawa-DC stores change: both Kingston stores (S22, S23) go to Mississauga, and two Ottawa stores are **split** between two DCs: S16 gets its pallets from Mississauga and Montreal, S21 mostly from Ottawa with 4 pallets from Montreal.

**Every flow is a whole number of pallets**, though the variables are continuous. This is no accident. The constraint matrix of a transportation problem has a special structure (**total unimodularity**): with whole-number supplies and demands, every corner of the feasible region has whole-number coordinates, so the simplex method can only land on whole numbers. Transportation, assignment, shortest-path and maximum-flow problems all share it, which is why this week's models need no integer variables.

> **Decision it supports:** how many pallets each DC sends to each store next week.
>
> **Other decisions you can support the same way:** which warehouse ships each online order · which power plant supplies each region · which school each neighbourhood's students attend.

### Try it 14.1

What would capacity cost? Using only today's output, how much more per week does the plan that fits cost than today's routing, and why is that number not a saving waiting to happen?

<details><summary>Solution</summary>

$39,089.81 − $35,194.71 = **$3,895.10 a week**. Today's routing is cheaper on paper only because Ottawa ships 223 pallets a week it has no room for; the overtime and the trailer yard that make that possible are not in the lane costs. The honest comparison is between plans that fit, or a plan that fits against the real cost of today's workaround. Day 15 prices the alternative: a bigger Ottawa DC.

</details>

## 14.3 Shadow prices: the delivered cost of a pallet

```python
    # ---------- 14.3 Shadow prices: what a pallet costs to deliver, store by store ----------
    print("DC capacity shadow prices ($ saved per extra pallet of capacity):",
          {d: round(-m.dual[m.supply[d]], 2) for d in m.D})
    prices = pd.Series({s: m.dual[m.demand[s]] for s in m.S}, name="delivered_cost").round(2)
    print(prices.to_frame().join(stores[["city"]]).sort_values("delivered_cost").iloc[[0, 1, -3, -2, -1]].to_string())
```

```text
DC capacity shadow prices ($ saved per extra pallet of capacity): {'DC-MIS': 0.0, 'DC-MTL': 26.18, 'DC-OTT': 48.18}
     delivered_cost         city
S10            4.39  Mississauga
S11            4.62  Mississauga
S36           59.89  Quebec City
S37           59.94  Quebec City
S38           60.34  Quebec City
```

Two kinds of shadow price, and each answers a business question:

- **DC capacity.** One more pallet of capacity in Ottawa would save $48.18 a week; in Montreal $26.18; in Mississauga nothing (it has spare room). The supply constraint is "≤", and in a minimization the solver reports its dual as a negative number, so we flip the sign to read it as a saving.
- **Store demand.** The shadow price of each store's demand constraint is the **delivered cost of one more pallet there**: $4.39 in Mississauga, $60.34 in Quebec City. These are location prices: what it costs Northbound, at the margin, to supply each store. Category managers can use them to set regional prices or minimum order sizes.

## 14.4 The assignment problem

Montreal has 8 delivery routes tomorrow and 8 drivers. Each driver knows some routes better: `driver_route_minutes` has the minutes each driver needs for each route. Give each driver exactly one route, minimizing total minutes. It is a transportation problem where every supply and demand is 1. Create `day14/assignment.py`:

```python
import duckdb
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from scipy.optimize import linear_sum_assignment

con = duckdb.connect("northbound.duckdb", read_only=True)
minutes = con.sql("SELECT * FROM driver_route_minutes").df().pivot(index="driver", columns="route", values="minutes")
print(minutes.to_string())

m = pyo.ConcreteModel()
m.D = pyo.Set(initialize=minutes.index.tolist())
m.R = pyo.Set(initialize=minutes.columns.tolist())
m.x = pyo.Var(m.D, m.R, bounds=(0, 1))            # NOT declared binary
m.one_route = pyo.Constraint(m.D, rule=lambda m, d: sum(m.x[d, r] for r in m.R) == 1)
m.one_driver = pyo.Constraint(m.R, rule=lambda m, r: sum(m.x[d, r] for d in m.D) == 1)
m.total = pyo.Objective(expr=sum(minutes.loc[d, r] * m.x[d, r] for d in m.D for r in m.R))
pyo.SolverFactory("highs").solve(m)
plan = {d: r for d in m.D for r in m.R if pyo.value(m.x[d, r]) > 0.5}
values = sorted({round(pyo.value(m.x[d, r]), 6) + 0.0 for d in m.D for r in m.R})
print(f"LP: {pyo.value(m.total):,.0f} minutes; values taken by x: {values}")
print(plan)

rows, cols = linear_sum_assignment(minutes.to_numpy())    # the Hungarian method
print(f"Hungarian method: {minutes.to_numpy()[rows, cols].sum():,} minutes")
print(f"each driver's first-choice route would need {minutes.min(axis=1).sum():,} minutes, but routes repeat: "
      f"{minutes.idxmin(axis=1).value_counts().to_dict()}")
```

```text
route    R1   R2   R3   R4   R5   R6   R7   R8
driver
D1      382  330  387  377  326  403  395  320
D2      332  398  427  349  387  425  426  285
D3      368  371  428  404  346  399  441  314
D4      394  372  450  411  364  410  380  313
D5      390  380  408  391  374  348  362  283
D6      411  318  407  386  320  336  387  349
D7      359  351  419  369  399  382  371  288
D8      375  361  409  425  360  371  362  288
LP: 2,768 minutes; values taken by x: [0.0, 1.0]
{'D1': 'R3', 'D2': 'R1', 'D3': 'R5', 'D4': 'R7', 'D5': 'R6', 'D6': 'R2', 'D7': 'R4', 'D8': 'R8'}
Hungarian method: 2,768 minutes
each driver's first-choice route would need 2,409 minutes, but routes repeat: {'R8': 7, 'R2': 1}
```

The variables are allowed any value from 0 to 1, yet every one comes out 0 or 1: total unimodularity again. `scipy.optimize.linear_sum_assignment` uses the **Hungarian method**, a specialized algorithm for exactly this problem, and agrees: **2,768 minutes**.

Why not give each driver his fastest route? Because seven of the eight are fastest on R8. The model resolves the conflict globally: D8 keeps R8, and everyone else gets the route where their loss is smallest.

### The planted bug

`day14/bug_supply.py` changes one character in the supply rule. Run it.

```python
m.del_component(m.supply)
m.supply = pyo.Constraint(m.D, rule=lambda m, d: sum(m.ship[d, s] for s in m.S) == dcs.pallets_per_week_capacity[d])
result = pyo.SolverFactory("highs").solve(m, load_solutions=False)
print(result.solver.termination_condition)
```

<details><summary>What happens, and why</summary>

```text
infeasible
```

`==` instead of `<=` says every DC must ship **exactly** its capacity: 3,800 pallets a week. The stores need 3,545, and the demand constraint says they get exactly that. No plan does both. When a model built from correct data is infeasible, compare the sense of every constraint (`==`, `<=`, `>=`) with the business rule it states.

</details>

## Mini-project: drivers with rules

Tomorrow's assignment has three rules: D3 has no licence for the tunnel on route R5; D6 must finish by 2 p.m., so at most 330 minutes; no driver may be on the road more than 380 minutes. Solve it in `day14/mini_project_drivers.py`. Does it still come out in whole numbers?

<details><summary>Solution</summary>

```python
def assign(rules=True, max_minutes=380):
    m = pyo.ConcreteModel()
    m.D = pyo.Set(initialize=minutes.index.tolist())
    m.R = pyo.Set(initialize=minutes.columns.tolist())
    m.x = pyo.Var(m.D, m.R, bounds=(0, 1))
    m.one_route = pyo.Constraint(m.D, rule=lambda m, d: sum(m.x[d, r] for r in m.R) == 1)
    m.one_driver = pyo.Constraint(m.R, rule=lambda m, r: sum(m.x[d, r] for d in m.D) == 1)
    if rules:
        for d in m.D:
            for r in m.R:
                too_long = minutes.loc[d, r] > max_minutes or (d == "D6" and minutes.loc[d, r] > 330)
                if too_long or (d, r) == ("D3", "R5"):
                    m.x[d, r].fix(0)                 # this pair is not allowed
    m.total = pyo.Objective(expr=sum(minutes.loc[d, r] * m.x[d, r] for d in m.D for r in m.R))
    result = pyo.SolverFactory("highs").solve(m, load_solutions=False)
    if result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return str(result.solver.termination_condition), None
    m.solutions.load_from(result)
    plan = {d: r for d in m.D for r in m.R if pyo.value(m.x[d, r]) > 0.5}
    whole = all(abs(pyo.value(m.x[d, r]) - round(pyo.value(m.x[d, r]))) < 1e-6 for d in m.D for r in m.R)
    longest = max(minutes.loc[d, r] for d, r in plan.items())
    return f"{pyo.value(m.total):,.0f} minutes, longest route {longest}, whole numbers: {whole}", plan


for rules, limit in ((False, None), (True, 380), (True, 390)):
    summary, plan = assign(rules, limit)
    print("no rules" if not rules else f"rules, at most {limit} minutes", "->", summary)
    if plan:
        print("   ", plan)
    else:
        print("    fastest time on each route:", minutes.min().to_dict())
```

```text
no rules -> 2,768 minutes, longest route 387, whole numbers: True
    {'D1': 'R3', 'D2': 'R1', 'D3': 'R5', 'D4': 'R7', 'D5': 'R6', 'D6': 'R2', 'D7': 'R4', 'D8': 'R8'}
rules, at most 380 minutes -> infeasible
    fastest time on each route: {'R1': 332, 'R2': 318, 'R3': 387, 'R4': 349, 'R5': 320, 'R6': 336, 'R7': 362, 'R8': 283}
rules, at most 390 minutes -> 2,784 minutes, longest route 387, whole numbers: True
    {'D1': 'R3', 'D2': 'R4', 'D3': 'R1', 'D4': 'R5', 'D5': 'R6', 'D6': 'R2', 'D7': 'R8', 'D8': 'R7'}
```

At 380 minutes the problem is **infeasible**: the fastest anyone does R3 is 387 minutes, so no assignment can cover it. The rule as written can't be met by any roster; Day 7's first check (a minimum above its maximum) catches this. At 390 minutes, the rules cost 16 minutes a day (2,784 against 2,768), and the answer is still in whole numbers: removing pairs (fixing variables to 0) keeps the special structure.

</details>

## Commit

```bash
git add .
git commit -m "Day 14: transportation and assignment"
git push
```

### Day 14 checklist

- [ ] I can write and solve a transportation problem from tables of supply, demand and lane costs.
- [ ] I know why transportation and assignment problems give whole-number answers without integer variables.
- [ ] I can read DC capacity duals as savings and store demand duals as delivered costs.
- [ ] I solved an assignment problem as an LP and checked it with the Hungarian method.
- [ ] I found an infeasibility caused by `==` where `<=` belonged.
- [ ] I can add forbidden pairs to an assignment and recognise a rule no roster can meet.

---

# DAY 15 — Mini-project: Assigning Stores to DCs

Day 14's plan split two stores between two DCs. Store managers hate that: two delivery schedules, two sets of paperwork, two trucks at the dock. Northbound's rule is **single sourcing**: each store gets everything from one DC. Today you build that plan with the business rules, compare it with today's routing and with Day 14's split plan, and price an expansion of the Ottawa DC.

| Time | Block |
|---|---|
| 0:00 – 0:30 | 15.1 The brief |
| 0:30 – 1:45 | 15.2 One DC per store, with binary variables |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 15.3 Which rules bind? |
| 2:45 – 3:45 | 15.4 What is a bigger Ottawa DC worth? |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | 15.5 The map and the memo |
| 4:45 – 5:00 | Commit with Git + checklist |

## 15.1 The brief

> **From:** VP Logistics
>
> Ottawa is 223 pallets a week over capacity, and I want a store-to-DC map that fits. Rules: every store is served by exactly one DC; Quebec stores need French-labelled stock, which Mississauga doesn't carry; fresh products may spend at most six hours on a truck. Tell me what the map costs, which stores move, and what it would be worth to expand Ottawa by 100, 200 or 300 pallets a week.

## 15.2 One DC per store, with binary variables

"Exactly one DC" is a yes/no decision per store and DC: **binary** variables, $$a_{d,s} \in \{0, 1\}$$. The pallets shipped are then $$\text{demand}_s \times a_{d,s}$$:

$$
\begin{aligned}
\min\quad & \sum_{d,s} \text{cost}_{d,s}\, \text{demand}_s\, a_{d,s} \\
\text{s.t.}\quad & \textstyle\sum_d a_{d,s} = 1 && \forall s \\
& \textstyle\sum_s \text{demand}_s\, a_{d,s} \le \text{capacity}_d && \forall d \\
& a_{d,s} = 0 && \text{for lanes over 6 hours, and Mississauga to Quebec stores} \\
& a_{d,s} \in \{0, 1\}
\end{aligned}
$$

This model does **not** have Day 14's special structure: the capacity rows weigh each store by its pallets. Without the binary requirement, the solver would happily split stores again. With `within=pyo.Binary`, HiGHS solves a **mixed-integer program** (MIP). How it does that (branch and bound) is Module 3; today we use it. Create `day15/store_assignment.py`:

```python
import os
import sys

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day14")
from transport import build_transport, load_network

pd.set_option("display.width", 140)
MAX_MINUTES = 360          # fresh products: at most six hours on the truck


def build_assignment(dcs, stores, lanes, capacity=None):
    """Single sourcing: assign[d, s] = 1 if DC d serves all of store s. A binary variable (Module 3 explains how
    the solver handles it; today we just use it)."""
    capacity = capacity or dcs.pallets_per_week_capacity.to_dict()
    m = pyo.ConcreteModel(name="store_assignment")
    m.D = pyo.Set(initialize=dcs.index.tolist())
    m.S = pyo.Set(initialize=stores.index.tolist())
    m.assign = pyo.Var(m.D, m.S, within=pyo.Binary)
    m.one_dc = pyo.Constraint(m.S, rule=lambda m, s: sum(m.assign[d, s] for d in m.D) == 1)
    m.capacity = pyo.Constraint(m.D, rule=lambda m, d: sum(stores.pallets_per_week[s] * m.assign[d, s] for s in m.S) <= capacity[d])
    for d in m.D:
        for s in m.S:
            french_rule = stores.province[s] == "QC" and not dcs.ships_french_labels[d]
            too_far = lanes.minutes[d, s] > MAX_MINUTES
            if french_rule or too_far:
                m.assign[d, s].fix(0)
    m.cost = pyo.Objective(expr=sum(lanes.cost_per_pallet[d, s] * stores.pallets_per_week[s] * m.assign[d, s]
                                    for d in m.D for s in m.S))
    return m


def chosen_dc(m):
    return {s: next(d for d in m.D if pyo.value(m.assign[d, s]) > 0.5) for s in m.S}
```

Forbidden lanes are handled by fixing their variables to 0 (`fix(0)`), as in the drivers mini-project. And the script:

```python
if __name__ == "__main__":
    con = duckdb.connect("northbound.duckdb")
    dcs, stores, lanes = load_network(con)
    solver = pyo.SolverFactory("highs")

    # ---------- 15.1 Three answers to compare ----------
    today = sum(lanes.cost_per_pallet[stores.current_dc[s], s] * stores.pallets_per_week[s] for s in stores.index)
    lp = build_transport(dcs, stores, lanes)
    solver.solve(lp)
    m = build_assignment(dcs, stores, lanes)
    solver.solve(m)
    print(f"today (Ottawa over capacity):     ${today:,.2f}")
    print(f"pallets split freely (Day 14):    ${pyo.value(lp.cost):,.2f}")
    print(f"one DC per store, all rules:      ${pyo.value(m.cost):,.2f}")

    plan = pd.DataFrame({"city": stores.city, "pallets": stores.pallets_per_week,
                         "today": stores.current_dc, "new": pd.Series(chosen_dc(m))})
    plan["new_cost"] = [lanes.cost_per_pallet[plan.new[s], s] * plan.pallets[s] for s in plan.index]
    print(plan.groupby("new").pallets.sum().to_frame("pallets").join(dcs.pallets_per_week_capacity).to_string())
    print("Stores that change DC:")
    print(plan[plan.today != plan.new].to_string())
```

```text
today (Ottawa over capacity):     $35,194.71
pallets split freely (Day 14):    $39,089.81
one DC per store, all rules:      $41,492.76
        pallets  pallets_per_week_capacity
new
DC-MIS     1365                       1600
DC-MTL     1480                       1500
DC-OTT      700                        700
Stores that change DC:
         city  pallets   today     new  new_cost
S20    Ottawa       87  DC-OTT  DC-MIS   4584.90
S23  Kingston       83  DC-OTT  DC-MIS   3120.80
S24  Gatineau       53  DC-OTT  DC-MTL   1482.94
```

Three plans, three costs:

| Plan | Weekly cost | Feasible? |
|---|---|---|
| Today's routing | $35,194.71 | No: Ottawa 223 pallets over |
| Pallets split freely (Day 14) | $39,089.81 | Yes, but two stores split |
| One DC per store, all rules | **$41,492.76** | Yes |

Single sourcing costs **$2,402.95 a week** more than splitting. The reason is visible in the moves: to get Ottawa under 700 pallets, a whole Ottawa store (S20, 87 pallets) must go to Mississauga, 438 km away, at $4,584.90 a week for that one store. When the model could split, it moved only the pallets it needed to.

## 15.3 Which rules bind?

```python
    # ---------- 15.2 Which rules bind? ----------
    excluded_far = [(d, s) for d in dcs.index for s in stores.index if lanes.minutes[d, s] > MAX_MINUTES]
    print(f"lanes over {MAX_MINUTES} minutes: {len(excluded_far)} of {len(lanes)}")
    m_free = build_assignment(dcs, stores, lanes)
    for v in m_free.assign.values():
        v.unfix()
    solver.solve(m_free)
    print(f"without the French-label and time rules: ${pyo.value(m_free.cost):,.2f}")
```

```text
lanes over 360 minutes: 38 of 120
without the French-label and time rules: $41,492.76
```

38 of 120 lanes are too long for fresh products, and Mississauga can't serve Quebec. Yet removing both rules leaves the cost unchanged: **neither rule binds**. The cheap lanes are short, and Mississauga is never the cheapest DC for a Quebec store anyway. That's worth saying in the memo: the rules cost nothing today, so nobody needs to argue about them.

## 15.4 What is a bigger Ottawa DC worth?

```python
    # ---------- 15.3 What-if: Ottawa DC grows to 900 pallets ----------
    for ott in (700, 800, 900, 1000):
        cap = dict(dcs.pallets_per_week_capacity, **{"DC-OTT": ott})
        mm = build_assignment(dcs, stores, lanes, cap)
        solver.solve(mm)
        print(f"  Ottawa {ott:>4} pallets: ${pyo.value(mm.cost):,.2f}  (saves ${pyo.value(m.cost) - pyo.value(mm.cost):,.2f} a week)")
```

```text
  Ottawa  700 pallets: $41,492.76  (saves $0.00 a week)
  Ottawa  800 pallets: $36,696.10  (saves $4,796.66 a week)
  Ottawa  900 pallets: $35,755.71  (saves $5,737.05 a week)
  Ottawa 1000 pallets: $35,194.71  (saves $6,298.05 a week)
```

A **100-pallet expansion saves $4,796.66 a week**, nearly $250,000 a year. The second 100 pallets add only $940; the third $561. At 1,000 pallets the plan returns to today's routing: today's map is what a big enough Ottawa DC would choose.

This is Day 13's diminishing returns again, but sharper: the first 100 pallets let the model bring the Ottawa store S20 home from Mississauga, which is worth far more than any later pallet. Single sourcing makes the value of capacity **lumpy**: it jumps when one more whole store fits.

### Try it 15.1

With the free-split model from Day 14, what would 100 more pallets in Ottawa save? Why is it less than $4,796.66?

<details><summary>Solution</summary>

```python
lp = build_transport(dcs, stores, lanes, dict(dcs.pallets_per_week_capacity, **{"DC-OTT": 800}))
pyo.SolverFactory("highs").solve(lp)
print(39_089.81 - pyo.value(lp.cost))
```

It saves **$2,525.60 a week**. Ottawa's shadow price ($48.18 a pallet) would predict $4,818, but it holds for only part of the 100 pallets (Day 12). And the single-sourcing plan starts from a worse position ($41,492.76), so it has more to gain: there, the expansion also undoes the expensive S20 move.

</details>

## 15.5 The map and the memo

```python
    # ---------- 15.4 Store and draw the assignment ----------
    con.sql("CREATE OR REPLACE TABLE store_dc_assignment AS SELECT * FROM plan")
    places = con.sql("SELECT store_id, latitude, longitude FROM store_logistics").df().set_index("store_id")
    dc_loc = con.sql("SELECT d.dc_id, p.latitude, p.longitude FROM dc_logistics d JOIN places p ON p.place = d.city").df().set_index("dc_id")
    colours = {"DC-MIS": "#4f7bd9", "DC-OTT": "#e08a2c", "DC-MTL": "#3ea86b"}
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for s, row in plan.iterrows():
        d = row.new
        ax.plot([dc_loc.longitude[d], places.longitude[s]], [dc_loc.latitude[d], places.latitude[s]],
                color=colours[d], lw=0.8, alpha=0.7, ls="-" if d == row.today else "--")
        ax.scatter(places.longitude[s], places.latitude[s], s=row.pallets / 2, color=colours[d], zorder=3)
    for d, row in dc_loc.iterrows():
        ax.scatter(row.longitude, row.latitude, marker="s", s=160, color=colours[d], edgecolor="black", zorder=4, label=d)
    ax.set_title("Store-to-DC assignment (dashed = store changes DC; dot size = pallets a week)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/day15_assignment.png", dpi=120)
    print("saved figures/day15_assignment.png")
    con.close()
```

![Store to DC assignment](figures/day15_assignment.png)

The assignment is stored in `store_dc_assignment` for the next days, and drawn: solid lines are stores that keep their DC, dashed lines stores that change.

Write `day15/memo.md` for the VP.

<details><summary>Model answer</summary>

**To:** VP Logistics **Re:** A store-to-DC map that fits

**Recommendation.** Move three stores: Ottawa S20 and Kingston S23 to Mississauga, Gatineau S24 to Montreal. Every store keeps a single DC, and Ottawa ships exactly its 700 pallets. Weekly shipping cost: **$41,492.76**, against $35,194.71 for today's map, which only works through Ottawa's overtime and trailer yard.

**The rules cost nothing.** French labelling and the six-hour limit don't change the plan: the cheapest DC for every store already meets them.

**Expand Ottawa by 100 pallets.** It saves **$4,797 a week (about $249,000 a year)** by bringing S20 home. A second 100 pallets adds only $940 a week, a third $561. If a 100-pallet expansion costs less than about $4,800 a week to run, do it; then compare its full cost with today's overtime.

**Caveat.** Lane costs are averages per pallet; a real delivery schedule groups stores into truck routes (Module 4).

</details>

## Commit

```bash
git add .
git commit -m "Day 15: single-sourcing store assignment, Ottawa expansion"
git push
```

### Day 15 checklist

- [ ] I can model single sourcing with binary variables and solve it with HiGHS.
- [ ] I compared today's routing, a split plan and a single-sourcing plan, and explained the differences.
- [ ] I checked which business rules bind, and reported the ones that don't.
- [ ] I priced capacity in steps and saw that single sourcing makes its value lumpy.
- [ ] I stored the assignment in the database and drew it on a map.
- [ ] My memo gives the VP a decision, its cost and its value.

---

# DAY 16 — Min-Cost Flow and Cross-Docks

Real goods don't just go from DCs to stores. Chilled products start at supplier plants, travel by full truck to DCs or **cross-docks** (sites where pallets are moved from incoming to outgoing trucks without being stored), and only then go to stores. Today you write one model for any such network: the **minimum-cost flow** problem, built from a table of nodes and a table of arcs. Transportation, assignment and tomorrow's shortest paths are all special cases.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 16.1 Nodes, arcs and flow conservation |
| 0:45 – 1:45 | 16.2 A model built from two tables |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 16.3 The chilled network: what flows where |
| 2:45 – 3:45 | 16.4 Node prices and what a cross-dock saves; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: keep both cross-docks? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 16.1 Nodes, arcs and flow conservation

A network is **nodes** (places) and **arcs** (one-way links). The chilled network has:

| Kind | Nodes | What happens there |
|---|---|---|
| Plant | Saint-Hyacinthe dairy (`PL-SHY`), Brampton chilled foods (`PL-BRA`) | Pallets start, up to the plant's weekly capacity |
| DC | `DC-MIS`, `DC-OTT`, `DC-MTL` | Pallets pass through, up to the DC's chilled throughput |
| Cross-dock | Kingston (`XD-KIN`), Trois-Rivières (`XD-TRV`) | Pallets pass through, up to the cross-dock's throughput |
| Store | 40 stores | Pallets stop: exactly the store's chilled demand |

Arcs have a cost per pallet and sometimes a capacity: full-truck **linehaul** from plants to DCs and cross-docks ($0.08 per pallet-km, limited by the trucks contracted), and store deliveries from DCs and cross-docks ($0.11 per pallet-km, 15% more for refrigerated trucks).

The model has one rule per node, **flow conservation**: what comes in minus what goes out equals what the node keeps.

$$
\sum_{(i,n)} f_{i,n} - \sum_{(n,j)} f_{n,j} =
\begin{cases}
\text{demand}_n & \text{store} \\
0 & \text{DC or cross-dock} \\
\ge -\text{capacity}_n & \text{plant (what leaves, up to capacity)}
\end{cases}
$$

## 16.2 A model built from two tables

Create `day16/network.py`. Nothing in it knows about plants or Kingston: it reads two tables and builds a constraint per node and a variable per arc.

```python
"""Day 16: a general minimum-cost flow model, built from a table of nodes and a table of arcs.

Import it:  from network import load_chilled, build_flow, flow_table
"""
import pandas as pd
import pyomo.environ as pyo


def load_chilled(con):
    nodes = con.sql("SELECT * FROM chilled_nodes").df().set_index("node")
    arcs = con.sql("SELECT * FROM chilled_arcs").df().set_index(["from_node", "to_node"])
    return nodes, arcs


def build_flow(nodes, arcs, closed=()):
    """Plants send pallets out (up to their capacity); DCs and cross-docks pass them on (up to their throughput);
    stores keep what they need. Arcs have a cost per pallet and, sometimes, a capacity."""
    m = pyo.ConcreteModel(name="chilled_flow")
    m.N = pyo.Set(initialize=nodes.index.tolist())
    m.A = pyo.Set(initialize=arcs.index.tolist(), dimen=2)

    def arc_bounds(m, i, j):
        cap = arcs.capacity_pallets[i, j]
        return (0, None if pd.isna(cap) else float(cap))

    m.flow = pyo.Var(m.A, bounds=arc_bounds)
    for i, j in m.A:
        if i in closed or j in closed:
            m.flow[i, j].fix(0)
    inflow = {n: [] for n in nodes.index}
    outflow = {n: [] for n in nodes.index}
    for i, j in arcs.index:
        outflow[i].append((i, j))
        inflow[j].append((i, j))

    def balance(m, n):
        net_in = sum(m.flow[a] for a in inflow[n]) - sum(m.flow[a] for a in outflow[n])
        kind = nodes.kind[n]
        if kind == "plant":
            return -net_in <= float(nodes.throughput_capacity[n])     # what leaves a plant, up to its capacity
        if kind == "store":
            return net_in == float(nodes.demand_pallets[n])           # what a store receives, exactly its demand
        return net_in == 0                                             # DCs and cross-docks keep nothing

    m.balance = pyo.Constraint(m.N, rule=balance)
    hubs = [n for n in nodes.index if nodes.kind[n] in ("dc", "cross_dock")]
    m.throughput = pyo.Constraint(hubs, rule=lambda m, n: sum(m.flow[a] for a in inflow[n]) <= float(nodes.throughput_capacity[n]))
    m.cost = pyo.Objective(expr=sum(float(arcs.cost_per_pallet[a]) * m.flow[a] for a in m.A))
    return m


def flow_table(m, nodes):
    rows = []
    for n in m.N:
        if nodes.kind[n] == "store":
            continue
        out = sum(pyo.value(m.flow[i, j]) for i, j in m.A if i == n)
        inn = sum(pyo.value(m.flow[i, j]) for i, j in m.A if j == n)
        rows.append((n, nodes.kind[n], nodes.place[n], round(inn, 1) + 0.0, round(out, 1) + 0.0, nodes.throughput_capacity[n]))
    return pd.DataFrame(rows, columns=["node", "kind", "place", "pallets_in", "pallets_out", "capacity"])
```

Points worth slowing down for:

- **Arc lists per node** (`inflow`, `outflow`) are built once. Searching all 156 arcs inside every node's rule would work, but get slow on big networks.
- `m.A` is a set of **pairs** (`dimen=2`), so `m.flow["DC-MTL", "S33"]` reads like the arc it is.
- `closed` fixes every arc into or out of a closed node to zero: closing a site is a data question, not a new model.

## 16.3 The chilled network: what flows where

Create `day16/run_chilled.py`:

```python
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day16")
from network import build_flow, flow_table, load_chilled

pd.set_option("display.width", 140)
con = duckdb.connect("northbound.duckdb", read_only=True)
nodes, arcs = load_chilled(con)
print(nodes.kind.value_counts().to_dict(), f"{len(arcs)} arcs, {nodes.demand_pallets.sum():.0f} pallets of chilled demand")

m = build_flow(nodes, arcs)
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
result = pyo.SolverFactory("highs").solve(m)
print(result.solver.termination_condition, f"weekly chilled cost ${pyo.value(m.cost):,.2f}")
print(flow_table(m, nodes).to_string(index=False))

# Which stores are served through a cross-dock?
via = {j: i for i, j in m.A if i.startswith("XD-") and pyo.value(m.flow[i, j]) > 1e-6}
print("served through a cross-dock:", {s: f"{x} ({nodes.place[s]})" for s, x in via.items()})
flows = [pyo.value(m.flow[a]) for a in m.A]
print("every flow a whole number of pallets:", all(abs(f - round(f)) < 1e-6 for f in flows))
```

```text
{'store': 40, 'dc': 3, 'plant': 2, 'cross_dock': 2} 156 arcs, 922 pallets of chilled demand
optimal weekly chilled cost $18,203.64
  node       kind           place  pallets_in  pallets_out  capacity
PL-SHY      plant Saint-Hyacinthe         0.0        578.0       950
PL-BRA      plant        Brampton         0.0        344.0       900
DC-MIS         dc     Mississauga       344.0        344.0       480
DC-OTT         dc          Ottawa       200.0        200.0       200
DC-MTL         dc        Montreal       255.0        255.0       460
XD-KIN cross_dock        Kingston        35.0         35.0       160
XD-TRV cross_dock  Trois-Rivieres       123.0        123.0       220
served through a cross-dock: {'S22': 'XD-KIN (Kingston)', 'S23': 'XD-KIN (Kingston)', 'S36': 'XD-TRV (Quebec City)', 'S37': 'XD-TRV (Quebec City)', 'S38': 'XD-TRV (Quebec City)', 'S39': 'XD-TRV (Sherbrooke)', 'S40': 'XD-TRV (Sherbrooke)'}
every flow a whole number of pallets: True
```

922 chilled pallets a week cost **$18,203.64** to move. Reading the flows:

- **Saint-Hyacinthe supplies 578 pallets, Brampton 344**: each plant serves its own side of the network. Neither plant is full.
- **The Ottawa DC is full** (200 of 200). Mississauga and Montreal have room.
- **Kingston's two stores come through the Kingston cross-dock** (35 pallets, by full truck from Mississauga). **Quebec City and Sherbrooke stores come through Trois-Rivières** (123 pallets, straight from the Saint-Hyacinthe plant), skipping the Montreal DC entirely.
- Flows are whole pallets again: min-cost flow shares the transportation problem's special structure.

> **Decision it supports:** which plant, DC and cross-dock each store's chilled pallets go through.
>
> **Other decisions you can support the same way:** routing parcels through sorting hubs · supply chains from farms through packers to stores · blood from donor clinics through labs to hospitals.

## 16.4 Node prices and what a cross-dock saves

```python
# Shadow prices: the cost of one more chilled pallet delivered at each store, and of hub capacity
prices = pd.Series({n: m.dual[m.balance[n]] for n in m.N if nodes.kind[n] == "store"}).round(2)
print("delivered cost of one more chilled pallet, cheapest and dearest stores:")
print(prices.sort_values().to_frame("cost").join(nodes.place).iloc[[0, 1, -2, -1]].to_string())
print("hub capacity worth ($ a pallet):", {n: round(-m.dual[m.throughput[n]], 2) for n in m.throughput})

# What does each cross-dock save?
for xd in ("XD-KIN", "XD-TRV"):
    closed = build_flow(nodes, arcs, closed=(xd,))
    pyo.SolverFactory("highs").solve(closed)
    print(f"without {xd}: ${pyo.value(closed.cost):,.2f}  (+${pyo.value(closed.cost) - pyo.value(m.cost):,.2f})")
```

```text
delivered cost of one more chilled pallet, cheapest and dearest stores:
      cost        place
S10   9.15  Mississauga
S11   9.41  Mississauga
S25  39.09     Gatineau
S24  39.48     Gatineau
hub capacity worth ($ a pallet): {'DC-MIS': 0.0, 'DC-OTT': 9.3, 'DC-MTL': 0.0, 'XD-KIN': 0.0, 'XD-TRV': 0.0}
without XD-KIN: $18,762.12  (+$558.48)
without XD-TRV: $19,391.57  (+$1,187.93)
```

- **The dearest chilled pallets go to Gatineau** ($39.48), not Quebec City: Ottawa's DC is full, so Gatineau is served from Montreal, 208 km away, while Quebec City gets cheap full trucks to Trois-Rivières.
- **Ottawa DC capacity is worth $9.30 a pallet**; every other hub has room.
- **Closing Kingston costs $558.48 a week, closing Trois-Rivières $1,187.93.** Those are what the cross-docks save in transport, before their own running costs.

### The planted bug

`day16/bug_sign.py` rewrites every store's balance as "out minus in equals demand". Run it.

```python
m = build_flow(nodes, arcs)
for n in m.N:
    if nodes.kind[n] == "store":
        out = sum(m.flow[i, j] for i, j in m.A if i == n)
        inn = sum(m.flow[i, j] for i, j in m.A if j == n)
        m.balance[n].set_value(out - inn == float(nodes.demand_pallets[n]))
print(pyo.SolverFactory("highs").solve(m, load_solutions=False).solver.termination_condition)
```

<details><summary>What happens, and why</summary>

```text
infeasible
```

With "out − in = demand", every store must **send out** its demand: stores become sources. Nothing can absorb those pallets (nothing downstream of a store), so the model is infeasible. Every network model needs one sign convention, stated once and used everywhere. Ours: inflow minus outflow equals what the node keeps.

</details>

## Mini-project: keep both cross-docks?

Each cross-dock costs money to run: Kingston $500 a week, Trois-Rivières $900. Compare all four combinations of open and closed, with transport and fixed costs, and recommend.

<details><summary>Solution</summary>

```python
import itertools
import sys

import duckdb
import pyomo.environ as pyo

sys.path.insert(0, "day16")
from network import build_flow, load_chilled

FIXED = {"XD-KIN": 500, "XD-TRV": 900}
con = duckdb.connect("northbound.duckdb", read_only=True)
nodes, arcs = load_chilled(con)
for keep in itertools.product([True, False], repeat=2):
    open_xd = [x for x, k in zip(FIXED, keep) if k]
    m = build_flow(nodes, arcs, closed=[x for x in FIXED if x not in open_xd])
    pyo.SolverFactory("highs").solve(m)
    fixed = sum(FIXED[x] for x in open_xd)
    print(f"open {', '.join(open_xd) or 'none':<16} flow ${pyo.value(m.cost):,.2f} + fixed ${fixed:,} = ${pyo.value(m.cost) + fixed:,.2f}")
```

```text
open XD-KIN, XD-TRV   flow $18,203.64 + fixed $1,400 = $19,603.64
open XD-KIN           flow $19,391.57 + fixed $500 = $19,891.57
open XD-TRV           flow $18,762.12 + fixed $900 = $19,662.12
open none             flow $19,950.05 + fixed $0 = $19,950.05
```

**Keep both**, at $19,603.64 a week in total. Trois-Rivières saves $1,187.93 for $900: a clear yes. Kingston saves $558.48 for $500: yes, but only by $58 a week. A small rise in Kingston's rent, or a small drop in Kingston's volume, would flip it. Here the savings simply add up (closing both costs $558.48 + $1,187.93 more), because the two cross-docks serve different stores. When sites compete for the same customers, the four combinations stop adding up, and with dozens of candidate sites you can't list them all: Module 3's facility location model chooses with binary variables.

</details>

## Commit

```bash
git add .
git commit -m "Day 16: min-cost flow from node and arc tables, cross-docks"
git push
```

### Day 16 checklist

- [ ] I can write flow conservation for plants, hubs and stores with one sign convention.
- [ ] My flow model is built from a node table and an arc table, and knows nothing else.
- [ ] I can read node shadow prices as delivered costs and hub duals as the value of capacity.
- [ ] I priced each cross-dock by closing it, and added fixed costs to decide.
- [ ] I know transportation and assignment are special cases of min-cost flow.
- [ ] I found a sign error by its infeasibility.

---

# DAY 17 — Shortest Paths and Bottlenecks

Where did the lane distances in `dc_store_lanes` come from? From shortest paths on the road network, computed by the data script. Today you compute them yourself, three ways, and then ask the opposite question: not "what is the shortest way?" but "how much can get through at all?": the **maximum flow**, and the **bottleneck** that limits it.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 17.1 The road network as a graph |
| 0:45 – 1:45 | 17.2 Shortest path as a flow; 17.3 where the lane distances come from |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 17.4 Maximum flow |
| 2:45 – 3:45 | 17.5 The minimum cut: finding the bottleneck; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: a winter storm closes the 401 |
| 4:45 – 5:00 | Commit with Git + checklist |

## 17.1 The road network as a graph

`road_segments` has 28 roads between 17 places, each with km, driving minutes and the trucks a week Northbound's carriers have contracted on it. **networkx** is Python's graph library: nodes, edges, and the classic algorithms. Create `day17/paths.py`:

```python
import duckdb
import networkx as nx
import pandas as pd
import pyomo.environ as pyo

pd.set_option("display.width", 140)
con = duckdb.connect("northbound.duckdb", read_only=True)
roads = con.sql("SELECT * FROM road_segments").df()

# ---------- 17.1 The road network as a graph ----------
G = nx.Graph()                         # roads go both ways
for r in roads.itertuples():
    G.add_edge(r.from_place, r.to_place, km=r.km, minutes=r.minutes, road=r.road)
print(f"{G.number_of_nodes()} places, {G.number_of_edges()} road segments")

for weight in ("km", "minutes"):
    path = nx.shortest_path(G, "Mississauga", "Quebec City", weight=weight)
    km = nx.path_weight(G, path, "km")
    minutes = nx.path_weight(G, path, "minutes")
    print(f"shortest by {weight:<7}: {' - '.join(path)}  ({km} km, {minutes} min)")
```

```text
17 places, 28 road segments
shortest by km     : Mississauga - Toronto - Belleville - Kingston - Cornwall - Montreal - Saint-Hyacinthe - Drummondville - Quebec City  (858 km, 570 min)
shortest by minutes: Mississauga - Toronto - Belleville - Kingston - Cornwall - Montreal - Trois-Rivieres - Quebec City  (863 km, 565 min)
```

**Dijkstra's algorithm** (what `shortest_path` uses for weighted graphs) grows a set of places whose shortest distance is known, always adding the nearest unknown one. The shortest route by **distance** (858 km, via Drummondville) is not the fastest: the route via Trois-Rivières is 5 km longer and 5 minutes quicker. A dispatcher wants minutes, a fuel budget wants km: the "shortest" path depends on what you measure.

## 17.2 Shortest path as a flow

The shortest path is a min-cost flow: send **one** truck from Mississauga to Quebec City, where each road costs its km. Roads go both ways, so each becomes two arcs.

```python
# ---------- 17.2 The same answer as a minimum-cost flow ----------
# Send one truck from Mississauga to Quebec City; each road can be used in either direction.
arcs = [(a, b) for a, b in G.edges()] + [(b, a) for a, b in G.edges()]
m = pyo.ConcreteModel()
m.N = pyo.Set(initialize=list(G.nodes()))
m.A = pyo.Set(initialize=arcs, dimen=2)
m.use = pyo.Var(m.A, within=pyo.NonNegativeReals)
supply = {n: 0 for n in G.nodes()}
supply["Mississauga"], supply["Quebec City"] = 1, -1
m.balance = pyo.Constraint(m.N, rule=lambda m, n: sum(m.use[n, j] for i, j in m.A if i == n)
                           - sum(m.use[i, n] for i, j in m.A if j == n) == supply[n])
m.km = pyo.Objective(expr=sum(G.edges[i, j]["km"] * m.use[i, j] for i, j in m.A))
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
pyo.SolverFactory("highs").solve(m)
used = [(i, j) for i, j in m.A if pyo.value(m.use[i, j]) > 0.5]
print(f"LP: {pyo.value(m.km):g} km using {used}")
# The duals are distances: dual(n) - dual(Quebec City) = km from n to Quebec City
d_end = m.dual[m.balance["Quebec City"]]
for n in ("Mississauga", "Kingston", "Montreal", "Drummondville"):
    print(f"  {n:<14} dual gives {m.dual[m.balance[n]] - d_end:6.1f} km to Quebec City; "
          f"networkx gives {nx.shortest_path_length(G, n, 'Quebec City', weight='km')}")
```

```text
LP: 858 km using [('Mississauga', 'Toronto'), ('Toronto', 'Belleville'), ('Belleville', 'Kingston'), ('Kingston', 'Cornwall'), ('Cornwall', 'Montreal'), ('Montreal', 'Saint-Hyacinthe'), ('Saint-Hyacinthe', 'Drummondville'), ('Drummondville', 'Quebec City')]
  Mississauga    dual gives  858.0 km to Quebec City; networkx gives 858
  Kingston       dual gives  560.0 km to Quebec City; networkx gives 560
  Montreal       dual gives  265.0 km to Quebec City; networkx gives 265
  Drummondville  dual gives  150.0 km to Quebec City; networkx gives 150
```

The LP finds the same 858 km. Its **duals** are distances: the shadow price of a place's balance, minus Quebec City's, is that place's distance to Quebec City. Each node's dual is the cost of getting one more truck from there to the end, which, for a shortest path, is the distance. Dijkstra's algorithm and the LP's dual are two ways of computing the same numbers.

## 17.3 Where the lane distances come from

`all_pairs_dijkstra_path_length` gives the road distance between every pair of places:

```python
# ---------- 17.3 All pairs: where the lane distances come from ----------
dist = dict(nx.all_pairs_dijkstra_path_length(G, weight="km"))
dcs = {"DC-MIS": "Mississauga", "DC-OTT": "Ottawa", "DC-MTL": "Montreal"}
table = pd.DataFrame({dc: {c: dist[city][c] for c in ["Toronto", "Hamilton", "Kingston", "Ottawa", "Gatineau",
                                                      "Montreal", "Laval", "Sherbrooke", "Quebec City"]}
                      for dc, city in dcs.items()})
print("road km from each DC's city:")
print(table.T.to_string())
check = con.sql("""
    SELECT l.dc_id, l.store_id, s.city, l.km
    FROM dc_store_lanes l JOIN stores s USING (store_id)
    WHERE l.store_id IN ('S01', 'S23', 'S36') ORDER BY l.store_id, l.dc_id""").df()
check["road_km"] = [dist[dcs[r.dc_id]][r.city] for r in check.itertuples()]
check["in_town_km"] = (check.km - check.road_km).round(1)
print(check.to_string(index=False))
```

```text
road km from each DC's city:
        Toronto  Hamilton  Kingston  Ottawa  Gatineau  Montreal  Laval  Sherbrooke  Quebec City
DC-MIS       28        55       298     438       446       593    613         743          858
DC-OTT      410       480       195       0         8       200    220         350          465
DC-MTL      565       635       295     200       208         0     20         150          265
 dc_id store_id        city    km  road_km  in_town_km
DC-MIS      S01     Toronto  31.6       28         3.6
DC-MTL      S01     Toronto 568.6      565         3.6
DC-OTT      S01     Toronto 413.6      410         3.6
DC-MIS      S23    Kingston 305.5      298         7.5
DC-MTL      S23    Kingston 302.5      295         7.5
DC-OTT      S23    Kingston 202.5      195         7.5
DC-MIS      S36 Quebec City 863.1      858         5.1
DC-MTL      S36 Quebec City 270.1      265         5.1
DC-OTT      S36 Quebec City 470.1      465         5.1
```

Each lane in `dc_store_lanes` is the road distance between the DC's city and the store's city, plus a few km in town to the store's door (`in_town_km`, the same for every DC). The Day 14 to 16 models rested on exactly this computation.

> **Decision it supports:** the route, and the distance and time behind every lane cost.
>
> **Other decisions you can support the same way:** emergency response times · the cheapest sequence of currency conversions · project critical paths (longest paths).

## 17.4 Maximum flow

Before Christmas, Northbound wants to move as many trucks as possible from the Mississauga DC to Quebec City. Each carrier contract allows a number of trucks a week on one road, eastbound: the network is now **directed**, with a **capacity** on each arc. The maximum flow is the most trucks a week the contracts allow, end to end. Create `day17/max_flow.py`:

```python
import duckdb
import networkx as nx
import pyomo.environ as pyo

con = duckdb.connect("northbound.duckdb", read_only=True)
roads = con.sql("SELECT from_place, to_place, truck_slots_per_week FROM road_segments").fetchall()

# Carrier contracts are eastbound: a directed graph, one arc per contract
G = nx.DiGraph()
for a, b, slots in roads:
    G.add_edge(a, b, capacity=slots)

# ---------- 17.4 Maximum flow as an LP ----------
m = pyo.ConcreteModel()
m.N = pyo.Set(initialize=list(G.nodes()))
m.A = pyo.Set(initialize=list(G.edges()), dimen=2)
m.trucks = pyo.Var(m.A, bounds=lambda m, i, j: (0, G.edges[i, j]["capacity"]))
m.through = pyo.Var(within=pyo.NonNegativeReals)        # trucks a week from start to end
start, end = "Mississauga", "Quebec City"
net = {n: (1 if n == start else -1 if n == end else 0) for n in G.nodes()}
m.balance = pyo.Constraint(m.N, rule=lambda m, n: sum(m.trucks[n, j] for i, j in m.A if i == n)
                           - sum(m.trucks[i, n] for i, j in m.A if j == n) == net[n] * m.through)
m.most = pyo.Objective(expr=m.through, sense=pyo.maximize)
m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
pyo.SolverFactory("highs").solve(m)
print(f"LP: at most {pyo.value(m.through):g} trucks a week from {start} to {end}")
busy = {a: pyo.value(m.trucks[a]) for a in m.A if pyo.value(m.trucks[a]) > 1e-6}
print("trucks on each contract used:", {f"{i}->{j}": round(v) for (i, j), v in busy.items()})
```

```text
LP: at most 33 trucks a week from Mississauga to Quebec City
trucks on each contract used: {'Mississauga->Toronto': 33, 'Toronto->Peterborough': 8, 'Toronto->Belleville': 25, 'Peterborough->Belleville': 2, 'Peterborough->Ottawa': 6, 'Belleville->Kingston': 27, 'Kingston->Ottawa': 12, 'Kingston->Cornwall': 15, 'Ottawa->Gatineau': 6, 'Ottawa->Montreal': 12, 'Cornwall->Montreal': 15, 'Montreal->Laval': 8, 'Montreal->Saint-Hyacinthe': 18, 'Montreal->Trois-Rivieres': 7, 'Gatineau->Montreal': 6, 'Laval->Trois-Rivieres': 8, 'Saint-Hyacinthe->Drummondville': 18, 'Drummondville->Quebec City': 18, 'Trois-Rivieres->Quebec City': 15}
```

`m.through` is a variable: the flow leaves Mississauga and arrives in Quebec City at the same rate, conserved everywhere in between. At most **33 trucks a week** can get through.

(Don't name a script `bottleneck.py`: pandas imports an optional package called `bottleneck`, and Python would find your file first. Your own file names can shadow packages: name scripts after what they do, not after common words.)

## 17.5 The minimum cut: finding the bottleneck

Why 33? The **max-flow min-cut theorem**: the maximum flow equals the smallest total capacity of a set of arcs whose removal disconnects start from end. That set is the **bottleneck**.

```python
# ---------- 17.5 networkx, and the minimum cut ----------
value, _ = nx.maximum_flow(G, start, end)
cut_value, (reach, rest) = nx.minimum_cut(G, start, end)
cut = sorted((a, b, G.edges[a, b]["capacity"]) for a in reach for b in G.successors(a) if b in rest)
print(f"networkx maximum flow {value}; minimum cut {cut_value}: {cut}")
print("places still reachable on the Mississauga side of the cut:", sorted(reach))
```

```text
networkx maximum flow 33; minimum cut 33: [('Kingston', 'Cornwall', 15), ('Kingston', 'Ottawa', 12), ('Peterborough', 'Ottawa', 6)]
places still reachable on the Mississauga side of the cut: ['Belleville', 'Brampton', 'Hamilton', 'Kingston', 'Mississauga', 'Peterborough', 'Toronto']
```

The bottleneck is **east of Kingston**: the 401 to Cornwall (15 trucks), Kingston to Ottawa (12) and Peterborough to Ottawa (6). Everything west of it can carry more than 33 a week; everything east of it too. Adding trucks anywhere else changes nothing. Max flow and min cut are a primal–dual pair, like Day 11's bakery and the rent for it: the flow model's dual is the cut.

### The planted bug

`day17/bug_units.py` loads the contracts with one copy-paste slip. Run it, then find the slip.

```python
import duckdb
import networkx as nx

con = duckdb.connect("northbound.duckdb", read_only=True)
G = nx.DiGraph()
for a, b, km, slots in con.sql("SELECT from_place, to_place, km, truck_slots_per_week FROM road_segments").fetchall():
    G.add_edge(a, b, capacity=km)
print("trucks a week from Mississauga to Quebec City:", nx.maximum_flow_value(G, "Mississauga", "Quebec City"))
```

<details><summary>What happens, and how to spot it</summary>

```text
trucks a week from Mississauga to Quebec City: 48
```

48 trucks: plausible, and wrong. The capacity is the road's **km**, not its truck slots. The minimum cut gives it away: with km as capacity, the bottleneck is the two short roads out of Mississauga (28 km to Toronto + 20 km to Brampton = 48). A bottleneck in the middle of the suburbs should make anyone suspicious. Day 9's lesson again: check the units, and **check that the explanation makes sense**, not just the number.

</details>

## Mini-project: a winter storm closes the 401

A storm closes the 401 between Kingston and Cornwall for a week. Recompute the fastest route and the maximum flow from Mississauga to Quebec City, find the new bottleneck, and test which single carrier contract would gain most from 10 more trucks a week.

<details><summary>Solution</summary>

```python
import duckdb
import networkx as nx

con = duckdb.connect("northbound.duckdb", read_only=True)
roads = con.sql("SELECT from_place, to_place, km, minutes, truck_slots_per_week FROM road_segments").fetchall()
START, END, CLOSED = "Mississauga", "Quebec City", {"Kingston", "Cornwall"}


def graphs(closed=False, extra=None):
    roadmap, contracts = nx.Graph(), nx.DiGraph()
    for a, b, km, minutes, slots in roads:
        if closed and {a, b} == CLOSED:
            continue
        roadmap.add_edge(a, b, km=km, minutes=minutes)
        contracts.add_edge(a, b, capacity=slots + (extra or {}).get((a, b), 0))
    return roadmap, contracts


for closed in (False, True):
    roadmap, contracts = graphs(closed)
    path = nx.shortest_path(roadmap, START, END, weight="minutes")
    print(f"{'storm' if closed else 'normal'}: fastest {nx.path_weight(roadmap, path, 'minutes')} min via "
          f"{' - '.join(path)}; at most {nx.maximum_flow_value(contracts, START, END)} trucks a week")

# During the storm, where is the bottleneck, and which single contract would gain most from 10 more trucks?
_, contracts = graphs(True)
base, (reach, rest) = nx.minimum_cut(contracts, START, END)
print("storm bottleneck:", sorted((a, b, contracts.edges[a, b]["capacity"]) for a in reach for b in contracts.successors(a) if b in rest))
gains = []
for a, b in contracts.edges():
    _, more = graphs(True, {(a, b): 10})
    gains.append((nx.maximum_flow_value(more, START, END) - base, f"{a}->{b}"))
print("best single contracts to extend by 10 trucks:", sorted(gains, reverse=True)[:4])
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

- The fastest route detours through Ottawa: **639 minutes instead of 565**.
- Capacity falls from 33 to **18 trucks a week**: the storm removed the 401's 15.
- The new bottleneck is the two roads into Ottawa (12 + 6). But 10 more trucks on either adds only **2**: as soon as one road into Ottawa widens, the next limit appears (Ottawa to Montreal and Gatineau to Montreal). To restore 33 trucks, Northbound needs several contracts at once, or a different route altogether (the Peterborough route, or rail).

The lesson of Day 13 in network form: relieving one bottleneck reveals the next. Always recompute the cut after a change.

</details>

## Commit

```bash
git add .
git commit -m "Day 17: shortest paths, maximum flow, minimum cut"
git push
```

Then run the tests. `tests/test_network.py` checks Days 14–17: the transportation and single-sourcing costs, the Hungarian method against the LP, flow conservation, the shortest path and max flow = min cut. Add `"day14"`, `"day15"`, `"day16"` to the folders in `conftest.py`.

```python
"""Module 2, Days 14-17: transportation, assignment, network flow, shortest paths and maximum flow."""
import networkx as nx
import numpy as np
import pyomo.environ as pyo
import pytest
from network import build_flow, load_chilled
from scipy.optimize import linear_sum_assignment
from store_assignment import build_assignment, chosen_dc
from transport import build_transport, load_network

TOL = 1e-6


@pytest.fixture(scope="module")
def network(con):
    return load_network(con)


def test_transportation_known_answer_and_whole_pallets(network):
    dcs, stores, lanes = network
    m = build_transport(dcs, stores, lanes)
    pyo.SolverFactory("highs").solve(m)
    assert pyo.value(m.cost) == pytest.approx(39_089.81, abs=0.01)
    for d in m.D:
        assert sum(pyo.value(m.ship[d, s]) for s in m.S) <= dcs.pallets_per_week_capacity[d] + TOL
    assert all(abs(pyo.value(v) - round(pyo.value(v))) < TOL for v in m.ship.values())


def test_single_sourcing_costs_at_least_the_split_plan(network):
    dcs, stores, lanes = network
    lp, mip = build_transport(dcs, stores, lanes), build_assignment(dcs, stores, lanes)
    pyo.SolverFactory("highs").solve(lp)
    pyo.SolverFactory("highs").solve(mip)
    assert pyo.value(mip.cost) >= pyo.value(lp.cost) - TOL
    assert pyo.value(mip.cost) == pytest.approx(41_492.76, abs=0.01)
    plan = chosen_dc(mip)
    assert all(not (stores.province[s] == "QC" and d == "DC-MIS") for s, d in plan.items())


def test_assignment_lp_matches_hungarian(con):
    minutes = con.sql("SELECT * FROM driver_route_minutes").df().pivot(index="driver", columns="route", values="minutes")
    rows, cols = linear_sum_assignment(minutes.to_numpy())
    m = pyo.ConcreteModel()
    m.x = pyo.Var(minutes.index, minutes.columns, bounds=(0, 1))
    m.a = pyo.Constraint(minutes.index, rule=lambda m, d: sum(m.x[d, r] for r in minutes.columns) == 1)
    m.b = pyo.Constraint(minutes.columns, rule=lambda m, r: sum(m.x[d, r] for d in minutes.index) == 1)
    m.o = pyo.Objective(expr=sum(minutes.loc[d, r] * m.x[d, r] for d in minutes.index for r in minutes.columns))
    pyo.SolverFactory("highs").solve(m)
    assert pyo.value(m.o) == pytest.approx(minutes.to_numpy()[rows, cols].sum())


def test_chilled_flow_is_conserved_and_known(con):
    nodes, arcs = load_chilled(con)
    m = build_flow(nodes, arcs)
    pyo.SolverFactory("highs").solve(m)
    assert pyo.value(m.cost) == pytest.approx(18_203.64, abs=0.01)
    shipped = sum(pyo.value(m.flow[i, j]) for i, j in m.A if nodes.kind[i] == "plant")
    assert shipped == pytest.approx(nodes.demand_pallets.sum())


def test_shortest_path_and_max_flow(con):
    roads = con.sql("SELECT * FROM road_segments").df()
    G = nx.Graph()
    D = nx.DiGraph()
    for r in roads.itertuples():
        G.add_edge(r.from_place, r.to_place, km=r.km)
        D.add_edge(r.from_place, r.to_place, capacity=r.truck_slots_per_week)
    assert nx.shortest_path_length(G, "Mississauga", "Quebec City", weight="km") == 858
    flow = nx.maximum_flow_value(D, "Mississauga", "Quebec City")
    cut, _ = nx.minimum_cut(D, "Mississauga", "Quebec City")
    assert flow == cut == 33
```

```text
31 passed
```

### Day 17 checklist

- [ ] I can load a road network into networkx and find shortest paths by km and by minutes.
- [ ] I solved a shortest path as a min-cost flow and read distances from its duals.
- [ ] I reproduced the lane distances from road segments.
- [ ] I can solve a maximum flow as an LP and with networkx.
- [ ] I can find the bottleneck with a minimum cut, and know max flow equals min cut.
- [ ] I recomputed the cut after a change, and saw the next bottleneck appear.
