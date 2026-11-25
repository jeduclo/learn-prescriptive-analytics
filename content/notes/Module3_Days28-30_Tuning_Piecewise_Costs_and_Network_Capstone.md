# DAY 28 — Making MIPs Fast: HiGHS vs Gurobi

Day 27 ended with a warning: the same model can solve at the root or after 16,000 nodes, depending on details that shouldn't matter. Today you measure solver performance properly (several runs, counting work rather than seconds), make the hard Christmas model easier by sizing it with its bound, and, optionally, compare HiGHS with Gurobi, the commercial solver many employers use.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 28.1 Measure: five seeds, one model |
| 0:45 – 1:45 | 28.2 A smaller model: size it with the bound |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 28.3 Gurobi (optional) |
| 2:45 – 3:45 | 28.4 A speed checklist; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: Christmas week |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Days 28–30: `christmas_week_orders` (Montreal's orders for December 21–24, Day 28), `flour_discounts`, `flour_all_units_offer`, `flour_delivery` and `overtime_tiers` (Day 29), `network_options` and `network_settings` (expansions and planning settings for the capstone, Day 30).

### Optional: Gurobi

Gurobi is optional in this course. `pip install gurobipy==14.0.0` installs it with a free licence limited to 2,000 variables and 2,000 constraints: enough for today's models, not for the Day 30 capstone. Students can get a full academic licence from Gurobi's website. Every script runs without it and says so.

## 28.1 Measure: five seeds, one model

A MIP solver makes many arbitrary choices (ties between branching variables, the order of cuts), and a **random seed** decides them. Changing the seed changes nothing about the problem, so it is the honest way to see how much luck is in one run. Measure **nodes and LP iterations**, not seconds: they are the same on every run of the same seed, while seconds depend on the computer and whatever else it is doing.

`run_highs` (Day 26) now also returns the LP iterations. Create `day28/tuning.py`:

```python
import sys

import duckdb
import pandas as pd

for folder in ("day24", "day26", "day28"):
    sys.path.insert(0, folder)
from gurobi_run import have_gurobi, run_gurobi
from highs_run import run_highs
from loads import lower_bound
from packing import build_packing

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
orders = con.sql("SELECT * FROM christmas_orders ORDER BY order_id").df().set_index("order_id")
capacity = con.sql("SELECT pallet_positions FROM truck_spec WHERE truck_type = 'trailer'").fetchall()[0][0]
MAX_KG, SEEDS = 16000, range(5)


def runs(solver, run, n, **options):
    rows = []
    for seed in SEEDS:
        seed_option = {"random_seed": seed} if solver == "HiGHS" else {"Seed": seed}
        r = run(build_packing(orders, capacity, n, max_kg=MAX_KG), **seed_option, **options)
        rows.append((solver, n, seed, r["status"], round(r["objective"]), round(r["bound"], 2), r["nodes"], r["lp_iterations"]))
    return rows


cols = ["solver", "trailers_allowed", "seed", "status", "trailers", "bound", "nodes", "lp_iterations"]

# ---------- 28.1 Measure: five seeds, same model ----------
print("--- HiGHS, 16 trailers allowed (as on Day 27)")
t = pd.DataFrame(runs("HiGHS", run_highs, 16, mip_max_nodes=20_000), columns=cols)
print(t.to_string(index=False))
print(f"nodes: median {t.nodes.median():,.0f}, smallest {t.nodes.min():,}, largest {t.nodes.max():,}")
```

```text
--- HiGHS, 16 trailers allowed (as on Day 27)
solver  trailers_allowed  seed  status  trailers  bound  nodes  lp_iterations
 HiGHS                16     0 Optimal        15   15.0   2161          38901
 HiGHS                16     1 Optimal        15   15.0  16124         253217
 HiGHS                16     2 Optimal        15   15.0      1           1198
 HiGHS                16     3 Optimal        15   15.0   4166          90431
 HiGHS                16     4 Optimal        15   15.0  11001         216987
nodes: median 4,166, smallest 1, largest 16,124
```

Same model, same answer (15 trailers), and the work ranges from **1 node to 16,124**. One run tells you almost nothing about a model's speed. Report the median and the worst case over several seeds, and compare models on the *same* seeds.

## 28.2 A smaller model: size it with the bound

The model allows 16 trailers (first fit decreasing's count), so every plan with 15 has many copies: any of the 16 trailers can be the empty one. The weight bound says at least 15 are needed. Ask a smaller question: **is 15 possible?** Allow exactly 15 trailers; if the solver proves that infeasible, try 16. Create `day28/loads.py`:

```python
"""Day 28: the fewest trailers for a day's orders, found by sizing the model with the lower bound.

Import it:  from loads import lower_bound, fewest_trailers
"""
import math

from packing import build_packing


def lower_bound(orders, capacity, max_kg):
    return max(math.ceil(orders.pallets.sum() / capacity), math.ceil(orders.weight_kg.sum() / max_kg))


def fewest_trailers(orders, capacity, max_kg, run, max_tries=3, **options):
    """Try n = lower bound trailers; if the solver proves n impossible, try n + 1, and so on.
    run is run_highs or run_gurobi; options go to the solver. Returns (trailers, proven, total nodes, tries)."""
    n, nodes = lower_bound(orders, capacity, max_kg), 0
    for tries in range(1, max_tries + 1):
        r = run(build_packing(orders, capacity, n, max_kg=max_kg), **options)
        nodes += max(r["nodes"], 0)
        if r["status"] == "Optimal":
            return n, True, nodes, tries
        if r["status"] != "Infeasible":
            return None, False, nodes, tries      # stopped by a limit: no answer either way
        n += 1
    return None, False, nodes, max_tries
```

```python
# ---------- 28.2 A smaller model: allow only as many trailers as the bound ----------
n = lower_bound(orders, capacity, MAX_KG)
print(f"--- HiGHS, lower bound: {n} trailers allowed")
t = pd.DataFrame(runs("HiGHS", run_highs, n, mip_max_nodes=20_000), columns=cols)
print(t.to_string(index=False))
print(f"nodes: median {t.nodes.median():,.0f}, smallest {t.nodes.min():,}, largest {t.nodes.max():,}")
```

```text
--- HiGHS, lower bound: 15 trailers allowed
solver  trailers_allowed  seed  status  trailers  bound  nodes  lp_iterations
 HiGHS                15     0 Optimal        15   15.0   2352          64871
 HiGHS                15     1 Optimal        15   15.0   1694          39965
 HiGHS                15     2 Optimal        15   15.0   1586          32442
 HiGHS                15     3 Optimal        15   15.0      0          12494
 HiGHS                15     4 Optimal        15   15.0      0           9732
nodes: median 1,586, smallest 0, largest 2,352
```

The median falls from 4,166 nodes to 1,586, and **the worst case from 16,124 to 2,352**. Fewer trailers means fewer variables and fewer equivalent plans. The best speed-ups usually come from the model, not from solver settings: tighter formulations (Day 27), smaller index sets, better bounds.

## 28.3 Gurobi (optional)

```python
def run_gurobi(m, **params):
    """Write the model as an LP file, solve it with Gurobi on one thread with the given parameters
    (for example Seed=1, NodeLimit=20000, MIPGap=0.01), and return what the solver reports."""
    path = os.path.join(tempfile.mkdtemp(), "model.lp")
    m.write(path, io_options={"symbolic_solver_labels": True})
    env = gp.Env(params={"OutputFlag": 0})
    g = gp.read(path, env)
    g.Params.Threads = 1
    for k, v in params.items():
        g.setParam(k, v)
    g.optimize()
    status = {gp.GRB.OPTIMAL: "Optimal", gp.GRB.NODE_LIMIT: "Node limit", gp.GRB.INFEASIBLE: "Infeasible",
              gp.GRB.TIME_LIMIT: "Time limit"}.get(g.Status, str(g.Status))
    return {"status": status, "objective": g.ObjVal if g.SolCount else float("nan"), "bound": g.ObjBound,
            "gap_pct": 100 * g.MIPGap if g.SolCount else float("inf"), "nodes": int(g.NodeCount),
            "lp_iterations": int(g.IterCount)}
```

```python
# ---------- 28.3 Gurobi (optional) ----------
if have_gurobi():
    print("--- Gurobi, one thread")
    t = pd.DataFrame(runs("Gurobi", run_gurobi, 16, NodeLimit=20_000) + runs("Gurobi", run_gurobi, n, NodeLimit=20_000),
                     columns=cols)
    print(t.to_string(index=False))
    print(t.groupby("trailers_allowed").nodes.agg(["median", "min", "max"]).to_string())
else:
    print("gurobipy is not installed: skipping Gurobi (pip install gurobipy for the free size-limited licence)")
```

```text
--- Gurobi, one thread
solver  trailers_allowed  seed     status  trailers  bound  nodes  lp_iterations
Gurobi                16     0 Node limit        16   15.0  20000          72130
Gurobi                16     1    Optimal        15   15.0   5183          22059
Gurobi                16     2    Optimal        15   15.0   4790          26509
Gurobi                16     3 Node limit        16   15.0  20000          70036
Gurobi                16     4    Optimal        15   15.0  14361          69260
Gurobi                15     0    Optimal        15   15.0      1           1924
Gurobi                15     1    Optimal        15   15.0    159           3238
Gurobi                15     2    Optimal        15   15.0    261           3082
Gurobi                15     3    Optimal        15   15.0   1321          12831
Gurobi                15     4    Optimal        15   15.0   1036          10693
                   median   min    max
trailers_allowed
15                  261.0     1   1321
16                14361.0  4790  20000
```

(Run on one thread so the counts are repeatable; yours may differ.) Two lessons:

- **On the 16-trailer model, Gurobi is not faster here**: two seeds out of five stop at the 20,000-node limit without proving 15. Commercial solvers are usually faster on large, hard models, often by a wide margin. On any one model, measure.
- **On the bound-sized model, Gurobi's worst case is 1,321 nodes**, its median 261. The reformulation helped both solvers. A better model is worth more than a better solver.

> **Decision it supports:** which solver and model to put into a planning process, based on evidence instead of reputation.
>
> **Other decisions you can support the same way:** whether a commercial licence pays for itself · which settings a production system should use · how long planners must wait for a plan.

## 28.4 A speed checklist

When a MIP is slow, work down this list, measuring over several seeds at each step:

1. **Is it the model?** Tight links, the smallest valid big-M, no needless symmetry (Day 27). Smaller index sets (28.2).
2. **Bounds.** Give every variable the tightest bounds you know (capacities, demand caps).
3. **A starting plan.** If you have a good plan (yesterday's, a heuristic's), pass it to the solver as a warm start, so pruning starts at once.
4. **Stopping rules.** Agree a gap with the business (Day 26): proving the last 0.1% can take most of the time.
5. **Solver settings.** Heuristic effort, cut effort, threads. Change one at a time, test on several instances.
6. **Another solver.** Last, and measured, not assumed.

### The planted bug

`day28/bug_gap.py` asks for Christmas Tuesday "to within 0.1%". Run it.

```python
con = duckdb.connect("northbound.duckdb", read_only=True)
orders = con.sql("SELECT * FROM christmas_orders ORDER BY order_id").df().set_index("order_id")
r = run_highs(build_packing(orders, 26, 16, max_kg=16000), mip_rel_gap=0.1)
print(f"{r['status']}: {r['objective']:.0f} trailers")
```

<details><summary>What happens, and why</summary>

```text
Optimal: 16 trailers
```

16 trailers, "Optimal". `mip_rel_gap` is a **fraction**: 0.1 means 10%, and 16 trailers against a bound of 15 is a 6.25% gap, inside the tolerance. 0.1% is `mip_rel_gap=0.001`. Gurobi's `MIPGap` is also a fraction. Whenever a status says "Optimal", check the gap you asked for, and read the bound.

</details>

## Mini-project: Christmas week

Montreal ships on December 21, 22, 23 and 24, all under the 16,000 kg limit. For each day, compare first fit decreasing with the bound-sized model (`fewest_trailers`), using HiGHS (and Gurobi if you have it). How many trailers does the model save over the week?

<details><summary>Solution</summary>

```python
"""Day 28 mini-project: Christmas week. Four days of Montreal orders under the 16,000 kg limit:
first fit decreasing against the bound-sized model, with each solver.

Run from the repository root (after day06/load_tables.py):  python day28/mini_project_week.py
"""
import sys

import duckdb
import pandas as pd

for folder in ("day24", "day26", "day28"):
    sys.path.insert(0, folder)
from gurobi_run import have_gurobi, run_gurobi
from highs_run import run_highs
from loads import fewest_trailers, lower_bound
from packing import first_fit_decreasing

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
week = con.sql("SELECT * FROM christmas_week_orders ORDER BY ship_date, order_id").df()
capacity, cost = con.sql("SELECT pallet_positions, cost_per_day FROM truck_spec WHERE truck_type = 'trailer'").fetchall()[0]
MAX_KG = 16000
solvers = [("HiGHS", run_highs, {"mip_max_nodes": 20_000})]
if have_gurobi():
    solvers.append(("Gurobi", run_gurobi, {"NodeLimit": 20_000}))
rows = []
for day, orders in week.groupby("ship_date"):
    orders = orders.set_index("order_id")
    ffd = len(first_fit_decreasing(orders.pallets.to_dict(), capacity, orders.weight_kg.to_dict(), MAX_KG))
    row = [str(day)[:10], len(orders), lower_bound(orders, capacity, MAX_KG), ffd]
    for name, run, options in solvers:
        trailers, proven, nodes, tries = fewest_trailers(orders, capacity, MAX_KG, run, **options)
        row += [trailers, nodes]
    rows.append(row)
cols = ["ship_date", "orders", "bound", "ffd"] + [f"{name.lower()}_{c}" for name, *_ in solvers for c in ("trailers", "nodes")]
t = pd.DataFrame(rows, columns=cols)
print(t.to_string(index=False))
saved = (t.ffd - t.highs_trailers).sum()
print(f"the model saves {saved} trailers in the week: ${saved * cost:,}")
```

```text
 ship_date  orders  bound  ffd  highs_trailers  highs_nodes  gurobi_trailers  gurobi_nodes
2026-12-21      44     16   16              16            1               16             1
2026-12-22      48     15   16              15         2352               15             1
2026-12-23      52     18   18              18            1               18             1
2026-12-24      40     16   17              16         4464               16          2739
the model saves 2 trailers in the week: $1,300
```

- On Monday and Wednesday, first fit decreasing already meets the bound: nothing to gain, and the model proves it in one node.
- On **Tuesday and Thursday the model saves a trailer each**: $1,300 for the week.
- With the model sized by the bound, both solvers answer every day within a few thousand nodes, and the answer is **proven**: no plan uses fewer trailers.

*Note to transport planning:* "For Christmas week, plan loads with the bound-sized model. It saves 2 trailers ($1,300) over first fit decreasing and proves that no loading uses fewer. Allow a 20,000-node limit per day; it was never reached."

</details>

## Commit

```bash
git add .
git commit -m "Day 28: measuring and speeding up MIPs, HiGHS vs Gurobi"
git push
```

### Day 28 checklist

- [ ] I measure solver work in nodes and LP iterations over several seeds, not one run in seconds.
- [ ] I can make a model smaller using a bound (fix the number of trailers, then increase).
- [ ] I can run the same Pyomo model with HiGHS and, optionally, Gurobi.
- [ ] I know reformulating usually helps more than switching solvers.
- [ ] I know `mip_rel_gap` is a fraction, and check the gap behind every "Optimal".

---

# DAY 29 — Piecewise-Linear Costs and Discounts

Real costs are rarely one price per unit. Overtime gets dearer after the first hours; suppliers give discounts for volume; trucks cost the same whether full or half-empty. Today you model costs made of straight pieces, and learn the rule that decides whether a piecewise cost is easy: **costs that rise per unit need no binaries; costs that fall per unit do.**

| Time | Block |
|---|---|
| 0:00 – 0:45 | 29.1 Rising costs: overtime tiers |
| 0:45 – 1:45 | 29.2 Falling costs: volume discounts |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 29.3 Pyomo's Piecewise; 29.4 all-units discounts and trucks |
| 2:45 – 3:45 | Convex or not; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: should the bakery sign the offer? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 29.1 Rising costs: overtime tiers

`overtime_tiers`: the first 20 packing hours of overtime cost $60 an hour (the evening shift), the next 20 cost $95 (the night shift). Split overtime into one variable per tier, each up to its hours, and add them to the capacity. Create `day29/piecewise.py`:

```python
d = load_data(con)
tiers = con.sql("SELECT * FROM overtime_tiers ORDER BY resource, tier").df()
print(tiers.to_string(index=False))


def with_overtime(d):
    m = build_model(d)
    m.T = pyo.Set(initialize=[(r.resource, r.tier) for r in tiers.itertuples()], dimen=2)
    hours = {(r.resource, r.tier): r.hours for r in tiers.itertuples()}
    m.rate = {(r.resource, r.tier): r.cost_per_hour for r in tiers.itertuples()}
    m.ot = pyo.Var(m.T, bounds=lambda m, r, k: (0, hours[r, k]))             # overtime hours in tier k
    for r in m.R:
        extra = sum(m.ot[t] for t in m.T if t[0] == r)
        m.capacity_limit[r].set_value(sum(m.use[r, p] * m.x[p] for p in m.P) <= m.capacity[r] + extra)
    m.overtime_cost = pyo.Expression(expr=sum(m.rate[t] * m.ot[t] for t in m.T))
    m.total_profit.deactivate()
    m.net = pyo.Objective(expr=m.total_profit.expr - m.overtime_cost, sense=pyo.maximize)
    return m


for name, scale in [("a normal week", 1.0), ("a December week, demand +30%", 1.3)]:
    week = {**d, "cap": {p: c * scale for p, c in d["cap"].items()}}
    m = with_overtime(week)
    solver.solve(m)
    used = ", ".join(f"{r} tier {k} {pyo.value(m.ot[r, k]):.2f} h" for r, k in m.T if pyo.value(m.ot[r, k]) > 1e-6)
    print(f"{name}: net ${pyo.value(m.net):,.2f}, overtime ${pyo.value(m.overtime_cost):,.2f}: {used}")
```

```text
     resource  tier  hours  cost_per_hour
  baker_hours     1     40             45
  baker_hours     2     40             70
packing_hours     1     20             60
packing_hours     2     20             95
a normal week: net $108,837.29, overtime $1,059.88: packing_hours tier 1 17.66 h
a December week, demand +30%: net $113,520.31, overtime $2,146.67: packing_hours tier 1 20.00 h, packing_hours tier 2 9.96 h
```

There is no rule saying "use tier 1 before tier 2", and none is needed: tier 1 is cheaper, so the LP fills it first. In a normal week the bakery buys 17.66 evening hours (the same answer as Day 13's packing overtime); in December, all 20 evening hours and 9.96 night hours. This works because the cost **rises** with each extra hour: a **convex** cost, minimized (or a concave profit, maximized).

## 29.2 Falling costs: volume discounts

In December the bakery needs 55 t of bread flour a week. `flour_discounts` lists **incremental** discounts: each band of tonnes is paid at its own price (Prairie Hard Red: the first 10 t at $520, the next 10 at $495, the next 10 at $470).

```text
supplier_id  from_tonnes  to_tonnes  price_per_tonne
     FL-DUR            0         10              590
     FL-DUR           10         30              540
     FL-OSW            0         15              410
     FL-OSW           15         30              395
     FL-PHR            0         10              520
     FL-PHR           10         20              495
     FL-PHR           20         30              470
     FL-QCO            0          8              640
```

Now the later bands are *cheaper*. Split the tonnes into bands as in 29.1, and the LP would fill the cheap $470 band first, without buying the first 20 t at full price. A falling cost per unit is **concave**: to force the bands to fill in order, each band gets a binary, $$r_{sk} = 1$$ if band $$k$$ is reached:

$$
y_{sk} \le w_{sk}\, r_{sk}, \qquad y_{s,k} \ge w_{s,k}\, r_{s,k+1}
$$

where $$y_{sk}$$ is the tonnes bought in band $$k$$ and $$w_{sk}$$ its width. Band $$k+1$$ can only be used once band $$k$$ is full. Create `day29/flour_costs.py`, which builds Day 5's blend with several price rules:

```python
"""Day 29: buying a week's bread flour when prices fall with volume.

Import it:  from flour_costs import load_flour, build_flour
"""
import pyomo.environ as pyo

NEED = 40   # tonnes of bread flour a week (Day 5)


def load_flour(con):
    flour = con.sql("SELECT * FROM flour_suppliers ORDER BY supplier_id").df().set_index("supplier_id")
    spec = con.sql("SELECT * FROM flour_spec").df().set_index("property")
    bands = con.sql("SELECT * FROM flour_discounts ORDER BY supplier_id, from_tonnes").df()
    offer = con.sql("SELECT * FROM flour_all_units_offer ORDER BY from_tonnes").df()
    delivery = con.sql("SELECT * FROM flour_delivery").df().set_index("supplier_id")
    return flour, spec, bands, offer, delivery


def build_flour(flour, spec, bands, prices="list", offer=None, delivery=None, need=NEED):
    """t[s] = tonnes from supplier s, blended to the spec (Day 5). prices:
    "list"         one price per tonne (Day 5)
    "incremental"  each band of tonnes at its own price; binaries make the bands fill in order
    "piecewise"    the same cost curve with Pyomo's Piecewise component
    "bands_lp"     the bands without binaries (wrong for discounts: see the planted bug)
    offer: an all-units offer (one supplier): every tonne at the price of the band the total falls in.
    delivery: whole trucks per supplier, each with a cost."""
    m = pyo.ConcreteModel(name="flour_purchase")
    m.S = pyo.Set(initialize=flour.index.tolist())
    m.t = pyo.Var(m.S, bounds=lambda m, s: (0, flour.max_tonnes_per_week[s]))
    m.need = pyo.Constraint(expr=sum(m.t[s] for s in m.S) == need)
    m.spec = pyo.ConstraintList()
    for prop, row in spec.iterrows():
        blend = sum(flour.loc[s, prop] * m.t[s] for s in m.S)
        if row.min_value == row.min_value:          # not NaN
            m.spec.add(blend >= row.min_value * need)
        if row.max_value == row.max_value:
            m.spec.add(blend <= row.max_value * need)
    offered = set() if offer is None else set(offer.supplier_id)
    cost = {}
    if prices == "list":
        for s in m.S:
            cost[s] = flour.price_per_tonne[s] * m.t[s]
    elif prices in ("incremental", "bands_lp"):
        idx = [(r.supplier_id, k) for k, r in bands.iterrows()]
        m.B = pyo.Set(initialize=idx, dimen=2)
        width = {(s, k): bands.to_tonnes[k] - bands.from_tonnes[k] for s, k in idx}
        m.y = pyo.Var(m.B, bounds=lambda m, s, k: (0, width[s, k]))          # tonnes bought in band k
        m.split = pyo.Constraint(m.S, rule=lambda m, s: m.t[s] == sum(m.y[b] for b in m.B if b[0] == s))
        if prices == "incremental":
            m.reach = pyo.Var(m.B, within=pyo.Binary)                        # reach[s, k] = 1 if band k is used
            m.only_if_reached = pyo.Constraint(m.B, rule=lambda m, s, k: m.y[s, k] <= width[s, k] * m.reach[s, k])
            order = [(b, nxt) for b, nxt in zip(idx, idx[1:]) if b[0] == nxt[0]]
            m.fill_in_order = pyo.Constraint(range(len(order)),             # band k+1 only once band k is full
                                             rule=lambda m, i: m.y[order[i][0]] >= width[order[i][0]] * m.reach[order[i][1]])
        for s in m.S:
            cost[s] = sum(bands.price_per_tonne[k] * m.y[s, k] for (s2, k) in idx if s2 == s)
    elif prices == "piecewise":
        m.c = pyo.Var(m.S)
        m.curve = pyo.Block(m.S)
        for s in m.S:
            b = bands[bands.supplier_id == s]
            points = [0.0] + [float(x) for x in b.to_tonnes]
            values = [0.0]
            for _, r in b.iterrows():
                values.append(values[-1] + r.price_per_tonne * (r.to_tonnes - r.from_tonnes))
            m.curve[s].pw = pyo.Piecewise(m.c[s], m.t[s], pw_pts=points, f_rule=values, pw_constr_type="EQ", pw_repn="INC")
            cost[s] = m.c[s]
    for s in offered:                                 # all-units offer: replaces that supplier's prices
        o = offer.reset_index(drop=True)
        m.tier = pyo.Var(o.index.tolist(), within=pyo.Binary)
        m.in_tier = pyo.Var(o.index.tolist(), within=pyo.NonNegativeReals)
        m.one_tier = pyo.Constraint(expr=sum(m.tier[k] for k in o.index) == 1)
        m.tier_lo = pyo.Constraint(o.index.tolist(), rule=lambda m, k: m.in_tier[k] >= o.from_tonnes[k] * m.tier[k])
        m.tier_hi = pyo.Constraint(o.index.tolist(), rule=lambda m, k: m.in_tier[k] <= o.to_tonnes[k] * m.tier[k])
        m.tier_total = pyo.Constraint(expr=m.t[s] == sum(m.in_tier[k] for k in o.index))
        cost[s] = sum(o.price_per_tonne[k] * m.in_tier[k] for k in o.index)
    m.flour_cost = pyo.Expression(expr=sum(cost[s] for s in m.S))
    if delivery is not None:
        m.trucks = pyo.Var(m.S, within=pyo.NonNegativeIntegers)
        m.full = pyo.Constraint(m.S, rule=lambda m, s: m.t[s] <= delivery.tonnes_per_truck[s] * m.trucks[s])
        m.delivery = pyo.Expression(expr=sum(delivery.cost_per_truck[s] * m.trucks[s] for s in m.S))
    else:
        m.delivery = pyo.Expression(expr=0)
    m.cost = pyo.Objective(expr=m.flour_cost + m.delivery)
    return m


def describe(m):
    tonnes = ", ".join(f"{s} {pyo.value(m.t[s]):.2f} t" for s in m.S if pyo.value(m.t[s]) > 1e-6)
    extra = f" + delivery ${pyo.value(m.delivery):,.2f}" if pyo.value(m.delivery) else ""
    return f"${pyo.value(m.cost):,.2f} (flour ${pyo.value(m.flour_cost):,.2f}{extra}): {tonnes}"
```

```python
# ---------- 29.2 Falling costs: incremental discounts ----------
flour, spec, bands, offer, delivery = load_flour(con)
NEED = 55    # tonnes a week in December
print(bands.to_string(index=False))
for prices in ("list", "incremental"):
    f = build_flour(flour, spec, bands, prices=prices, need=NEED)
    solver.solve(f)
    print(f"{prices:<12}", describe(f))
```

```text
list         $25,931.82 (flour $25,931.82): FL-DUR 0.45 t, FL-OSW 24.55 t, FL-PHR 30.00 t
incremental  $25,038.64 (flour $25,038.64): FL-DUR 0.45 t, FL-OSW 24.55 t, FL-PHR 30.00 t
```

With discounts the same tonnes cost **$893 a week less** ($25,038.64 instead of $25,931.82). The plan doesn't change: the flour spec, not the price, decides the blend here.

> **Decision it supports:** how much to buy from each supplier when price depends on volume.
>
> **Other decisions you can support the same way:** freight with rates by weight band · energy contracts with tiered tariffs · advertising bought in volume.

## 29.3 Pyomo's Piecewise

Pyomo builds these formulations for you from breakpoints: `Piecewise(cost, tonnes, pw_pts=..., f_rule=...)`, with a choice of formulation (`pw_repn`). `"INC"` is the incremental formulation of 29.2; `"SOS2"` uses special ordered sets, which some solvers handle natively.

```python
# ---------- 29.3 The same curve with Pyomo's Piecewise ----------
f = build_flour(flour, spec, bands, prices="piecewise", need=NEED)
solver.solve(f)
print(f"{'piecewise':<12}", describe(f))
```

```text
piecewise    $25,038.64 (flour $25,038.64): FL-DUR 0.45 t, FL-OSW 24.55 t, FL-PHR 30.00 t
```

Same answer. Write it by hand once to understand it; use `Piecewise` afterwards.

## 29.4 All-units discounts and delivery trucks

Ontario Soft White offers an **all-units** deal: the price of *every* tonne depends on the week's total, $392 a tonne from 15 t, $372 from 25 t. The cost jumps *down* at 15 t and 25 t: it is not even continuous. The model picks one tier with a binary, and the tonnes must fall inside it (`tier`, `in_tier` in `flour_costs.py`). Deliveries come in whole trucks (`flour_delivery`): a general integer per supplier, like Day 25's linehaul.

```python
# ---------- 29.4 All-units discounts and delivery trucks ----------
print("--- Ontario Soft White's all-units offer")
print(offer.to_string(index=False))
print("--- delivery trucks")
print(delivery.to_string())
for name, kw in [("all-units offer", dict(offer=offer)), ("delivery trucks", dict(delivery=delivery)),
                 ("offer and trucks", dict(offer=offer, delivery=delivery))]:
    f = build_flour(flour, spec, bands, prices="incremental", need=NEED, **kw)
    solver.solve(f)
    print(f"{name:<17}", describe(f))
```

```text
--- Ontario Soft White's all-units offer
supplier_id  from_tonnes  to_tonnes  price_per_tonne
     FL-OSW            0         15              410
     FL-OSW           15         25              392
     FL-OSW           25         30              372
--- delivery trucks
             tonnes_per_truck  cost_per_truck
supplier_id
FL-PHR                     12             380
FL-OSW                     12             220
FL-QCO                      8             260
FL-DUR                     12             450
all-units offer   $24,450.00 (flour $24,450.00): FL-DUR 2.50 t, FL-OSW 25.00 t, FL-PHR 27.50 t
delivery trucks   $27,035.00 (flour $25,195.00 + delivery $1,840.00): FL-OSW 24.00 t, FL-PHR 30.00 t, FL-QCO 1.00 t
offer and trucks  $26,601.27 (flour $24,731.27 + delivery $1,870.00): FL-DUR 5.36 t, FL-OSW 25.64 t, FL-PHR 24.00 t
```

- **The offer changes the plan**: Ontario Soft White goes to exactly 25.00 t to reach the $372 tier, and durum makes up the protein. $24,450 a week, $589 less.
- **Trucks change the plan too**: 24 t of soft white (two full trucks) instead of 24.55, and 1 t from Quebec Organic instead of 0.45 t of durum: one Quebec truck ($260) costs less than one durum truck ($450).
- With both, the plan fills two Prairie trucks exactly (24.00 t) and puts 25.64 t on soft white.

### Math corner (optional): when does a piecewise cost need binaries?

A function is **convex** if the segment between any two points of its graph lies on or above the graph: slopes rise. Minimizing a convex piecewise-linear cost is an LP, because the cheaper pieces are always used first (29.1). Concave costs (falling slopes: discounts, economies of scale) and discontinuous ones (all-units, fixed charges) make the feasible set non-convex, and linear programming can only describe convex sets. They need binaries (or SOS2 constraints), and the solver needs branch and bound. The same rule works for maximization with profits: concave profit is easy, convex profit isn't. Checking the direction of the slopes before modelling tells you whether a cost is an LP or a MIP.

### The planted bug

`day29/bug_discount_lp.py` writes the discount bands without binaries: "the solver will fill them in order". Run it.

```python
con = duckdb.connect("northbound.duckdb", read_only=True)
flour, spec, bands, offer, delivery = load_flour(con)
m = build_flour(flour, spec, bands, prices="bands_lp", need=55)
pyo.SolverFactory("highs").solve(m)
print("model says:", describe(m))
print("bands used:", {f"{s} {bands.from_tonnes[k]}-{bands.to_tonnes[k]} t": round(pyo.value(m.y[s, k]), 2)
                     for s, k in m.B if pyo.value(m.y[s, k]) > 1e-6})
check = build_flour(flour, spec, bands, prices="incremental", need=55)      # what the same tonnes really cost
for s in check.S:
    check.t[s].fix(pyo.value(m.t[s]))
pyo.SolverFactory("highs").solve(check)
print("the same tonnes at the supplier's real prices:", describe(check))
```

<details><summary>What happens, and why</summary>

```text
model says: $24,897.73 (flour $24,897.73): FL-DUR 8.64 t, FL-OSW 26.36 t, FL-PHR 20.00 t
bands used: {'FL-DUR 10-30 t': 8.64, 'FL-OSW 0-15 t': 11.36, 'FL-OSW 15-30 t': 15.0, 'FL-PHR 10-20 t': 10.0, 'FL-PHR 20-30 t': 10.0}
the same tonnes at the supplier's real prices: $25,884.09 (flour $25,884.09): FL-DUR 8.64 t, FL-OSW 26.36 t, FL-PHR 20.00 t
```

The model claims $24,897.73, cheaper than the real best plan ($25,038.64). Look at the bands: 15 t of soft white at the $395 band and only 11.36 t at the $410 band; Prairie's $495 and $470 bands without its first, $520 band. The supplier would charge **$25,884.09** for those tonnes: $845 a week more than the real best plan. Discounts are concave, and an LP fills cheap bands first. The model's cost is a fiction, and so is its plan. Check: the result is *below* a correctly priced plan, which can't happen.

</details>

## Mini-project: should the bakery sign the offer?

The offer is for a year. Compare the weekly cost, with delivery trucks, with and without the offer, for needs of 40 t (a normal week) to 60 t (the busiest December week). How much does it save, and when does it change what the bakery buys?

<details><summary>Solution</summary>

```python
"""Day 29 mini-project: should the bakery sign Ontario Soft White's all-units offer?
Weekly needs from 40 t (a normal week) to 60 t (the busiest December week), with delivery trucks.

Run from the repository root (after day06/load_tables.py):  python day29/mini_project_offer.py
"""
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day29")
from flour_costs import build_flour, load_flour

pd.set_option("display.width", 140)
con = duckdb.connect("northbound.duckdb", read_only=True)
flour, spec, bands, offer, delivery = load_flour(con)
rows = []
for need in (40, 45, 50, 55, 60):
    out = [need]
    for kw in ({}, {"offer": offer}):
        m = build_flour(flour, spec, bands, prices="incremental", delivery=delivery, need=need, **kw)
        pyo.SolverFactory("highs").solve(m)
        out += [round(pyo.value(m.cost), 2), round(pyo.value(m.t["FL-OSW"]), 2)]
    rows.append(out + [round(out[1] - out[3], 2)])
t = pd.DataFrame(rows, columns=["need_t", "cost_now", "osw_t_now", "cost_offer", "osw_t_offer", "saving"])
print(t.to_string(index=False))
```

```text
 need_t  cost_now  osw_t_now  cost_offer  osw_t_offer  saving
     40  19641.67      17.78    19363.33        17.78  278.34
     45  22205.00      20.00    21920.00        20.00  285.00
     50  24388.33      22.22    24096.67        22.22  291.66
     55  27035.00      24.00    26601.27        25.64  433.73
     60  29706.82      27.27    28854.55        27.27  852.27
```

- In a normal week (40 t), the offer saves **$278**: the same 17.78 t of soft white, at $392 instead of the incremental prices.
- The saving grows with volume, and at 55 t it **changes the plan** (25.64 t of soft white, to reach the $372 tier): $434. At 60 t: $852.
- An all-units offer can't cost more than the incremental prices here: in every band, the offer's price is at or below the incremental one. So the decision is whether the contract's other terms (exclusivity, minimum volumes) are worth about $280 to $850 a week.

*Memo to purchasing:* "Sign Ontario Soft White's offer if it carries no minimum volume: it saves $14,000 to $15,000 a year at normal volumes, more in December. The saving rises in weeks above 50 t, when buying 25 t from them reaches the lowest tier."

</details>

## Commit

```bash
git add .
git commit -m "Day 29: piecewise-linear costs and discounts"
git push
```

### Day 29 checklist

- [ ] I model rising piecewise costs with one variable per piece, without binaries.
- [ ] I model falling (incremental) discounts with binaries that fill bands in order.
- [ ] I can model an all-units discount and whole-truck delivery costs.
- [ ] I can use Pyomo's `Piecewise`.
- [ ] I check the slopes (convex or concave) before choosing LP or MIP.
- [ ] I know a cost below a correctly priced plan is a sign of a broken model.

---

# DAY 30 — Capstone: Redesigning the DC Network

The module's capstone puts Days 21–29 together. Northbound grows about 3% a year. The board asks for a five-year plan for the distribution network: which DCs to run, which expansions to build and when, whether to open new sites, and what it costs, with proof that the plan is the best one (or how close it is). The deliverable is a plan, a cost case and a memo.

| Time | Block |
|---|---|
| 0:00 – 0:30 | 30.1 The brief, the model and a known-answer check |
| 0:30 – 1:15 | 30.2 Doing nothing |
| 1:15 – 1:45 | 30.3 The five-year plan |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 30.4 The gap report |
| 2:45 – 3:45 | 30.5 Scenarios |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | 30.6 The memo |
| 4:45 – 5:00 | Commit, tag, checklist |

## 30.1 The brief

- **Demand:** each store's pallets (Day 14) grow 3% a year from 2027 to 2031.
- **Sites:** Day 23's three DCs and seven candidates. An existing DC can close (one-time cost $1.5 million: severance and lease exit); a new site can open (one-time $3 million: fit-out and moving) and pays its weekly fixed cost from then on.
- **Expansions** (`network_options`): Ottawa +300 pallets a week ($1.4 million once, $2,500 a week), Montreal +300 ($1.8 million, $3,000), Mississauga +400 ($2 million, $3,200).
- **Rules:** as on Day 23: single sourcing each year, French labels, six hours on the road.
- **Cost case:** present value at the start of 2027, discounted at 8% a year; 52 weeks a year.

The model is Day 23's location model with a year index, plus Day 21's yes/no logic for timing: an existing DC that closes stays closed, a new site that opens stays open, an expansion stays built, and one-time costs are paid in the year something changes. Create `day30/redesign.py`:

```python
"""Day 30 capstone: redesign the DC network for the next five years.
Which sites run each year, which expansions to build and when, which site serves each store each year.

Import it:  from redesign import load_redesign, build_redesign, yearly_table
"""
import pyomo.environ as pyo
import pandas as pd

MAX_MINUTES = 360


def load_redesign(con):
    sites = con.sql("SELECT * FROM candidate_sites ORDER BY site_id").df().set_index("site_id")
    stores = con.sql("""SELECT s.store_id, s.city, s.province, l.pallets_per_week
                        FROM stores s JOIN store_logistics l USING (store_id) ORDER BY s.store_id""").df().set_index("store_id")
    lanes = con.sql("SELECT * FROM site_store_lanes").df().set_index(["site_id", "store_id"])
    options = con.sql("SELECT * FROM network_options ORDER BY option_id").df().set_index("option_id")
    settings = {k: v for k, v in con.sql("SELECT setting, value FROM network_settings").fetchall()}
    return sites, stores, lanes, options, settings


def build_redesign(sites, stores, lanes, options, settings, years=None, growth=None, rate=None,
                   allowed_options=None, allow_new_sites=True, allow_closing=True):
    """Minimize the present value of five years of network costs (52 weeks a year, discounted to the start of 2027)."""
    years = int(years or settings["years"])
    growth = settings["growth_per_year"] if growth is None else growth
    rate = settings["discount_rate"] if rate is None else rate
    weeks = settings["weeks_per_year"]
    m = pyo.ConcreteModel(name="network_redesign")
    m.I = pyo.Set(initialize=sites.index.tolist())
    m.J = pyo.Set(initialize=stores.index.tolist())
    m.T = pyo.RangeSet(1, years)
    m.E = pyo.Set(initialize=[e for e in options.index if allowed_options is None or e in allowed_options])
    existing = [i for i in m.I if sites.kind[i] == "existing DC"]
    pv = {t: 1 / (1 + rate) ** (t - 1) for t in m.T}                     # costs of year t valued at the start of year 1
    demand = {(j, t): stores.pallets_per_week[j] * (1 + growth) ** (t - 1) for j in m.J for t in m.T}

    m.open = pyo.Var(m.I, m.T, within=pyo.Binary)
    m.change = pyo.Var(m.I, m.T, within=pyo.NonNegativeReals)            # 1 in the year a site opens or closes
    m.built = pyo.Var(m.E, m.T, within=pyo.Binary)                       # expansion in place in year t
    m.build = pyo.Var(m.E, m.T, within=pyo.NonNegativeReals)             # 1 in the year it is built
    m.serve = pyo.Var(m.I, m.J, m.T, within=pyo.Binary)

    def before(m, i, t):
        return (1 if i in existing else 0) if t == 1 else m.open[i, t - 1]
    m.new_stay_open = pyo.Constraint([(i, t) for i in m.I for t in m.T if i not in existing],
                                     rule=lambda m, i, t: m.open[i, t] >= before(m, i, t))
    m.closed_stay_closed = pyo.Constraint([(i, t) for i in existing for t in m.T],
                                          rule=lambda m, i, t: m.open[i, t] <= before(m, i, t))
    m.count_change = pyo.Constraint(m.I, m.T, rule=lambda m, i, t: m.change[i, t] >= (m.open[i, t] - before(m, i, t)) *
                                    (1 if i not in existing else -1))
    if not allow_new_sites:
        for i in m.I:
            if i not in existing:
                for t in m.T:
                    m.open[i, t].fix(0)
    if not allow_closing:
        for i in existing:
            for t in m.T:
                m.open[i, t].fix(1)
    m.expansion_stays = pyo.Constraint(m.E, m.T, rule=lambda m, e, t: pyo.Constraint.Skip if t == 1
                                       else m.built[e, t] >= m.built[e, t - 1])
    m.count_build = pyo.Constraint(m.E, m.T, rule=lambda m, e, t: m.build[e, t] >= m.built[e, t] - (0 if t == 1 else m.built[e, t - 1]))
    m.expand_open_site = pyo.Constraint(m.E, m.T, rule=lambda m, e, t: m.built[e, t] <= m.open[options.site_id[e], t])

    m.every_store = pyo.Constraint(m.J, m.T, rule=lambda m, j, t: sum(m.serve[i, j, t] for i in m.I) == 1)
    m.only_if_open = pyo.Constraint(m.I, m.J, m.T, rule=lambda m, i, j, t: m.serve[i, j, t] <= m.open[i, t])
    m.capacity = pyo.Constraint(m.I, m.T, rule=lambda m, i, t:
                                sum(demand[j, t] * m.serve[i, j, t] for j in m.J)
                                <= sites.capacity_pallets[i] * m.open[i, t]
                                + sum(options.extra_pallets[e] * m.built[e, t] for e in m.E if options.site_id[e] == i))
    for i in m.I:
        for j in m.J:
            if (stores.province[j] == "QC" and not sites.ships_french_labels[i]) or lanes.minutes[i, j] > MAX_MINUTES:
                for t in m.T:
                    m.serve[i, j, t].fix(0)

    m.weekly = pyo.Expression(m.T, rule=lambda m, t:
                              sum(sites.fixed_cost_per_week[i] * m.open[i, t] for i in m.I)
                              + sum(options.weekly_cost[e] * m.built[e, t] for e in m.E)
                              + sum(lanes.cost_per_pallet[i, j] * demand[j, t] * m.serve[i, j, t] for i in m.I for j in m.J))
    m.one_time = pyo.Expression(m.T, rule=lambda m, t:
                                sum((settings["close_cost_existing_dc"] if i in existing else settings["open_cost_new_site"])
                                    * m.change[i, t] for i in m.I)
                                + sum(options.one_time_cost[e] * m.build[e, t] for e in m.E))
    m.pv = pyo.Objective(expr=sum(pv[t] * (weeks * m.weekly[t] + m.one_time[t]) for t in m.T))
    m.demand = demand
    return m


def yearly_table(m, sites, options):
    rows = []
    for t in m.T:
        running = [i for i in m.I if pyo.value(m.open[i, t]) > 0.5]
        built = [e for e in m.E if pyo.value(m.built[e, t]) > 0.5]
        pallets = sum(m.demand[j, t] for j in m.J)
        cap = sum(sites.capacity_pallets[i] for i in running) + sum(options.extra_pallets[e] for e in built)
        rows.append((2026 + t, round(pallets), cap, ", ".join(i.split("-")[1] for i in running), ", ".join(built) or "-",
                     round(pyo.value(m.weekly[t]), 2), round(pyo.value(m.one_time[t])) + 0))
    return pd.DataFrame(rows, columns=["year", "pallets_a_week", "capacity", "sites", "expansions", "weekly_cost", "one_time"])
```

Every year's costs are multiplied by $$1 / 1.08^{t-1}$$: a dollar spent in 2031 is worth 74 cents today. Then `day30/network_plan.py`, starting with a **known-answer check**: one year, no growth, nothing to build must give Day 23's network at 52 times its weekly cost.

```python
for folder in ("day23", "day26", "day30"):
    sys.path.insert(0, folder)
from facility import build_location
from highs_run import run_highs
from redesign import build_redesign, load_redesign, yearly_table

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
sites, stores, lanes, options, settings = load_redesign(con)
solver = pyo.SolverFactory("highs")
print(options.to_string())
print(settings)

# ---------- 30.1 Check: one year, no growth = Day 23 ----------
m = build_redesign(sites, stores, lanes, options, settings, years=1, growth=0)
solver.solve(m)
print(f"one year, no growth: ${pyo.value(m.pv):,.2f} = 52 x ${pyo.value(m.pv) / 52:,.2f} a week (Day 23: $92,492.76)")
```

```text
          site_id    kind  extra_pallets  one_time_cost  weekly_cost
option_id
EXP-MIS    DC-MIS  expand            400        2000000         3200
EXP-MTL    DC-MTL  expand            300        1800000         3000
EXP-OTT    DC-OTT  expand            300        1400000         2500
{'first_year': 2027.0, 'years': 5.0, 'growth_per_year': 0.03, 'discount_rate': 0.08, 'weeks_per_year': 52.0, 'open_cost_new_site': 3000000.0, 'close_cost_existing_dc': 1500000.0}
one year, no growth: $4,809,623.52 = 52 x $92,492.76 a week (Day 23: $92,492.76)
```

The model reproduces Day 23 to the cent. Only now is it worth trusting with five years.

## 30.2 Doing nothing

```python
# ---------- 30.2 Doing nothing ----------
existing = sites[sites.kind == "existing DC"]
for t in range(1, int(settings["years"]) + 1):
    grown = stores.assign(pallets_per_week=stores.pallets_per_week * (1 + settings["growth_per_year"]) ** (t - 1))
    d = build_location(existing, grown, lanes)
    r = solver.solve(d, load_solutions=False)
    ok = r.solver.termination_condition == pyo.TerminationCondition.optimal
    print(f"{2026 + t}: {grown.pallets_per_week.sum():,.0f} pallets a week on 3,800 of capacity: "
          f"{'fits' if ok else 'NO PLAN: some store cannot be served'}")
```

```text
2027: 3,545 pallets a week on 3,800 of capacity: fits
2028: 3,651 pallets a week on 3,800 of capacity: fits
2029: 3,761 pallets a week on 3,800 of capacity: fits
2030: 3,874 pallets a week on 3,800 of capacity: NO PLAN: some store cannot be served
2031: 3,990 pallets a week on 3,800 of capacity: NO PLAN: some store cannot be served
```

**By 2030, today's network can't serve every store**: 3,874 pallets a week need more than 3,800 of capacity, and single sourcing needs some room to spare. Doing nothing is not an option; the question is what to do, and when.

## 30.3 The five-year plan

```python
# ---------- 30.3 The five-year plan ----------
m = build_redesign(sites, stores, lanes, options, settings)
solver.solve(m)
plan = yearly_table(m, sites, options)
print(f"five-year cost, present value at the start of 2027: ${pyo.value(m.pv):,.2f}")
print(plan.to_string(index=False))
moves = sum(pyo.value(m.serve[i, j, t]) > 0.5 and pyo.value(m.serve[i, j, t - 1]) < 0.5
            for i in m.I for j in m.J for t in m.T if t > 1)
print(f"stores that change DC from one year to the next: {moves}")
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, i in zip(axes, ["DC-MIS", "DC-OTT", "DC-MTL"]):
    years = [2026 + t for t in m.T]
    load = [sum(m.demand[j, t] * pyo.value(m.serve[i, j, t]) for j in m.J) for t in m.T]
    cap = [sites.capacity_pallets[i] + sum(options.extra_pallets[e] * pyo.value(m.built[e, t])
                                           for e in m.E if options.site_id[e] == i) for t in m.T]
    ax.bar(years, load, color="#4f7bd9", label="pallets a week")
    ax.plot(years, cap, "--", color="#e08a2c", lw=2, label="capacity")
    ax.set_title(f"{i} {sites.place[i]}")
    ax.set_xticks(years)
    ax.set_ylim(0, 1800)
axes[0].set_ylabel("pallets a week")
axes[0].legend(loc="lower left")
fig.suptitle("Five-year plan: load and capacity by DC")
fig.tight_layout()
fig.savefig("figures/day30_network_plan.png", dpi=110)
print("wrote figures/day30_network_plan.png")
```

```text
five-year cost, present value at the start of 2027: $22,194,811.21
 year  pallets_a_week  capacity         sites expansions  weekly_cost  one_time
 2027            3545      4100 MIS, MTL, OTT    EXP-OTT     88694.71   1400000
 2028            3651      4100 MIS, MTL, OTT    EXP-OTT     89750.55         0
 2029            3761      4100 MIS, MTL, OTT    EXP-OTT     92711.44         0
 2030            3874      4100 MIS, MTL, OTT    EXP-OTT     94764.76         0
 2031            3990      4100 MIS, MTL, OTT    EXP-OTT     99847.25         0
stores that change DC from one year to the next: 9
wrote figures/day30_network_plan.png
```

![Load and capacity by DC](figures/day30_network_plan.png)

- **Expand Ottawa in 2027**, and change nothing else. Present value of five years: **$22.19 million**.
- The expansion pays from the first year: the weekly cost falls from $92,492.76 to $88,694.71 *including* the expansion's $2,500 a week, because Ottawa-area stores no longer travel from Mississauga (the problem found on Day 14).
- **No new site and no closure** pays within five years at $3 million to open and $1.5 million to close.
- By 2031 the three DCs are almost full: **the next decision is due around 2030** (see the scenarios).
- **Nine store moves** between DCs over the five years (as stores grow, the model re-balances). Each move costs effort the model doesn't see; Module 8 comes back to plan stability.

## 30.4 The gap report

```python
# ---------- 30.4 Gap report ----------
for opts in ({}, {"mip_rel_gap": 0.01}):
    h = run_highs(build_redesign(sites, stores, lanes, options, settings), **opts)
    print(f"HiGHS {opts or 'defaults'}: {h['status']}, ${h['objective']:,.2f}, bound ${h['bound']:,.2f}, "
          f"gap {h['gap_pct']:.3f}%, {h['nodes']} nodes")
```

```text
HiGHS defaults: Optimal, $22,194,811.21, bound $22,194,811.21, gap 0.000%, 1 nodes
HiGHS {'mip_rel_gap': 0.01}: Optimal, $22,211,959.08, bound $22,052,924.17, gap 0.716%, 1 nodes
```

HiGHS proves the plan optimal (gap 0%). With a 1% gap, it stops sooner with a plan $17,148 more expensive (0.08%), and can only promise that no plan costs less than $22.05 million. For a five-year, $22 million decision, wait for the proof: it takes seconds here. On a bigger model (more years, more candidate sites, weekly instead of yearly periods) it may not, and then the report must say "this plan costs at most X% more than the best possible".

## 30.5 Scenarios

```python
# ---------- 30.5 Scenarios ----------
rows = []
for name, kw in [("plan (growth 3%, rate 8%)", {}), ("growth 1%", {"growth": 0.01}), ("growth 5%", {"growth": 0.05}),
                 ("discount rate 12%", {"rate": 0.12}), ("no Ottawa expansion", {"allowed_options": ["EXP-MTL", "EXP-MIS"]}),
                 ("new sites free to open", {}), ]:
    s = settings if name != "new sites free to open" else {**settings, "open_cost_new_site": 0}
    m = build_redesign(sites, stores, lanes, options, s, **kw)
    solver.solve(m)
    t = yearly_table(m, sites, options)
    first = {e: int(t.year[t.expansions.str.contains(e)].min()) for e in options.index if t.expansions.str.contains(e).any()}
    rows.append((name, round(pyo.value(m.pv), 2), ", ".join(f"{e} {y}" for e, y in first.items()) or "-",
                 t.sites.iloc[-1]))
print(pd.DataFrame(rows, columns=["scenario", "present_value", "expansions", "sites_in_2031"]).to_string(index=False))
```

```text
                 scenario  present_value                 expansions      sites_in_2031
plan (growth 3%, rate 8%)    22194811.21               EXP-OTT 2027      MIS, MTL, OTT
                growth 1%    20996293.73                          -      MIS, MTL, OTT
                growth 5%    23987941.78 EXP-MTL 2030, EXP-OTT 2027      MIS, MTL, OTT
        discount rate 12%    20829515.86               EXP-OTT 2027      MIS, MTL, OTT
      no Ottawa expansion    23026170.97               EXP-MTL 2029      MIS, MTL, OTT
   new sites free to open    21661271.52                          - MIS, MTL, OTT, QUE
```

- **Growth decides the plan.** At 1% a year, nothing needs building before 2031. At 5%, Ottawa in 2027 and Montreal in 2030.
- **The discount rate doesn't**: at 12% the plan is the same.
- **Without the Ottawa expansion**, the best plan expands Montreal in 2029 and costs **$831,360 more** over five years: Ottawa's expansion is worth that much.
- **If new sites cost nothing to open**, Quebec City opens. It is $3 million of one-time costs that keeps it out: a new DC needs a longer horizon than five years to pay.

## 30.6 The memo

<details><summary>A model memo</summary>

*To: the Board. Re: the DC network, 2027–2031.*

**Recommendation.** Expand the Ottawa DC by 300 pallets a week in 2027 ($1.4 million, then $2,500 a week). Keep the three DCs; open no new site in this period.

**Why.** At 3% growth a year, today's network can't serve every store by 2030. Expanding Ottawa solves that and lowers weekly costs at once, because Ottawa-area stores now served from Mississauga can be served locally: weekly costs fall by about $3,800 a week in 2027 after paying for the expansion. Five-year cost, present value: **$22.2 million**. Without the Ottawa expansion, the best alternative (expanding Montreal in 2029) costs $0.8 million more.

**How sure.** The plan is proven optimal for the stated costs and growth (gap 0%). It doesn't change with the discount rate (8% or 12%). It depends on growth: at 1% a year nothing is needed yet; at 5%, Montreal must also expand in 2030.

**What to watch.** By 2031 all three DCs are nearly full. We recommend reviewing the network in 2029 with actual growth, when a new site (Quebec City is the most promising: Days 23 and 30) or a Montreal expansion will be the next step.

**Not in the model.** Construction risk and timing within the year; the cost of moving stores between DCs (nine moves in five years); uncertainty in growth beyond the three scenarios (Module 6 treats it properly).

</details>

## Module 3 wrap-up

| Day | Idea | Northbound decision |
|---|---|---|
| 21 | Binary variables, logic as constraints | Which bakery investments to buy |
| 22 | Fixed costs, big-M, minimum runs | Which products to make each week |
| 23 | Facility location | Is today's DC network right? |
| 24 | Knapsack, bin packing, cutting stock | Rush loads, trailers, cake bars |
| 25 | Integer trucks, location in a flow network | Which cross-docks to run |
| 26 | Branch and bound, cuts, gaps | How solvers prove a plan is best |
| 27 | Strong formulations, symmetry | Why some models solve and others don't |
| 28 | Measuring, sizing, solver choice | Christmas week loads |
| 29 | Piecewise costs, discounts | Overtime, flour contracts |
| 30 | Multi-year location | The five-year network plan |

## Commit and tag

```bash
git add .
git commit -m "Day 30: capstone, five-year DC network plan"
git tag v3.0-module3
git push --tags
```

Then run the tests: `tests/test_costs_and_plans.py` checks Days 28–30 (the bound-sized packing, the discount models against each other, the all-units offer, and the capstone against Day 23). Add `"day28"`, `"day29"`, `"day30"` to the folders in `conftest.py`.

```python
"""Module 3, Days 28-30: bound-sized packing, piecewise costs and the five-year network plan."""
import pyomo.environ as pyo
import pytest
from flour_costs import build_flour, load_flour
from highs_run import run_highs
from loads import fewest_trailers, lower_bound
from redesign import build_redesign, load_redesign


def test_fewest_trailers_monday(con):
    week = con.sql("SELECT * FROM christmas_week_orders WHERE ship_date = '2026-12-21'").df().set_index("order_id")
    trailers, proven, _, tries = fewest_trailers(week, 26, 16000, run_highs)
    assert proven and tries == 1 and trailers == lower_bound(week, 26, 16000) == 16


def test_discount_models_agree_and_lp_bands_are_too_cheap(con):
    flour, spec, bands, offer, delivery = load_flour(con)
    cost = {}
    for prices in ("list", "incremental", "piecewise", "bands_lp"):
        m = build_flour(flour, spec, bands, prices=prices, need=55)
        pyo.SolverFactory("highs").solve(m)
        cost[prices] = pyo.value(m.cost)
    assert cost["incremental"] == pytest.approx(cost["piecewise"], abs=0.01) == pytest.approx(25_038.64, abs=0.01)
    assert cost["bands_lp"] < cost["incremental"] < cost["list"]


def test_all_units_offer_never_costs_more(con):
    flour, spec, bands, offer, delivery = load_flour(con)
    for need in (40, 55):
        costs = []
        for kw in ({}, {"offer": offer}):
            m = build_flour(flour, spec, bands, prices="incremental", delivery=delivery, need=need, **kw)
            pyo.SolverFactory("highs").solve(m)
            costs.append(pyo.value(m.cost))
        assert costs[1] <= costs[0] + 1e-6


def test_redesign_one_year_is_day23(con):
    sites, stores, lanes, options, settings = load_redesign(con)
    m = build_redesign(sites, stores, lanes, options, settings, years=1, growth=0)
    pyo.SolverFactory("highs").solve(m)
    assert pyo.value(m.pv) == pytest.approx(52 * 92_492.76, abs=0.05)
```

```text
49 passed
```

### Day 30 checklist

- [ ] I extended a single-period model to five years with timing logic (open once, close once, build once).
- [ ] I checked the multi-year model against a known single-year answer before using it.
- [ ] I showed that doing nothing fails, and when.
- [ ] I reported the plan with its present value and its gap.
- [ ] I tested the plan against growth, discount rate and unavailable options.
- [ ] I wrote a board memo with the recommendation, why, how sure, what to watch and what the model leaves out.
