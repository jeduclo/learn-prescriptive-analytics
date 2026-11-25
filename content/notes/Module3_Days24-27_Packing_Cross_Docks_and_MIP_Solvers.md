# DAY 24 — Knapsack, Bin Packing and Cutting Stock

Three classic integer problems run through every warehouse and bakery. The **knapsack** asks what fits in one container. **Bin packing** asks how few containers hold everything. **Cutting stock** asks how to cut standard pieces into the sizes customers want. Today you meet all three at Northbound: the Monday rush truck, Tuesday's trailers and the bakery's cake bars. Each time a quick rule of thumb does well, and a model does better or proves the rule of thumb can't be beaten.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 24.1–24.3 The knapsack: the rush truck |
| 0:45 – 1:45 | 24.4–24.5 Bin packing: Tuesday's trailers |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 24.6–24.7 Cutting stock: cake bars from slabs |
| 2:45 – 3:45 | Relaxations of the three problems; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: spring thaw |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Days 24–27: `truck_spec` (a 26-pallet trailer and a 20-pallet straight truck, with weight limits and day costs), `rush_orders` (16 Monday top-up orders for Montreal stores), `truck_orders` (24 Tuesday orders), `christmas_orders` (48 orders for the Tuesday before Christmas, Days 26–27), `cake_slab` and `cake_bars` (sheet-cake slabs and the bars cut from them), `crossdock_candidates` and `crossdock_arcs` (six cross-dock sites for the chilled network, Day 25).

## 24.1 Greedy loading

Sixteen Montreal stores ran short over the weekend. Monday's rush truck (20 pallet positions, 12,000 kg) can carry some of their top-up orders; each order delivered saves the sales it brings (`value_if_delivered`). Which orders go? Create `day24/knapsack.py`. First, the dispatcher's rule: best value per pallet first, skipping what no longer fits.

```python
    # ---------- 24.1 Greedy: best value per pallet first ----------
    room, greedy = pallets, []
    for o in orders.sort_values("value_per_pallet", ascending=False).index:
        if orders.pallets[o] <= room:
            greedy.append(o)
            room -= orders.pallets[o]
    print(f"greedy: {orders.value_if_delivered[greedy].sum():,} with {orders.pallets[greedy].sum()} pallets: {', '.join(sorted(greedy))}")
```

```text
order_id
RO-01         S28        4       2800                 900            225.00
RO-02         S32        6       4280                2110            351.67
RO-03         S29        5       3300                1730            346.00
RO-04         S32        3       1840                 940            313.33
RO-05         S30        2       1240                 460            230.00
RO-06         S37        1        720                 380            380.00
RO-07         S38        6       4200                2390            398.33
RO-08         S28        6       3680                2300            383.33
RO-09         S38        1        530                 260            260.00
RO-10         S37        1        710                 350            350.00
RO-11         S31        4       2130                 790            197.50
RO-12         S32        4       2080                 800            200.00
RO-13         S29        4       1880                1400            350.00
RO-14         S39        1        490                 280            280.00
RO-15         S38        5       3450                1560            312.00
RO-16         S39        3       2000                1080            360.00
16 orders, 56 pallets, $17,730 of value; the rush truck takes 20 pallets and 12,000 kg
greedy: 7,440 with 20 pallets: RO-04, RO-06, RO-07, RO-08, RO-10, RO-16
```

## 24.2 The knapsack model

One binary per order, $$t_o = 1$$ if order $$o$$ goes on the truck:

$$
\max \sum_o \text{value}_o\, t_o \quad \text{s.t.} \quad \sum_o \text{pallets}_o\, t_o \le 20, \qquad t_o \in \{0, 1\}
$$

```python
def build_knapsack(orders, pallets, max_kg=None, domain=pyo.Binary):
    """take[o] = 1 if order o goes on the trailer. Maximize the value delivered within the pallet positions
    (and, if max_kg is given, the weight limit)."""
    m = pyo.ConcreteModel(name="knapsack")
    m.O = pyo.Set(initialize=orders.index.tolist())
    m.take = pyo.Var(m.O, within=domain, bounds=(0, 1))
    m.positions = pyo.Constraint(expr=sum(orders.pallets[o] * m.take[o] for o in m.O) <= pallets)
    if max_kg is not None:
        m.weight = pyo.Constraint(expr=sum(orders.weight_kg[o] * m.take[o] for o in m.O) <= max_kg)
    m.value = pyo.Objective(expr=sum(orders.value_if_delivered[o] * m.take[o] for o in m.O), sense=pyo.maximize)
    return m
```

```python
    # ---------- 24.2 The knapsack model ----------
    m = build_knapsack(orders, pallets)
    solver.solve(m)
    best = loaded(m)
    print(f"knapsack: {pyo.value(m.value):,.0f} with {orders.pallets[best].sum()} pallets, {orders.weight_kg[best].sum():,} kg: {', '.join(best)}")
    print("greedy loads but the model leaves:", ", ".join(sorted(set(greedy) - set(best))),
          "| the model loads instead:", ", ".join(sorted(set(best) - set(greedy))))
    value, chosen = dp_knapsack(orders.pallets.tolist(), orders.value_if_delivered.tolist(), pallets)
    print(f"dynamic programming: {value:,} with {', '.join(orders.index[k] for k in chosen)}")
    r = build_knapsack(orders, pallets, domain=pyo.Reals)
    solver.solve(r)
    print("relaxation:", f"{pyo.value(r.value):,.2f};", "fractional:",
          {o: round(pyo.value(r.take[o]), 3) for o in r.O if 1e-6 < pyo.value(r.take[o]) < 1 - 1e-6})
```

```text
knapsack: 7,550 with 20 pallets, 12,480 kg: RO-06, RO-07, RO-08, RO-13, RO-16
greedy loads but the model leaves: RO-04, RO-10 | the model loads instead: RO-13
dynamic programming: 7,550 with RO-06, RO-07, RO-08, RO-13, RO-16
relaxation: 7,556.67; fractional: {'RO-02': 0.667}
```

The model delivers **$7,550**, $110 more than the dispatcher's rule. Greedy filled the truck with RO-04 (3 pallets) and RO-10 (1); the model swaps them for RO-13 (4 pallets, $1,400). Greedy looks at each order alone; the model looks at what fits together.

**Dynamic programming** gives the same answer another way: the best value with $$c$$ pallet positions, built up one order at a time. It is fast when the capacity is a small whole number, and a good known-answer check for the model (Day 9). It doesn't extend easily to a second limit; the model does.

The relaxation takes two-thirds of RO-02 and earns $7,556.67: only $6.67 above the best whole load. That is why the knapsack is "easy" for MIP solvers, as long as there is only one limit.

> **Decision it supports:** which items to take when they can't all fit.
>
> **Other decisions you can support the same way:** which projects to fund from a budget · which promotions get the end-of-aisle display · which orders to rush when capacity is short.

## 24.3 Two limits

The truck also has a weight limit. One more constraint:

```python
    # ---------- 24.3 A second limit: weight ----------
    w = build_knapsack(orders, pallets, max_kg=max_kg)
    solver.solve(w)
    both = loaded(w)
    print(f"with the {max_kg:,} kg limit: {pyo.value(w.value):,.0f}, {orders.pallets[both].sum()} pallets, "
          f"{orders.weight_kg[both].sum():,} kg: {', '.join(both)}")
```

```text
with the 12,000 kg limit: 7,170, 19 pallets, 11,760 kg: RO-07, RO-08, RO-13, RO-16
```

The best load from 24.2 weighed 12,480 kg, over the limit. With both limits the truck carries **$7,170** of orders in 19 pallets and 11,760 kg: weight, not space, decides. With two limits, "value per pallet" and "value per kilo" rank the orders differently, and no simple rule fixes that. The model handles it with one line.

## 24.4 First fit decreasing

Tuesday's 24 Montreal orders (200 pallets) go on 26-pallet trailers. How few trailers? Create `day24/packing.py`. First the warehouse's rule, **first fit decreasing** (FFD): biggest order first, onto the first trailer with room.

```python
def first_fit_decreasing(sizes, capacity, weights=None, max_kg=None):
    """Biggest order first; put each order on the first truck with room; open a new truck when none has."""
    trucks = []                                   # each truck: [pallets used, kg used, [orders]]
    for o in sorted(sizes, key=lambda o: (-sizes[o], o)):
        for t in trucks:
            if t[0] + sizes[o] <= capacity and (max_kg is None or t[1] + weights[o] <= max_kg):
                t[0] += sizes[o]
                t[1] += weights[o] if weights else 0
                t[2].append(o)
                break
        else:
            trucks.append([sizes[o], weights[o] if weights else 0, [o]])
    return trucks
```

## 24.5 The bin-packing model

$$
\begin{aligned}
\min\quad & \textstyle\sum_k u_k \\
\text{s.t.}\quad & \textstyle\sum_k p_{ok} = 1 && \forall o \text{ (every order on one trailer)} \\
& \textstyle\sum_o \text{pallets}_o\, p_{ok} \le 26\, u_k && \forall k \text{ (only on a trailer that leaves)} \\
& p_{ok},\ u_k \in \{0, 1\}
\end{aligned}
$$

The trailers $$k$$ go up to FFD's count: the model never needs more than a plan we already have.

```python
def build_packing(orders, capacity, n_trucks, max_kg=None):
    """put[o, k] = 1 if order o goes on truck k; use[k] = 1 if truck k leaves. Minimize the trucks used."""
    m = pyo.ConcreteModel(name="bin_packing")
    m.O = pyo.Set(initialize=orders.index.tolist())
    m.K = pyo.RangeSet(1, n_trucks)
    m.put = pyo.Var(m.O, m.K, within=pyo.Binary)
    m.use = pyo.Var(m.K, within=pyo.Binary)
    m.one_truck = pyo.Constraint(m.O, rule=lambda m, o: sum(m.put[o, k] for k in m.K) == 1)
    m.positions = pyo.Constraint(m.K, rule=lambda m, k: sum(orders.pallets[o] * m.put[o, k] for o in m.O) <= capacity * m.use[k])
    if max_kg is not None:
        m.weight = pyo.Constraint(m.K, rule=lambda m, k: sum(orders.weight_kg[o] * m.put[o, k] for o in m.O) <= max_kg * m.use[k])
    m.trucks = pyo.Objective(expr=sum(m.use[k] for k in m.K))
    return m
```

```python
if __name__ == "__main__":
    con = duckdb.connect("northbound.duckdb", read_only=True)
    orders = con.sql("SELECT * FROM truck_orders ORDER BY order_id").df().set_index("order_id")
    capacity, max_kg = con.sql("SELECT pallet_positions, max_kg FROM truck_spec WHERE truck_type = 'trailer'").fetchall()[0]
    total = orders.pallets.sum()
    print(f"{len(orders)} orders, {total} pallets; a trailer takes {capacity}: at least {total / capacity:.2f} "
          f"-> {math.ceil(total / capacity)} trailers")

    # ---------- 24.4 First fit decreasing ----------
    ffd = first_fit_decreasing(orders.pallets.to_dict(), capacity)
    print(f"first fit decreasing: {len(ffd)} trailers, pallets per trailer {[t[0] for t in ffd]}")

    # ---------- 24.5 The bin-packing model ----------
    m = build_packing(orders, capacity, n_trucks=len(ffd))
    pyo.SolverFactory("highs").solve(m)
    print(f"bin-packing model: {pyo.value(m.trucks):.0f} trailers")
    print(trucks_used(m, orders).to_string(index=False))
```

```text
24 orders, 200 pallets; a trailer takes 26: at least 7.69 -> 8 trailers
first fit decreasing: 8 trailers, pallets per trailer [26, 26, 24, 26, 26, 24, 25, 23]
bin-packing model: 8 trailers
 truck  orders  pallets    kg                  order_ids
     1       4       26 16540 TO-06, TO-07, TO-08, TO-17
     2       4       25 16010 TO-02, TO-11, TO-14, TO-18
     3       2       26 15440               TO-03, TO-22
     4       3       24 13330        TO-01, TO-15, TO-19
     5       3       25 17100        TO-13, TO-21, TO-23
     6       3       26 14390        TO-09, TO-16, TO-24
     7       3       24 14620        TO-05, TO-10, TO-20
     8       2       24 15210               TO-04, TO-12
```

FFD needs 8 trailers, and 200 pallets need at least $$\lceil 200 / 26 \rceil = 8$$. So **FFD's plan is already optimal**, and the simple bound proves it without a solver. Always compute a bound: it tells you when to stop looking. (The model's plan differs from FFD's, but uses the same number of trailers: bin packing usually has many optimal plans. Day 27 comes back to that.)

> **Decision it supports:** how few vehicles, containers or shifts can hold a set of jobs.
>
> **Other decisions you can support the same way:** how many pallets for an order of cases · how many servers for a set of programs · how many exam rooms for a set of classes.

## 24.6 Cutting patterns

The bakery bakes sheet cake in 120 cm slabs ($18 each) and cuts them into bars of five lengths for the cafés. Create `day24/cutting.py`. A **pattern** is one way to cut a slab: say, one 24 cm bar, one 45 and one 50 (119 cm, 1 cm wasted). The code lists every *maximal* pattern, one where not even the shortest bar fits in the leftover.

```python
def patterns(lengths, slab, fits=lambda used, slab: used <= slab):
    """Every way to cut one slab: how many bars of each length, as long as they fit.
    Only maximal patterns are kept: ones where not even the shortest bar would fit in what is left."""
    names = list(lengths)
    found = []

    def extend(k, counts, used):
        if k == len(names):
            if not fits(used + min(lengths.values()), slab) and any(counts):
                found.append(dict(zip(names, counts)))
            return
        n = 0
        while fits(used + n * lengths[names[k]], slab):
            extend(k + 1, counts + [n], used + n * lengths[names[k]])
            n += 1

    extend(0, [], 0)
    return found
```

```python
if __name__ == "__main__":
    con = duckdb.connect("northbound.duckdb", read_only=True)
    bars = con.sql("SELECT * FROM cake_bars ORDER BY length_cm").df().set_index("bar")
    slab, cost = con.sql("SELECT * FROM cake_slab").fetchall()[0]
    print(bars.to_string())
    lengths, demand = bars.length_cm.to_dict(), bars.bars_per_week.to_dict()
    need = sum(lengths[b] * demand[b] for b in lengths)
    print(f"bars need {need:,} cm of cake: at least {need / slab:.2f} -> {math.ceil(need / slab)} slabs of {slab} cm")
    today = {b: math.ceil(demand[b] / (slab // lengths[b])) for b in lengths}
    print(f"today, one kind of bar per slab: {today} = {sum(today.values())} slabs, ${sum(today.values()) * cost:,.2f} a week")

    # ---------- 24.6 Cutting patterns ----------
    pats = patterns(lengths, slab)
    print(f"{len(pats)} maximal patterns, for example:")
    for p in pats[:4]:
        print("   ", {b: n for b, n in p.items() if n})
```

```text
                 product  length_cm  bars_per_week
bar
BAR-18         Lemon bar         18             64
BAR-24       Brownie bar         24             52
BAR-30        Carrot bar         30             40
BAR-45         Opera bar         45             28
BAR-50  Black forest bar         50             22
bars need 5,960 cm of cake: at least 49.67 -> 50 slabs of 120 cm
today, one kind of bar per slab: {'BAR-18': 11, 'BAR-24': 11, 'BAR-30': 10, 'BAR-45': 14, 'BAR-50': 11} = 57 slabs, $1,026.00 a week
35 maximal patterns, for example:
    {'BAR-30': 1, 'BAR-45': 2}
    {'BAR-30': 2, 'BAR-50': 1}
    {'BAR-30': 2, 'BAR-45': 1}
    {'BAR-30': 4}
```

Today the bakery cuts each slab into one kind of bar: 57 slabs a week.

## 24.7 The cutting-stock model

One integer variable per pattern, $$c_p$$ = how many slabs to cut with pattern $$p$$; $$a_{bp}$$ = bars of length $$b$$ in pattern $$p$$:

$$
\min \sum_p c_p \quad \text{s.t.} \quad \sum_p a_{bp}\, c_p \ge \text{demand}_b \ \ \forall b, \qquad c_p \in \{0, 1, 2, \dots\}
$$

```python
def build_cutting(pats, demand, domain=pyo.NonNegativeIntegers):
    """cut[p] = number of slabs cut with pattern p. Make at least the bars needed, from as few slabs as possible."""
    m = pyo.ConcreteModel(name="cutting_stock")
    m.P = pyo.RangeSet(0, len(pats) - 1)
    m.B = pyo.Set(initialize=list(demand))
    m.cut = pyo.Var(m.P, within=domain)
    m.enough = pyo.Constraint(m.B, rule=lambda m, b: sum(pats[p][b] * m.cut[p] for p in m.P) >= demand[b])
    m.slabs = pyo.Objective(expr=sum(m.cut[p] for p in m.P))
    return m
```

```python
    # ---------- 24.7 The cutting-stock model ----------
    solver = pyo.SolverFactory("highs")
    m = build_cutting(pats, demand)
    solver.solve(m)
    print(f"cutting-stock model: {pyo.value(m.slabs):.0f} slabs, ${pyo.value(m.slabs) * cost:,.2f} a week")
    print(show(m, pats, lengths, slab).to_string(index=False))
    r = build_cutting(pats, demand, domain=pyo.NonNegativeReals)
    solver.solve(r)
    rounded = sum(math.ceil(pyo.value(r.cut[p]) - 1e-9) for p in r.P)
    print(f"relaxation: {pyo.value(r.slabs):.2f} slabs; rounding every pattern up: {rounded} slabs")
    print(show(r, pats, lengths, slab).to_string(index=False))
```

```text
cutting-stock model: 50 slabs, $900.00 a week
pattern (bars x cm)  slabs  waste_cm
        1x30 + 2x45    3.0         0
               4x30    4.0         0
 1x24 + 1x45 + 1x50   22.0         1
 1x18 + 3x24 + 1x30   10.0         0
        5x18 + 1x30   11.0         0
relaxation: 49.85 slabs; rounding every pattern up: 51 slabs
pattern (bars x cm)  slabs  waste_cm
        1x30 + 2x45   3.00         0
               4x30   4.05         0
 1x24 + 1x45 + 1x50  22.00         1
 1x18 + 3x24 + 1x30  10.00         0
        5x18 + 1x30  10.80         0
```

**50 slabs**, the least possible ($$\lceil 5{,}960 / 120 \rceil$$), with 22 cm of trim in the whole week: **7 slabs and $126 a week less** than one kind of bar per slab. The relaxation's 49.85 slabs is a good bound, but rounding each pattern up gives 51: rounding is not a method, again (Day 21).

With five bar lengths there are only 35 patterns. With 50 lengths there would be millions; real cutting-stock solvers generate patterns as they need them (**column generation**, see the math corner).

> **Decision it supports:** how to cut standard stock into the sizes needed, with the least waste.
>
> **Other decisions you can support the same way:** cutting steel bars, paper rolls, fabric or timber · splitting a work week into shifts · loading a truck in layers.

### The planted bug

`day24/bug_patterns.py` uses a colleague's pattern generator, "so bars never touch the end of the slab". Run it.

```python
pats = patterns(lengths, slab, fits=lambda used, slab: used < slab)
m = build_cutting(pats, demand)
pyo.SolverFactory("highs").solve(m)
print(f"{len(pats)} patterns; optimal: {pyo.value(m.slabs):.0f} slabs, ${pyo.value(m.slabs) * cost:,.2f} a week")
```

<details><summary>What happens, and why</summary>

```text
32 patterns; optimal: 52 slabs, $936.00 a week
```

`used < slab` instead of `used <= slab` drops every pattern that uses exactly 120 cm, and four of the five patterns in the best plan use the slab exactly. The model is still "optimal": for the patterns it was given. **The model can only choose from what you generate.** Test generators on a case you can check by hand (does 4 × 30 cm appear?), and compare the best plan with the bound: 52 slabs against a bound of 50 is a warning.

</details>

### Math corner (optional): column generation

The cutting-stock relaxation has a dual price $$\pi_b$$ for each bar length: what one more bar of that length costs in slabs. A new pattern is worth adding only if its bars are worth more than one slab: $$\sum_b \pi_b\, a_b > 1$$. Finding the best such pattern is a knapsack (bar lengths as sizes, $$\pi_b$$ as values, 120 cm as capacity). Column generation alternates: solve the relaxation with the patterns so far, solve a knapsack for a new pattern, add it, repeat until no pattern is worth more than 1. Then solve the integer model on the patterns found. This is how industrial cutting and crew-scheduling models handle millions of possible patterns (Gilmore and Gomory, 1961).

## Mini-project: spring thaw

From March to May, Quebec lowers truck weight limits on roads softened by the thaw. Planning asks: how many trailers do Tuesday's orders need at weight limits of 20,000, 18,000, 16,000, 15,000 and 14,000 kg? Does first fit decreasing (by pallets, checking both limits) still do the job? Show the 16,000 kg plan.

<details><summary>Solution</summary>

```python
"""Day 24 mini-project: spring thaw. From March to May, Quebec limits truck weights on thawing roads.
How many trailers do Tuesday's Montreal orders need at each weight limit, and does first fit decreasing still do?

Run from the repository root (after day06/load_tables.py):  python day24/mini_project_thaw.py
"""
import sys

import duckdb
import pyomo.environ as pyo

sys.path.insert(0, "day24")
from packing import build_packing, first_fit_decreasing, trucks_used

con = duckdb.connect("northbound.duckdb", read_only=True)
orders = con.sql("SELECT * FROM truck_orders ORDER BY order_id").df().set_index("order_id")
capacity, cost = con.sql("SELECT pallet_positions, cost_per_day FROM truck_spec WHERE truck_type = 'trailer'").fetchall()[0]
total_kg = orders.weight_kg.sum()
for max_kg in (20000, 18000, 16000, 15000, 14000):
    ffd = first_fit_decreasing(orders.pallets.to_dict(), capacity, orders.weight_kg.to_dict(), max_kg)
    m = build_packing(orders, capacity, n_trucks=len(ffd), max_kg=max_kg)
    pyo.SolverFactory("highs").solve(m)
    best = round(pyo.value(m.trucks))
    print(f"{max_kg:>6,} kg: weight alone needs {total_kg / max_kg:.2f}; first fit decreasing {len(ffd)}, "
          f"model {best} trailers (${best * cost:,} a day)")
    if max_kg == 16000:
        plan = trucks_used(m, orders)
print("the 16,000 kg plan:")
print(plan.to_string(index=False))
```

```text
20,000 kg: weight alone needs 6.13; first fit decreasing 8, model 8 trailers ($5,200 a day)
18,000 kg: weight alone needs 6.81; first fit decreasing 8, model 8 trailers ($5,200 a day)
16,000 kg: weight alone needs 7.67; first fit decreasing 9, model 8 trailers ($5,200 a day)
15,000 kg: weight alone needs 8.18; first fit decreasing 9, model 9 trailers ($5,850 a day)
14,000 kg: weight alone needs 8.76; first fit decreasing 10, model 9 trailers ($5,850 a day)
the 16,000 kg plan:
 truck  orders  pallets    kg                  order_ids
     1       2       25 15310               TO-12, TO-16
     2       2       26 15440               TO-03, TO-22
     3       4       26 15100 TO-01, TO-05, TO-15, TO-21
     4       3       26 13500        TO-07, TO-09, TO-17
     5       3       26 15910        TO-02, TO-11, TO-23
     6       4       24 15630 TO-10, TO-13, TO-14, TO-24
     7       3       25 15940        TO-18, TO-19, TO-20
     8       3       22 15810        TO-04, TO-06, TO-08
```

- Down to 18,000 kg nothing changes: pallets decide.
- **At 16,000 kg, FFD needs 9 trailers but 8 are enough**: the model pairs heavy and light orders, which FFD (sorting by pallets only) never considers. The model saves a trailer, $650, every Tuesday of the thaw.
- At 15,000 kg the 9th trailer can't be avoided: weight alone needs 8.18. At 14,000 kg FFD uses 10, the model 9.

*Memo to transport planning:* "During the thaw, plan loads with the model, not with first fit decreasing. At the expected 16,000 kg limit, Tuesday's orders fit on 8 trailers instead of 9."

</details>

## Commit

```bash
git add .
git commit -m "Day 24: knapsack, bin packing, cutting stock"
git push
```

### Day 24 checklist

- [ ] I can write a knapsack, a bin-packing and a cutting-stock model.
- [ ] I compared each with the rule of thumb a planner would use.
- [ ] I compute a simple bound (pallets ÷ capacity) to know when a plan can't be beaten.
- [ ] I know a second limit breaks simple ranking rules but not the model.
- [ ] I know a pattern-based model can only choose the patterns it is given.
- [ ] I can explain what column generation adds (math corner).

---

# DAY 25 — Mini-project: Which Cross-Docks to Open

On Day 16 you sent chilled pallets through two cross-docks, Kingston and Trois-Rivières, and checked by hand whether each was worth its weekly cost. Transport now has six candidate sites and a complaint: "the model pays linehaul by the pallet, but we pay by the truck". Today is a full mini-project: rebuild the chilled network as a MIP with cross-docks to choose and whole trucks to pay for, and recommend which cross-docks to run.

| Time | Block |
|---|---|
| 0:00 – 0:30 | 25.1 The brief, and today's network with whole trucks |
| 0:30 – 1:45 | 25.2 Six candidates in one model |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 25.3 Check: all 64 combinations |
| 2:45 – 3:45 | 25.4–25.5 How close is each candidate? Scenarios |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | 25.6 The memo |
| 4:45 – 5:00 | Commit with Git + checklist |

## 25.1 The brief

- **Network:** Day 16's chilled network (two plants, three DCs, 40 stores, 922 pallets a week), with six candidate cross-docks (`crossdock_candidates`): Kingston and Trois-Rivières (running today), Cornwall, Belleville, Drummondville and Sherbrooke. Each has a weekly fixed cost and a throughput limit.
- **Arcs** (`crossdock_arcs`): linehaul arcs from plants and DCs to each cross-dock, paid **by the truck** (26 pallets, $$\$26 \times (2.50 + 0.08\,\text{km})$$, so a full truck costs what Day 16 charged for 26 pallets), and delivery arcs from each cross-dock to stores within 260 km, paid by the pallet. Quebec stores need French labels, so only Quebec-side cross-docks and Cornwall may serve them.
- **Model:** min-cost flow (Day 16) plus $$o_x \in \{0,1\}$$ for each cross-dock and $$n_a \in \{0, 1, 2, \dots\}$$ trucks on each linehaul arc, with $$\text{flow}_a \le 26\, n_a$$. The plant-to-DC linehaul is paid by the truck too.

Create `day25/crossdocks.py`:

```python
"""Day 25 mini-project: which cross-docks should the chilled network use?
Six candidate cross-docks, each with a weekly fixed cost; linehaul trucks paid by the truck, not by the pallet.

Import it:  from crossdocks import load_crossdock_network, build_crossdocks, report
"""
import pandas as pd
import pyomo.environ as pyo

TRUCK_PALLETS = 26


def load_crossdock_network(con):
    nodes = con.sql("SELECT * FROM chilled_nodes WHERE kind <> 'cross_dock'").df().set_index("node")
    stores = con.sql("SELECT store_id, province FROM stores").df().set_index("store_id")
    xds = con.sql("SELECT * FROM crossdock_candidates ORDER BY xd_id").df().set_index("xd_id")
    old = con.sql("""SELECT from_node, to_node, km, cost_per_pallet, capacity_pallets FROM chilled_arcs
                     WHERE from_node NOT LIKE 'XD-%' AND to_node NOT LIKE 'XD-%'""").df()
    old["arc_type"] = ["linehaul" if f.startswith("PL-") else "delivery" for f in old.from_node]
    old["cost_per_truck"] = (old.cost_per_pallet * TRUCK_PALLETS).where(old.arc_type == "linehaul")
    new = con.sql("SELECT * FROM crossdock_arcs").df()
    new["capacity_pallets"] = float("nan")
    arcs = pd.concat([old, new], ignore_index=True).set_index(["from_node", "to_node"])
    arcs["cost_per_pallet"] = arcs.cost_per_pallet.where(arcs.arc_type == "delivery",
                                                         arcs.cost_per_truck / TRUCK_PALLETS)
    return nodes, stores, xds, arcs


def build_crossdocks(nodes, xds, arcs, trucks=True, allowed=None, forced=(), cost_scale=1.0, demand_scale=1.0,
                     domain=pyo.Binary):
    """open[x] = 1 if cross-dock x runs this year. flow[a] = pallets a week on arc a.
    With trucks=True, every linehaul arc pays for whole trucks: trucks[a] >= flow[a] / 26."""
    m = pyo.ConcreteModel(name="crossdocks")
    m.X = pyo.Set(initialize=xds.index.tolist())
    m.A = pyo.Set(initialize=arcs.index.tolist(), dimen=2)
    m.L = pyo.Set(initialize=[a for a in arcs.index if arcs.arc_type[a] == "linehaul"], dimen=2)
    m.open = pyo.Var(m.X, within=domain)
    m.flow = pyo.Var(m.A, within=pyo.NonNegativeReals)
    for a in m.A:
        if not pd.isna(arcs.capacity_pallets[a]):
            m.flow[a].setub(float(arcs.capacity_pallets[a]))
    for x in m.X:
        if allowed is not None and x not in allowed:
            m.open[x].fix(0)
        if x in forced:
            m.open[x].fix(1)
    demand = {n: float(nodes.demand_pallets[n]) * demand_scale for n in nodes.index if nodes.kind[n] == "store"}
    inflow = {n: [a for a in arcs.index if a[1] == n] for n in list(nodes.index) + list(xds.index)}
    outflow = {n: [a for a in arcs.index if a[0] == n] for n in list(nodes.index) + list(xds.index)}

    def balance(m, n):
        net_in = sum(m.flow[a] for a in inflow[n]) - sum(m.flow[a] for a in outflow[n])
        if n in demand:
            return net_in == demand[n]
        if nodes.kind.get(n) == "plant":
            return -net_in <= float(nodes.throughput_capacity[n])
        return net_in == 0
    m.N = pyo.Set(initialize=list(nodes.index) + list(xds.index))
    m.balance = pyo.Constraint(m.N, rule=balance)
    dcs = [n for n in nodes.index if nodes.kind[n] == "dc"]
    m.dc_throughput = pyo.Constraint(dcs, rule=lambda m, n: sum(m.flow[a] for a in inflow[n]) <= float(nodes.throughput_capacity[n]))
    m.xd_throughput = pyo.Constraint(m.X, rule=lambda m, x: sum(m.flow[a] for a in inflow[x]) <= float(xds.throughput_pallets[x]) * m.open[x])
    # strong link: a store can only get pallets from an open cross-dock (at most all it needs)
    m.only_if_open = pyo.Constraint([a for a in arcs.index if a[0] in xds.index],
                                    rule=lambda m, x, s: m.flow[x, s] <= demand[s] * m.open[x])
    m.fixed = pyo.Expression(expr=sum(float(xds.fixed_cost_per_week[x]) * m.open[x] for x in m.X) + 0.0)
    m.delivery = pyo.Expression(expr=cost_scale * sum(float(arcs.cost_per_pallet[a]) * m.flow[a]
                                                      for a in m.A if a not in m.L))
    if trucks:
        m.trucks = pyo.Var(m.L, within=pyo.NonNegativeIntegers)
        m.full = pyo.Constraint(m.L, rule=lambda m, i, j: m.flow[i, j] <= TRUCK_PALLETS * m.trucks[i, j])
        m.linehaul = pyo.Expression(expr=cost_scale * sum(float(arcs.cost_per_truck[a]) * m.trucks[a] for a in m.L))
    else:
        m.linehaul = pyo.Expression(expr=cost_scale * sum(float(arcs.cost_per_pallet[a]) * m.flow[a] for a in m.L))
    m.cost = pyo.Objective(expr=m.fixed + m.linehaul + m.delivery)
    return m


def report(m, xds):
    """One line of costs, and the cross-docks used with the pallets through each."""
    used = [(x, round(sum(pyo.value(m.flow[a]) for a in m.A if a[1] == x), 1) + 0.0) for x in m.X if pyo.value(m.open[x]) > 0.5]
    text = ", ".join(f"{xds.place[x]} {p:g}" for x, p in used) or "none"
    return (f"${pyo.value(m.cost):,.2f} = fixed {pyo.value(m.fixed) + 0.0:,.0f} + linehaul {pyo.value(m.linehaul):,.2f} "
            f"+ delivery {pyo.value(m.delivery):,.2f}; cross-docks: {text}")
```

Three things to notice. `trucks` is a general **integer** variable, not a binary. The strong link from Day 23 (`only_if_open`: each store's delivery from a cross-dock is at most its demand times $$o_x$$) comes on top of the throughput row. And `trucks=False` gives back Day 16's per-pallet model, so you can check the new model against an answer you trust.

Then `day25/run_crossdocks.py`:

```python
import itertools
import sys

import duckdb
import pandas as pd
import pyomo.environ as pyo

sys.path.insert(0, "day25")
from crossdocks import build_crossdocks, load_crossdock_network, report

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
nodes, stores, xds, arcs = load_crossdock_network(con)
solver = pyo.SolverFactory("highs")


def solve(**kw):
    m = build_crossdocks(nodes, xds, arcs, **kw)
    solver.solve(m)
    return m


print(xds.to_string())
print(f"{len(arcs)} arcs: {(arcs.arc_type == 'linehaul').sum()} linehaul (paid by the truck), "
      f"{(arcs.arc_type == 'delivery').sum()} store deliveries (paid by the pallet)")

# ---------- 25.1 Today's network, two ways ----------
today = ("XD-KIN", "XD-TRV")
m = solve(trucks=False, allowed=today, forced=today)
print("today, per-pallet linehaul (Day 16):", report(m, xds))
m = solve(allowed=today, forced=today)
print("today, whole trucks:                ", report(m, xds))
for a in m.L:
    if pyo.value(m.trucks[a]) > 0.5:
        print(f"    {a[0]} -> {a[1]}: {pyo.value(m.flow[a]):.0f} pallets on {pyo.value(m.trucks[a]):.0f} truck(s)")
today_cost = pyo.value(m.cost)
```

```text
                 place  fixed_cost_per_week  throughput_pallets
xd_id
XD-BEL      Belleville                  550                 140
XD-COR        Cornwall                  650                 180
XD-DRU   Drummondville                  700                 200
XD-KIN        Kingston                  500                 160
XD-SHE      Sherbrooke                  600                 120
XD-TRV  Trois-Rivieres                  900                 220
227 arcs: 19 linehaul (paid by the truck), 208 store deliveries (paid by the pallet)
today, per-pallet linehaul (Day 16): $19,603.64 = fixed 1,400 + linehaul 10,391.30 + delivery 7,812.34; cross-docks: Kingston 35, Trois-Rivieres 123
today, whole trucks:                 $19,993.74 = fixed 1,400 + linehaul 10,304.84 + delivery 8,288.90; cross-docks: Kingston 26, Trois-Rivieres 104
    PL-SHY -> DC-OTT: 200 pallets on 8 truck(s)
    PL-SHY -> DC-MTL: 283 pallets on 11 truck(s)
    PL-BRA -> DC-MIS: 335 pallets on 13 truck(s)
    DC-MIS -> XD-KIN: 26 pallets on 1 truck(s)
    PL-SHY -> XD-TRV: 104 pallets on 4 truck(s)
```

The first line is **exactly Day 16's answer**, $19,603.64: the new model is right before we change anything. Paying by the truck, the same network costs **$19,993.74, $390 a week more**, and the flows change: Kingston gets exactly 26 pallets (one full truck) instead of 35 (one and a third trucks), and Trois-Rivières 104 (four trucks) instead of 123. The per-pallet model sent partial trucks as if they were free.

## 25.2 Six candidates in one model

```python
# ---------- 25.2 Six candidates ----------
m = solve(trucks=False)
print("best, per-pallet linehaul:", report(m, xds))
best = solve()
print("best, whole trucks:       ", report(best, xds))
print(f"saves ${today_cost - pyo.value(best.cost):,.2f} a week against today")
```

```text
best, per-pallet linehaul: $18,445.14 = fixed 1,200 + linehaul 9,702.50 + delivery 7,542.64; cross-docks: Drummondville 123, Kingston 35
best, whole trucks:        $18,757.88 = fixed 700 + linehaul 9,027.20 + delivery 9,030.68; cross-docks: Drummondville 124
saves $1,235.86 a week against today
```

- **Per pallet**, the model would open Drummondville and Kingston.
- **By the truck**, it opens **Drummondville only**, for $18,757.88: **$1,235.86 a week less than today** (about $64,000 a year). Kingston's 26 to 35 pallets a week don't pay for a cross-dock once each truck must be paid in full.
- Drummondville replaces Trois-Rivières: it is on the plant's route from Saint-Hyacinthe, cheaper to run ($700 against $900), and serves the three Quebec City and two Sherbrooke stores, straight from the plant at Saint-Hyacinthe.

## 25.3 Check: all 64 combinations

```python
# ---------- 25.3 Check by trying all 64 combinations ----------
results = []
for keep in itertools.product([0, 1], repeat=len(xds)):
    chosen = tuple(x for x, k in zip(xds.index, keep) if k)
    m = solve(allowed=chosen, forced=chosen)
    results.append((pyo.value(m.cost), ", ".join(c[3:] for c in chosen) or "none"))
results.sort()
print("best five of 64:", "; ".join(f"{c} ${v:,.2f}" for v, c in results[:5]))
```

```text
best five of 64: DRU $18,757.88; DRU, KIN $18,842.95; BEL, DRU $18,989.67; COR, DRU $19,173.46; DRU, SHE $19,186.06
```

Sixty-four solves, the same best answer as one MIP. Note how close the runners-up are: adding Kingston costs only $85 a week more.

## 25.4 How close is each candidate?

```python
# ---------- 25.4 Force each candidate open ----------
rows = []
for x in xds.index:
    m = solve(forced=(x,))
    extra = pyo.value(m.cost) - pyo.value(best.cost)
    used = ", ".join(y[3:] for y in m.X if pyo.value(m.open[y]) > 0.5)
    rows.append((x, xds.place[x], xds.fixed_cost_per_week[x], round(extra, 2) + 0.0, used))
print(pd.DataFrame(rows, columns=["forced", "place", "fixed_cost", "extra_cost", "open"]).to_string(index=False))
```

```text
forced          place  fixed_cost  extra_cost     open
XD-BEL     Belleville         550      231.79 BEL, DRU
XD-COR       Cornwall         650      415.58 COR, DRU
XD-DRU  Drummondville         700        0.00      DRU
XD-KIN       Kingston         500       85.07 DRU, KIN
XD-SHE     Sherbrooke         600      428.18 DRU, SHE
XD-TRV Trois-Rivieres         900      882.20 SHE, TRV
```

Kingston is $85 a week from paying; Trois-Rivières, today's biggest cross-dock, is $882 a week from paying. Forced open, Trois-Rivières drags Sherbrooke in with it, not Drummondville.

## 25.5 Scenarios

```python
# ---------- 25.5 Scenarios ----------
for name, kw in [("fuel and wages +30%", dict(cost_scale=1.3)), ("chilled demand +15%", dict(demand_scale=1.15)),
                 ("both", dict(cost_scale=1.3, demand_scale=1.15)), ("no cross-docks at all", dict(allowed=()))]:
    print(f"{name:<22}", report(solve(**kw), xds))
```

```text
fuel and wages +30%    $24,135.83 = fixed 1,200 + linehaul 12,625.65 + delivery 10,310.18; cross-docks: Drummondville 124, Kingston 26
chilled demand +15%    $21,760.11 = fixed 700 + linehaul 9,383.40 + delivery 11,676.71; cross-docks: Drummondville 150.3
both                   $28,038.70 = fixed 1,200 + linehaul 13,088.71 + delivery 13,749.99; cross-docks: Drummondville 150.3, Kingston 26
no cross-docks at all  $20,194.89 = fixed 0 + linehaul 9,079.20 + delivery 11,115.69; cross-docks: none
```

- **Drummondville is in every scenario.** It is a robust choice.
- **Kingston comes back if transport costs rise 30%** (fuel and drivers' wages), because longer direct runs cost more. With 15% more demand alone, it stays closed.
- **No cross-docks at all** costs $20,194.89: cross-docks are worth $1,437 a week, and almost all of that comes from Drummondville.

## 25.6 The memo

<details><summary>A model memo</summary>

*To: Director of Transport. Re: chilled cross-docks.*

"We recommend running one chilled cross-dock, in **Drummondville**, instead of the two in Kingston and Trois-Rivières. The model saves **$1,236 a week** ($64,000 a year) before one-time costs, mainly by moving Quebec stores' chilled pallets through a cheaper site on the plant's route.

Kingston is a close call: it costs $85 a week more than it saves at today's prices, and becomes worth keeping if transport costs rise by 30%. We suggest keeping the Kingston lease on a short renewal rather than closing it at once.

Trois-Rivières costs $882 a week more than it saves; close it once Drummondville runs.

Our earlier analysis (Day 16) paid linehaul by the pallet and overstated the value of small flows. This study pays for whole trucks, as transport does."

</details>

## Commit

```bash
git add .
git commit -m "Day 25: cross-dock location with whole trucks"
git push
```

### Day 25 checklist

- [ ] I extended a min-cost flow with yes/no sites and integer trucks.
- [ ] I checked the new model reproduces an answer I trust (Day 16) before using it.
- [ ] I can explain why paying by the truck changes the answer.
- [ ] I checked one MIP against all 64 combinations.
- [ ] I tested the answer under scenarios and found which choices are robust.
- [ ] I wrote a recommendation with the close calls flagged.

---

# DAY 26 — How MIP Solvers Work

HiGHS has answered every Module 3 model in under a second. Today you open the box: you write **branch and bound** yourself, watch it search, add a **cutting plane** by hand, and see what a real solver adds on top. Then you meet a model HiGHS can't finish quickly, and learn the controls that decide when "good enough" is good enough.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 26.1 Branch and bound by hand |
| 0:45 – 1:45 | 26.2 A textbook branch and bound for any model |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 26.3 Cutting planes |
| 2:45 – 3:45 | 26.4 What HiGHS adds, and its controls; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: how good is good enough? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 26.1 Branch and bound by hand

Take the rush truck from Day 24 (pallets only). Its relaxation needs no solver: take orders by value per pallet, and a fraction of the first one that doesn't fit. That fraction is the only thing between the relaxation and a real load, so **branch** on it: one branch where the order is taken, one where it is left behind. Each branch has its own relaxation, and so its own **bound**. A branch whose bound can't beat the best load found so far is **pruned**: nothing below it can be better.

Create `day26/by_hand.py`:

```python
def lp_bound(orders, capacity, fixed):
    """Best value of the relaxation when some orders are fixed: fixed[o] = 1 (on the truck) or 0 (left behind).
    Returns (bound, the order taken as a fraction or None); bound is None if the fixed orders don't fit."""
    room = capacity - sum(orders[o][0] for o, f in fixed.items() if f == 1)
    if room < 0:
        return None, None
    value = sum(orders[o][1] for o, f in fixed.items() if f == 1)
    for o in sorted((o for o in orders if o not in fixed), key=lambda o: (-orders[o][1] / orders[o][0], o)):
        pallets, v = orders[o]
        if pallets <= room:
            room -= pallets
            value += v
        else:
            return value + v * room / pallets, (o if room > 0 else None)
    return value, None


def branch_and_bound_knapsack(orders, capacity, bound=lp_bound, show=print):
    """Depth first: try 'take it' before 'leave it'. Prune a node if its bound can't beat the best load so far."""
    best = {"value": 0, "load": None, "nodes": 0}

    def node(fixed, depth):
        best["nodes"] += 1
        b, frac = bound(orders, capacity, fixed)
        where = ", ".join(f"{o}={f}" for o, f in fixed.items()) or "root"
        if b is None:
            show(f"{'  ' * depth}{where}: doesn't fit")
            return
        if b <= best["value"]:
            show(f"{'  ' * depth}{where}: bound {b:,.2f} <= best {best['value']:,.0f}: prune")
            return
        if frac is None:            # the relaxation took whole orders only: an integer plan
            load = sorted([o for o, f in fixed.items() if f == 1] + greedy_rest(orders, capacity, fixed))
            best.update(value=b, load=load)
            show(f"{'  ' * depth}{where}: bound {b:,.2f} is a whole load: new best {', '.join(load)}")
            return
        show(f"{'  ' * depth}{where}: bound {b:,.2f}, {frac} fractional: branch")
        node({**fixed, frac: 1}, depth + 1)
        node({**fixed, frac: 0}, depth + 1)

    node({}, 0)
    return best


def greedy_rest(orders, capacity, fixed):
    room = capacity - sum(orders[o][0] for o, f in fixed.items() if f == 1)
    taken = []
    for o in sorted((o for o in orders if o not in fixed), key=lambda o: (-orders[o][1] / orders[o][0], o)):
        if orders[o][0] <= room:
            room -= orders[o][0]
            taken.append(o)
    return taken
```

```python
    # ---------- 26.1 The search tree ----------
    best = branch_and_bound_knapsack(orders, capacity)
    print(f"best {best['value']:,.0f}: {', '.join(best['load'])}; {best['nodes']} nodes "
          f"instead of {2 ** len(orders):,} combinations")
```

```text
root: bound 7,556.67, RO-02 fractional: branch
  RO-02=1: bound 7,540.00, RO-16 fractional: branch
    RO-02=1, RO-16=1: bound 7,496.67, RO-08 fractional: branch
      RO-02=1, RO-16=1, RO-08=1: bound 7,481.67, RO-07 fractional: branch
        RO-02=1, RO-16=1, RO-08=1, RO-07=1: doesn't fit
        RO-02=1, RO-16=1, RO-08=1, RO-07=0: bound 7,270.00, RO-13 fractional: branch
          RO-02=1, RO-16=1, RO-08=1, RO-07=0, RO-13=1: bound 7,270.00 is a whole load: new best RO-02, RO-06, RO-08, RO-13, RO-16
          RO-02=1, RO-16=1, RO-08=1, RO-07=0, RO-13=0: bound 7,258.00 <= best 7,270: prune
      RO-02=1, RO-16=1, RO-08=0: bound 7,360.00, RO-13 fractional: branch
        RO-02=1, RO-16=1, RO-08=0, RO-13=1: bound 7,360.00 is a whole load: new best RO-02, RO-06, RO-07, RO-13, RO-16
        RO-02=1, RO-16=1, RO-08=0, RO-13=0: bound 7,348.00 <= best 7,360: prune
    RO-02=1, RO-16=0: bound 7,530.00 is a whole load: new best RO-02, RO-06, RO-07, RO-08, RO-10
  RO-02=0: bound 7,550.00, RO-13 fractional: branch
    RO-02=0, RO-13=1: bound 7,550.00 is a whole load: new best RO-06, RO-07, RO-08, RO-13, RO-16
    RO-02=0, RO-13=0: bound 7,538.00 <= best 7,550: prune
best 7,550: RO-06, RO-07, RO-08, RO-13, RO-16; 15 nodes instead of 65,536 combinations
```

Read the tree from the top. The root bound is $7,556.67 with RO-02 fractional. Taking RO-02 leads, after several branches, to loads worth $7,270, $7,360 and $7,530. Leaving RO-02 behind gives a whole load worth **$7,550** at once, and that ends the search: the last open branch has a bound of $7,538, below $7,550. **15 nodes instead of 65,536 combinations.** The bound is what lets the search skip almost everything.

## 26.2 A textbook branch and bound for any model

The same idea works for any Pyomo model: relax the integer variables, solve the LP, pick a fractional variable $$v = 2.4$$, and branch with $$v \le 2$$ and $$v \ge 3$$ (for a binary at 0.6: $$v = 0$$ and $$v = 1$$). Create `day26/bb.py`:

```python
"""Day 26: a textbook branch and bound for any Pyomo model with binary or integer variables.
No presolve, no cuts, no heuristics: just LP relaxations, branching and pruning, so you can watch the search.

Import it:  from bb import branch_and_bound
"""
import heapq
import itertools
import math

import pyomo.environ as pyo

TOL = 1e-6


def branch_and_bound(m, order="best", max_nodes=10_000, gap=0.0, log=None):
    """Solve m by branch and bound. order: "best" (open node with the best bound first) or "depth" (newest node first).
    Stops when no open node could beat the best plan by more than gap (a fraction of it), or after max_nodes nodes.
    Returns the best value found, the proven bound, the number of nodes solved, and whether the search finished."""
    ints = {v.name: v for v in m.component_data_objects(pyo.Var) if v.is_integer() and not v.fixed}
    start = {n: (v.lb if v.lb is not None else 0, v.ub, v.domain) for n, v in ints.items()}
    for v in ints.values():
        v.domain = pyo.Reals                                 # the LP relaxation: same bounds, any value in between
    obj = next(m.component_data_objects(pyo.Objective, active=True))
    sign = 1 if obj.sense == pyo.maximize else -1             # work as if maximizing: bigger is better
    solver = pyo.SolverFactory("highs")
    best, plan, nodes = -math.inf, None, 0
    gap_pruned = -math.inf                                    # the best bound among nodes dropped only because of the gap
    tie = itertools.count()
    todo = [(0, next(tie), math.inf, {})]                     # (priority, tie-break, parent's bound, {name: (lb, ub)})

    def good_enough(bound):
        nonlocal gap_pruned
        if bound <= best + gap * abs(best) + TOL:
            if bound > best + TOL:
                gap_pruned = max(gap_pruned, bound)
            return True
        return False

    while todo and nodes < max_nodes:
        _, _, parent_bound, fixes = heapq.heappop(todo)
        if good_enough(parent_bound):
            continue                                          # pruned without solving: a better plan is known already
        nodes += 1
        for n, v in ints.items():
            lo, hi = fixes.get(n, start[n][:2])
            v.setlb(lo)
            v.setub(hi)
        res = solver.solve(m, load_solutions=False)
        if res.solver.termination_condition != pyo.TerminationCondition.optimal:
            bound, status = None, "infeasible"
        else:
            m.solutions.load_from(res)
            bound = sign * pyo.value(obj)
            frac = [n for n, v in ints.items() if abs(pyo.value(v) - round(pyo.value(v))) > TOL]
            if good_enough(bound):
                status = "pruned by bound"
            elif not frac:
                best, plan = bound, {n: round(pyo.value(v)) for n, v in ints.items()}
                status = "integer: new best"
            else:                                             # branch on the most fractional variable
                n = max(frac, key=lambda n: (0.5 - abs(pyo.value(ints[n]) % 1 - 0.5), n))
                x = pyo.value(ints[n])
                lo, hi = fixes.get(n, start[n][:2])
                for child in ({**fixes, n: (lo, math.floor(x))}, {**fixes, n: (math.ceil(x), hi)}):
                    k = next(tie)
                    heapq.heappush(todo, (-bound if order == "best" else -k, k, bound, child))
                status = f"branch on {n} = {x:.3f}"
        if log is not None:
            log.append((nodes, len(fixes), "-" if bound is None else round(sign * bound, 2),
                        status, "-" if plan is None else round(sign * best, 2)))
    still_open = [b for _, _, b, _ in todo if not good_enough(b)]     # (also records open nodes within the gap)
    proven = max([best, gap_pruned] + still_open)
    for n, v in ints.items():                                 # put the model back, with the best plan loaded
        v.domain = start[n][2]
        v.setlb(start[n][0])
        v.setub(start[n][1])
        if plan is not None:
            v.set_value(plan[n])
    return {"value": sign * best, "bound": sign * proven, "nodes": nodes, "finished": not any(not good_enough(t[2]) for t in todo)}
```

Two choices shape the search. **Which variable** to branch on (here: the most fractional) and **which node next**: *best first* (the open node with the best bound) proves optimality fastest; *depth first* (the newest node) dives towards whole solutions and keeps few nodes in memory.

```python
# ---------- 26.2 The same search on any Pyomo model ----------
for kg in (None, max_kg):
    for order in ("best", "depth"):
        m = build_knapsack(orders, pallets, max_kg=kg)
        r = branch_and_bound(m, order=order)
        print(f"{'pallets only' if kg is None else 'pallets and weight':<18} {order:>5} first: {r['value']:,.0f} "
              f"in {r['nodes']} nodes")
log = []
branch_and_bound(build_knapsack(orders, pallets, max_kg=max_kg), log=log)
print("first nodes of the pallets-and-weight search (node, fixings, bound, what happened, best so far):")
for row in log[:8]:
    print("   ", row)
```

```text
pallets only        best first: 7,550 in 5 nodes
pallets only       depth first: 7,550 in 17 nodes
pallets and weight  best first: 7,170 in 131 nodes
pallets and weight depth first: 7,170 in 257 nodes
first nodes of the pallets-and-weight search (node, fixings, bound, what happened, best so far):
    (1, 0, 7315.0, 'branch on take[RO-16] = 0.875', '-')
    (2, 1, 7289.97, 'branch on take[RO-03] = 0.312', '-')
    (3, 1, 7307.74, 'branch on take[RO-07] = 0.940', '-')
    (4, 2, 7133.3, 'branch on take[RO-03] = 0.979', '-')
    (5, 2, 7307.14, 'branch on take[RO-14] = 0.490', '-')
    (6, 3, 7296.67, 'branch on take[RO-06] = 0.333', '-')
    (7, 3, 7293.75, 'branch on take[RO-08] = 0.932', '-')
    (8, 4, 7295.82, 'branch on take[RO-03] = 0.073', '-')
```

Best first needs 5 nodes for one limit, 131 for two; depth first 17 and 257. The second limit makes the relaxation much less like a whole load, so the tree grows. Read the log: each node fixes one more variable, and the bound falls slowly from $7,315.

## 26.3 Cutting planes

Instead of branching, you can **cut**: add a constraint that every whole load satisfies but the current fractional solution breaks. For a knapsack, the classic cut is the **cover inequality**: if a set of orders together exceeds the truck, at most all-but-one of them can go.

```python
# ---------- 26.3 A cutting plane ----------
m = build_knapsack(orders, pallets, domain=pyo.Reals)
pyo.SolverFactory("highs").solve(m)
print(f"relaxation {pyo.value(m.value):,.2f}:", {o: round(pyo.value(m.take[o]), 3) for o in m.O if pyo.value(m.take[o]) > 1e-6})
cover = [o for o in m.O if pyo.value(m.take[o]) > 1e-6]
print(f"cover {', '.join(cover)}: {orders.pallets[cover].sum()} pallets > {pallets}, so at most {len(cover) - 1} of them")
m.cover = pyo.Constraint(expr=sum(m.take[o] for o in cover) <= len(cover) - 1)
pyo.SolverFactory("highs").solve(m)
print(f"relaxation with the cut {pyo.value(m.value):,.2f}:",
      {o: round(pyo.value(m.take[o]), 3) for o in m.O if pyo.value(m.take[o]) > 1e-6})
for kg in (None, max_kg):
    for cut in (False, True):
        k = build_knapsack(orders, pallets, max_kg=kg)
        if cut:
            k.cover = pyo.Constraint(expr=sum(k.take[o] for o in cover) <= len(cover) - 1)
        print(f"{'pallets only' if kg is None else 'pallets and weight'}, {'with' if cut else 'without'} the cut: "
              f"{branch_and_bound(k)['nodes']} nodes")
```

```text
relaxation 7,556.67: {'RO-02': 0.667, 'RO-06': 1.0, 'RO-07': 1.0, 'RO-08': 1.0, 'RO-16': 1.0}
cover RO-02, RO-06, RO-07, RO-08, RO-16: 22 pallets > 20, so at most 4 of them
relaxation with the cut 7,550.00: {'RO-06': 1.0, 'RO-07': 1.0, 'RO-08': 1.0, 'RO-13': 1.0, 'RO-16': 1.0}
pallets only, without the cut: 5 nodes
pallets only, with the cut: 1 nodes
pallets and weight, without the cut: 131 nodes
pallets and weight, with the cut: 131 nodes
```

One cut makes the pallets-only relaxation a whole load: **the best one, at the root, with no branching**. With both limits, this one cut doesn't help: the relaxation moves to other fractions and other covers would be needed. Real solvers generate many families of cuts (covers, cliques, mixed-integer rounding, Gomory cuts) in rounds at the root and sometimes deeper. **Branch and cut** = branch and bound + cuts.

## 26.4 What HiGHS adds, and its controls

Now Day 23's capacitated location model (10 binaries for sites and 400 for store assignments):

```python
# ---------- 26.4 What HiGHS adds ----------
sites, stores, lanes = load_sites(con)
m = build_location(sites, stores, lanes)
r = branch_and_bound(m, max_nodes=300)
print(f"Day 23 location, textbook search, 300 nodes: best ${r['value']:,.2f}, bound ${r['bound']:,.2f}, "
      f"gap {100 * (r['value'] - r['bound']) / r['value']:.1f}%")
for opts in ({}, {"presolve": "off"}, {"presolve": "off", "mip_heuristic_effort": 0.0}):
    h = run_highs(build_location(sites, stores, lanes), **opts)
    print(f"Day 23 location, HiGHS {opts or 'defaults'}: ${h['objective']:,.2f} proven in {h['nodes']} node(s)")
```

```text
Day 23 location, textbook search, 300 nodes: best $95,148.96, bound $91,048.80, gap 4.3%
Day 23 location, HiGHS defaults: $92,492.76 proven in 1 node(s)
Day 23 location, HiGHS {'presolve': 'off'}: $92,492.76 proven in 1 node(s)
Day 23 location, HiGHS {'presolve': 'off', 'mip_heuristic_effort': 0.0}: $92,492.76 proven in 1 node(s)
```

After 300 nodes, the textbook search has a plan costing $95,148.96 and a gap of 4.3%. HiGHS proves $92,492.76 at the root, even with presolve and its heuristics turned off: its cuts close the gap. What a modern solver adds to the textbook method:

| Component | What it does |
|---|---|
| Presolve | Removes and tightens variables and rows before the search (fixed variables, redundant rows, tighter bounds and big-Ms) |
| Cuts | Tightens the relaxation at the root (covers, cliques, Gomory, MIR…) |
| Heuristics | Finds good whole plans early (rounding, feasibility jump, RINS, RENS…), so pruning starts early |
| Branching rules | Pseudo-costs: branch on variables that moved the bound most in the past |
| Symmetry handling | Detects interchangeable variables (Day 27) |
| Restarts, conflict analysis | Learns from infeasible nodes, restarts with what was learned |

Not every model falls at the root. The Tuesday before Christmas, Montreal has 48 orders and the thaw limit of 16,000 kg applies:

```python
xmas = con.sql("SELECT * FROM christmas_orders ORDER BY order_id").df().set_index("order_id")
trailer, kg16 = con.sql("SELECT pallet_positions, max_kg FROM truck_spec WHERE truck_type = 'trailer'").fetchall()[0][0], 16000
ffd = first_fit_decreasing(xmas.pallets.to_dict(), trailer, xmas.weight_kg.to_dict(), kg16)
print(f"Christmas Tuesday: {len(xmas)} orders; at least {xmas.pallets.sum() / trailer:.2f} trailers by pallets, "
      f"{xmas.weight_kg.sum() / kg16:.2f} by weight; first fit decreasing {len(ffd)}")
for opts in ({"mip_max_nodes": 200}, {"mip_max_nodes": 2000}, {"mip_rel_gap": 0.07}):
    h = run_highs(build_packing(xmas, trailer, len(ffd), max_kg=kg16), **opts)
    print(f"    HiGHS {opts}: {h['status']}, {h['objective']:.0f} trailers, bound {h['bound']:.2f}, "
          f"gap {h['gap_pct']:.2f}%, {h['nodes']} nodes")
```

```text
Christmas Tuesday: 48 orders; at least 13.27 trailers by pallets, 14.73 by weight; first fit decreasing 16
    HiGHS {'mip_max_nodes': 200}: Solution limit reached, 16 trailers, bound 15.00, gap 6.25%, 244 nodes
    HiGHS {'mip_max_nodes': 2000}: Solution limit reached, 16 trailers, bound 15.00, gap 6.25%, 2093 nodes
    HiGHS {'mip_rel_gap': 0.07}: Optimal, 16 trailers, bound 15.00, gap 6.25%, 1 nodes
```

HiGHS finds 16 trailers at once and proves no plan needs fewer than 15 (the weight bound, 14.73, rounded up), but after 2,000 nodes it still can't tell whether 15 is possible. The **gap** is 1 trailer in 16: 6.25%. Three controls decide when to stop:

- `mip_rel_gap` (default 0.01%): stop when the gap is below this. At 7% HiGHS stops at once and calls 16 "optimal": within tolerance, not proven best.
- `mip_max_nodes` (and `time_limit`): stop after so many nodes (or seconds) and keep the best plan found. Node limits give the same answer on every run; time limits depend on the computer.
- `mip_abs_gap`: the same in the objective's units. For trailers, any gap below 1 means proven.

Day 27 finishes the Christmas problem. (Node counts can differ slightly on your computer, with a different number of threads.)

### The planted bug

`day26/bug_bound.py` speeds up the hand-made search with a "quicker" bound. Run it.

```python
def quick_bound(orders, capacity, fixed):
    room = capacity - sum(orders[o][0] for o, f in fixed.items() if f == 1)
    if room < 0:
        return None, None
    value = 0
    for o in sorted((o for o in orders if o not in fixed), key=lambda o: (-orders[o][1] / orders[o][0], o)):
        pallets, v = orders[o]
        if pallets <= room:
            room -= pallets
            value += v
        else:
            return value + v * room / pallets, (o if room > 0 else None)
    return value, None
```

<details><summary>What happens, and why</summary>

```text
best 7,440: RO-04, RO-06, RO-07, RO-08, RO-10, RO-16; 21 nodes
```

$7,440, the dispatcher's greedy load from Day 24, not the best one. `quick_bound` starts `value` at 0 and forgets the value of the orders already fixed on the truck. Its bounds are too low, so branches containing better loads look worse than the best so far and are pruned. **A bound must never be below what the branch can really reach** (for a maximization). A wrong bound doesn't crash: it silently prunes the optimum. Test a search against a brute-force answer on small cases (Day 9's known-answer tests).

</details>

## Mini-project: how good is good enough?

Run the textbook search on the rush truck with both limits, best first and depth first, stopping (a) at gaps of 0%, 1%, 2% and 5%, and (b) after 10, 25 and 50 nodes. Report nodes, the best load found and the proven bound. What does each stopping rule buy?

<details><summary>Solution</summary>

```python
"""Day 26 mini-project: how good is good enough? Stop the textbook search at a gap, or after a number of nodes,
on the rush truck with both limits.

Run from the repository root (after day06/load_tables.py):  python day26/mini_project_gap.py
"""
import sys

import duckdb
import pandas as pd

for folder in ("day24", "day26"):
    sys.path.insert(0, folder)
from bb import branch_and_bound
from knapsack import build_knapsack

con = duckdb.connect("northbound.duckdb", read_only=True)
orders = con.sql("SELECT * FROM rush_orders ORDER BY order_id").df().set_index("order_id")
pallets, max_kg = con.sql("SELECT pallet_positions, max_kg FROM truck_spec WHERE truck_type = 'straight'").fetchall()[0]
rows = []
for order in ("best", "depth"):
    for gap in (0.0, 0.01, 0.02, 0.05):
        r = branch_and_bound(build_knapsack(orders, pallets, max_kg=max_kg), order=order, gap=gap)
        rows.append((order, f"gap {gap:.0%}", r["nodes"], r["value"], r["bound"]))
    for limit in (10, 25, 50):
        r = branch_and_bound(build_knapsack(orders, pallets, max_kg=max_kg), order=order, max_nodes=limit)
        rows.append((order, f"{limit} nodes", r["nodes"], r["value"], r["bound"]))
t = pd.DataFrame(rows, columns=["search", "stop at", "nodes", "best_found", "proven_bound"])
t["best_found"] = t.best_found.map(lambda v: "none" if v == float("-inf") else f"{v:,.0f}")
t["proven_bound"] = t.proven_bound.round(2)
print(t.to_string(index=False))
```

```text
search  stop at  nodes best_found  proven_bound
  best   gap 0%    131      7,170       7170.00
  best   gap 1%     59      7,170       7241.11
  best   gap 2%     54      7,170       7247.14
  best   gap 5%     54      7,170       7247.14
  best 10 nodes     10       none       7295.82
  best 25 nodes     25       none       7276.20
  best 50 nodes     50       none       7249.17
 depth   gap 0%    257      7,170       7170.00
 depth   gap 1%    137      7,170       7241.11
 depth   gap 2%    115      7,170       7307.74
 depth   gap 5%     67      7,170       7315.00
 depth 10 nodes     10       none       7315.00
 depth 25 nodes     25       none       7315.00
 depth 50 nodes     50      6,690       7315.00
```

- **A gap of 1% halves the work** (59 nodes instead of 131) and here still finds the best load, $7,170. The bound says only that no load beats $7,241: the last 1% is what proof costs.
- **Best first finds no load at all in 50 nodes**: it keeps exploring the most promising fractional nodes. **Depth first** finds a load ($6,690) by node 50, but its bound stays at the root's $7,315. Each search is good at one half of the job: dives find plans, best-first proves them. Real solvers mix both, and add heuristics so a good plan appears in the first seconds.
- For a manager: **always report the plan and the gap** ("this plan costs at most 2% more than the best possible"). An "optimal" plan with a 5% tolerance is a 5%-plan.

</details>

## Commit

```bash
git add .
git commit -m "Day 26: branch and bound, cuts, solver controls"
git push
```

### Day 26 checklist

- [ ] I can run branch and bound by hand: relax, branch, bound, prune.
- [ ] I know best-first and depth-first search, and what each is good at.
- [ ] I can write a cover cut and say why it is valid.
- [ ] I can name what presolve, cuts, heuristics and symmetry handling add to the textbook method.
- [ ] I can set `mip_rel_gap` and node limits, and report a plan with its gap.
- [ ] I know a bound that is too low prunes the optimum silently.

---

# DAY 27 — Strong Formulations and Symmetry

There are many ways to write the same MIP: they have the same whole-number plans, but different relaxations. The tighter the relaxation, the smaller the search tree. Today you measure that with Day 26's textbook search, which hides nothing, and with HiGHS, which hides a lot, until it can't. You compare strong and weak links, big and small M, and symmetric and symmetry-broken packing, and finish the Christmas problem.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 27.1 Strong and weak links |
| 0:45 – 1:45 | 27.2 Big-M, again; 27.3 Symmetry |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 27.4 Capacitated location: four formulations |
| 2:45 – 3:45 | Valid inequalities; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: Christmas Tuesday, solved |
| 4:45 – 5:00 | Commit with Git + checklist |

## 27.1 Strong and weak links

A formulation is **stronger** than another if its relaxation allows fewer fractional points while keeping the same whole-number plans. Its root bound is closer to the best plan, so the search prunes earlier. Create `day27/tighten.py`, with functions that change Module 3's models:

```python
"""Day 27: tighter formulations. Helpers to weaken, tighten and de-symmetrize the Module 3 models.

Import it:  from tighten import weak_link, total_capacity, break_symmetry, root_bound
"""
import pyomo.environ as pyo


def weak_link(m):
    """Facility location with one linking row per site instead of one per site and store:
    sum_j serve[i, j] <= (number of stores) * open[i]. Same integer plans, weaker relaxation."""
    m.del_component(m.only_if_open)
    m.only_if_open = pyo.Constraint(m.I, rule=lambda m, i: sum(m.serve[i, j] for j in m.J) <= len(m.J) * m.open[i])
    return m


def total_capacity(m, sites, stores, margin=1.0):
    """A valid inequality (with margin=1): the sites that open must have room for all the pallets together."""
    m.total_capacity = pyo.Constraint(expr=sum(sites.capacity_pallets[i] * m.open[i] for i in m.I)
                                      >= margin * stores.pallets_per_week.sum())
    return m


def break_symmetry(m):
    """Bin packing: trucks are identical, so (1) use trucks in order: use[k] >= use[k+1]; and
    (2) order number i (counting from 1) goes on one of the first i trucks."""
    trucks = list(m.K)
    m.in_order = pyo.Constraint(trucks[:-1], rule=lambda m, k: m.use[k] >= m.use[k + 1])
    for i, o in enumerate(m.O, start=1):
        for k in trucks:
            if k > i:
                m.put[o, k].fix(0)
    return m


def root_bound(m):
    """The LP relaxation's value: relax every integer variable, solve, then put the domains back."""
    ints = [v for v in m.component_data_objects(pyo.Var) if v.is_integer()]
    saved = [(v.domain, v.lb, v.ub) for v in ints]
    for v, (_, lb, ub) in zip(ints, saved):
        v.domain = pyo.Reals                      # any value between the variable's bounds (0 and 1 for a binary)
        v.setlb(lb)
        v.setub(ub)
    pyo.SolverFactory("highs").solve(m)
    value = pyo.value(next(m.component_data_objects(pyo.Objective, active=True)))
    for v, (d, lb, ub) in zip(ints, saved):
        v.domain = d
        v.setlb(lb)
        v.setub(ub)
    return value
```

`weak_link` replaces Day 23's 400 rows $$s_{ij} \le o_i$$ with 10 rows $$\sum_j s_{ij} \le 40\, o_i$$. Same whole-number plans: if a site is closed, it serves nobody; if open, it may serve all 40 stores. But in the relaxation, $$o_i = 0.1$$ lets a site serve four whole stores. Create `day27/formulations.py`:

```python
def measure(name, make, max_nodes=10_000):
    root = root_bound(make())
    r = branch_and_bound(make(), max_nodes=max_nodes)
    h = run_highs(make())
    found = "none" if abs(r["value"]) == float("inf") else round(r["value"], 2)
    return (name, round(root, 2), found, round(r["bound"], 2), r["nodes"], "yes" if r["finished"] else "no",
            round(h["objective"], 2), h["nodes"])


cols = ["formulation", "root_bound", "textbook_best", "textbook_bound", "textbook_nodes", "finished", "highs", "highs_nodes"]

# ---------- 27.1 Strong and weak links ----------
print("--- uncapacitated location")
rows = [measure("strong: serve[i,j] <= open[i]", lambda: build_location(sites, stores, lanes, capacitated=False)),
        measure("weak: sum_j serve[i,j] <= 40 open[i]", lambda: weak_link(build_location(sites, stores, lanes, capacitated=False)))]
print(pd.DataFrame(rows, columns=cols).to_string(index=False))
```

```text
--- uncapacitated location
                         formulation  root_bound  textbook_best  textbook_bound  textbook_nodes finished   highs  highs_nodes
       strong: serve[i,j] <= open[i]    85450.50        85450.5         85450.5               1      yes 85450.5            1
weak: sum_j serve[i,j] <= 40 open[i]    37636.36        85450.5         85450.5             145      yes 85450.5            1
```

On the uncapacitated model, the strong link's relaxation is **already the best plan** ($85,450.50, 1 node). The weak link's root bound is $37,636.36, less than half, and the textbook search needs 145 nodes. HiGHS solves both at the root: its presolve and cuts rebuild what the weak formulation threw away. On large models they can't always, so write the strong form.

**Fewer rows is not better.** The strong form has 400 linking rows instead of 10, and solves far faster.

## 27.2 Big-M, again

Day 22 showed that a large M weakens the relaxation. Now count the nodes:

```python
# ---------- 27.2 Big-M, again ----------
d = load_data(con)
setup = con.sql("SELECT * FROM bakery_range_setup").df().set_index("sku")
print("--- set-up model (Day 22)")
rows = [measure(f"M = {label}", lambda M=M: build_with_setups(d, setup, big_m=M))
        for label, M in [("demand cap", None), ("100", 100), ("10,000", 10_000), ("1,000,000", 1_000_000)]]
print(pd.DataFrame(rows, columns=cols).to_string(index=False))
```

```text
--- set-up model (Day 22)
   formulation  root_bound  textbook_best  textbook_bound  textbook_nodes finished     highs  highs_nodes
M = demand cap   102372.08      102242.38       102242.38              11      yes 102242.38            1
       M = 100   107076.82      102242.38       102242.38            1065      yes 102242.38            1
    M = 10,000   108338.54      102242.38       102242.38            1557      yes 102242.38            1
 M = 1,000,000   108351.15      102242.38       102242.38            1543      yes 102242.38            1
```

**11 nodes** with M = each product's demand cap; **over 1,000** with M = 100 or more. Same answer every time. HiGHS needs 1 node for all four: its presolve and cuts tighten what a loose M leaves loose. Don't count on that when M multiplies something the solver can't bound.

## 27.3 Symmetry

In bin packing, trailers are identical: any plan can be relabelled by swapping trailer 1 and trailer 3, and the relaxation can't tell the copies apart. The search explores each copy of every dead end. **Symmetry breaking** keeps one copy: use trailers in order ($$u_k \ge u_{k+1}$$), and put order number $$i$$ on one of the first $$i$$ trailers (any plan can be relabelled this way).

```python
# ---------- 27.3 Symmetry ----------
orders = con.sql("SELECT * FROM truck_orders ORDER BY order_id LIMIT 8").df().set_index("order_id")
trailer = con.sql("SELECT pallet_positions FROM truck_spec WHERE truck_type = 'trailer'").fetchall()[0][0]
n = len(first_fit_decreasing(orders.pallets.to_dict(), trailer, orders.weight_kg.to_dict(), 16000))
print(f"--- bin packing: first 8 Tuesday orders, 16,000 kg limit, {n} trucks available")
rows = [measure("trucks interchangeable", lambda: build_packing(orders, trailer, n, max_kg=16000)),
        measure("with symmetry breaking", lambda: break_symmetry(build_packing(orders, trailer, n, max_kg=16000)))]
print(pd.DataFrame(rows, columns=cols).to_string(index=False))
```

```text
--- bin packing: first 8 Tuesday orders, 16,000 kg limit, 4 trucks available
           formulation  root_bound  textbook_best  textbook_bound  textbook_nodes finished  highs  highs_nodes
trucks interchangeable        2.65            3.0             3.0            3581      yes    3.0            1
with symmetry breaking        2.65            3.0             3.0             491      yes    3.0            1
```

With only 8 orders and 4 trailers, the textbook search needs **3,581 nodes** to prove that 3 trailers is the minimum; with symmetry breaking, **491**. HiGHS detects the symmetry itself and solves both at the root.

## 27.4 Capacitated location: four formulations

Back to Day 23's capacitated, single-sourced model, with a second idea: a **valid inequality**. Any whole plan must open enough capacity for all pallets: $$\sum_i \text{capacity}_i\, o_i \ge 3{,}545$$. It cuts off no whole plan, so it is safe to add, and it may cut off fractional ones.

```python
# ---------- 27.4 Capacitated location: four formulations ----------
print("--- capacitated location (Day 23)")
rows = [measure(name, make, max_nodes=300) for name, make in [
    ("weak link", lambda: weak_link(build_location(sites, stores, lanes))),
    ("weak link + total capacity", lambda: total_capacity(weak_link(build_location(sites, stores, lanes)), sites, stores)),
    ("strong link (Day 23)", lambda: build_location(sites, stores, lanes)),
    ("strong link + total capacity", lambda: total_capacity(build_location(sites, stores, lanes), sites, stores))]]
print(pd.DataFrame(rows, columns=cols).to_string(index=False))
```

```text
--- capacitated location (Day 23)
                 formulation  root_bound textbook_best  textbook_bound  textbook_nodes finished    highs  highs_nodes
                   weak link    72158.68          none        90501.30             300       no 92492.76            0
  weak link + total capacity    72158.68          none        90396.27             300       no 92492.76            5
        strong link (Day 23)    89717.62      95148.96        91048.80             300       no 92492.76            1
strong link + total capacity    89717.62      92974.23        90942.86             300       no 92492.76            1
```

- **The weak link's root bound is $72,158.68; the strong link's $89,717.62**, 3% from the best plan.
- In 300 nodes, the textbook search finds **no plan at all** with the weak link. With the strong link it finds $95,148.96; adding the total-capacity inequality, $92,974.23, about 0.5% above the best.
- Here the inequality doesn't move the root bound (the relaxation already opens enough capacity), but it changes the path of the search: with the strong link it found a better plan in the same 300 nodes.
- HiGHS: 1 node with the strong link, up to 5 with the weak one.

### The planted bug

`day27/bug_cut.py` adds the total-capacity inequality "with a 10% safety margin". Run it.

```python
con = duckdb.connect("northbound.duckdb", read_only=True)
sites, stores, lanes = load_sites(con)
m = total_capacity(build_location(sites, stores, lanes), sites, stores, margin=1.10)
pyo.SolverFactory("highs").solve(m)
print(f"optimal: ${pyo.value(m.cost):,.2f} a week = fixed ${pyo.value(m.fixed):,.0f} + transport ${pyo.value(m.transport):,.2f}")
print(summary(m, sites, stores).to_string(index=False))
```

<details><summary>What happens, and why</summary>

```text
optimal: $95,148.96 a week = fixed $64,500 + transport $30,648.96
   site       place  open  stores  pallets  capacity
 DC-MIS Mississauga   1.0      15   1195.0      1600
 DC-MTL    Montreal   1.0      15   1380.0      1500
 DC-OTT      Ottawa   1.0       7    685.0       700
NEW-QUE Quebec City   1.0       3    285.0       700
```

"Optimal", $95,148.96 a week: Quebec City opens, $2,656 a week more than the real best plan. Today's three DCs have 3,800 pallets of capacity, only 7% more than the 3,545 needed, so a 10% margin forbids the best plan. The inequality is not valid: it **changes the problem** instead of tightening it. A cut must hold for every whole plan you would accept. If you want a safety margin, that is a business rule: write it as one, report it as one, and price it (here $2,656 a week).

</details>

## Mini-project: Christmas Tuesday, solved

Day 26 left Christmas Tuesday at 16 trailers with a bound of 15 after 2,000 nodes. Give HiGHS 20,000 nodes and compare three orderings of the orders (as listed, heaviest first, most pallets first), each with and without `break_symmetry`. Then report the plan and what it saves.

<details><summary>Solution</summary>

```python
"""Day 27 mini-project: Christmas Tuesday, solved. Day 26 left it at 16 trailers with a bound of 15.

Run from the repository root (after day06/load_tables.py):  python day27/mini_project_christmas.py
"""
import sys

import duckdb
import pyomo.environ as pyo

for folder in ("day24", "day26", "day27"):
    sys.path.insert(0, folder)
from highs_run import run_highs
from packing import build_packing, first_fit_decreasing, trucks_used
from tighten import break_symmetry

con = duckdb.connect("northbound.duckdb", read_only=True)
orders = con.sql("SELECT * FROM christmas_orders ORDER BY order_id").df().set_index("order_id")
trailer, cost = con.sql("SELECT pallet_positions, cost_per_day FROM truck_spec WHERE truck_type = 'trailer'").fetchall()[0]
n = len(first_fit_decreasing(orders.pallets.to_dict(), trailer, orders.weight_kg.to_dict(), 16000))
for name, sort_by in [("orders as listed", None), ("heaviest first", ["weight_kg", "pallets"]),
                      ("most pallets first", ["pallets", "weight_kg"])]:
    o = orders if sort_by is None else orders.sort_values(sort_by, ascending=False)
    for sym in (False, True):
        m = build_packing(o, trailer, n, max_kg=16000)
        if sym:
            break_symmetry(m)
        h = run_highs(m, mip_max_nodes=20_000)
        print(f"{name:<18} {'symmetry broken' if sym else 'as Day 24':<16} {h['status']:<24} {h['objective']:.0f} trailers, "
              f"bound {h['bound']:.2f}, {h['nodes']:,} nodes")
m = break_symmetry(build_packing(orders.sort_values(["weight_kg", "pallets"], ascending=False), trailer, n, max_kg=16000))
pyo.SolverFactory("highs").solve(m)
plan = trucks_used(m, orders)
print(f"{len(plan)} trailers, ${len(plan) * cost:,} for the day (first fit decreasing: {n}, ${n * cost:,})")
print(plan.to_string(index=False))
```

```text
orders as listed   as Day 24        Optimal                  15 trailers, bound 15.00, 2,161 nodes
orders as listed   symmetry broken  Optimal                  15 trailers, bound 15.00, 8,963 nodes
heaviest first     as Day 24        Optimal                  15 trailers, bound 15.00, 1 nodes
heaviest first     symmetry broken  Optimal                  15 trailers, bound 15.00, 3,575 nodes
most pallets first as Day 24        Optimal                  15 trailers, bound 15.00, 1,642 nodes
most pallets first symmetry broken  Optimal                  15 trailers, bound 15.00, 4,970 nodes
15 trailers, $9,750 for the day (first fit decreasing: 16, $10,400)
 truck  orders  pallets    kg                         order_ids
     1       2       25 15590                      XO-24, XO-14
     2       3       19 15530               XO-22, XO-46, XO-13
     3       3       20 15290               XO-40, XO-19, XO-43
     4       2       22 15830                      XO-27, XO-02
     5       3       20 15690               XO-11, XO-10, XO-23
     6       3       25 15380               XO-03, XO-33, XO-21
     7       3       24 15470               XO-25, XO-20, XO-34
     8       3       20 15840               XO-44, XO-12, XO-16
     9       3       24 15700               XO-05, XO-01, XO-36
    10       3       26 15990               XO-17, XO-41, XO-06
    11       4       22 15680        XO-18, XO-29, XO-47, XO-26
    12       3       25 15910               XO-32, XO-39, XO-45
    13       4       24 15960        XO-30, XO-38, XO-08, XO-48
    14       4       26 15990        XO-28, XO-35, XO-04, XO-07
    15       5       23 15860 XO-42, XO-09, XO-31, XO-37, XO-15
```

- **15 trailers is possible**, and every variant proves it within 20,000 nodes. Day 26's 2,000 nodes stopped just short (the plain model needs 2,161). The plan saves one trailer, **$650 for the day**, over first fit decreasing.
- **Symmetry breaking made HiGHS slower here** (8,963 nodes instead of 2,161 as listed). HiGHS already handles symmetry its own way, and the added rows fight with it. A textbook improvement is not always an improvement for a modern solver: measure.
- **The order of the rows changes everything**: the same model, listed heaviest first, solves at the root; listed by pallets, 1,642 nodes. This is **performance variability**: small, meaningless changes send the solver down different paths. Never judge a formulation on one run of one instance; test several orderings or instances (or `random_seed` values) before concluding.

*Note to transport planning:* "Plan Christmas loads with the model, with a 20,000-node limit: on the Tuesday before Christmas it saves a trailer over the current loading rule."

</details>

## Commit

```bash
git add .
git commit -m "Day 27: formulation strength, symmetry, variability"
git push
```

Then run the tests: `tests/test_search.py` checks Days 24–27 (the knapsack three ways, the textbook search against HiGHS, cutting patterns, the cross-dock model against Day 16, weak and strong links). Add `"day24"` to `"day27"` to the folders in `conftest.py`.

```python
"""Module 3, Days 24-27: knapsack, packing, cutting, cross-docks, branch and bound and formulations."""
import pyomo.environ as pyo
import pytest
from bakery import load_data
from bb import branch_and_bound
from by_hand import branch_and_bound_knapsack
from crossdocks import build_crossdocks, load_crossdock_network
from cutting import build_cutting, patterns
from facility import build_location, load_sites
from knapsack import build_knapsack, dp_knapsack
from setup_model import build_with_setups
from tighten import root_bound, weak_link


@pytest.fixture(scope="module")
def rush(con):
    return con.sql("SELECT * FROM rush_orders ORDER BY order_id").df().set_index("order_id")


def test_knapsack_three_ways_agree(rush):
    m = build_knapsack(rush, 20)
    pyo.SolverFactory("highs").solve(m)
    value, _ = dp_knapsack(rush.pallets.tolist(), rush.value_if_delivered.tolist(), 20)
    hand = branch_and_bound_knapsack({o: (rush.pallets[o], rush.value_if_delivered[o]) for o in rush.index}, 20,
                                     show=lambda line: None)
    assert pyo.value(m.value) == value == hand["value"] == 7_550


def test_textbook_search_matches_highs(con, rush):
    k = build_knapsack(rush, 20, max_kg=12_000)
    assert branch_and_bound(k)["value"] == pytest.approx(7_170)
    d = load_data(con)
    setup = con.sql("SELECT * FROM bakery_range_setup").df().set_index("sku")
    r = branch_and_bound(build_with_setups(d, setup))
    assert r["finished"] and r["value"] == pytest.approx(102_242.38, abs=0.01)


def test_cutting_patterns_fit_and_cover_demand(con):
    bars = con.sql("SELECT * FROM cake_bars").df().set_index("bar")
    lengths, demand = bars.length_cm.to_dict(), bars.bars_per_week.to_dict()
    pats = patterns(lengths, 120)
    assert all(sum(p[b] * lengths[b] for b in p) <= 120 for p in pats)
    m = build_cutting(pats, demand)
    pyo.SolverFactory("highs").solve(m)
    assert pyo.value(m.slabs) == pytest.approx(50)


def test_crossdocks_reproduce_day16_then_improve(con):
    nodes, stores, xds, arcs = load_crossdock_network(con)
    today = ("XD-KIN", "XD-TRV")
    m = build_crossdocks(nodes, xds, arcs, trucks=False, allowed=today, forced=today)
    pyo.SolverFactory("highs").solve(m)
    assert pyo.value(m.cost) == pytest.approx(19_603.64, abs=0.01)          # Day 16's answer
    best = build_crossdocks(nodes, xds, arcs)
    pyo.SolverFactory("highs").solve(best)
    assert pyo.value(best.cost) == pytest.approx(18_757.88, abs=0.01)
    assert [x for x in best.X if pyo.value(best.open[x]) > 0.5] == ["XD-DRU"]


def test_weak_link_has_weaker_bound_same_answer(con):
    sites, stores, lanes = load_sites(con)
    strong = build_location(sites, stores, lanes, capacitated=False)
    weak = weak_link(build_location(sites, stores, lanes, capacitated=False))
    assert root_bound(weak) < root_bound(strong) - 1_000
    for m in (strong, weak):
        pyo.SolverFactory("highs").solve(m)
    assert pyo.value(weak.cost) == pytest.approx(pyo.value(strong.cost), abs=0.01)
```

```text
45 passed
```

### Day 27 checklist

- [ ] I can say what makes one formulation stronger than another, and measure it by the root bound.
- [ ] I prefer per-pair linking rows and the smallest valid M.
- [ ] I can break symmetry in bin packing, and know a modern solver may do it better itself.
- [ ] I can add a valid inequality, and check that it cuts off no plan I would accept.
- [ ] I compare formulations on several runs, because of performance variability.
