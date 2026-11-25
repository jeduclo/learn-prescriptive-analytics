# DAY 38 — Heuristics and Metaheuristics

OR-Tools' routing search (Days 36–37) is a **metaheuristic**: a general strategy for improving plans without proving anything. Today you write two yourself, **local search** and **simulated annealing**, to see what's inside, and learn how to compare heuristics honestly (several seeds, the right objective, a bound when one exists).

| Time | Block |
|---|---|
| 0:00 – 0:45 | 38.1 Local search: neighbourhoods and local optima |
| 0:45 – 1:45 | 38.2 Simulated annealing |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 38.3 Comparing methods; 38.4 the right objective |
| 2:45 – 3:45 | When to use a heuristic; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: Christmas trailers by annealing |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Day 40: `delivery_days` (next week's racks by day), `delivery_drivers` and `driver_rules`.

## 38.1 Local search: neighbourhoods and local optima

A **neighbourhood** is the set of plans one small change away. For van routes, three classic moves:

- **relocate**: move one stop to another position, in its route or another one;
- **swap**: exchange two stops between routes;
- **2-opt**: reverse a piece of one route (Day 36).

**Local search** takes the best neighbour while it is better, and stops when none is: a **local optimum**. Create `day38/metaheuristics.py`:

```python
import math
import random


def route_km(route, km):
    path = [0] + route + [0]
    return sum(km[a][b] for a, b in zip(path, path[1:]))


def total_km(routes, km):
    return sum(route_km(r, km) for r in routes)


def load(route, racks):
    return sum(racks[n] for n in route)


def neighbours(routes, racks, capacity):
    """Every plan one move away: relocate a stop (to any position, in any route), swap two stops in different
    routes, or reverse a piece of one route (2-opt)."""
    n = len(routes)
    for a in range(n):
        for i, stop in enumerate(routes[a]):
            for b in range(n):
                if b != a and load(routes[b], racks) + racks[stop] > capacity:
                    continue
                for j in range(len(routes[b]) + (0 if b == a else 1)):
                    if b == a and j == i:
                        continue
                    new = [r[:] for r in routes]
                    new[a].pop(i)
                    new[b].insert(j, stop)
                    yield "relocate", new
    for a in range(n):
        for b in range(a + 1, n):
            for i, x in enumerate(routes[a]):
                for j, y in enumerate(routes[b]):
                    if (load(routes[a], racks) - racks[x] + racks[y] <= capacity
                            and load(routes[b], racks) - racks[y] + racks[x] <= capacity):
                        new = [r[:] for r in routes]
                        new[a][i], new[b][j] = y, x
                        yield "swap", new
    for a in range(n):
        for i in range(len(routes[a]) - 1):
            for j in range(i + 1, len(routes[a])):
                new = [r[:] for r in routes]
                new[a][i:j + 1] = reversed(new[a][i:j + 1])
                yield "2-opt", new


def local_search(routes, km, racks, capacity, log=None):
    """Best improvement: take the best neighbour while it is shorter. Stops at a local optimum."""
    current, cost = [r[:] for r in routes], total_km(routes, km)
    while True:
        best = min(((total_km(new, km), kind, new) for kind, new in neighbours(current, racks, capacity)),
                   key=lambda x: x[0], default=None)
        if best is None or best[0] >= cost - 1e-9:
            return [r for r in current if r], cost
        cost, kind, current = best
        current = [r for r in current if r]
        if log is not None:
            log.append((kind, round(cost, 2)))


def random_move(routes, racks, capacity, rng):
    """One random relocate, swap or 2-opt move that respects capacity (None if the try fails)."""
    new = [r[:] for r in routes]
    kind = rng.choice(["relocate", "swap", "2-opt"])
    a = rng.randrange(len(new))
    if not new[a]:
        return None
    i = rng.randrange(len(new[a]))
    if kind == "relocate":
        b = rng.randrange(len(new) + 1)                        # len(new): open a new route
        if b == len(new):
            new.append([])
        stop = new[a][i]
        if b != a and load(new[b], racks) + racks[stop] > capacity:
            return None
        new[a].pop(i)
        new[b].insert(rng.randrange(len(new[b]) + 1), stop)
    elif kind == "swap":
        b = rng.randrange(len(new))
        if b == a or not new[b]:
            return None
        j = rng.randrange(len(new[b]))
        x, y = new[a][i], new[b][j]
        if load(new[a], racks) - racks[x] + racks[y] > capacity or load(new[b], racks) - racks[y] + racks[x] > capacity:
            return None
        new[a][i], new[b][j] = y, x
    else:
        j = rng.randrange(len(new[a]))
        i, j = min(i, j), max(i, j)
        new[a][i:j + 1] = reversed(new[a][i:j + 1])
    return [r for r in new if r]


def simulated_annealing(routes, km, racks, capacity, seed=0, start_temp=5.0, cooling=0.9995, steps=40_000,
                        vehicle_km=0.0):
    """Accept every better move, and a worse one with probability exp(-worse / temperature). The temperature
    falls a little each step, so the search wanders early and settles late. vehicle_km: a penalty per route."""
    rng = random.Random(seed)

    def cost(rs):
        return total_km(rs, km) + vehicle_km * len(rs)

    current, current_cost = [r[:] for r in routes], cost(routes)
    best, best_cost = current, current_cost
    temp = start_temp
    for _ in range(steps):
        new = random_move(current, racks, capacity, rng)
        temp *= cooling
        if new is None:
            continue
        delta = cost(new) - current_cost
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current, current_cost = new, current_cost + delta
            if current_cost < best_cost - 1e-9:
                best, best_cost = [r[:] for r in current], current_cost
    return best, total_km(best, km)
```

Then `day38/run_heuristics.py`:

```python
# ---------- 38.1 Local search ----------
for name, start in [("sweep", sweep(stops, ids, racks, VAN)), ("savings", savings(km, racks, VAN))]:
    log = []
    routes, cost = local_search(start, km, racks, VAN, log=log)
    print(f"{name}: {total_km(start, km):.2f} km -> local search {cost:.2f} km after {len(log)} moves "
          f"({', '.join(sorted({k for k, _ in log})) or 'none: already a local optimum'})")
    rows += [(name, len(start), round(total_km(start, km), 2)), (f"{name} + local search", len(routes), round(cost, 2))]
```

```text
sweep: 344.29 km -> local search 276.89 km after 21 moves (2-opt, relocate, swap)
savings: 266.93 km -> local search 266.93 km after 0 moves (none: already a local optimum)
```

From the dispatcher's sweep routes, 21 moves cut 344 km to 277 km and two vans. From the savings routes, **no single move helps**: they are already a local optimum, yet 3% longer than OR-Tools' 258 km. Local search stops at the first valley it finds.

## 38.2 Simulated annealing

**Simulated annealing** (Kirkpatrick and others, 1983) escapes valleys by sometimes accepting a worse plan: a move that adds $$\Delta$$ km is accepted with probability $$e^{-\Delta / T}$$. The **temperature** $$T$$ starts high (the search wanders) and falls step by step (the search settles). The name comes from cooling metal slowly so its atoms settle into a strong structure.

```python
# ---------- 38.2 Simulated annealing ----------
start = savings(km, racks, VAN)
for steps in (40_000, 200_000):
    results = [float(simulated_annealing(start, km, racks, VAN, seed=s, steps=steps, cooling=1e-4 ** (1 / steps))[1])
               for s in range(10)]
    print(f"annealing from savings, {steps:,} steps, 10 seeds: best {min(results):.2f}, median "
          f"{statistics.median(results):.2f}, worst {max(results):.2f} km")
    rows.append((f"annealing, {steps:,} steps (best of 10 seeds)", 10, round(min(results), 2)))
    rows.append((f"annealing, {steps:,} steps (median of 10)", 10, round(statistics.median(results), 2)))
routes = solve_cvrp(km, racks, [VAN] * 15, solutions=1000)
rows.append(("OR-Tools, 1,000 solutions (Day 36)", len(routes), round(total_km(routes, km), 2)))
```

```text
annealing from savings, 40,000 steps, 10 seeds: best 258.30, median 266.93, worst 266.93 km
annealing from savings, 200,000 steps, 10 seeds: best 258.30, median 260.15, worst 266.93 km
```

- The best of 10 seeds reaches **258.30 km**, the same as OR-Tools.
- The **median** matters as much as the best: with 40,000 steps more than half the seeds never improve on the savings plan; with 200,000 the median is 260.15 km.
- A heuristic's result is a distribution. Report the best and the spread over several seeds, and fix the seeds so others can repeat the runs.

## 38.3 Comparing methods

```python
# ---------- 38.3 Comparing methods, and a question for an exact model ----------
print(pd.DataFrame(rows, columns=["method", "vans", "km"]).to_string(index=False))
van_day, van_km = con.sql("SELECT cost_per_day, cost_per_km FROM delivery_fleet WHERE vehicle_type = 'van'").fetchall()[0]
m = cp_model.CpModel()                                          # can 9 vans carry the racks at all?
x = {(n, v): m.new_bool_var(f"x{n}_{v}") for n in range(1, len(ids)) for v in range(9)}
for n in range(1, len(ids)):
    m.add_exactly_one(x[n, v] for v in range(9))
for v in range(9):
    m.add(sum(racks[n] * x[n, v] for n in range(1, len(ids))) <= VAN)
s = cp_model.CpSolver()
s.parameters.num_workers = 8
s.parameters.interleave_search = True
print(f"can 9 vans of {VAN} racks carry the {sum(racks)} racks (ignoring the roads)? {s.status_name(s.solve(m))}")
```

```text
                                     method  vans     km
                                      sweep    12 344.29
                       sweep + local search    10 276.89
                                    savings    10 266.93
                     savings + local search    10 266.93
 annealing, 40,000 steps (best of 10 seeds)    10 258.30
     annealing, 40,000 steps (median of 10)    10 266.93
annealing, 200,000 steps (best of 10 seeds)    10 258.30
    annealing, 200,000 steps (median of 10)    10 260.15
         OR-Tools, 1,000 solutions (Day 36)    10 258.30
can 9 vans of 18 racks carry the 161 racks (ignoring the roads)? OPTIMAL
```

Every method uses 10 vans, but a quick exact model of the packing alone says **9 vans can carry the racks**. None of the heuristics looked for 9 vans, because the objective was kilometres, and the tenth van makes the routes shorter.

## 38.4 The right objective

```python
# ---------- 38.4 The right objective: a van costs a day charge too ----------
penalty = van_day / van_km                                       # one more van costs as much as this many km
print(f"a van's day charge ${van_day:.0f} = {penalty:.0f} km at ${van_km:.2f}/km")
best = None
for seed in range(10):
    r, k = simulated_annealing(start, km, racks, VAN, seed=seed, steps=200_000, cooling=1e-4 ** (1 / 200_000),
                               vehicle_km=penalty)
    cost = van_day * len(r) + van_km * float(k)
    if best is None or cost < best[0]:
        best = (cost, len(r), float(k), seed)
print(f"annealing on cost (best of 10 seeds): {best[1]} vans, {best[2]:.2f} km, ${best[0]:,.2f} a day (seed {best[3]})")
routes = solve_cvrp(km, racks, [VAN] * 15, solutions=1000, fixed_cost=1000 * penalty)
print(f"OR-Tools on cost: {len(routes)} vans, {total_km(routes, km):.2f} km, "
      f"${van_day * len(routes) + van_km * total_km(routes, km):,.2f} a day")
print(f"annealing on km (38.2): 10 vans, 258.30 km, ${van_day * 10 + van_km * 258.30:,.2f} a day")
```

```text
a van's day charge $90 = 225 km at $0.40/km
annealing on cost (best of 10 seeds): 9 vans, 261.68 km, $914.67 a day (seed 7)
OR-Tools on cost: 9 vans, 265.02 km, $916.01 a day
annealing on km (38.2): 10 vans, 258.30 km, $1,003.32 a day
```

A van costs $90 a day, as much as 225 km of driving. With that in the objective, annealing finds **9 vans and 261.68 km: $914.67 a day, against $1,003.32** for the shortest 10-van plan. OR-Tools, given the same fixed cost, finds 9 vans too. The best heuristic is worthless if it optimizes the wrong thing.

### When to use a heuristic

| Use a heuristic when | Use an exact method when |
|---|---|
| The problem is large or messy (thousands of stops, odd rules) | The problem fits a MIP or CP model and solves in time |
| A good plan now beats a proven plan later | A proof matters (contracts, audits, big money) |
| You need many plans quickly (what-ifs, every hour) | You need duals, bounds or gaps |

Often the answer is both: an exact model for a piece (38.3's packing check), a heuristic for the whole, and a bound to say how good the heuristic is.

> **Decision it supports:** good plans for problems too large or messy for exact methods.
>
> **Other decisions you can support the same way:** timetables for a whole university · chip layouts · warehouse slotting · large crew schedules.

### The planted bug

`day38/bug_capacity.py`: a colleague's "faster" relocate move for annealing. Run it.

```python
def relocate(routes, rng):
    new = [r[:] for r in routes]
    a, b = rng.randrange(len(new)), rng.randrange(len(new))
    if a == b or not new[a]:
        return None
    stop = new[a].pop(rng.randrange(len(new[a])))
    if load(new[a], racks) + racks[stop] > 18:                 # does the stop fit?
        return None
    new[b].insert(rng.randrange(len(new[b]) + 1), stop)
    return [r for r in new if r]
```

<details><summary>What happens, and why</summary>

```text
annealing result: 5 vans, 254.49 km
racks per van: [22, 37, 24, 51, 27]
```

Five vans and 254 km: better than anything so far, and impossible. One van carries 51 racks. The capacity check looks at route `a` *after removing the stop* instead of route `b`, which receives it, so every move passes. A heuristic you write yourself has no solver checking its constraints: **check every result independently** (`load` of every route), as on Days 32 and 37.

</details>

## Mini-project: Christmas trailers by annealing

Days 26–28: HiGHS needed up to 16,000 nodes to prove that Christmas Tuesday fits on 15 trailers. Write an annealing search that tries to load the 48 orders on exactly 15 trailers, minimizing the overflow (pallets over 26 plus tonnes over 16). Run 10 seeds. Who proves 15 is the best?

<details><summary>Solution</summary>

```python
def anneal_packing(pallets, kg, n_trucks, start, cap=26, max_kg=16000, seed=0, steps=200_000, start_temp=2.0):
    """Simulated annealing for loading orders on exactly n_trucks trailers. The cost is how far the trailers are
    over their limits (pallets over 26, plus tonnes over 16); a cost of 0 means every trailer is legal.
    start: {order: trailer}. Returns (assignment, overflow, steps used)."""
    rng = random.Random(seed)
    orders = sorted(pallets)
    put = dict(start)
    lp, lw = [0] * n_trucks, [0] * n_trucks
    for o, k in put.items():
        lp[k] += pallets[o]
        lw[k] += kg[o]

    def overflow():
        return sum(max(0, p - cap) + max(0, (w - max_kg) / 1000) for p, w in zip(lp, lw))

    current, temp = overflow(), start_temp
    cooling = 1e-4 ** (1 / steps)
    for step in range(steps):
        if current == 0:
            return put, 0.0, step
        temp *= cooling
        if rng.random() < 0.5:                               # move one order to another trailer
            o = rng.choice(orders)
            a, b = put[o], rng.randrange(n_trucks)
            if a == b:
                continue
            changes = [(a, -pallets[o], -kg[o]), (b, pallets[o], kg[o])]
            swap = {o: b}
        else:                                                # swap two orders on different trailers
            o1, o2 = rng.sample(orders, 2)
            a, b = put[o1], put[o2]
            if a == b:
                continue
            dp, dw = pallets[o2] - pallets[o1], kg[o2] - kg[o1]
            changes = [(a, dp, dw), (b, -dp, -dw)]
            swap = {o1: b, o2: a}
        for k, dp, dw in changes:
            lp[k] += dp
            lw[k] += dw
        new = overflow()
        if new <= current or rng.random() < math.exp(-(new - current) / temp):
            current = new
            put.update(swap)
        else:
            for k, dp, dw in changes:
                lp[k] -= dp
                lw[k] -= dw
    return put, current, steps
```

```python
"""Day 38 mini-project: Christmas Tuesday's trailers (Days 26-28) by simulated annealing.
HiGHS needed up to 16,000 nodes to prove 15 trailers. Can a heuristic find 15, and who proves it is the best?

Run from the repository root (after day06/load_tables.py):  python day38/mini_project_trailers.py
"""
import math
import sys

import duckdb

for folder in ("day24", "day38"):
    sys.path.insert(0, folder)
from metaheuristics import anneal_packing
from packing import first_fit_decreasing

con = duckdb.connect("northbound.duckdb", read_only=True)
orders = con.sql("SELECT * FROM christmas_orders ORDER BY order_id").df().set_index("order_id")
pallets, kg = orders.pallets.to_dict(), orders.weight_kg.to_dict()
ffd = first_fit_decreasing(pallets, 26, kg, 16000)
bound = max(math.ceil(sum(pallets.values()) / 26), math.ceil(sum(kg.values()) / 16000))
print(f"first fit decreasing: {len(ffd)} trailers; no plan can use fewer than {bound} (the weight bound)")
start = {}
for k, (_, _, on) in enumerate(ffd):                  # FFD's plan, with the 16th trailer's orders spread over the 15
    for o in on:
        start[o] = k if k < bound else 0
for seed in range(10):
    put, overflow, steps = anneal_packing(pallets, kg, bound, start, seed=seed)
    loads = [(sum(pallets[o] for o in put if put[o] == k), sum(kg[o] for o in put if put[o] == k)) for k in range(bound)]
    legal = all(p <= 26 and w <= 16000 for p, w in loads)
    print(f"seed {seed}: {'15 legal trailers' if legal else f'overflow {overflow:.2f}'} after {steps:,} steps; "
          f"fullest {max(p for p, _ in loads)} pallets, heaviest {max(w for _, w in loads):,} kg")
```

```text
first fit decreasing: 16 trailers; no plan can use fewer than 15 (the weight bound)
seed 0: 15 legal trailers after 85,197 steps; fullest 26 pallets, heaviest 16,000 kg
seed 1: 15 legal trailers after 72,655 steps; fullest 26 pallets, heaviest 15,990 kg
seed 2: 15 legal trailers after 84,952 steps; fullest 24 pallets, heaviest 16,000 kg
seed 3: 15 legal trailers after 62,692 steps; fullest 26 pallets, heaviest 15,960 kg
seed 4: 15 legal trailers after 71,783 steps; fullest 25 pallets, heaviest 15,970 kg
seed 5: 15 legal trailers after 88,185 steps; fullest 26 pallets, heaviest 15,970 kg
seed 6: 15 legal trailers after 68,034 steps; fullest 26 pallets, heaviest 15,990 kg
seed 7: 15 legal trailers after 70,383 steps; fullest 26 pallets, heaviest 15,990 kg
seed 8: 15 legal trailers after 72,439 steps; fullest 26 pallets, heaviest 15,990 kg
seed 9: 15 legal trailers after 62,597 steps; fullest 26 pallets, heaviest 15,950 kg
```

- **Every seed finds 15 legal trailers**, each in well under a second (62,000 to 88,000 steps).
- Annealing can't prove anything, but it doesn't need to: the **weight bound** (Day 24) says no plan uses fewer than 15 trailers. A heuristic plan that meets a bound is **proven optimal**.
- The lesson for hard MIPs: give the solver a good plan as a warm start (Day 28's checklist), or find one by heuristic and check it against a bound.

</details>

## Commit

```bash
git add .
git commit -m "Day 38: local search and simulated annealing"
git push
```

### Day 38 checklist

- [ ] I can write a local search with relocate, swap and 2-opt moves.
- [ ] I can write simulated annealing and choose a starting temperature and cooling.
- [ ] I report a heuristic's best and spread over several fixed seeds.
- [ ] I put the real costs (day charges, not only km) into a heuristic's objective.
- [ ] I check every heuristic plan independently.
- [ ] I prove a heuristic plan optimal when it meets a bound.

---

# DAY 39 — Decomposition and Rolling Horizons

Real planning problems are too big, too uncertain or too slow-changing to solve in one piece. Two ways of splitting them run through every planning system: **decomposition** (solve parts separately, then together: cluster first, route second) and **rolling horizons** (plan ahead, carry out the first period, plan again with what you now know). Today you measure what each costs, and how to re-plan when something changes without upsetting everyone.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 39.1 Cluster first, route second |
| 0:45 – 1:45 | 39.2 Rolling horizons: how far to look ahead |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 39.3 Why short horizons lose |
| 2:45 – 3:45 | State; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: three people off sick |
| 4:45 – 5:00 | Commit with Git + checklist |

## 39.1 Cluster first, route second

Split vehicle routing in two: first decide **which stops share a vehicle** (a packing problem), then the **order** within each group (a small TSP, solved exactly). Create `day39/decompose.py`:

```python
"""Day 39: splitting big plans into smaller ones. Cluster-first route-second for the vans, and rolling
horizons for the 12-week bakery plan (Day 8).

Import it:  from decompose import route_clusters, rolling_bakery, evaluate_bakery
"""
import numpy as np
import pyomo.environ as pyo

from planner import build, load_plan_data, solve
from routing import tour_km, tsp_exact


def route_clusters(clusters, km):
    """Route second: the shortest tour through each cluster, proven with CP-SAT (Day 36)."""
    routes = []
    for c in clusters:
        nodes = [0] + list(c)
        sub = np.array([[km[a][b] for b in nodes] for a in nodes])
        tour, status, _ = tsp_exact(sub)
        assert status == "OPTIMAL"
        routes.append([nodes[i] for i in tour])
    return routes


def rolling_bakery(con, lookahead, weeks=12, fix=("make", "sell", "keep")):
    """Plan weeks t .. t+lookahead-1, keep only week t's decisions, move on. Earlier weeks stay as decided:
    the variables named in `fix` are fixed to the values already chosen."""
    decided = {}
    for t in range(1, weeks + 1):
        horizon = min(t + lookahead - 1, weeks)
        d = load_plan_data(con, weeks=horizon)
        m = build(d)
        for (name, p, w), value in decided.items():
            if name in fix:
                getattr(m, name)[p, w].fix(value)
        solve(m)
        for name in ("make", "sell", "keep"):
            for p in m.P:
                decided[name, p, t] = max(0.0, round(pyo.value(getattr(m, name)[p, t]), 9))   # no -1e-14
        last = m
    return decided, last


def evaluate_bakery(con, decided, weeks=12):
    """The real profit of a 12-week plan: the full model with every decision fixed."""
    d = load_plan_data(con, weeks=weeks)
    m = build(d)
    for (name, p, w), value in decided.items():
        getattr(m, name)[p, w].fix(value)
    solve(m)
    return pyo.value(m.profit)
```

Then `day39/run_decomposition.py`:

```python
# ---------- 39.1 Cluster first, route second ----------
stops, ids, km, minutes = load_run(con)
racks = [int(stops.racks[s]) for s in ids]
rows = []
for name, clusters in [("sweep", sweep(stops, ids, racks, 18)), ("savings", savings(km, racks, 18)),
                       ("OR-Tools (Day 36)", solve_cvrp(km, racks, [18] * 15, solutions=1000))]:
    routed = route_clusters(clusters, km)
    rows.append((name, len(clusters), round(total_km(clusters, km), 2), round(total_km(routed, km), 2)))
print(pd.DataFrame(rows, columns=["clusters_from", "vans", "km_as_found", "km_each_cluster_routed_optimally"])
      .to_string(index=False))
```

```text
    clusters_from  vans  km_as_found  km_each_cluster_routed_optimally
            sweep    12       344.29                            330.54
          savings    10       266.93                            266.93
OR-Tools (Day 36)    10       258.30                            258.30
```

- Re-routing the sweep's clusters optimally saves 14 km, but they stay 72 km worse than OR-Tools: **the clusters decide most of the cost**.
- The savings and OR-Tools routes are already optimal within their clusters. Their difference is all in the grouping.

Decomposition makes each part easy, but the parts must be cut where the problem is loosely connected. Here, clustering and routing are tightly connected, so a joint method (OR-Tools, annealing) does better.

## 39.2 Rolling horizons: how far to look ahead

Day 8's 12-week bakery plan in one model is the benchmark. A **rolling** plan sees only `lookahead` weeks at a time: plan weeks $$t$$ to $$t + L - 1$$, keep week $$t$$, move to $$t+1$$. Everything already decided stays fixed (`rolling_bakery` fixes make, sell and keep for past weeks).

```python
# ---------- 39.2 Rolling horizons ----------
d = load_plan_data(con, weeks=12)
full = build(d)
solve(full)
best = pyo.value(full.profit)
print(f"12 weeks in one model: ${best:,.2f}")
rows = []
for lookahead in (1, 2, 3, 4, 6, 12):
    decided, _ = rolling_bakery(con, lookahead)
    profit = evaluate_bakery(con, decided)
    rows.append((lookahead, round(profit, 2), round(best - profit, 2) + 0.0))
print(pd.DataFrame(rows, columns=["weeks_seen_each_time", "profit_over_12_weeks", "lost_against_one_model"])
      .to_string(index=False))
```

```text
12 weeks in one model: $1,229,959.05
 weeks_seen_each_time  profit_over_12_weeks  lost_against_one_model
                    1            1227461.02                 2498.03
                    2            1229276.87                  682.18
                    3            1229700.22                  258.82
                    4            1229942.29                   16.75
                    6            1229956.59                    2.45
                   12            1229959.05                    0.00
```

Seeing one week at a time loses **$2,498** over 12 weeks; two weeks $682; **four weeks only $17**; six weeks $2. Beyond a few weeks, more lookahead buys almost nothing here, because what the bakery does now affects the next 2 to 4 weeks (shelf life up to 4 weeks, the Christmas peak). A good lookahead is a little longer than the effects of today's decisions.

## 39.3 Why short horizons lose

```python
# ---------- 39.3 Why short horizons lose: the end of the window ----------
decided, _ = rolling_bakery(con, 1)
stock = {t: sum(v for (name, p, w), v in decided.items() if name == "keep" and w == t) for t in range(1, 13)}
plan = {t: sum(pyo.value(full.keep[p, t]) for p in full.P) for t in range(1, 13)}
print("batches kept at the end of each week (one week seen / one model):")
print("   ", " ".join(f"w{t}:{stock[t]:.0f}/{plan[t]:.0f}" for t in range(1, 13)))
```

```text
batches kept at the end of each week (one week seen / one model):
    w1:0/62 w2:0/120 w3:0/140 w4:0/0 w5:0/0 w6:0/9 w7:0/26 w8:0/14 w9:0/7 w10:0/6 w11:0/25 w12:0/0
```

With one week in view, the plan never stores anything: stock at the end of the window is worth nothing to the model. The full plan builds up 140 batches by the end of week 3, ahead of the Christmas peak. This is the **end-of-horizon effect**. Fixes: look further ahead, or give leftover stock a value at the end of the window.

> **Decision it supports:** how far ahead to plan, and how often to re-plan.
>
> **Other decisions you can support the same way:** production planning every week · supply plans every month · rosters every period · energy dispatch every hour.

### State

What a rolling plan carries from one period to the next is its **state**: stock on hand, who worked last Sunday (Day 32), nights already worked this month (Day 35), trucks already on the road. Get the state wrong and the next plan is built on a world that doesn't exist.

### The planted bug

`day39/bug_history.py` re-plans "everything up to the look-ahead each week, to stay flexible". Run it.

```python
decided, last = rolling_bakery(con, 2, fix=())                 # nothing already decided is kept
try:
    print(f"profit: ${evaluate_bakery(con, decided):,.2f}")
except RuntimeError as error:
    print("the 12-week plan, checked:", error)
```

<details><summary>What happens, and why</summary>

```text
the 12-week plan, checked: no optimal plan: infeasible
5 product-weeks sell more than they have, e.g.
    week 3 BK-27: sells 47.20 batches, had 17.67 in store + made 18.27
    week 7 BK-18: sells 27.90 batches, had 2.32 in store + made 23.05
    week 8 BK-18: sells 27.50 batches, had 2.54 in store + made 17.32
    week 9 BK-27: sells 25.40 batches, had 0.00 in store + made 24.64
```

Nothing already decided is kept, so each week's plan re-decides the past. Week 7's plan assumes pastries (BK-18) were stored in week 6 that the week-6 plan didn't keep. Put together, the 12 weeks are **infeasible**: in 5 product-weeks the bakery sells more than it has. The past is not a decision. Fix what has happened, and plan only what hasn't.

</details>

## Mini-project: three people off sick

December's roster (Day 35) is published. On the Sunday before week 2, three people report sick for weeks 2 and 3: E07 (a forklift driver), E19 and E23. Week 1 has been worked. Re-plan weeks 2 to 4 week by week, (a) freely, (b) charging $25 for every shift that differs from the published roster. Compare pay and how many shifts change.

<details><summary>Solution</summary>

Day 35's `build_four_weeks` takes two more options for this: `unavailable` (employee and week) and `previous` with `change_cost` (a penalty for each shift added or removed against an earlier roster). `rolling` can start at any week, with the nights worked so far.

```python
"""Day 39 mini-project: December's roster is published (Day 35). On the Sunday before week 2, three people
report sick for weeks 2 and 3: E07 (forklift), E19 and E23 (pickers). Re-plan weeks 2-4, week by week,
keeping week 1 as worked. Compare a free re-plan with one that pays $25 for every shift it changes.

Run from the repository root (after day06/load_tables.py):  python day39/mini_project_sick.py   (about 1.5 minutes)
"""
import sys

import duckdb
import pandas as pd

for folder in ("day32", "day35"):
    sys.path.insert(0, folder)
from four_weeks import rolling
from roster import load_roster_data

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
staff, shifts, coverage, rules, last_sunday = load_roster_data(con)
requests = con.sql("SELECT * FROM staff_requests").df()
december = {"peak": {4: 0.10}}
published, roster = rolling(staff, shifts, coverage, rules, last_sunday, requests=requests, **december)
print(f"published December roster: ${published['pay']:,.2f}, {published['agency_shifts']} agency shifts")

week1 = roster[roster.week == 1]
sunday = week1[week1.day == "Sun"].set_index("employee_id")["shift"].to_dict()
after_week1 = {e: sunday.get(e, "off") for e in staff.index}
nights = week1[week1["shift"] == "N"].groupby("employee_id").size().to_dict()
pay_week1 = rolling(staff, shifts, coverage, rules, last_sunday, weeks=1, requests=requests, **december)[0]["pay"]
sick = {(e, w) for e in ("E07", "E19", "E23") for w in (2, 3)}
published_later = roster[roster.week >= 2]
lost = published_later[[(e, w) in sick for e, w in zip(published_later.employee_id, published_later.week)]]
print(f"shifts the sick staff were due to work in weeks 2-3: {len(lost)}")
rows = []
for name, cost in [("free re-plan", 0.0), ("re-plan, $25 a changed shift", 25.0)]:
    out, new = rolling(staff, shifts, coverage, rules, after_week1, first=2, nights=nights, unavailable=sick,
                       previous=roster, change_cost=cost, **december)
    rows.append((name, round(pay_week1 + out["pay"], 2), out["agency_shifts"], out["changes"], out["most_nights"]))
print(pd.DataFrame(rows, columns=["plan", "december_pay", "agency_shifts_weeks_2_4", "shifts_changed", "most_nights"])
      .to_string(index=False))
```

```text
published December roster: $125,861.40, 8 agency shifts
shifts the sick staff were due to work in weeks 2-3: 30
                        plan  december_pay  agency_shifts_weeks_2_4  shifts_changed  most_nights
                free re-plan     132179.96                        8             682            8
re-plan, $25 a changed shift     132219.96                        8              62            8
```

- Covering 30 lost shifts raises December's pay by about **$6,300** (sick full-timers are still paid their guaranteed shifts in this model, and their shifts go to part-timers and overtime).
- A **free re-plan changes 682 shifts**: the solver doesn't care which of many equally cheap rosters it returns, so most of the published roster changes.
- **Charging $25 a change** keeps all but **62 shifts** as published, for **$40 more**. That is the price of a stable plan: almost nothing.

*Memo to the DC manager:* "When someone falls sick, re-plan with a change penalty. Covering this week's three absences costs about $6,300 either way; with the penalty, only 62 shifts change instead of 682, for $40. Publish only the changed shifts."

</details>

## Commit

```bash
git add .
git commit -m "Day 39: decomposition and rolling horizons"
git push
```

### Day 39 checklist

- [ ] I can split routing into clustering and routing, and say when that loses.
- [ ] I can run a rolling horizon and measure the cost of a short lookahead.
- [ ] I can explain the end-of-horizon effect and how to soften it.
- [ ] I carry the state correctly from one period to the next.
- [ ] I can re-plan with a penalty for changes, to keep plans stable.

---

# DAY 40 — Capstone: Next Week's Roster and Routes

Module 4's capstone puts it together for next week, Monday to Saturday: the morning bread routes for each day (Days 36–38), the drivers who drive them (Days 31–33), and the warehouse roster (Days 32–33), with a weekly budget, one what-if, and a memo that also settles the questions left open on Days 34 to 39.

| Time | Block |
|---|---|
| 0:00 – 0:30 | 40.1 The brief |
| 0:30 – 1:30 | 40.1 Routes for every day |
| 1:30 – 1:45 | Break |
| 1:45 – 2:45 | 40.2 The drivers' roster |
| 2:45 – 3:15 | 40.3 The warehouse roster; 40.4 the budget and a what-if |
| 3:15 – 3:30 | Break |
| 3:30 – 4:30 | 40.5 The memo |
| 4:30 – 5:00 | Module wrap-up, commit, tag |

## 40.1 The brief

- **Routes:** each day's racks are the usual racks times the day's factor (`delivery_days`: 0.90 on Monday to 1.20 on Friday). Today's fleet, windows and downtown rule (Day 37).
- **Drivers** (`delivery_drivers`): 7 full-timers (4 or 5 days a week), 4 part-timers (at most 3 days); 4 hold a truck licence. A driver's day is the route plus 30 minutes of checks, paid at least 4 hours.
- **Warehouse:** Day 33's roster with the union's proposal (at most 2 nights, at most 1 weekend day).

Create `day40/next_week.py`:

```python
"""Day 40 capstone: next week's bread run, Monday to Saturday, and the drivers who drive it.

Import it:  from next_week import day_stops, route_week, build_driver_roster, solve_drivers
"""
import pandas as pd
from ortools.sat.python import cp_model

from vrptw import simulate, solve_vrptw


def day_stops(stops, factor):
    """That day's racks: the usual racks times the day's factor, at least one rack per stop."""
    s = stops.copy()
    s["racks"] = [0 if kind == "depot" else max(1, round(r * factor)) for kind, r in zip(s.kind, s.racks)]
    return s


def route_week(stops, ids, km, minutes, fleet, days, **kw):
    """Route every day (Day 37's model). Returns one row per route, and one summary row per day."""
    routes, summary = [], []
    for day, factor in days:
        s = day_stops(stops, factor)
        plan = solve_vrptw(s, ids, km, minutes, fleet, **kw)
        cost, total_km, problems = simulate(plan, s, ids, km, minutes, fleet, window_shift=kw.get("window_shift", 0))
        for k, r in enumerate(plan, start=1):
            routes.append((day, f"{day}-{k}", r["vehicle"], r["type"], r["leaves"], r["back"], len(r["stops"])))
        summary.append((day, int(s.racks.sum()), len(plan), sum(r["type"] == "truck" for r in plan), round(total_km, 2),
                        cost, len(problems)))
    return (pd.DataFrame(routes, columns=["day", "route", "vehicle", "type", "leaves", "back", "stops"]),
            pd.DataFrame(summary, columns=["day", "racks", "vehicles", "trucks", "km", "fleet_cost", "problems"]))


def build_driver_roster(routes, drivers, rules):
    """drive[d, route] = 1 if driver d drives that route. Every route gets one driver with the right licence;
    one route a day; full-timers work 4 or 5 days, part-timers at most 3. A driver's day is the route plus 30
    minutes of checks, paid at least 4 hours. Minimize the week's driver pay (cents)."""
    m = cp_model.CpModel()
    extra, minimum = rules["pre_and_post_trip_minutes"], 60 * rules["min_paid_hours_per_day"]
    drive, pay = {}, []
    for d in drivers.index:
        for r in routes.itertuples():
            if r.type == "truck" and not drivers.truck_licence[d]:
                continue
            drive[d, r.route] = x = m.new_bool_var(f"{d}_{r.route}")
            paid_minutes = max(minimum, r.back - r.leaves + extra)
            pay.append(int(round(100 * drivers.hourly_wage[d] * paid_minutes / 60)) * x)
    for r in routes.itertuples():
        m.add_exactly_one(drive[d, r.route] for d in drivers.index if (d, r.route) in drive)
    for d in drivers.index:
        days = []
        for day, g in routes.groupby("day", sort=False):
            works = m.new_bool_var(f"{d}_works_{day}")
            m.add(sum(drive[d, r] for r in g.route if (d, r) in drive) == works)    # at most one route a day
            days.append(works)
        if drivers.contract[d] == "full-time":
            m.add(sum(days) >= int(rules["full_time_min_days"]))
            m.add(sum(days) <= int(rules["full_time_max_days"]))
        else:
            m.add(sum(days) <= int(rules["part_time_max_days"]))
    m.minimize(sum(pay))
    return m, drive


def solve_drivers(m, drive, routes):
    s = cp_model.CpSolver()
    s.parameters.num_workers = 8
    s.parameters.interleave_search = True
    s.parameters.max_deterministic_time = 20
    status = s.solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return s.status_name(status), None, None
    who = {r: d for (d, r), x in drive.items() if s.value(x)}
    table = routes.assign(driver=routes.route.map(who))
    return s.status_name(status), round(s.objective_value / 100, 2), table
```

The plan is built in sequence: routes first, then drivers for the routes. That's a **decomposition** (Day 39): a cheaper route plan could need more drivers, and the sequence can't see that. Here it's safe because driver pay depends mostly on how many routes there are, which the route model already minimizes through the day charges.

Then `day40/run_capstone.py`:

```python
# ---------- 40.1 Routes for every day ----------
routes, summary = route_week(stops, ids, km, minutes, fleet, days)
print(summary.to_string(index=False))
print(f"week: {summary.vehicles.sum()} vehicle-days ({summary.trucks.sum()} truck-days), {summary.km.sum():,.2f} km, "
      f"fleet ${summary.fleet_cost.sum():,.2f}")
```

```text
day  racks  vehicles  trucks     km  fleet_cost  problems
Mon    151         6       3 218.25      831.51         0
Tue    156         6       3 232.49      837.94         0
Wed    161         7       2 229.91      864.50         0
Thu    166         7       3 232.41      923.84         0
Fri    190         8       3 238.31     1024.26         0
Sat    181         8       3 238.97     1016.81         0
week: 42 vehicle-days (17 truck-days), 1,390.34 km, fleet $5,498.86
```

Six to eight vehicles a day, every window kept every day; **$5,498.86** of fleet costs for the week. The trucks run 17 of 18 possible truck-days: trucks are the cheapest way to carry racks (Day 36).

## 40.2 Who drives: the drivers' roster

```python
# ---------- 40.2 Who drives: the drivers' roster ----------
print(drivers.groupby(["contract", "truck_licence"]).size().rename("drivers").to_string())
m, drive = build_driver_roster(routes, drivers, rules)
status, driver_pay, table = solve_drivers(m, drive, routes)
print(f"drivers: {status}, ${driver_pay:,.2f} for the week")
grid = table.assign(slot=table.vehicle + " " + table.leaves.map(hhmm) + "-" + table.back.map(hhmm))
grid = grid.pivot(index="driver", columns="day", values="slot").reindex(index=drivers.index,
                                                                     columns=[d for d, _ in days]).fillna("off")
print(grid.to_string())
print("days worked:", grid.ne("off").sum(axis=1).to_dict())
```

```text
contract   truck_licence
full-time  0                4
           1                3
part-time  0                3
           1                1
drivers: OPTIMAL, $4,515.40 for the week
day                        Mon                  Tue                  Wed                  Thu                  Fri                  Sat
driver_id
DR01                       off  truck-1 04:19-07:41  truck-1 04:40-08:00  truck-3 05:15-07:37  truck-2 05:28-07:40  truck-3 05:28-07:40
DR02       truck-1 04:55-07:41  truck-2 05:05-07:40                  off  truck-2 04:46-06:27  truck-1 04:53-07:09  truck-2 04:41-07:41
DR03       truck-2 04:41-07:50                  off                  off  truck-1 04:59-07:50  truck-3 04:52-08:00  truck-1 05:35-07:14
DR04         van-3 05:29-08:00    van-4 04:58-07:17                  off    van-4 04:58-07:17    van-5 06:30-07:30    van-5 06:06-06:59
DR05         van-1 04:27-07:10                  off    van-3 05:38-07:44    van-5 06:34-07:34    van-3 05:56-07:49                  off
DR06                       off    van-2 04:50-07:31    van-1 04:51-06:59    van-2 05:18-07:49    van-6 05:35-06:46    van-6 05:27-07:04
DR07                       off    van-5 05:10-07:06    van-4 05:21-06:30    van-6 04:41-06:44    van-1 04:59-06:30    van-1 06:01-06:59
DR08       truck-3 05:28-07:30  truck-3 05:19-07:12  truck-2 05:18-07:40                  off                  off                  off
DR09         van-2 05:28-06:35                  off    van-5 05:19-07:37                  off                  off    van-3 04:43-07:07
DR10                       off                  off    van-6 05:04-06:27                  off    van-2 05:58-07:50    van-2 04:50-07:13
DR11                       off                  off                  off                  off                  off                  off
days worked: {'DR01': 5, 'DR02': 5, 'DR03': 4, 'DR04': 5, 'DR05': 4, 'DR06': 5, 'DR07': 5, 'DR08': 3, 'DR09': 3, 'DR10': 3, 'DR11': 0}
```

- **$4,515.40** of driver pay, proven optimal.
- The truck routes set the pace: 17 truck routes need the 4 truck licences almost every day (DR01 and DR02 drive five days, DR08 three).
- DR11, the most expensive part-timer, isn't needed this week.

## 40.3 The warehouse; 40.4 the budget and a what-if

```python
# ---------- 40.3 The warehouse roster for the same week ----------
staff, shifts, coverage, wrules, last_sunday = load_roster_data(con)
requests = con.sql("SELECT * FROM staff_requests").df()
warehouse = plan_week(staff, shifts, coverage, wrules, last_sunday, requests, max_nights=2, max_weekend_days=1)
print(f"warehouse (Day 33, union rules): {warehouse['status']}, ${warehouse['pay']:,.2f}, "
      f"{warehouse['broken']} requests broken")

# ---------- 40.4 The week's budget, and one what-if ----------
budget = {"delivery fleet": summary.fleet_cost.sum(), "delivery drivers": driver_pay, "warehouse staff": warehouse["pay"]}
for k, v in budget.items():
    print(f"{k:<18} ${v:>11,.2f}")
print(f"{'total':<18} ${sum(budget.values()):>11,.2f}")
later_routes, later = route_week(stops, ids, km, minutes, fleet, days, window_shift=30)
m2, drive2 = build_driver_roster(later_routes, drivers, rules)
status2, pay2, _ = solve_drivers(m2, drive2, later_routes)
print(f"if cafes took deliveries 30 minutes later: {later.vehicles.sum()} vehicle-days, fleet ${later.fleet_cost.sum():,.2f}, "
      f"drivers ${pay2:,.2f} ({status2}): saves ${summary.fleet_cost.sum() + driver_pay - later.fleet_cost.sum() - pay2:,.2f} a week")
```

```text
warehouse (Day 33, union rules): OPTIMAL, $28,896.88, 2 requests broken
delivery fleet     $   5,498.86
delivery drivers   $   4,515.40
warehouse staff    $  28,896.88
total              $  38,911.14
if cafes took deliveries 30 minutes later: 41 vehicle-days, fleet $5,462.44, drivers $4,411.52 (OPTIMAL): saves $140.30 a week
```

The week's delivery and warehouse labour cost **$38,911.14**. If cafés took deliveries 30 minutes later (Day 37), one vehicle-day fewer and shorter driver days would save **$140 a week** (about $7,300 a year).

## 40.5 The memo

<details><summary>A model memo</summary>

*To: the Operations Director. Re: next week's plan, and five decisions from Module 4.*

**Next week.** Delivery routes for Monday to Saturday, the drivers' roster and the warehouse roster are attached. Every café and store window is met, every labour rule holds, and the union's warehouse proposal is applied. Cost: fleet $5,499, drivers $4,515, warehouse staff $28,897; **total $38,911**.

**Five decisions:**

1. **Start the bakery's night run at 21:30, not 21:00** (Day 34). Everything is still packed by 03:43, before the vans load at 04:00. Starting at 22:00 would push dispatch towards 05:00, which costs about $61 a day on the road (Day 37).
2. **Ask the cafés for 30 minutes more** on their delivery windows (Days 37, 40): about $140 a week with the current fleet.
3. **Certify two pickers as forklift drivers** ($3,600 once, Day 35). It makes every-other-weekend off possible, even in peak weeks.
4. **Re-plan rosters with a change penalty** when people are absent (Day 39): the same cost, a tenth of the changes.
5. **Plan the bakery four weeks ahead**, re-planning weekly (Day 39): one-week plans lose about $2,500 a quarter by never building stock before peaks.

**How sure.** The drivers' roster is proven optimal for these routes. The routes come from a heuristic search (multi-start, fixed limits): repeatable, not proven; differences under about $5 a day between options are noise (Day 37). Labour rules follow Ontario's general hours-of-work rules as modelled on Day 32; they should be confirmed for our agreements.

**Not in the model.** Traffic and weather on the day; driver preferences and seniority; sick days (re-plan as on Day 39).

</details>

## Module 4 wrap-up

| Day | Idea | Northbound decision |
|---|---|---|
| 31 | CP-SAT, global constraints | Driver assignment with depot rules |
| 32 | Coverage and labour rules | A legal warehouse week |
| 33 | Soft constraints, fairness | Requests and the union's proposal |
| 34 | Interval variables, job shop | Tonight's bakery run |
| 35 | Multi-week rosters | December, and two new forklift drivers |
| 36 | TSP, CVRP | Van routes |
| 37 | Time windows, mixed fleets | The real morning run |
| 38 | Local search, annealing | Better routes; Christmas trailers |
| 39 | Decomposition, rolling horizons | Lookahead; re-planning for sickness |
| 40 | Putting it together | Next week's routes and rosters |

## Commit and tag

```bash
git add .
git commit -m "Day 40: capstone, next week's routes and rosters"
git tag v4.0-module4
git push --tags
```

Then run the tests: `tests/test_heuristics_and_plans.py` checks Days 38–40 (legal heuristic routes, 15 Christmas trailers, rolling horizons and the infeasible no-state plan, the drivers' rules on a small example). Add `"day38"`, `"day39"`, `"day40"` to the folders in `conftest.py`.

```python
"""Module 4, Days 38-40: heuristics, rolling horizons and the drivers' roster."""
import pandas as pd
import pytest
from decompose import evaluate_bakery, rolling_bakery
from metaheuristics import anneal_packing, local_search, simulated_annealing, total_km
from next_week import build_driver_roster, solve_drivers
from packing import first_fit_decreasing
from routing import load_run, savings, sweep


def test_local_search_and_annealing_keep_routes_legal(con):
    stops, ids, km, minutes = load_run(con)
    racks = [int(stops.racks[s]) for s in ids]
    start = sweep(stops, ids, racks, 18)
    improved, cost = local_search(start, km, racks, 18)
    assert cost < total_km(start, km)
    routes, k = simulated_annealing(savings(km, racks, 18), km, racks, 18, seed=0, steps=20_000)
    for r in (improved, routes):
        assert sorted(n for x in r for n in x) == list(range(1, len(ids)))
        assert all(sum(racks[n] for n in x) <= 18 for x in r)


def test_annealing_loads_christmas_on_15_trailers(con):
    o = con.sql("SELECT * FROM christmas_orders").df().set_index("order_id")
    pallets, kg = o.pallets.to_dict(), o.weight_kg.to_dict()
    ffd = first_fit_decreasing(pallets, 26, kg, 16000)
    start = {x: (k if k < 15 else 0) for k, (_, _, on) in enumerate(ffd) for x in on}
    put, overflow, _ = anneal_packing(pallets, kg, 15, start, seed=0)
    assert overflow == 0
    for k in range(15):
        assert sum(pallets[x] for x in put if put[x] == k) <= 26 and sum(kg[x] for x in put if put[x] == k) <= 16000


def test_rolling_horizons(con):
    full, _ = rolling_bakery(con, 12)
    short, _ = rolling_bakery(con, 1)
    assert evaluate_bakery(con, short) < evaluate_bakery(con, full) - 1000
    no_state, _ = rolling_bakery(con, 2, fix=())
    with pytest.raises(RuntimeError):
        evaluate_bakery(con, no_state)


def test_driver_roster_rules():
    routes = pd.DataFrame([(day, f"{day}-{k}", f"v{k}", "truck" if k == 1 else "van", 300, 420, 5)
                           for day in ("Mon", "Tue", "Wed") for k in (1, 2)],
                          columns=["day", "route", "vehicle", "type", "leaves", "back", "stops"])
    drivers = pd.DataFrame({"contract": ["part-time", "part-time", "part-time"], "truck_licence": [1, 0, 0],
                            "hourly_wage": [30.0, 25.0, 26.0]}, index=["A", "B", "C"])
    rules = {"pre_and_post_trip_minutes": 30, "min_paid_hours_per_day": 4, "full_time_min_days": 4,
             "full_time_max_days": 5, "part_time_max_days": 3}
    status, pay, table = solve_drivers(*build_driver_roster(routes, drivers, rules), routes)
    assert status == "OPTIMAL" and (table[table.type == "truck"].driver == "A").all()
    assert table.groupby(["driver", "day"]).size().max() == 1
    assert pay == pytest.approx(3 * 4 * 30.0 + 3 * 4 * 25.0)          # B is cheaper than C for the vans
```

```text
61 passed
```

### Day 40 checklist

- [ ] I built a week's plan from several models, in sequence, and said what the sequence can't see.
- [ ] I checked every day's routes against windows, capacity and street access.
- [ ] I assigned drivers to routes with licences, contracts and paid hours.
- [ ] I priced one what-if across the whole week.
- [ ] I wrote one memo that turns a module of analyses into decisions.
