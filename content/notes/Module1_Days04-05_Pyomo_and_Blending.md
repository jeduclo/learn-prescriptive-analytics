# DAY 4 — Your First Pyomo Model

`linprog` wanted the model as anonymous matrices. Today you write the same bakery model in **Pyomo**, where every variable, constraint and number has a name, and the model reads like the mathematics you wrote on Day 1. By the end you can build, solve, inspect and trust a Pyomo model.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 4.1 Why a modelling language; the bakery in ten lines |
| 0:45 – 1:45 | 4.2 Sets, parameters and indexed constraints |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 4.3 What the solver receives, and what it says |
| 2:45 – 3:45 | 4.4 A report you can trust, and a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: the rye loaf in Pyomo |
| 4:45 – 5:00 | Commit with Git + checklist |

## 4.1 Why a modelling language

Look back at Day 2's `linprog` call: `c`, `A_ub`, `b_ub`, `bounds`. Which row of `A_ub` is the oven? Which column is croissants? You had to remember, and on Day 2 the order of `b_ub` was baker, oven, starter only because `pivot` sorted the resources alphabetically. With 30 products and 4 resources, that bookkeeping is where errors hide.

A **modelling language** lets you write the model with names, the way you write it on paper. It then builds the matrices and passes them to a solver. **Pyomo** is the open-source modelling language for Python; it talks to HiGHS, Gurobi, CPLEX and others, so the same model runs on any of them.

| Paper | Pyomo |
|---|---|
| Decision variables | `pyo.Var` |
| Objective | `pyo.Objective` |
| Constraints | `pyo.Constraint` |
| Index sets (products, resources) | `pyo.Set` |
| Data (profit, capacity) | `pyo.Param` |

Create `day04/first_model.py`. First, the model exactly as on Day 1, one variable per product:

```python
import pyomo.environ as pyo

# ---------- 4.1 One variable per product, written out by hand ----------
m = pyo.ConcreteModel(name="bakery_scalar")
m.S = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 35))   # sourdough batches
m.W = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 45))   # whole-wheat batches
m.C = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 31))   # croissant batches

m.profit = pyo.Objective(expr=300 * m.S + 200 * m.W + 250 * m.C, sense=pyo.maximize)
m.oven = pyo.Constraint(expr=2 * m.S + 1 * m.W + 1.5 * m.C <= 100)
m.bakers = pyo.Constraint(expr=1 * m.S + 1 * m.W + 1.5 * m.C <= 80)
m.starter = pyo.Constraint(expr=m.S <= 40)

solver = pyo.SolverFactory("highs")
result = solver.solve(m)
print("termination:", result.solver.termination_condition)
print(f"S = {pyo.value(m.S):g}, W = {pyo.value(m.W):g}, C = {pyo.value(m.C):g}")
print(f"profit = ${pyo.value(m.profit):,.2f}")
```

```text
termination: optimal
S = 20, W = 45, C = 10
profit = $17,500.00
```

The same $17,500 plan, and every line reads like the paper model. Things to notice:

- `bounds=(0, 35)` puts the demand cap straight on the variable.
- `sense=pyo.maximize`: no more flipping signs as with `linprog`.
- `pyo.value(m.S)` reads a value after solving. `m.S` on its own is the variable object, not a number.
- `result.solver.termination_condition` is the status. Check it every time.

> **Decision it supports:** the weekly bake plan, in a form a colleague can read and check line by line.
>
> **Other decisions you can support the same way:** any small model you need to explain to a manager · a quick check of a model someone built in a spreadsheet.

## 4.2 Sets, parameters and indexed constraints

Writing one variable per product does not scale. The real bakery has 30 products. We write the model **once** over sets, and feed the data separately:

$$
\begin{aligned}
\max\quad & \sum_{p \in P} \text{profit}_p \, x_p \\
\text{s.t.}\quad & \sum_{p \in P} \text{use}_{r,p} \, x_p \le \text{capacity}_r && \forall r \in R \\
& 0 \le x_p \le \text{cap}_p && \forall p \in P
\end{aligned}
$$

$$P$$ is the set of products, $$R$$ the set of resources. The model has one capacity constraint **per resource** and one demand constraint **per product**, however many there are.

```python
# ---------- 4.2 Sets and parameters: one model for any list of products ----------
products = ["BK-SD", "BK-WW", "BK-CR"]
resources = ["oven_hours", "baker_hours", "sourdough_starter"]
profit = {"BK-SD": 300, "BK-WW": 200, "BK-CR": 250}
cap = {"BK-SD": 35, "BK-WW": 45, "BK-CR": 31}
capacity = {"oven_hours": 100, "baker_hours": 80, "sourdough_starter": 40}
use = {("oven_hours", "BK-SD"): 2.0, ("oven_hours", "BK-WW"): 1.0, ("oven_hours", "BK-CR"): 1.5,
       ("baker_hours", "BK-SD"): 1.0, ("baker_hours", "BK-WW"): 1.0, ("baker_hours", "BK-CR"): 1.5,
       ("sourdough_starter", "BK-SD"): 1.0}

m = pyo.ConcreteModel(name="bakery")
m.P = pyo.Set(initialize=products)                       # products
m.R = pyo.Set(initialize=resources)                      # resources
m.profit = pyo.Param(m.P, initialize=profit)             # $ per batch
m.cap = pyo.Param(m.P, initialize=cap)                   # demand cap, batches
m.capacity = pyo.Param(m.R, initialize=capacity)         # per week
m.use = pyo.Param(m.R, m.P, initialize=use, default=0)   # use per batch; missing = 0

m.x = pyo.Var(m.P, within=pyo.NonNegativeReals)          # batches to bake


def demand_rule(m, p):
    return m.x[p] <= m.cap[p]


def capacity_rule(m, r):
    return sum(m.use[r, p] * m.x[p] for p in m.P) <= m.capacity[r]


m.demand = pyo.Constraint(m.P, rule=demand_rule)
m.capacity_limit = pyo.Constraint(m.R, rule=capacity_rule)
m.total_profit = pyo.Objective(expr=sum(m.profit[p] * m.x[p] for p in m.P), sense=pyo.maximize)

m.capacity_limit.pprint()

result = solver.solve(m)
print("termination:", result.solver.termination_condition)
for p in m.P:
    print(f"  {p}: {pyo.value(m.x[p]):g} batches")
print(f"profit = ${pyo.value(m.total_profit):,.2f}")
```

```text
capacity_limit : Size=3, Index=R, Active=True
    Key               : Lower : Body                                   : Upper : Active
          baker_hours :  -Inf :     x[BK-SD] + x[BK-WW] + 1.5*x[BK-CR] :  80.0 :   True
           oven_hours :  -Inf : 2.0*x[BK-SD] + x[BK-WW] + 1.5*x[BK-CR] : 100.0 :   True
    sourdough_starter :  -Inf :     x[BK-SD] + 0*x[BK-WW] + 0*x[BK-CR] :  40.0 :   True
termination: optimal
  BK-SD: 20 batches
  BK-WW: 45 batches
  BK-CR: 10 batches
profit = $17,500.00
```

How it works:

- `pyo.Param(m.R, m.P, initialize=use, default=0)`: the recipe is a dictionary keyed by (resource, product). Combinations not listed, like starter for whole-wheat, take the default 0. This mirrors the long recipe file from Day 1.
- A **rule** is a function that returns one constraint for one index. Pyomo calls `capacity_rule(m, r)` once for each resource and stores the results in `m.capacity_limit`, indexed by resource.
- `m.capacity_limit.pprint()` prints every constraint Pyomo built. **Read it.** It is the fastest way to catch a wrong coefficient. Here each row matches the recipe.

> **Decision it supports:** the same plan, from a model that does not change when products are added.
>
> **Other decisions you can support the same way:** a staffing plan over a set of shifts · a media plan over a set of channels.

### Try it 4.1

Packing becomes a fourth resource: sourdough uses 0.5 packing hours a batch, whole-wheat 0.5, croissants 0.6, and the packing line has 35 hours next week. Add it by changing only the data (`resources`, `capacity`, `use`). What is the new plan?

<details><summary>Solution</summary>

```python
resources = ["oven_hours", "baker_hours", "sourdough_starter", "packing_hours"]
capacity["packing_hours"] = 35
use.update({("packing_hours", "BK-SD"): 0.5, ("packing_hours", "BK-WW"): 0.5, ("packing_hours", "BK-CR"): 0.6})
```

The model code does not change. The new plan is **sourdough 30, whole-wheat 40, croissant 0: $17,000**. Packing (0.5 × 30 + 0.5 × 40 = 35 hours) and the oven are now full, and croissants drop out altogether. A new limit can change the whole mix, not just shave it.

</details>

## 4.3 What the solver receives, and what it says

Two ways to see what is going on between Pyomo and HiGHS.

```python
# ---------- 4.3 What the solver receives, and what it says ----------
print("--- HiGHS log ---")
pyo.SolverFactory("highs").solve(m, tee=True)      # a new solver object, so it starts from scratch
m.write("day04/bakery.lp", io_options={"symbolic_solver_labels": True})
print(open("day04/bakery.lp").read())
```

First, `tee=True` prints the solver's log:

```text
--- HiGHS log ---
Hessian has dimension 3 but no nonzeros, so is ignored
LP has 6 rows; 3 cols; 10 nonzeros
Coefficient ranges:
  Matrix  [1e+00, 2e+00]
  Cost    [2e+02, 3e+02]
  Bound   [0e+00, 0e+00]
  RHS     [3e+01, 1e+02]
Presolving model
6 rows, 3 cols, 10 nonzeros 0s
Initial sweep identifies 0 + 4 + 0 = 4 / 6 empty + singleton + redundant rows
2 rows, 3 cols, 6 nonzeros 0s
2 rows, 3 cols, 6 nonzeros 0s
Presolve reductions: rows 2(-4); columns 3(-0); nonzeros 6(-4) 
Solving the presolved LP
Choosing to use dual simplex solver
Using dual simplex solver
  Iteration        Objective     Infeasibilities num(sum)
          0     0.0000000000e+00 Ph1: 0(0) 0.0s
          2    -1.7500000000e+04 Pr: 0(0) 0.0s
Performed postsolve
Solving the original LP from the solution after postsolve
Model status        : Optimal
Simplex   iterations: 2
Objective value     :  1.7500000000e+04
P-D objective error :  0.0000000000e+00
HiGHS run time      :          0.00
```

Read it top to bottom:

- **6 rows, 3 cols:** 6 constraints (3 demand + 3 capacity), 3 variables.
- **Coefficient ranges:** the smallest and largest numbers in the model. When these span many powers of ten (say 1e-4 to 1e+7), solvers struggle; Day 9 is about scaling.
- **Presolve** removed 4 of the 6 rows: the 3 single-variable demand rows became bounds, and the starter row (S ≤ 40) is implied by the cap S ≤ 35. Real solvers simplify your model before solving it.
- **Dual simplex, 2 iterations:** the solver walked across 2 corners (Day 3) to the best one. The objective shows as −17,500 because HiGHS minimizes internally.
- **Model status: Optimal.** This is the line that matters.

Second, `m.write("day04/bakery.lp")` saves the model in **LP format**, the plain-text form solvers read:

```text
\* Source Pyomo model name=bakery *\

max 
total_profit:
+300 x(BK_SD)
+200 x(BK_WW)
+250 x(BK_CR)

s.t.

c_u_demand(BK_SD)_:
+1 x(BK_SD)
<= 35

c_u_demand(BK_WW)_:
+1 x(BK_WW)
<= 45

c_u_demand(BK_CR)_:
+1 x(BK_CR)
<= 31

c_u_capacity_limit(oven_hours)_:
+2.0 x(BK_SD)
+1 x(BK_WW)
+1.5 x(BK_CR)
<= 100

c_u_capacity_limit(baker_hours)_:
+1 x(BK_SD)
+1 x(BK_WW)
+1.5 x(BK_CR)
<= 80

c_u_capacity_limit(sourdough_starter)_:
+1 x(BK_SD)
<= 40

bounds
   0 <= x(BK_SD) <= +inf
   0 <= x(BK_WW) <= +inf
   0 <= x(BK_CR) <= +inf
end
```

This file is the model with no Python in the way. Commit it: when someone changes the model, `git diff day04/bakery.lp` shows exactly which coefficient moved. You can also send it to a colleague who uses another solver.

### Try it 4.2: change a number, re-solve

Rebuilding a model for every what-if is slow on big models. Declare the capacity as `mutable=True`, then change it and re-solve. Run `day04/oven_sweep.py`:

```python
m.capacity = pyo.Param(m.R, initialize={"oven_hours": 100, "baker_hours": 80, "sourdough_starter": 40}, mutable=True)
```

and later:

```python
solver = pyo.SolverFactory("highs")
for hours in (100, 105, 110, 115, 120, 130):
    m.capacity["oven_hours"] = hours
    solver.solve(m)
    plan = [round(pyo.value(m.x[p]), 2) + 0.0 for p in m.P]
    print(f"oven {hours:>3} h: profit ${pyo.value(m.total_profit):>10,.2f}  plan {plan}")
```

What happens to profit as oven hours go from 100 to 130? Where does extra oven time stop paying?

<details><summary>Solution</summary>

```text
oven 100 h: profit $ 17,500.00  plan [20.0, 45.0, 10.0]
oven 105 h: profit $ 18,166.67  plan [25.0, 45.0, 6.67]
oven 110 h: profit $ 18,833.33  plan [30.0, 45.0, 3.33]
oven 115 h: profit $ 19,500.00  plan [35.0, 45.0, 0.0]
oven 120 h: profit $ 19,500.00  plan [35.0, 45.0, 0.0]
oven 130 h: profit $ 19,500.00  plan [35.0, 45.0, 0.0]
```

Each extra oven hour is worth $133.33 (Day 2's number) **up to 115 hours**, where sourdough reaches its demand cap of 35. Beyond that, extra oven time is worth nothing. A shadow price holds only over a range: Day 12 gets that range straight from the solver. (`+ 0.0` turns −0.0 into 0.0 in the printout.)

</details>

## 4.4 A report you can trust

A solver answer is only useful with three checks: the status, the slack of every resource, and the objective recomputed from the plan. Put them in one function you reuse:

```python
# ---------- 4.4 A report you can trust ----------
def report(m, result):
    tc = result.solver.termination_condition
    if tc != pyo.TerminationCondition.optimal:
        print("NOT OPTIMAL:", tc)
        return
    print(f"{'resource':<18}{'used':>8}{'capacity':>10}{'slack':>8}")
    for r in m.R:
        used = pyo.value(m.capacity_limit[r].body)
        print(f"{r:<18}{used:>8.2f}{pyo.value(m.capacity[r]):>10.2f}{pyo.value(m.capacity[r]) - used:>8.2f}")
    recomputed = sum(pyo.value(m.profit[p]) * pyo.value(m.x[p]) for p in m.P)
    print(f"objective ${pyo.value(m.total_profit):,.2f}, recomputed ${recomputed:,.2f}")


report(m, result)
```

```text
resource              used  capacity   slack
oven_hours          100.00    100.00    0.00
baker_hours          80.00     80.00    0.00
sourdough_starter    20.00     40.00   20.00
objective $17,500.00, recomputed $17,500.00
```

`m.capacity_limit[r].body` is the left-hand side of the constraint; `pyo.value` evaluates it at the solution. Recomputing the objective by hand looks redundant until the day the objective expression has a bug.

> **Decision it supports:** handing a plan to the bakery manager with the evidence that it fits.
>
> **Other decisions you can support the same way:** any model output that leaves your laptop.

### The planted bug

`day04/bug_typo.py` solves without error and returns a plausible plan. **Before reading the solution**, run it, compare with yesterday's answer, and find the problem using `pprint`.

```python
import pyomo.environ as pyo

m = pyo.ConcreteModel()
m.P = pyo.Set(initialize=["BK-SD", "BK-WW", "BK-CR"])
m.R = pyo.Set(initialize=["oven_hours", "baker_hours", "sourdough_starter"])
m.profit = pyo.Param(m.P, initialize={"BK-SD": 300, "BK-WW": 200, "BK-CR": 250})
m.capacity = pyo.Param(m.R, initialize={"oven_hours": 100, "baker_hours": 80, "sourdough_starter": 40})
m.use = pyo.Param(m.R, m.P, default=0, initialize={
    ("oven_hours", "BK-SD"): 2.0, ("oven_hours", "BK-WW"): 1.0, ("oven_hours", "BK-CR"): 15,
    ("baker_hours", "BK-SD"): 1.0, ("baker_hours", "BK-WW"): 1.0, ("baker_hours", "BK-CR"): 1.5,
    ("sourdough_starter", "BK-SD"): 1.0})
m.x = pyo.Var(m.P, bounds=lambda m, p: (0, {"BK-SD": 35, "BK-WW": 45, "BK-CR": 31}[p]))
m.capacity_limit = pyo.Constraint(m.R, rule=lambda m, r: sum(m.use[r, p] * m.x[p] for p in m.P) <= m.capacity[r])
m.total_profit = pyo.Objective(expr=sum(m.profit[p] * m.x[p] for p in m.P), sense=pyo.maximize)

result = pyo.SolverFactory("highs").solve(m)
print(result.solver.termination_condition)
for p in m.P:
    print(f"  {p}: {pyo.value(m.x[p]):g}")
print(f"profit ${pyo.value(m.total_profit):,.2f}")
```

<details><summary>What happens, and how to find it</summary>

```text
optimal
  BK-SD: 27.5
  BK-WW: 45
  BK-CR: 0
profit $17,250.00
```

Optimal, plausible, and wrong: croissants disappear and profit is $250 lower than yesterday. Add `m.capacity_limit.pprint()` and the oven row reads `2.0*x[BK-SD] + x[BK-WW] + 15*x[BK-CR]`: someone typed **15** oven hours for croissants instead of 1.5. At 15 hours a batch, croissants are hopeless, so the solver drops them.

No solver can catch a data typo. Your defences: print the model, check that the plan's product mix makes sense, and (Day 6) validate data before it reaches the model.

</details>

## Mini-project: the rye loaf in Pyomo

Rewrite Day 2's rye mini-project in Pyomo, in `day04/rye_pyomo.py`, with:

1. all data in one dictionary `data` (profit, caps, capacities, recipe);
2. a function `build(data)` that returns the model, with **no numbers inside it**;
3. the same answer as Day 2.

Then add rye to the data, not to the code.

<details><summary>Solution</summary>

```python
"""Day 4 mini-project: the four-product bakery (with rye) in Pyomo, from dictionaries.

Run from the repository root:  python day04/rye_pyomo.py
"""
import pyomo.environ as pyo

data = {
    "profit": {"BK-SD": 300, "BK-WW": 200, "BK-CR": 250, "BK-RY": 240},
    "cap": {"BK-SD": 35, "BK-WW": 45, "BK-CR": 31, "BK-RY": 15},
    "capacity": {"oven_hours": 100, "baker_hours": 80, "sourdough_starter": 40},
    "use": {("oven_hours", "BK-SD"): 2.0, ("oven_hours", "BK-WW"): 1.0, ("oven_hours", "BK-CR"): 1.5, ("oven_hours", "BK-RY"): 1.5,
            ("baker_hours", "BK-SD"): 1.0, ("baker_hours", "BK-WW"): 1.0, ("baker_hours", "BK-CR"): 1.5, ("baker_hours", "BK-RY"): 1.0,
            ("sourdough_starter", "BK-SD"): 1.0, ("sourdough_starter", "BK-RY"): 0.5},
}


def build(data):
    m = pyo.ConcreteModel()
    m.P = pyo.Set(initialize=list(data["profit"]))
    m.R = pyo.Set(initialize=list(data["capacity"]))
    m.profit = pyo.Param(m.P, initialize=data["profit"])
    m.capacity = pyo.Param(m.R, initialize=data["capacity"])
    m.use = pyo.Param(m.R, m.P, initialize=data["use"], default=0)
    m.x = pyo.Var(m.P, bounds=lambda m, p: (0, data["cap"][p]))
    m.capacity_limit = pyo.Constraint(m.R, rule=lambda m, r: sum(m.use[r, p] * m.x[p] for p in m.P) <= m.capacity[r])
    m.total_profit = pyo.Objective(expr=sum(m.profit[p] * m.x[p] for p in m.P), sense=pyo.maximize)
    return m


m = build(data)
result = pyo.SolverFactory("highs").solve(m)
print(result.solver.termination_condition)
for p in m.P:
    print(f"  {p}: {pyo.value(m.x[p]):g}")
print(f"profit ${pyo.value(m.total_profit):,.2f}")
```

```text
optimal
  BK-SD: 12.5
  BK-WW: 45
  BK-CR: 5
  BK-RY: 15
profit $17,600.00
```

The same $17,600 as Day 2. `bounds=lambda m, p: (0, data["cap"][p])` sets each variable's bounds from the data. Because `build` contains no numbers, adding a fifth product is a data change only. Day 6 takes this one step further and reads `data` from DuckDB.

</details>

## Commit

```bash
git add .
git commit -m "Day 4: first Pyomo models, solver log, LP file, report"
git push
```

### Day 4 checklist

- [ ] I can write a linear program in Pyomo with `Var`, `Objective` and `Constraint`.
- [ ] I can index a model over sets and feed it data through `Param`, with defaults for missing combinations.
- [ ] I check the termination condition before reading any value.
- [ ] I can read a HiGHS log and an LP file.
- [ ] I can change a mutable parameter and re-solve.
- [ ] I found a data typo with `pprint`.

---

# DAY 5 — Mini-project: Blending and Meal Kits

Today is a project day. In the morning you blend flour from four suppliers to a quality specification at least cost. In the afternoon you design a Northbound Fresh meal kit: first the cheapest kit that meets the nutrition rules, then one that customers would actually buy. Both are classic **blending** problems: choose how much of each ingredient goes into a mix so that the mix's properties meet a spec.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 5.1 Blending flour: shares of one tonne |
| 0:45 – 1:45 | 5.2 A week of flour, with supplier limits |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 5.3 The cheapest meal kit (the diet problem) |
| 2:45 – 3:45 | 5.4 Rules that make a kit people would buy; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: the kit memo and nutrition label |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

The data script has grown. Open the **Course data** page, copy the new `make_data.py` over your old one, and rerun it:

```bash
python make_data.py
```

```text
data/dcs.csv: 3 rows
data/stores.csv: 40 rows
data/bakery_products.csv: 3 rows
data/bakery_recipe.csv: 7 rows
data/bakery_resources.csv: 3 rows
data/bakery_current_plan.csv: 3 rows
data/bakery_sales_weekly.csv: 12,360 rows
data/flour_suppliers.csv: 4 rows
data/flour_spec.csv: 3 rows
data/mealkit_ingredients.csv: 13 rows
data/mealkit_rules.csv: 6 rows
data/bakery_range.csv: 30 rows
data/bakery_range_recipe.csv: 94 rows
data/bakery_range_resources.csv: 4 rows
data/bakery_range_forecast.csv: 30 rows
data/bakery_range_minimums.csv: 35 rows
data/plan_weeks.csv: 12 rows
data/plan_demand.csv: 360 rows
data/bakery_storage.csv: 4 rows
data/plan_settings.csv: 8 rows
data/places.csv: 17 rows
data/road_segments.csv: 28 rows
data/store_logistics.csv: 40 rows
data/dc_logistics.csv: 3 rows
data/dc_store_lanes.csv: 120 rows
data/driver_route_minutes.csv: 64 rows
data/chilled_nodes.csv: 47 rows
data/chilled_arcs.csv: 156 rows
data/store_pallets_history.csv: 2,120 rows
data/store_week_actual.csv: 160 rows
data/store_week_forecast.csv: 160 rows
data/dc_weeks.csv: 12 rows
data/distribution_costs.csv: 5 rows
data/bakery_investments.csv: 4 rows
data/bakery_range_setup.csv: 30 rows
data/candidate_sites.csv: 10 rows
data/site_store_lanes.csv: 400 rows
data/truck_spec.csv: 2 rows
data/rush_orders.csv: 16 rows
data/truck_orders.csv: 24 rows
data/christmas_orders.csv: 48 rows
data/cake_slab.csv: 1 rows
data/cake_bars.csv: 5 rows
data/crossdock_candidates.csv: 6 rows
data/crossdock_arcs.csv: 101 rows
data/christmas_week_orders.csv: 184 rows
data/flour_discounts.csv: 8 rows
data/flour_all_units_offer.csv: 3 rows
data/flour_delivery.csv: 4 rows
data/overtime_tiers.csv: 4 rows
data/network_options.csv: 3 rows
data/network_settings.csv: 7 rows
data/dc_staff.csv: 34 rows
data/dc_shifts.csv: 3 rows
data/dc_coverage.csv: 42 rows
data/dc_rules.csv: 8 rows
data/dc_last_sunday.csv: 34 rows
data/dc_current_roster.csv: 142 rows
data/staff_requests.csv: 47 rows
data/bakery_machines.csv: 8 rows
data/bakery_night_jobs.csv: 18 rows
data/bakery_operations.csv: 72 rows
data/bakery_rules.csv: 5 rows
data/delivery_stops.csv: 35 rows
data/delivery_matrix.csv: 1,225 rows
data/delivery_fleet.csv: 2 rows
data/forklift_certification.csv: 2 rows
data/delivery_days.csv: 6 rows
data/delivery_drivers.csv: 11 rows
data/driver_rules.csv: 5 rows
data/trip_log.csv: 1,872 rows
data/bakery_disruptions.csv: 2 rows
data/receiving_log.csv: 306 rows
data/receiving_settings.csv: 5 rows
data/dock_shift_options.csv: 10 rows
data/stress_scenarios.csv: 6 rows
data/yule_log_history.csv: 118 rows
data/yule_log_forecast.csv: 40 rows
data/yule_log_economics.csv: 5 rows
data/dc_items.csv: 6 rows
data/dc_suppliers.csv: 3 rows
data/dc_daily_demand.csv: 1,092 rows
data/purchase_orders.csv: 156 rows
data/inventory_settings.csv: 5 rows
data/dc_supplier_offers.csv: 2 rows
data/bakery_two_stage.csv: 4 rows
data/bakery_oven_log.csv: 780 rows
data/bakery_robust.csv: 1 rows
data/mdp_settings.csv: 3 rows
data/replenishment_scenarios.csv: 4 rows
```

Later lessons add files to the same script, so your list may be longer than the one above. The first seven files are byte-for-byte the same as before: new data uses its own random stream, so your Days 1–4 results do not change. Rerun `python day01/ladder.py` too: it rewrites `data/forecast_next_week.csv`, which Day 6 needs.

## 5.1 Blending flour: shares of one tonne

The bakery's bread flour must have at least 12.5% protein (for gluten), at most 0.60% ash (for a light crumb) and at most 14.0% moisture (for shelf life). Four suppliers each sell flour with different properties and prices:

```text
                       supplier  price_per_tonne  protein_pct  ash_pct  moisture_pct  max_tonnes_per_week
supplier_id                                                                                              
FL-PHR         Prairie Hard Red              520         14.5     0.62          13.0                   30
FL-OSW       Ontario Soft White              410         10.0     0.48          14.2                   30
FL-QCO           Quebec Organic              640         13.0     0.55          13.5                    8
FL-DUR           Imported Durum              590         15.5     0.80          12.5                   30
```

No single flour works on its own: Prairie and Durum have too much ash, Ontario too little protein, Quebec is dear. A blend can do better. If $$f_s$$ is the share of supplier $$s$$ in one tonne, the blend's protein is the weighted average $$\sum_s \text{protein}_s f_s$$, and so on for every property. That is linear in $$f$$, so this is an LP:

$$
\begin{aligned}
\min\quad & \sum_s \text{price}_s \, f_s \\
\text{s.t.}\quad & \sum_s f_s = 1 \\
& \sum_s \text{protein}_s \, f_s \ge 12.5,\quad \sum_s \text{ash}_s \, f_s \le 0.60,\quad \sum_s \text{moisture}_s \, f_s \le 14.0 \\
& 0 \le f_s \le 1
\end{aligned}
$$

Create `day05/flour_blend.py`:

```python
import pandas as pd

import pyomo.environ as pyo

flour = pd.read_csv("data/flour_suppliers.csv").set_index("supplier_id")
spec = pd.read_csv("data/flour_spec.csv").set_index("property")
print(flour.drop(columns="province").to_string())
print(spec.to_string())
solver = pyo.SolverFactory("highs")


def spec_constraints(m, amount, total):
    """Blend quality = weighted average of supplier qualities, written without division."""
    m.spec = pyo.ConstraintList()
    for prop, row in spec.iterrows():
        blend = sum(flour.loc[s, prop] * amount[s] for s in m.S)
        if pd.notna(row.min_value):
            m.spec.add(blend >= row.min_value * total)
        if pd.notna(row.max_value):
            m.spec.add(blend <= row.max_value * total)


# ---------- 5.1 Shares of one tonne ----------
m = pyo.ConcreteModel()
m.S = pyo.Set(initialize=flour.index.tolist())
m.share = pyo.Var(m.S, bounds=(0, 1))
m.one_tonne = pyo.Constraint(expr=sum(m.share[s] for s in m.S) == 1)
spec_constraints(m, m.share, 1)
m.cost = pyo.Objective(expr=sum(flour.loc[s, "price_per_tonne"] * m.share[s] for s in m.S))
print(solver.solve(m).solver.termination_condition)
print(f"cost per tonne ${pyo.value(m.cost):,.2f}")
for s in m.S:
    print(f"  {s} {flour.loc[s, 'supplier']:<20} {pyo.value(m.share[s]):.4f}")
for prop in spec.index:
    print(f"  blend {prop}: {sum(flour.loc[s, prop] * pyo.value(m.share[s]) for s in m.S):.3f}")
```

```text
optimal
cost per tonne $471.11
  FL-PHR Prairie Hard Red     0.5556
  FL-OSW Ontario Soft White   0.4444
  FL-QCO Quebec Organic       0.0000
  FL-DUR Imported Durum       0.0000
  blend protein_pct: 12.500
  blend ash_pct: 0.558
  blend moisture_pct: 13.533
```

Five parts Prairie to four parts Ontario, at **$471.11 a tonne**: cheaper than any single flour that meets the spec (Quebec alone, at $640, is the only one). The protein rule binds at exactly 12.5%; ash and moisture have room to spare.

Two habits in `spec_constraints`:

- The spec is **data** (`flour_spec.csv`), so the quality team can change a limit without touching code.
- Each rule is written as `blend >= min * total`, never `blend / total >= min`. With shares, `total` is 1; in 5.2 it is a variable, and dividing by a variable makes the model nonlinear (you'll hit this in 5.4).

> **Decision it supports:** which suppliers to buy flour from, and in what proportions.
>
> **Other decisions you can support the same way:** animal feed, fertilizer, fuel or coffee blends · a portfolio with limits on average risk.

### Try it 5.1

Marketing wants to print "made with Quebec flour" on the bags, which requires at least 20% Quebec Organic. What does the label cost per tonne? And what happens if the quality team tightens ash to 0.55%, or to 0.50%?

<details><summary>Solution</summary>

Add `m.local = pyo.Constraint(expr=m.share["FL-QCO"] >= 0.2)` and re-solve; for ash, change `0.6` to `0.55` in `flour_spec.csv` (or in a copy of the spec in your script).

- **20% Quebec:** $502.44 a tonne (Prairie 0.4222, Ontario 0.3778, Quebec 0.2). The label costs **$31.33 a tonne**.
- **Ash ≤ 0.55%:** $523.33 a tonne, one third from each of Prairie, Ontario and Quebec. Ash and protein now both bind.
- **Ash ≤ 0.50%:** **infeasible**. Only Ontario flour is below 0.50% ash, and it has 10% protein; no blend meets both rules. Day 7 is about finding out which rules conflict.

</details>

## 5.2 A week of flour, with supplier limits

Next week the bakery needs 40 tonnes, and Quebec Organic can deliver at most 8. Now the decision is tonnes per supplier, and the spec applies to the average over the total:

```python
# ---------- 5.2 Tonnes for a week, with supplier limits ----------
NEED = 40   # tonnes of bread flour next week
m = pyo.ConcreteModel()
m.S = pyo.Set(initialize=flour.index.tolist())
m.t = pyo.Var(m.S, bounds=lambda m, s: (0, flour.loc[s, "max_tonnes_per_week"]))
total = sum(m.t[s] for s in m.S)
m.need = pyo.Constraint(expr=total == NEED)
spec_constraints(m, m.t, total)
m.cost = pyo.Objective(expr=sum(flour.loc[s, "price_per_tonne"] * m.t[s] for s in m.S))
print(solver.solve(m).solver.termination_condition)
print(f"weekly flour cost ${pyo.value(m.cost):,.2f} for {NEED} t")
for s in m.S:
    print(f"  {s}: {pyo.value(m.t[s]):.3f} t")
```

```text
optimal
weekly flour cost $18,844.44 for 40 t
  FL-PHR: 22.222 t
  FL-OSW: 17.778 t
  FL-QCO: 0.000 t
  FL-DUR: 0.000 t
```

The same 5 : 4 blend, 40 × $471.11 = $18,844.44. Here `total` is an expression in the variables, and `blend >= 12.5 * total` stays linear. The supplier limits don't bind this week; they will when Prairie has a bad harvest (try setting its limit to 15).

## 5.3 The cheapest meal kit: the diet problem

Northbound Fresh wants to sell a **Family Meal Kit**: ingredients for 4 servings in one box. The nutrition team set rules per kit:

```text
nutrient,min_per_kit,max_per_kit
kcal,2400,3200
protein_g,120,
fat_g,,110
fibre_g,40,
sodium_mg,,2400
weight_g,1800,2600
```

`mealkit_ingredients.csv` has 13 ingredients, each with a food group, cost per 100 g, nutrients per 100 g (approximate values) and the most that fits in a box. The decision is grams of each ingredient, $$g_i$$; every nutrient is $$\sum_i \text{nutrient}_i \, g_i / 100$$.

This is the **diet problem**, one of the first problems linear programming was tested on: George Stigler posed it in 1945 for a cheapest healthy diet, and it was solved with the simplex method in 1947.

We will build several versions of the kit today, so the model goes in a small module, `day05/kit_model.py`:

```python
"""Day 5: the meal kit model, shared by mealkit.py and price_rules.py."""
import pandas as pd
import pyomo.environ as pyo

food = pd.read_csv("data/mealkit_ingredients.csv").set_index("ingredient")
rules = pd.read_csv("data/mealkit_rules.csv").set_index("nutrient")
NUTRIENTS = ["kcal", "protein_g", "fat_g", "fibre_g", "sodium_mg"]
solver = pyo.SolverFactory("highs")


def build():
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=food.index.tolist())
    m.g = pyo.Var(m.I, bounds=lambda m, i: (0, food.loc[i, "max_g"]))    # grams per kit
    m.weight = pyo.Expression(expr=sum(m.g[i] for i in m.I))
    m.amount = pyo.Expression(NUTRIENTS, rule=lambda m, n: sum(food.loc[i, n] * m.g[i] / 100 for i in m.I))
    m.rules = pyo.ConstraintList()
    for n, row in rules.iterrows():
        e = m.weight if n == "weight_g" else m.amount[n]
        if pd.notna(row.min_per_kit):
            m.rules.add(e >= row.min_per_kit)
        if pd.notna(row.max_per_kit):
            m.rules.add(e <= row.max_per_kit)
    m.cost = pyo.Objective(expr=sum(food.loc[i, "cost_per_100g"] * m.g[i] / 100 for i in m.I))
    return m


def show(m, title):
    print(f"--- {title}: ${pyo.value(m.cost):.2f} per kit, {pyo.value(m.weight):,.0f} g")
    for i in m.I:
        if pyo.value(m.g[i]) > 1e-6:
            print(f"  {i:<18} {pyo.value(m.g[i]):7.1f} g")
    print("  " + ", ".join(f"{n} {pyo.value(m.amount[n]):,.1f}" for n in NUTRIENTS))
```

Then `day05/mealkit.py` uses it:

```python
import sys

import pyomo.environ as pyo

sys.path.insert(0, "day05")
from kit_model import build, food, show, solver

# ---------- 5.3 The cheapest kit that meets the nutrition rules ----------
m = build()
print(solver.solve(m).solver.termination_condition)
show(m, "Cheapest kit")
```

```text
optimal
--- Cheapest kit: $6.23 per kit, 1,800 g
  Brown rice           110.3 g
  Dry lentils          298.0 g
  Canned chickpeas     391.7 g
  Carrots              600.0 g
  Onions               400.0 g
  kcal 2,400.0, protein_g 120.0, fat_g 18.0, fibre_g 83.7, sodium_mg 1,393.4
```

`pyo.Expression` names a piece of the model (the kit's weight, each nutrient amount) so constraints and reports can reuse it.

$6.23 for four servings, nutritionally fine, and nobody would buy it: rice, lentils, chickpeas, carrots and onions, no sauce, no meat. Stigler's 1945 diet was mostly flour, cabbage and dried beans for the same reason. **The model only knows the rules you give it.** A cheap, strange answer means rules are missing, not that the model is wrong.

### Try it 5.2

Which nutrition rules bind in the cheapest kit? Which ingredients sit at their box limit?

<details><summary>Solution</summary>

Energy is at its minimum (2,400 kcal), protein at its minimum (120 g) and weight at its minimum (1,800 g). Fat (18 g of 110), fibre (83.7 of 40 minimum) and sodium have slack. Carrots (600 g) and onions (400 g) are at their box limits: they are the cheapest grams of weight. So the cheapest kit is driven by three things: hit the calories, hit the protein, fill the box cheaply.

</details>

## 5.4 Rules that make a kit people would buy

The product team describes the kit they want: a whole-wheat pasta kit, with a real tomato sauce, meat, plenty of vegetables, and no single ingredient dominating. Each wish becomes a linear rule:

| Wish | Rule |
|---|---|
| A pasta kit | 350 g ≤ pasta ≤ 500 g, no rice |
| A real sauce | tomato sauce ≥ 400 g |
| Meat in it | beef + chicken ≥ 400 g |
| Plenty of vegetables | vegetables ≥ 35% of the kit's weight |
| Nothing dominates | each ingredient ≤ 30% of the weight |

The share rules are ratios ("35% of the weight"), and we write them without division: vegetables ≥ 0.35 × weight.

```python
# ---------- 5.4 Rules that make it a pasta kit people would buy ----------
m = build()
veg = [i for i in m.I if food.loc[i, "food_group"] == "vegetable"]
m.pasta = pyo.Constraint(expr=pyo.inequality(350, m.g["Whole-wheat pasta"], 500))
m.sauce = pyo.Constraint(expr=m.g["Tomato sauce"] >= 400)
m.meat = pyo.Constraint(expr=m.g["Lean ground beef"] + m.g["Chicken thigh"] >= 400)
m.no_rice = pyo.Constraint(expr=m.g["Brown rice"] == 0)
m.veg_share = pyo.Constraint(expr=sum(m.g[i] for i in veg) >= 0.35 * m.weight)
m.max_share = pyo.Constraint(m.I, rule=lambda m, i: m.g[i] <= 0.30 * m.weight)
print(solver.solve(m).solver.termination_condition)
show(m, "Pasta kit")
```

```text
optimal
--- Pasta kit: $10.30 per kit, 1,800 g
  Whole-wheat pasta    373.5 g
  Chicken thigh        400.0 g
  Tomato sauce         400.0 g
  Carrots              226.5 g
  Onions               400.0 g
  kcal 2,400.0, protein_g 138.2, fat_g 55.0, fibre_g 52.8, sodium_mg 2,251.0
```

`pyo.inequality(350, x, 500)` is a two-sided rule in one line. The kit now costs **$10.30**, and every extra dollar over $6.23 buys something the customer asked for. The model picked chicken over beef (cheaper per gram of protein) and carrots and onions as vegetables (cheapest), with the sauce counting as a vegetable too.

> **Decision it supports:** a meal kit recipe at the lowest cost that meets nutrition and product rules.
>
> **Other decisions you can support the same way:** hospital or school menus · pet food formulas · a fleet's fuel mix with emission limits.

### Try it 5.3

Customers expect something green. Require at least 150 g of spinach or broccoli in total. What does it cost? Then require the meat to be beef (≥ 400 g of beef): what now?

<details><summary>Solution</summary>

```python
m.green = pyo.Constraint(expr=m.g["Spinach"] + m.g["Broccoli"] >= 150)
```

```text
--- Pasta kit with greens: $10.91 per kit, 1,800 g
  Whole-wheat pasta    376.9 g
  Chicken thigh        400.0 g
  Tomato sauce         400.0 g
  Broccoli             150.0 g
  Carrots               73.1 g
  Onions               400.0 g
  kcal 2,400.0, protein_g 141.5, fat_g 55.4, fibre_g 52.7, sodium_mg 2,194.8
```

Greens add **$0.61** a kit (broccoli, the cheaper green, replaces carrots). Adding `m.beef = pyo.Constraint(expr=m.g["Lean ground beef"] >= 400)` gives **$12.03**: a beef kit costs $1.12 more than a chicken one.

</details>

### The planted bug

`day05/bug_ratio.py` writes the vegetable share the way you would say it, "vegetables divided by weight is at least 35%":

```python
veg = [i for i in m.I if food.loc[i, "food_group"] == "vegetable"]
m.veg_share = pyo.Constraint(expr=sum(m.g[i] for i in veg) / weight >= 0.35)
```

Run it.

<details><summary>What happens, and the fix</summary>

```text
pyomo.contrib.solver.common.util.IncompatibleModelError: Highs interface does not support expressions of degree None
```

Dividing by `weight`, which contains variables, makes the constraint **nonlinear** ("degree None" means "not a polynomial"). HiGHS solves linear and quadratic models only, so it refuses. The fix is to multiply both sides by the weight, which is positive:

```python
m.veg_share = pyo.Constraint(expr=sum(m.g[i] for i in veg) >= 0.35 * weight)
```

Every "share of a total" rule can be linearized this way, as long as the total cannot be negative.

</details>

## Mini-project: the kit memo

Northbound Fresh plans to sell the kit at $24.99. Using your pasta kit with greens (Try it 5.3), write `day05/mealkit_memo.md` for the category manager:

1. The recipe in grams, and the ingredient cost per kit and per serving.
2. A nutrition label per serving (one quarter of the kit): kcal, protein, fat, fibre, sodium.
3. The margin per kit before packing and labour.
4. Which two rules cost the most, and what dropping each would save (re-solve without it).
5. One sentence on what the model does not know (taste, supply, food safety).

<details><summary>Model answer</summary>

**1.** Pasta 376.9 g, chicken thigh 400 g, tomato sauce 400 g, broccoli 150 g, carrots 73.1 g, onions 400 g; **$10.91 a kit, $2.73 a serving**.

**2.** Per serving: 600 kcal, 35.4 g protein, 13.9 g fat, 13.2 g fibre, 549 mg sodium.

**3.** $24.99 − $10.91 = **$14.08** before packing and labour.

**4.** Drop one rule at a time and re-solve (`day05/price_rules.py`, a loop over the rules):

```python
def kit(drop=None):
    m = build()
    veg = [i for i in m.I if food.loc[i, "food_group"] == "vegetable"]
    product_rules = {
        "pasta 350-500 g": lambda: pyo.Constraint(expr=pyo.inequality(350, m.g["Whole-wheat pasta"], 500)),
        "sauce >= 400 g": lambda: pyo.Constraint(expr=m.g["Tomato sauce"] >= 400),
        "meat >= 400 g": lambda: pyo.Constraint(expr=m.g["Lean ground beef"] + m.g["Chicken thigh"] >= 400),
        "no rice": lambda: pyo.Constraint(expr=m.g["Brown rice"] == 0),
        "vegetables >= 35%": lambda: pyo.Constraint(expr=sum(m.g[i] for i in veg) >= 0.35 * m.weight),
        "each <= 30%": lambda: pyo.Constraint(m.I, rule=lambda m, i: m.g[i] <= 0.30 * m.weight),
        "greens >= 150 g": lambda: pyo.Constraint(expr=m.g["Spinach"] + m.g["Broccoli"] >= 150),
    }
    for k, (name, make) in enumerate(product_rules.items()):
        if name != drop:
            setattr(m, f"rule_{k}", make())
    solver.solve(m)
    return pyo.value(m.cost), list(product_rules)


base, names = kit()
print(f"all rules: ${base:.2f}")
for name in names:
    cost, _ = kit(drop=name)
    print(f"  without {name:<18} ${cost:5.2f}  saves ${base - cost:.2f}")
```

```text
all rules: $10.91
  without pasta 350-500 g    $10.91  saves $0.00
  without sauce >= 400 g     $10.47  saves $0.44
  without meat >= 400 g      $ 7.25  saves $3.66
  without no rice            $10.88  saves $0.03
  without vegetables >= 35%  $10.91  saves $0.00
  without each <= 30%        $10.91  saves $0.00
  without greens >= 150 g    $10.30  saves $0.61
```

The **meat rule costs $3.66 a kit**, a third of the ingredient cost; greens cost $0.61 and the sauce $0.44. Rules that save $0.00 are not binding: the kit already meets them, so dropping them changes nothing.

**5.** "The model meets the nutrition and product rules at least cost; it does not know whether the kit tastes good, so the test kitchen must approve the recipe before launch."

</details>

## Commit

```bash
git add .
git commit -m "Day 5: flour blending and meal kit models"
git push
```

### Day 5 checklist

- [ ] I can model a blend whose properties are weighted averages of its ingredients.
- [ ] I write share rules as "part ≥ share × total", never with division.
- [ ] I keep specs and rules in data files, not in code.
- [ ] I know why the cheapest diet is strange, and fix it with rules, not by hand.
- [ ] I can price a business rule by solving with and without it.
- [ ] I can read the error HiGHS gives for a nonlinear constraint.
