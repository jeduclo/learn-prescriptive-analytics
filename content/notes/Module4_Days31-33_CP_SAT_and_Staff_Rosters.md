# DAY 31 — Constraint Programming with CP-SAT

Module 4 is about schedules and routes: who works when, which truck goes where, and in what order. These problems are full of logic ("if this, then not that", "all different", "no two in a row") that a MIP can express only awkwardly. **Constraint programming** (CP) expresses it directly. Today you meet **CP-SAT**, Google OR-Tools' solver, which combines constraint programming, SAT solving and linear relaxations. It is among the strongest free solvers for scheduling.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 31.1 Domains and a global constraint |
| 0:45 – 1:45 | 31.2 Business rules as logic |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 31.3 Other objectives; counting solutions |
| 2:45 – 3:45 | 31.4 CP-SAT or MIP?; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: Day 15's assignment in CP-SAT |
| 4:45 – 5:00 | Commit with Git + checklist |

### Set-up

Install OR-Tools in your virtual environment (it is in `requirements.txt` from today):

```bash
pip install ortools==9.15.6755
```

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Days 31–33: the Montreal DC's warehouse staff (`dc_staff`), shifts (`dc_shifts`), people needed per shift (`dc_coverage`), labour rules (`dc_rules`), who worked last Sunday (`dc_last_sunday`), the supervisor's roster for next week (`dc_current_roster`) and staff requests (`staff_requests`).

## 31.1 Domains and a global constraint

Day 14's eight drivers and eight routes, written the CP way. Instead of 64 yes/no variables, one integer variable per driver: **which route** (0 to 7). One **global constraint**, `AllDifferent`, says no two drivers share a route. An **element** constraint looks up each driver's minutes on the chosen route. Create `day31/first_models.py`:

```python
import duckdb
from ortools.sat.python import cp_model

con = duckdb.connect("northbound.duckdb", read_only=True)
rows = con.sql("SELECT driver, route, minutes FROM driver_route_minutes ORDER BY driver, route").fetchall()
drivers = sorted({d for d, _, _ in rows})
routes = sorted({r for _, r, _ in rows})
minutes = {(d, r): int(t) for d, r, t in rows}


def solver(workers=8):
    s = cp_model.CpSolver()
    s.parameters.num_workers = workers
    s.parameters.interleave_search = True        # deterministic: the same answer on every run
    s.parameters.max_time_in_seconds = 60
    return s


# ---------- 31.1 Variables with domains, a global constraint ----------
m = cp_model.CpModel()
route_of = {d: m.new_int_var(0, len(routes) - 1, f"route_of_{d}") for d in drivers}    # which route each driver takes
m.add_all_different(route_of.values())                                                   # no two drivers on one route
time_of = {d: m.new_int_var(0, 1000, f"minutes_{d}") for d in drivers}
for d in drivers:
    m.add_element(route_of[d], [minutes[d, r] for r in routes], time_of[d])             # time_of[d] = minutes[d][route_of[d]]
m.minimize(sum(time_of.values()))
s = solver()
print(s.status_name(s.solve(m)), f"{s.objective_value:,.0f} minutes:", {d: routes[s.value(route_of[d])] for d in drivers})
```

```text
OPTIMAL 2,768 minutes: {'D1': 'R3', 'D2': 'R1', 'D3': 'R5', 'D4': 'R7', 'D5': 'R6', 'D6': 'R2', 'D7': 'R4', 'D8': 'R8'}
```

The same 2,768 minutes as Day 14. Three things are different from a MIP:

- **Only whole numbers.** CP-SAT's variables and coefficients are integers. Costs in dollars and cents must be scaled (31.4).
- **Global constraints** (`AllDifferent`, `Element`, `Circuit`, `NoOverlap`, `Cumulative`…) carry meaning the solver exploits directly.
- **Search, not only relaxations.** CP-SAT propagates each decision through the constraints, learns from dead ends (as SAT solvers do), and runs several strategies in parallel (`num_workers`). With `interleave_search=True` the parallel search is deterministic: the same answer on every run. (On another computer or operating system an equally good plan may come out, with the same cost: when plans tie, any of them is a correct answer.)

## 31.2 Business rules as logic

Real driver rules from the Montreal depot:

- D8 is new: not R8, which crosses the busy port area.
- D2 and D7 share a car: both start from the same yard (R3, R4, R7 and R8 leave from the south yard).
- At most one of the three senior drivers (D1, D4, D7) in the south yard.
- Nobody over 390 minutes (Day 14).

```python
# ---------- 31.2 Business rules as logic ----------
take = {(d, r): m.new_bool_var(f"{d}_{r}") for d in drivers for r in routes}             # take[d, r] <=> route_of[d] == r
for d in drivers:
    for k, r in enumerate(routes):
        m.add(route_of[d] == k).only_enforce_if(take[d, r])
        m.add(route_of[d] != k).only_enforce_if(take[d, r].Not())
south = {d: m.new_bool_var(f"south_{d}") for d in drivers}                               # R3, R4, R7, R8 leave from the south yard
for d in drivers:
    m.add(sum(take[d, r] for r in ("R3", "R4", "R7", "R8")) == 1).only_enforce_if(south[d])
    m.add(sum(take[d, r] for r in ("R3", "R4", "R7", "R8")) == 0).only_enforce_if(south[d].Not())
m.add(take["D8", "R8"] == 0)                                    # D8 is new: R8 crosses the busy port area
m.add(south["D2"] == south["D7"])                               # D2 and D7 share a car: same yard
m.add(sum(south[d] for d in ("D1", "D4", "D7")) <= 1)           # at most one of the three senior drivers in the south yard
for d in drivers:
    m.add(time_of[d] <= 390)                                    # nobody over 6.5 hours (Day 14)
s = solver()
print(s.status_name(s.solve(m)), f"{s.objective_value:,.0f} minutes with the rules:",
      {d: routes[s.value(route_of[d])] for d in drivers})
```

```text
OPTIMAL 2,844 minutes with the rules: {'D1': 'R3', 'D2': 'R1', 'D3': 'R8', 'D4': 'R5', 'D5': 'R6', 'D6': 'R4', 'D7': 'R2', 'D8': 'R7'}
```

`only_enforce_if` attaches a constraint to a yes/no variable: `take[d, r]` is linked to `route_of[d]` in both directions, and `south[d]` to the four south-yard routes. The rules cost **76 minutes a day** (2,844 against 2,768). In a MIP, "share a car" and "same yard" need extra binaries and big-M rows; here each is one line.

## 31.3 Other objectives; counting solutions

```python
# ---------- 31.3 Other objectives: the longest route, and how many best answers ----------
longest = m.new_int_var(0, 1000, "longest")
m.add_max_equality(longest, list(time_of.values()))
m.minimize(longest)
s = solver()
s.solve(m)
best_longest = int(s.objective_value)
print(f"shortest possible longest route: {best_longest} minutes")
m.add(longest <= best_longest)
m.minimize(sum(time_of.values()))                               # then the least total time among those
s = solver()
s.solve(m)
best_total = int(s.objective_value)
print(f"least total with nobody over {best_longest}: {best_total:,} minutes")


class Count(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        super().__init__()
        self.n = 0

    def on_solution_callback(self):
        self.n += 1


m.clear_objective()
m.add(sum(time_of.values()) <= best_total)
s = cp_model.CpSolver()
s.parameters.enumerate_all_solutions = True
counter = Count()
s.solve(m, counter)
print(f"assignments that reach both: {counter.n}")
```

```text
shortest possible longest route: 387 minutes
least total with nobody over 387: 2,844 minutes
assignments that reach both: 1
```

- **Minimize the longest route**: `add_max_equality` builds the maximum directly. 387 minutes is the floor (nobody drives R3 faster), and the plan from 31.2 already reaches it.
- **Lexicographic objectives**: fix the first objective's best value as a constraint, then optimize the second.
- **Counting solutions**: with `enumerate_all_solutions` and a callback, CP-SAT lists every solution of a model without an objective. Only **one** assignment reaches both 387 and 2,844: the planners have no alternatives to choose from. When several plans tie, counting them tells you how much freedom is left for other wishes (Day 33).

> **Decision it supports:** assignments with many logical rules (who may do what, who goes with whom).
>
> **Other decisions you can support the same way:** exam timetables · seating plans · matching volunteers to tasks · sports schedules.

## 31.4 CP-SAT or MIP?

The Christmas trailers again (Days 26–28), as a CP-SAT model:

```python
# ---------- 31.4 CP-SAT on the Christmas trailers (Day 27) ----------
orders = con.sql("SELECT * FROM christmas_orders ORDER BY order_id").df().set_index("order_id")
for n in (16, 15):
    b = cp_model.CpModel()
    put = {(o, k): b.new_bool_var(f"put_{o}_{k}") for o in orders.index for k in range(n)}
    use = [b.new_bool_var(f"use_{k}") for k in range(n)]
    for o in orders.index:
        b.add_exactly_one(put[o, k] for k in range(n))
    for k in range(n):
        b.add(sum(int(orders.pallets[o]) * put[o, k] for o in orders.index) <= 26 * use[k])
        b.add(sum(int(orders.weight_kg[o]) * put[o, k] for o in orders.index) <= 16000 * use[k])
    b.minimize(sum(use))
    s = solver()
    status = s.solve(b)
    print(f"Christmas, {n} trailers allowed: {s.status_name(status)}, {s.objective_value:.0f} trailers, "
          f"bound {s.best_objective_bound:.0f}")
```

```text
Christmas, 16 trailers allowed: OPTIMAL, 15 trailers, bound 15
Christmas, 15 trailers allowed: OPTIMAL, 15 trailers, bound 15
```

CP-SAT proves 15 trailers in a few seconds with either model (16 or 15 trailers allowed), where HiGHS needed up to 16,000 nodes on some seeds (Day 28). A rough guide:

| Use CP-SAT when | Use a MIP solver when |
|---|---|
| Decisions are yes/no or small integers | There are many continuous variables (flows, amounts) |
| Rules are logical: if-then, all-different, sequences, no overlap | Costs and limits are mostly linear |
| Time is in whole units (minutes, shifts) | The LP relaxation is tight (Days 23, 27) |
| Finding any feasible plan is the hard part | You need duals and ranging (Module 2) |

Both are worth trying on a new problem: measure, as on Day 28.

### The planted bug

`day31/bug_rounding.py` solves Day 15's assignment in CP-SAT: "it wants integers, so `int()` the costs". Run it.

```python
con = duckdb.connect("northbound.duckdb", read_only=True)
dcs, stores, lanes = load_network(con)
lanes = lanes.assign(cost_per_pallet=lanes.cost_per_pallet.astype(int))     # whole dollars a pallet
m, assign = build_cp_assignment(dcs, stores, lanes, to_int=int)
status, value, _, plan = solve_and_price(m, assign, lanes, stores)
print(f"{status}: ${value:,.0f} a week")
real = load_network(con)[2]
print(f"the same plan at the real lane costs: ${sum(real.cost_per_pallet[d, s] * stores.pallets_per_week[s] for s, d in plan.items()):,.2f}")
```

<details><summary>What happens, and why</summary>

```text
OPTIMAL: $39,395 a week
the same plan at the real lane costs: $41,492.76
```

`int()` truncates: $4.39 a pallet becomes $4, $60.34 becomes $60. The model reports **$39,395 a week, $2,098 less** than the plan really costs. Here the plan happens to be the same; with closer costs, truncation changes which plan wins too. Scale to the precision you need (cents: multiply by 100, then round), say so in the code, and **price the final plan at the real costs** as a check.

</details>

## Mini-project: Day 15's assignment in CP-SAT

Write Day 15's single-sourcing model (each store to exactly one DC, DC capacities, French labels, six hours on the road) in CP-SAT with costs in cents, and check it against HiGHS.

<details><summary>Solution</summary>

```python
"""Day 31: Day 15's store-to-DC assignment written for CP-SAT, which only knows whole numbers.

Import it:  from cp_assignment import build_cp_assignment
"""
from ortools.sat.python import cp_model

MAX_MINUTES = 360


def build_cp_assignment(dcs, stores, lanes, to_int=lambda dollars: round(100 * dollars)):
    """assign[d, s] = 1 if DC d serves store s. Costs become whole numbers with to_int (default: cents)."""
    m = cp_model.CpModel()
    assign = {}
    for d in dcs.index:
        for s in stores.index:
            french_rule = stores.province[s] == "QC" and not dcs.ships_french_labels[d]
            if not french_rule and lanes.minutes[d, s] <= MAX_MINUTES:
                assign[d, s] = m.new_bool_var(f"assign_{d}_{s}")
    for s in stores.index:
        m.add_exactly_one(assign[d, s] for d in dcs.index if (d, s) in assign)
    for d in dcs.index:
        m.add(sum(int(stores.pallets_per_week[s]) * assign[d, s] for s in stores.index if (d, s) in assign)
              <= int(dcs.pallets_per_week_capacity[d]))
    m.minimize(sum(to_int(lanes.cost_per_pallet[d, s] * stores.pallets_per_week[s]) * x for (d, s), x in assign.items()))
    return m, assign


def solve_and_price(m, assign, lanes, stores):
    """Solve, then price the plan at the real (decimal) lane costs."""
    s = cp_model.CpSolver()
    s.parameters.num_workers = 8
    s.parameters.interleave_search = True
    status = s.solve(m)
    chosen = [(d, st) for (d, st), x in assign.items() if s.value(x)]
    real = sum(lanes.cost_per_pallet[d, st] * stores.pallets_per_week[st] for d, st in chosen)
    return s.status_name(status), s.objective_value, round(real, 2), dict((st, d) for d, st in chosen)
```

```python
"""Day 31 mini-project: Day 15's single-sourcing assignment with CP-SAT. Same answer as the MIP?

Run from the repository root (after day06/load_tables.py):  python day31/mini_project_assignment.py
"""
import sys

import duckdb
import pyomo.environ as pyo

for folder in ("day14", "day15", "day31"):
    sys.path.insert(0, folder)
from cp_assignment import build_cp_assignment, solve_and_price
from store_assignment import build_assignment
from transport import load_network

con = duckdb.connect("northbound.duckdb", read_only=True)
dcs, stores, lanes = load_network(con)
m, assign = build_cp_assignment(dcs, stores, lanes)
status, cents, real, plan = solve_and_price(m, assign, lanes, stores)
print(f"CP-SAT: {status}, objective {cents:,.0f} cents = ${real:,.2f} a week")
mip = build_assignment(dcs, stores, lanes)
pyo.SolverFactory("highs").solve(mip)
mip_plan = {s: d for d in mip.D for s in mip.S if pyo.value(mip.assign[d, s]) > 0.5}
print(f"HiGHS (Day 15): ${pyo.value(mip.cost):,.2f} a week")
print("stores assigned differently:", sorted(s for s in plan if plan[s] != mip_plan[s]) or "none")
```

```text
CP-SAT: OPTIMAL, objective 4,149,276 cents = $41,492.76 a week
HiGHS (Day 15): $41,492.76 a week
stores assigned differently: none
```

The same plan and the same cost to the cent: 4,149,276 cents = $41,492.76. `add_exactly_one` is CP-SAT's "one of these is 1". Two solvers agreeing is a strong check on both models.

</details>

## Commit

```bash
git add .
git commit -m "Day 31: CP-SAT"
git push
```

### Day 31 checklist

- [ ] I can write a CP-SAT model with integer domains, `AllDifferent` and `Element`.
- [ ] I can attach constraints to yes/no variables with `only_enforce_if`.
- [ ] I can minimize a maximum and solve objectives in order.
- [ ] I can count solutions with a callback.
- [ ] I scale money to cents, round (not truncate), and price the final plan at real costs.
- [ ] I can say when I'd try CP-SAT before a MIP solver.

---

# DAY 32 — Staff Scheduling I: Coverage and Labour Rules

The Montreal DC's warehouse runs three shifts a day, seven days a week, with 34 people: forklift drivers (who can also pick) and pickers. Every week the supervisor builds the roster by hand. Today you build it with CP-SAT: enough people on every shift, every legal rule respected, at least pay. Then you check the supervisor's roster with the same rules.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 32.1 The data and the rules |
| 0:45 – 1:45 | 32.2 Coverage only |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 32.3 With the labour rules |
| 2:45 – 3:45 | 32.4 The supervisor's roster; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: the Christmas peak |
| 4:45 – 5:00 | Commit with Git + checklist |

## 32.1 The data and the rules

Shifts: day 07:00–15:00, evening 15:00–23:00 ($1.50 an hour premium), night 23:00–07:00 ($2.75). Full-timers are guaranteed 40 hours (5 shifts) a week; part-timers work at most 3 shifts.

The general rules of Ontario's *Employment Standards Act, 2000* (ESA) for hours of work, as the province's guides summarize them:

- At most **8 hours a day** (or the regular work day, if longer) and **48 hours a week**, unless the employee agrees in writing to more.
- **11 consecutive hours off in every 24-hour period**, and at least **24 consecutive hours off each work week** (or 48 in every two weeks).
- **Overtime pay at 1.5 times the regular rate for hours over 44** in a work week.

(These are general rules with exceptions and special rules for some industries and jobs. A real roster system needs them checked by someone qualified; this is a course model, not legal advice.)

With 8-hour shifts, the rules become simple constraints:

| Rule | In the model |
|---|---|
| 48 hours a week | at most 6 shifts; this also guarantees a day off, so 24 hours off in the week |
| 11 hours off in every 24 | no evening shift followed by a day shift (8 hours off), no night followed by a day or an evening (0 or 8 hours) |
| Overtime after 44 hours | the 6th shift has 4 overtime hours at time and a half |

`dc_rules` stores these numbers, so a rule change is a data change. Create `day32/roster.py`:

```python
"""Day 32: a legal weekly roster for the Montreal DC warehouse, with CP-SAT.

Import it:  from roster import load_roster_data, build_roster, solve_roster, check_roster, roster_cost
"""
import pandas as pd
from ortools.sat.python import cp_model

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SHIFTS = ["D", "E", "N"]
# Shift pairs on consecutive days that leave less than 11 hours off in some 24-hour period:
# evening (ends 23:00) then day (starts 07:00): 8 hours; night (ends 07:00) then day (07:00): none;
# night (ends 07:00) then evening (starts 15:00): 8 hours.
TOO_CLOSE = [("E", "D"), ("N", "D"), ("N", "E")]


def load_roster_data(con):
    staff = con.sql("SELECT * FROM dc_staff ORDER BY employee_id").df().set_index("employee_id")
    shifts = con.sql("SELECT * FROM dc_shifts").df().set_index("shift")
    coverage = {(r.day, r.shift, r.role): int(r.people) for r in con.sql("SELECT * FROM dc_coverage").df().itertuples()}
    rules = {r: v for r, v, _ in con.sql("SELECT * FROM dc_rules").fetchall()}
    last_sunday = dict(con.sql("SELECT * FROM dc_last_sunday").fetchall())
    return staff, shifts, coverage, rules, last_sunday


def can_do(staff, e, role):
    """Forklift drivers can also pick; pickers can't drive a forklift."""
    return role == "picker" or staff.role[e] == "forklift"


def build_roster(staff, shifts, coverage, rules, last_sunday, labour_rules=True, carry_over=True, agency_rate=None):
    """work[e, d, s, r] = 1 if employee e works shift s on day d in role r. Minimize the week's pay.
    CP-SAT works with whole numbers, so pay is counted in half-cents (time and a half adds half a wage)."""
    m = cp_model.CpModel()
    E = list(staff.index)
    roles = ["picker", "forklift"]
    work = {(e, d, s, r): m.new_bool_var(f"work_{e}_{d}_{s}_{r}")
            for e in E for d in DAYS for s in SHIFTS for r in roles if can_do(staff, e, r)}
    on = {(e, d, s): sum(work[e, d, s, r] for r in roles if (e, d, s, r) in work) for e in E for d in DAYS for s in SHIFTS}
    for e in E:
        for d in DAYS:
            m.add(sum(on[e, d, s] for s in SHIFTS) <= 1)                           # one shift a day
    agency = {}
    if agency_rate is not None:                                                      # agency pickers (not drivers)
        agency = {(d, s, r): m.new_int_var(0, 20, f"agency_{d}_{s}") for (d, s, r) in coverage if r == "picker"}
    for (d, s, r), people in coverage.items():
        m.add(sum(work[e, d, s, r] for e in E if (e, d, s, r) in work) + agency.get((d, s, r), 0) >= people)  # enough people
    hours = int(rules["hours_per_shift"])
    shifts_worked = {e: sum(on[e, d, s] for d in DAYS for s in SHIFTS) for e in E}
    if labour_rules:
        max_shifts = int(rules["max_hours_per_week"]) // hours                    # 48 hours: 6 shifts, so a day off
        for e in E:
            limit = int(rules["part_time_max_shifts"]) if staff.contract[e] == "part-time" else max_shifts
            m.add(shifts_worked[e] <= limit)
            for a, b in zip(DAYS, DAYS[1:]):
                for s1, s2 in TOO_CLOSE:
                    m.add(on[e, a, s1] + on[e, b, s2] <= 1)                          # 11 hours off in every 24
            if carry_over and last_sunday[e] in SHIFTS:
                for s1, s2 in TOO_CLOSE:
                    if s1 == last_sunday[e]:
                        m.add(on[e, "Mon", s2] == 0)                                 # last Sunday counts too
    # pay, in half-cents: full-timers are paid at least their guaranteed shifts; time and a half after 44 hours
    cost = []
    paid = {}
    for e in E:
        wage = int(round(200 * staff.hourly_wage[e]))                             # half-cents an hour
        if staff.contract[e] == "full-time":
            paid[e] = m.new_int_var(int(rules["full_time_min_shifts"]), 7, f"paid_{e}")
            m.add(paid[e] >= shifts_worked[e])
        else:
            paid[e] = shifts_worked[e]
        overtime = m.new_int_var(0, 7 * hours, f"overtime_{e}")
        m.add(overtime >= hours * shifts_worked[e] - int(rules["overtime_after_hours"]))
        cost.append(wage * hours * paid[e] + (wage // 2) * overtime)                 # the extra half for overtime hours
        cost += [int(round(200 * shifts.premium_per_hour[s])) * hours * on[e, d, s] for d in DAYS for s in SHIFTS]
    if agency:
        cost.append(int(round(200 * agency_rate)) * hours * sum(agency.values()))
    m.pay = sum(cost)                                                                # kept for Day 33
    m.minimize(m.pay)
    m.agency = agency
    return m, work


def dollars(half_cents):
    return round(half_cents / 200, 2)


def solve_roster(m, work, seconds=120):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_workers = 8
    solver.parameters.interleave_search = True      # deterministic parallel search: the same roster on every run
    status = solver.solve(m)
    rows = [(e, d, s, r) for (e, d, s, r), v in work.items() if solver.value(v)] if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else []
    return solver, status, pd.DataFrame(rows, columns=["employee_id", "day", "shift", "role"])


def check_roster(roster, staff, coverage, rules, last_sunday):
    """An independent check of a roster against every rule. Returns a list of problems (empty if legal)."""
    problems = []
    for (d, s, r), people in coverage.items():
        n = ((roster["day"] == d) & (roster["shift"] == s) & (roster["role"] == r)).sum()
        if n < people:
            problems.append(f"{d} {s} {r}: {n} of {people}")
    for e, g in roster.groupby("employee_id"):
        if g["day"].duplicated().any():
            problems.append(f"{e}: two shifts on one day")
        if any(not can_do(staff, e, r) for r in g.role):
            problems.append(f"{e}: works a role they can't do")
        limit = int(rules["part_time_max_shifts"]) if staff.contract[e] == "part-time" else int(rules["max_hours_per_week"]) // 8
        if len(g) > limit:
            problems.append(f"{e}: {len(g)} shifts, over {limit}")
        day_shift = dict(zip(g["day"], g["shift"]))
        prev = [("Sun-1", last_sunday[e])] + [(d, day_shift.get(d)) for d in DAYS]
        for (a, s1), (b, s2) in zip(prev, prev[1:]):
            if (s1, s2) in TOO_CLOSE:
                problems.append(f"{e}: {s1} on {a} then {s2} on {b}: less than 11 hours off")
    return problems


def roster_cost(roster, staff, shifts, rules):
    total = 0.0
    for e in staff.index:
        g = roster[roster.employee_id == e]
        n, wage = len(g), staff.hourly_wage[e]
        paid = max(n, int(rules["full_time_min_shifts"])) if staff.contract[e] == "full-time" else n
        total += wage * 8 * paid + 0.5 * wage * max(0, 8 * n - rules["overtime_after_hours"])
        total += sum(8 * shifts.premium_per_hour[s] for s in g["shift"])
    return round(total, 2)
```

Four points about this model:

- `work[e, d, s, r]` exists only for roles the employee can do (`can_do`): a picker has no forklift variables at all.
- The pay is in **half-cents** so that time and a half stays a whole number (31.4).
- `check_roster` is an **independent checker**, written separately from the model. Every roster, the model's or a person's, goes through it.
- The rest rule needs **last Sunday**: someone who finished at 23:00 on Sunday can't start at 07:00 on Monday.

Then `day32/run_roster.py`:

```python
# ---------- 32.1 The data ----------
print(staff.groupby(["role", "contract"]).size().rename("people").to_string())
cov = pd.Series(coverage).rename_axis(["day", "shift", "role"]).unstack("day")[DAYS]
print(cov.to_string())
print(f"{sum(coverage.values())} shifts to fill; full-timers are paid for at least {rules['full_time_min_shifts']:.0f} shifts")
```

```text
role      contract
forklift  full-time    10
          part-time     2
picker    full-time    16
          part-time     6
day             Mon  Tue  Wed  Thu  Fri  Sat  Sun
shift role
D     forklift    3    3    3    3    3    2    2
      picker      7    7    7    7    7    5    5
E     forklift    2    2    2    2    2    2    2
      picker      6    6    6    6    6    4    4
N     forklift    1    1    1    1    1    1    1
      picker      3    3    3    3    3    2    2
142 shifts to fill; full-timers are paid for at least 5 shifts
```

## 32.2 Coverage only

```python
# ---------- 32.2 Coverage only ----------
m, work = build_roster(staff, shifts, coverage, rules, last_sunday, labour_rules=False)
solver, status, roster = solve_roster(m, work)
problems = check_roster(roster, staff, coverage, rules, last_sunday)
print(f"coverage only: {solver.status_name(status)}, ${dollars(solver.objective_value):,.2f}; {len(problems)} rule breaks, e.g.")
for p in problems[:5]:
    print("   ", p)
```

```text
coverage only: OPTIMAL, $28,836.40; 30 rule breaks, e.g.
    E01: N on Tue then D on Wed: less than 11 hours off
    E02: E on Sun-1 then D on Mon: less than 11 hours off
    E06: N on Thu then E on Fri: less than 11 hours off
    E07: N on Sun-1 then E on Mon: less than 11 hours off
    E07: E on Wed then D on Thu: less than 11 hours off
```

Cheapest possible, and illegal: 30 rule breaks, all of them too little rest. Without the rules, the model happily puts someone on a night shift and the next day's 07:00 shift.

## 32.3 With the labour rules

```python
# ---------- 32.3 With the labour rules ----------
m, work = build_roster(staff, shifts, coverage, rules, last_sunday)
solver, status, roster = solve_roster(m, work)
problems = check_roster(roster, staff, coverage, rules, last_sunday)
print(f"with the rules: {solver.status_name(status)}, ${dollars(solver.objective_value):,.2f} "
      f"(checked again: ${roster_cost(roster, staff, shifts, rules):,.2f}); {len(problems)} rule breaks")
per = roster.groupby("employee_id").size().reindex(staff.index, fill_value=0)
print("shifts per person:", per.groupby(staff.contract).value_counts().sort_index().to_dict())
grid = roster.pivot(index="employee_id", columns="day", values="shift").reindex(columns=DAYS).fillna(".")
print(grid.head(12).to_string())
```

```text
with the rules: OPTIMAL, $28,876.48 (checked again: $28,876.48); 0 rule breaks
shifts per person: {('full-time', 5): 26, ('part-time', 0): 4, ('part-time', 3): 4}
day         Mon Tue Wed Thu Fri Sat Sun
employee_id
E01           E   E   E   .   E   N   .
E02           .   E   E   E   E   .   D
E03           D   D   D   .   D   D   .
E04           E   .   D   D   D   .   N
E06           D   .   N   N   N   .   E
E07           N   .   D   E   E   E   .
E08           N   N   N   N   .   .   E
E09           D   D   N   N   .   D   .
E10           D   D   .   D   N   .   D
E12           E   .   .   D   D   E   E
E13           N   .   D   D   N   .   D
E14           N   .   D   D   D   .   D
```

**$28,876.48**: legality costs only $40.08 a week here. The checker agrees with the model on both the rules and the pay. Every full-timer works exactly their 5 guaranteed shifts, and only 4 part-timers are needed. Read the grid: no E followed by D, no N followed by D or E, at least one day off each.

## 32.4 The supervisor's roster

```python
# ---------- 32.4 Against the supervisor's roster ----------
current = con.sql("SELECT * FROM dc_current_roster").df()
problems = check_roster(current, staff, coverage, rules, last_sunday)
print(f"the supervisor's roster: ${roster_cost(current, staff, shifts, rules):,.2f}; {len(problems)} rule breaks:")
for p in problems:
    print("   ", p)
cur = current.groupby("employee_id").size().reindex(staff.index, fill_value=0)
print("shifts per person:", cur.groupby(staff.contract).value_counts().sort_index().to_dict())
```

```text
the supervisor's roster: $31,380.16; 5 rule breaks:
    E02: E on Sun-1 then D on Mon: less than 11 hours off
    E13: N on Sun-1 then D on Mon: less than 11 hours off
    E14: E on Sun-1 then D on Mon: less than 11 hours off
    E15: E on Sun-1 then D on Mon: less than 11 hours off
    E19: N on Sun-1 then D on Mon: less than 11 hours off
shifts per person: {('full-time', 4): 12, ('full-time', 5): 14, ('part-time', 3): 8}
```

- The hand-made roster costs **$31,380.16: $2,503.68 a week more** (about $130,000 a year). Twelve full-timers work only 4 shifts but are paid for 5, while all 8 part-timers work 3 shifts.
- It breaks the rest rule **5 times, all on Monday morning**: the supervisor built the week without looking at who worked last Sunday evening or night.

> **Decision it supports:** who works which shift, legally and at least cost.
>
> **Other decisions you can support the same way:** nurse rosters · call-centre shifts · store staffing · airline crew days off.

### The planted bug

`day32/bug_carry_over.py` builds the roster "only looking at next week". Run it.

```python
con = duckdb.connect("northbound.duckdb", read_only=True)
staff, shifts, coverage, rules, last_sunday = load_roster_data(con)
m, work = build_roster(staff, shifts, coverage, rules, last_sunday, carry_over=False)
solver, status, roster = solve_roster(m, work)
print(f"{solver.status_name(status)}: ${dollars(solver.objective_value):,.2f}")
for p in check_roster(roster, staff, coverage, rules, last_sunday):
    print("   ", p)
```

<details><summary>What happens, and why</summary>

```text
OPTIMAL: $28,876.48
    E08: N on Sun-1 then E on Mon: less than 11 hours off
    E14: E on Sun-1 then D on Mon: less than 11 hours off
    E15: E on Sun-1 then D on Mon: less than 11 hours off
    E24: E on Sun-1 then D on Mon: less than 11 hours off
    E27: E on Sun-1 then D on Mon: less than 11 hours off
    E34: N on Sun-1 then E on Mon: less than 11 hours off
```

Same cost, "optimal", and six breaks of the 11-hour rule on Monday, the same mistake as the supervisor's. A week is not independent of the week before: **schedules have state at the boundary** (last week's shifts, hours already worked, days off already taken). Rolling plans (Day 39) carry it forward every week. The independent checker caught what the model didn't know.

</details>

## Mini-project: the Christmas peak

In December, the DC needs 10% more people on every shift (rounded up), and in the busiest week 20% more. Can the staff cover it legally? If not, the DC can book agency pickers at $36 an hour (agencies can't supply certified forklift drivers). What does each week cost, and how many agency shifts are needed?

<details><summary>Solution</summary>

```python
"""Day 32 mini-project: Christmas peak weeks need 10% and 20% more people on every shift (rounded up).
Can the staff cover it legally? With agency pickers at $36 an hour, what does each week cost?

Run from the repository root (after day06/load_tables.py):  python day32/mini_project_peak.py
"""
import math
import sys

import duckdb
from ortools.sat.python import cp_model

sys.path.insert(0, "day32")
from roster import DAYS, build_roster, check_roster, dollars, load_roster_data, solve_roster

con = duckdb.connect("northbound.duckdb", read_only=True)
staff, shifts, coverage, rules, last_sunday = load_roster_data(con)
for extra in (0.0, 0.10, 0.20):
    week = {k: math.ceil(v * (1 + extra)) for k, v in coverage.items()}
    for agency in (None, 36.0):
        m, work = build_roster(staff, shifts, week, rules, last_sunday, agency_rate=agency)
        solver, status, roster = solve_roster(m, work)
        label = f"+{extra:.0%}, {sum(week.values())} shifts, {'with' if agency else 'no'} agency:"
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(label, solver.status_name(status))
            continue
        n = roster.groupby("employee_id").size()
        temps = sum(solver.value(v) for v in m.agency.values())
        breaks = [p for p in check_roster(roster, staff, week, rules, last_sunday) if not p.startswith(tuple(DAYS))]
        legal = not breaks                                       # (coverage gaps filled by agency are not rule breaks)
        print(label, f"${dollars(solver.objective_value):,.2f}; agency shifts {temps}; staff on 6 shifts {(n == 6).sum()}; "
              f"legal: {legal}")
```

```text
+0%, 142 shifts, no agency: $28,876.48; agency shifts 0; staff on 6 shifts 0; legal: True
+0%, 142 shifts, with agency: $28,876.48; agency shifts 0; staff on 6 shifts 0; legal: True
+10%, 184 shifts, no agency: INFEASIBLE
+10%, 184 shifts, with agency: $39,231.96; agency shifts 8; staff on 6 shifts 22; legal: True
+20%, 194 shifts, no agency: INFEASIBLE
+20%, 194 shifts, with agency: $41,951.96; agency shifts 18; staff on 6 shifts 22; legal: True
```

- **The staff alone can't cover either peak week**: 184 shifts against at most 180 (26 full-timers × 6 + 8 part-timers × 3). The model says so in seconds ("infeasible"), not after a week of trying.
- With agency pickers, the +10% week costs **$39,231.96** with 8 agency shifts, and 22 full-timers work a 6th shift (with 4 hours of overtime each). Overtime is cheaper than agency, up to the 48-hour limit.
- The +20% week needs 18 agency shifts: $41,951.96.

*Note to the DC manager:* "For peak weeks, book 8 agency picker shifts (+10%) or 18 (+20%) now, and plan 22 sixth shifts with overtime. No roster without agency staff is legal at these levels."

</details>

## Commit

```bash
git add .
git commit -m "Day 32: legal weekly roster with CP-SAT"
git push
```

### Day 32 checklist

- [ ] I can turn hours-of-work rules into roster constraints (shift limits, forbidden shift pairs).
- [ ] I keep rule values in data, not in code.
- [ ] I check every roster with an independent checker, the model's included.
- [ ] I carry last week's shifts into the new week.
- [ ] I compared a hand-made roster with the model's on cost and legality.
- [ ] I can tell when a requirement is impossible, and price the way out (agency staff).

Sources: [Ontario: Your guide to the Employment Standards Act, hours of work](https://www.ontario.ca/document/your-guide-employment-standards-act-0/hours-work) · [Ontario: Information about hours of work and overtime pay (2024)](https://www.ontario.ca/files/2024-12/mlitsd-info-about-how-and-ot-en-2024-12-12.pdf) · [Ontario: Industries and jobs with exemptions or special rules, terms and definitions](https://ontario.ca/document/industries-and-jobs-exemptions-or-special-rules/terms-and-definitions)

---

# DAY 33 — Staff Scheduling II: Preferences and Fairness

A legal, cheap roster can still be a bad one: it ignores the day off someone asked for, and gives the same few people every night shift. Today you add requests and fairness to Day 32's model. You will see that much of this is free, and price what isn't.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 33.1 Requests as soft constraints |
| 0:45 – 1:45 | 33.2 Fairness |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 33.3 A peak week: when requests cost money |
| 2:45 – 3:45 | Weights, units and a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: the union's proposal |
| 4:45 – 5:00 | Commit with Git + checklist |

## 33.1 Requests as soft constraints

`staff_requests` holds next week's 47 requests: days off (mostly Fridays and weekends), "no nights", and preferred shifts, some with priority 2. A **hard** constraint must hold; a **soft** constraint may break, at a price in the objective. Each request gets a yes/no variable `broken`; the objective adds a cost for each broken request (times its priority). Create `day33/preferences.py`:

```python
"""Day 33: requests and fairness on top of Day 32's roster model.

Import it:  from preferences import on_shift, add_requests, nights_and_weekends
"""
from roster import DAYS, SHIFTS


def on_shift(work):
    """on[e, d, s]: 1 if e works shift s on day d (in any role)."""
    on = {}
    for (e, d, s, r), v in work.items():
        on.setdefault((e, d, s), []).append(v)
    return {k: sum(v) for k, v in on.items()}


def add_requests(m, work, requests):
    """One yes/no 'broken' variable per request: day off, no nights, or a preferred shift.
    Returns [(employee, request, priority, broken)]."""
    on = on_shift(work)
    out = []
    for r in requests.itertuples():
        e = r.employee_id
        broken = m.new_bool_var(f"broken_{e}_{r.request}_{r.Index}")
        if r.request == "day_off":
            shifts = [on[e, r.day, s] for s in SHIFTS]
        elif r.request == "no_nights":
            shifts = [on[e, d, "N"] for d in DAYS]
        else:                                                   # prefer_shift: every other shift breaks it
            shifts = [on[e, d, s] for d in DAYS for s in SHIFTS if s != r.shift]
        for x in shifts:
            m.add(broken >= x)                                  # working any of them breaks the request
        out.append((e, r.request, int(r.priority), broken))
    return out


def nights_and_weekends(m, work, staff):
    """Integer variables: nights[e] and weekend_days[e] for each full-timer."""
    on = on_shift(work)
    nights, weekends = {}, {}
    for e in staff.index[staff.contract == "full-time"]:
        nights[e] = m.new_int_var(0, 7, f"nights_{e}")
        m.add(nights[e] == sum(on[e, d, "N"] for d in DAYS))
        weekends[e] = m.new_int_var(0, 2, f"weekend_{e}")
        m.add(weekends[e] == sum(on[e, d, s] for d in ("Sat", "Sun") for s in SHIFTS))
    return nights, weekends


def plan_week(staff, shifts, coverage, rules, last_sunday, requests, weight=50.0, max_nights=None,
              max_weekend_days=None, agency_rate=None):
    """Day 32's roster with requests (each broken request costs `weight` dollars, times its priority)
    and optional fairness caps. Returns a summary dict."""
    from ortools.sat.python import cp_model
    from roster import build_roster, dollars, solve_roster
    m, work = build_roster(staff, shifts, coverage, rules, last_sunday, agency_rate=agency_rate)
    broken = add_requests(m, work, requests)
    nights, weekends = nights_and_weekends(m, work, staff)
    if max_nights is not None:
        for v in nights.values():
            m.add(v <= max_nights)
    if max_weekend_days is not None:
        for v in weekends.values():
            m.add(v <= max_weekend_days)
    penalty = sum(p * b for _, _, p, b in broken)
    m.minimize(m.pay + int(round(200 * weight)) * penalty)               # half-cents, like the pay
    solver, status, roster = solve_roster(m, work)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": solver.status_name(status)}
    night_counts = sorted((solver.value(v) for v in nights.values()), reverse=True)
    return {"status": solver.status_name(status), "pay": dollars(solver.value(m.pay)),
            "broken": sum(solver.value(b) for *_, b in broken),
            "broken_list": [(e, r) for e, r, _, b in broken if solver.value(b)],
            "most_nights": night_counts[0], "nights_top5": night_counts[:5],
            "most_weekend_days": max(solver.value(v) for v in weekends.values()),
            "both_weekend_days": sum(solver.value(v) == 2 for v in weekends.values()),
            "agency": sum(solver.value(v) for v in m.agency.values()), "roster": roster}
```

`plan_week` returns pay, broken requests, and fairness measures. Then `day33/run_preferences.py`:

```python
# ---------- 33.1 Requests as soft constraints ----------
print(requests.request.value_counts().to_string(), f"\n{len(requests)} requests, weighted {requests.priority.sum()}")
print("day-off requests by day:", requests[requests.request == "day_off"].day.value_counts().to_dict())
print("--- requests with no price, and at $50 each")
show([(f"each broken request costs ${w}", plan_week(*data, weight=w)) for w in (0, 50)])
```

```text
request
day_off         28
no_nights       13
prefer_shift     6
47 requests, weighted 58
day-off requests by day: {'Sun': 10, 'Sat': 8, 'Fri': 5, 'Mon': 2, 'Thu': 2, 'Wed': 1}
--- requests with no price, and at $50 each
                         plan      pay  broken  most_nights     nights_top5  most_weekend_days  both_weekend_days
 each broken request costs $0 28876.48      47            4 [4, 3, 3, 2, 2]                  2                  6
each broken request costs $50 28876.48       0            5 [5, 5, 4, 3, 2]                  2                  8
```

**Granting all 47 requests costs nothing**: same pay, $28,876.48. With a weight of 0, the model broke all of them, not because it had to but because nothing told it otherwise. A solver is indifferent to everything that isn't in the model: **anything people care about must be in the objective or the constraints.**

## 33.2 Fairness

Look at `nights_top5` above: with all requests granted, two people work 5 nights each. Fairness rules put caps on that: at most 2 nights (or 1), at most 1 weekend day.

```python
# ---------- 33.2 Fairness caps ----------
print("--- fairness caps, requests at $50 each")
show([(name, plan_week(*data, **kw)) for name, kw in [
    ("requests only", {}), ("at most 2 nights", {"max_nights": 2}), ("at most 1 night", {"max_nights": 1}),
    ("at most 1 weekend day", {"max_weekend_days": 1}), ("2 nights, 1 weekend day", {"max_nights": 2, "max_weekend_days": 1})]])
r = plan_week(*data, max_nights=1)
print("requests broken with at most 1 night each:", r["broken_list"])
```

```text
--- fairness caps, requests at $50 each
                   plan      pay  broken  most_nights     nights_top5  most_weekend_days  both_weekend_days
          requests only 28876.48       0            5 [5, 5, 4, 3, 2]                  2                  8
       at most 2 nights 28896.88       1            2 [2, 2, 2, 2, 2]                  2                  5
        at most 1 night 28876.48       6            1 [1, 1, 1, 1, 1]                  2                  7
  at most 1 weekend day 28876.48       1            4 [4, 4, 4, 3, 3]                  1                  0
2 nights, 1 weekend day 28896.88       2            2 [2, 2, 2, 2, 2]                  1                  0
requests broken with at most 1 night each: [('E03', 'no_nights'), ('E06', 'no_nights'), ('E07', 'day_off'), ('E08', 'prefer_shift'), ('E09', 'no_nights'), ('E10', 'prefer_shift')]
```

- **At most 2 nights** each: $20.40 more a week and 1 broken request.
- **At most 1 night** each: no extra pay, but **6 broken requests**, half of them "no nights" requests: if nobody may work more than one night, everyone must take a turn, including those who asked not to.
- **At most 1 weekend day** each: 1 broken request; together with 2 nights, 2 broken and $20.40.

Fairness and requests pull against each other. The model doesn't decide which matters more: it shows the price of each choice, and people decide. Other ways to write fairness: minimize the *largest* number of nights (min-max, as in 31.3), or the spread between most and fewest; or balance over several weeks (Day 35).

## 33.3 A peak week: when requests cost money

In the +10% week (Day 32's mini-project), staff are short and requests compete with coverage.

```python
# ---------- 33.3 A peak week: when requests cost money ----------
peak = {k: math.ceil(v * 1.10) for k, v in coverage.items()}
peak_data = (staff, shifts, peak, rules, last_sunday, requests)
print("--- peak week (+10%), with agency pickers")
rows = []
for w in (0, 5, 10, 25, 50, 200):
    r = plan_week(*peak_data, weight=w, agency_rate=36.0)
    rows.append((w, r["pay"], r["broken"], r["agency"]))
t = pd.DataFrame(rows, columns=["dollars_per_request", "pay", "broken", "agency_shifts"])
print(t.to_string(index=False))
```

```text
--- peak week (+10%), with agency pickers
 dollars_per_request      pay  broken  agency_shifts
                   0 39231.96      47              8
                   5 39236.56       6              9
                  10 39236.56       6              9
                  25 39289.76       3             10
                  50 39397.46       0             13
                 200 39397.46       0             13
```

- Each step of the weight is a point on a **trade-off curve**: $5 a request already grants 41 of 47 requests for $4.60 more; granting all 47 costs **$165.50 more** a week (5 more agency shifts).
- The weight is a **price**: "how many dollars is one broken request worth to us?" Managers can answer that. Sweep it and show the curve, rather than picking one number in the code.

> **Decision it supports:** how much to spend to honour staff requests and share unpopular shifts fairly.
>
> **Other decisions you can support the same way:** class timetables with teacher preferences · on-call rotas · holiday allocation.

### The planted bug

`day33/bug_units.py` sets "a broken request costs $50" in the peak week. Run it.

```python
m, work = build_roster(staff, shifts, peak, rules, last_sunday, agency_rate=36.0)
broken = add_requests(m, work, requests)
m.minimize(m.pay + 50 * sum(p * b for _, _, p, b in broken))     # $50 a broken request
solver, status, roster = solve_roster(m, work)
print(f"{solver.status_name(status)}: pay ${dollars(solver.value(m.pay)):,.2f}, "
      f"requests broken {sum(solver.value(b) for *_, b in broken)} of {len(broken)}")
```

<details><summary>What happens, and why</summary>

```text
OPTIMAL: pay $39,231.96, requests broken 7 of 47
```

The pay is the cheapest possible and 7 requests are broken, to save $165.50: about $24 a request, not $50. The pay is in **half-cents** (Day 32), so a penalty of 50 is 50 half-cents: **$0.25 a request**. To the model, a broken request was worth a quarter. Keep every term of an objective in the same unit, convert in one place (`plan_week` multiplies by 200), and test that a large weight really changes the answer.

</details>

## Mini-project: the union's proposal

The union proposes: nobody works more than 2 nights a week, and no full-timer works both weekend days. What would it cost in a normal week and in a +10% peak week (with agency pickers allowed), in pay and in broken requests? Recommend.

<details><summary>Solution</summary>

```python
"""Day 33 mini-project: the union proposes "nobody works more than 2 nights a week, and no full-timer works
both weekend days". What would it cost in a normal week and in a peak week, in pay and in broken requests?

Run from the repository root (after day06/load_tables.py):  python day33/mini_project_union.py   (about 1 minute)
"""
import math
import sys

import duckdb
import pandas as pd

for folder in ("day32", "day33"):
    sys.path.insert(0, folder)
from preferences import plan_week
from roster import load_roster_data

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
staff, shifts, coverage, rules, last_sunday = load_roster_data(con)
requests = con.sql("SELECT * FROM staff_requests").df()
rows = []
for week, cov in [("normal", coverage), ("peak +10%", {k: math.ceil(v * 1.10) for k, v in coverage.items()})]:
    for name, kw in [("today's rules", {}), ("union proposal", {"max_nights": 2, "max_weekend_days": 1})]:
        r = plan_week(staff, shifts, cov, rules, last_sunday, requests, agency_rate=36.0, **kw)
        rows.append((week, name, r["status"], r.get("pay"), r.get("broken"), r.get("agency"), r.get("most_nights"),
                     r.get("both_weekend_days")))
print(pd.DataFrame(rows, columns=["week", "rules", "status", "pay", "broken", "agency_shifts", "most_nights",
                                  "both_weekend_days"]).to_string(index=False))
```

```text
     week          rules     status      pay  broken  agency_shifts  most_nights  both_weekend_days
   normal  today's rules    OPTIMAL 28876.48     0.0            0.0          5.0                6.0
   normal union proposal    OPTIMAL 28896.88     2.0            0.0          2.0                0.0
peak +10%  today's rules    OPTIMAL 39397.46     0.0           13.0          5.0               11.0
peak +10% union proposal INFEASIBLE      NaN     NaN            NaN          NaN                NaN
```

- **In a normal week the proposal costs $20.40 and 2 broken requests**: almost free. Nobody works more than 2 nights (today up to 5) and no one works both weekend days (today 6 people do).
- **In the peak week it is impossible**, even with agency pickers. The weekend forklift shifts are the bottleneck: at +10% the weekend needs 16 forklift shifts, and with one weekend day each for 10 full-time drivers plus two part-timers' 2 days each, the drivers can cover at most 14. Agency staff can't drive forklifts.

*Memo to the DC manager:* "Accept the union's proposal for normal weeks: it costs about $20 a week. For peak weeks, either certify two more pickers as forklift drivers, or agree with the union that drivers may work both weekend days in December. Without one of the two, no legal peak roster meets the proposal."

</details>

## Commit

```bash
git add .
git commit -m "Day 33: requests and fairness"
git push
```

Then run the tests: `tests/test_scheduling.py` checks Days 31–33 (CP-SAT against Day 15, the legal roster against the checker, the Monday carry-over, free requests). Add `"day31"`, `"day32"`, `"day33"` to the folders in `conftest.py`.

```python
"""Module 4, Days 31-33: CP-SAT, legal rosters, requests and fairness."""
import pytest
from cp_assignment import build_cp_assignment, solve_and_price
from preferences import plan_week
from roster import build_roster, check_roster, dollars, load_roster_data, roster_cost, solve_roster
from transport import load_network


def test_cp_sat_assignment_matches_day15(con):
    dcs, stores, lanes = load_network(con)
    m, assign = build_cp_assignment(dcs, stores, lanes)
    status, cents, real, _ = solve_and_price(m, assign, lanes, stores)
    assert status == "OPTIMAL" and round(cents) == 4_149_276 and real == pytest.approx(41_492.76, abs=0.01)


@pytest.fixture(scope="module")
def roster_data(con):
    return load_roster_data(con)


def test_legal_roster_passes_the_independent_check(roster_data):
    staff, shifts, coverage, rules, last_sunday = roster_data
    m, work = build_roster(*roster_data)
    solver, status, roster = solve_roster(m, work)
    assert check_roster(roster, staff, coverage, rules, last_sunday) == []
    assert dollars(solver.objective_value) == pytest.approx(roster_cost(roster, staff, shifts, rules), abs=0.01)
    assert dollars(solver.objective_value) == pytest.approx(28_876.48, abs=0.01)


def test_forgetting_last_sunday_breaks_the_rest_rule(roster_data):
    staff, shifts, coverage, rules, last_sunday = roster_data
    m, work = build_roster(*roster_data, carry_over=False)
    _, _, roster = solve_roster(m, work)
    problems = check_roster(roster, staff, coverage, rules, last_sunday)
    assert problems and all("Sun-1" in p for p in problems)


def test_all_requests_granted_at_no_extra_pay(con, roster_data):
    requests = con.sql("SELECT * FROM staff_requests").df()
    r = plan_week(*roster_data, requests, weight=50)
    assert r["broken"] == 0 and r["pay"] == pytest.approx(28_876.48, abs=0.01)
```

```text
53 passed
```

### Day 33 checklist

- [ ] I can write requests as soft constraints with a broken-request variable and a price.
- [ ] I check what is free before arguing about what costs money.
- [ ] I can add fairness caps and see how they trade against requests.
- [ ] I sweep a weight to draw a trade-off curve instead of guessing one number.
- [ ] I keep every term of an objective in the same unit.
- [ ] I can explain why a proposal is infeasible, and what would make it feasible.
