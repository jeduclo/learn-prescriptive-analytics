# DAY 34 — Job-Shop Scheduling on the Bakery Lines

Every night the central bakery runs 18 production jobs (sourdough, croissants, cakes, cookies) through mixers, a laminator, a proofer, two ovens, a decorating table and the packing line. Each job passes its steps in a fixed order, and each machine does one thing at a time. That is a **job shop**, one of the classic hard scheduling problems. Today you schedule it with CP-SAT's **interval variables**, add the bakery's real rules (proofer racks, dough that can't wait), and compare with the head baker's plan.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 34.1 The data and the head baker's plan |
| 0:45 – 1:45 | 34.2 Interval variables and the job-shop model |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 34.3 What the rules cost; 34.4 a short-staffed night |
| 2:45 – 3:45 | Reading a Gantt chart; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: can the night crew start later? |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Days 34–37: `bakery_machines`, `bakery_night_jobs`, `bakery_operations` and `bakery_rules` (tonight's run, Day 34), `forklift_certification` (Day 35), `delivery_stops`, `delivery_matrix` and `delivery_fleet` (tomorrow morning's bread run, Days 36–37).

## 34.1 The data and the head baker's plan

Times are minutes after 21:00. Bread and viennoiserie must be packed by **04:00**, when the delivery vans load (Days 36–37); cakes and cookies by 05:00 or 06:00. Create `day34/bakery_shop.py`:

```python
"""Day 34: tonight's bakery run as a job shop, with CP-SAT interval variables.

Import it:  from bakery_shop import load_shop, build_schedule, solve_schedule, dispatch_fifo, check_schedule, clock
"""
import pandas as pd
from ortools.sat.python import cp_model

HORIZON = 900          # minutes after 21:00


def clock(minute, start=1260):
    t = (start + int(minute)) % 1440
    return f"{t // 60:02d}:{t % 60:02d}"


def load_shop(con):
    jobs = con.sql("SELECT * FROM bakery_night_jobs ORDER BY job_id").df().set_index("job_id")
    ops = con.sql("SELECT * FROM bakery_operations ORDER BY job_id, step").df()
    machines = con.sql("SELECT * FROM bakery_machines").df().set_index("machine_id")
    rules = dict(con.sql("SELECT * FROM bakery_rules").fetchall())
    return jobs, ops, machines, rules


def build_schedule(jobs, ops, machines, rules, one_of_each=False, proofer_limit=True, max_wait=True,
                   objective="makespan", oven_down=None):
    """start/end of every operation. one_of_each: only MIX-1 and OVEN-1 work (a short-staffed night).
    objective: "makespan" (finish everything as early as possible) or "lateness" (total minutes late, then makespan).
    oven_down: (from, to) minutes when OVEN-2 is out of service."""
    m = cp_model.CpModel()
    start, end, choice = {}, {}, {}
    on_machine = {mid: [] for mid in machines.index}
    proofer = []
    for r in ops.itertuples():
        key = (r.job_id, r.step)
        start[key] = m.new_int_var(0, HORIZON, f"start_{r.job_id}_{r.step}")
        end[key] = m.new_int_var(0, HORIZON, f"end_{r.job_id}_{r.step}")
        interval = m.new_interval_var(start[key], r.minutes, end[key], f"op_{r.job_id}_{r.step}")
        group = [mid for mid in machines.index if machines.machine_group[mid] == r.machine_group]
        if one_of_each:
            group = group[:1]
        if r.machine_group == "proofer":
            proofer.append(interval)
        elif len(group) == 1:
            on_machine[group[0]].append(interval)
            choice[key] = {group[0]: True}
        elif len(group) > 1:                                       # either machine: one optional interval each
            picks = {}
            for mid in group:
                p = m.new_bool_var(f"{r.job_id}_{r.step}_on_{mid}")
                on_machine[mid].append(m.new_optional_interval_var(start[key], r.minutes, end[key], p, f"{mid}_{key}"))
                picks[mid] = p
            m.add_exactly_one(picks.values())
            choice[key] = picks
    if oven_down is not None:
        on_machine["OVEN-2"].append(m.new_interval_var(oven_down[0], oven_down[1] - oven_down[0], oven_down[1], "oven2_down"))
    for mid, intervals in on_machine.items():
        if mid != "PROOFER":
            m.add_no_overlap(intervals)                            # one operation at a time on each machine
    if proofer_limit:
        m.add_cumulative(proofer, [1] * len(proofer), int(machines.capacity["PROOFER"]))   # 4 racks in the proofer
    for job, g in ops.groupby("job_id"):
        g = g.sort_values("step")
        for a, b in zip(g.itertuples(), list(g.itertuples())[1:]):
            m.add(start[job, b.step] >= end[job, a.step])          # in order: each step after the one before
            if max_wait and a.operation == "proof" and b.operation == "bake":
                m.add(start[job, b.step] <= end[job, a.step] + int(rules["max_wait_proof_to_bake"]))
    finish = {j: end[j, int(ops[ops.job_id == j].step.max())] for j in jobs.index}
    makespan = m.new_int_var(0, HORIZON, "makespan")
    m.add_max_equality(makespan, list(finish.values()))
    late = {j: m.new_int_var(0, HORIZON, f"late_{j}") for j in jobs.index}
    for j in jobs.index:
        m.add(late[j] >= finish[j] - int(jobs.due_minute[j]))
    if objective == "makespan":
        m.minimize(makespan)
    else:
        m.minimize(1000 * sum(late.values()) + makespan)
    m.vars = {"start": start, "end": end, "choice": choice, "makespan": makespan, "late": late}
    return m


def solve_schedule(m, ops, work=10.0):
    """Deterministic search with a fixed amount of work (CP-SAT's "deterministic time"), so every run gives the
    same schedule on the same OR-Tools version. About 20 seconds on a laptop."""
    s = cp_model.CpSolver()
    s.parameters.max_deterministic_time = work
    s.parameters.num_workers = 8
    s.parameters.interleave_search = True
    status = s.solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return s, status, None
    rows = []
    for r in ops.itertuples():
        key = (r.job_id, r.step)
        picks = m.vars["choice"].get(key)
        machine = "PROOFER" if r.machine_group == "proofer" else ("-" if picks is None else
                                                                   next(mid for mid, p in picks.items() if p is True or s.value(p)))
        rows.append((r.job_id, r.step, r.operation, machine, s.value(m.vars["start"][key]), s.value(m.vars["end"][key])))
    return s, status, pd.DataFrame(rows, columns=["job_id", "step", "operation", "machine", "start", "end"])


def dispatch_fifo(jobs, ops, machines, one_of_each=False):
    """The head baker's way: jobs in list order, each step as soon as its machine (or a proofer rack) is free."""
    free = {mid: 0 for mid in machines.index if mid != "PROOFER"}
    proofing = []                                                  # (start, end) of everything in the proofer
    rows = []
    for job in jobs.index:
        ready = 0
        for r in ops[ops.job_id == job].sort_values("step").itertuples():
            group = [mid for mid in free if machines.machine_group[mid] == r.machine_group]
            if one_of_each:
                group = group[:1]
            if r.machine_group == "proofer":
                t = ready
                while sum(a < t + r.minutes and t < b for a, b in proofing) >= machines.capacity["PROOFER"]:
                    t = min(b for a, b in proofing if b > t)
                proofing.append((t, t + r.minutes))
                machine = "PROOFER"
            elif group:
                machine = min(group, key=lambda mid: (max(free[mid], ready), mid))
                t = max(free[machine], ready)
                free[machine] = t + r.minutes
            else:
                t, machine = ready, "-"
            rows.append((job, r.step, r.operation, machine, t, t + r.minutes))
            ready = t + r.minutes
    return pd.DataFrame(rows, columns=["job_id", "step", "operation", "machine", "start", "end"])


def check_schedule(sched, jobs, machines, rules):
    """Independent check: order of steps, one job per machine at a time, proofer racks, proofing wait. Returns problems."""
    problems = []
    for job, g in sched.sort_values("step").groupby("job_id"):
        rows = list(g.itertuples())
        for a, b in zip(rows, rows[1:]):
            if b.start < a.end:
                problems.append(f"{job}: {b.operation} starts before {a.operation} ends")
            if a.operation == "proof" and b.operation == "bake" and b.start - a.end > rules["max_wait_proof_to_bake"]:
                problems.append(f"{job}: waits {b.start - a.end} min after proofing (over-proofed)")
    for mid, g in sched[~sched.machine.isin(["PROOFER", "-"])].groupby("machine"):
        g = g.sort_values("start")
        for a, b in zip(g.itertuples(), list(g.itertuples())[1:]):
            if b.start < a.end:
                problems.append(f"{mid}: {a.job_id} and {b.job_id} overlap")
    p = sched[sched.machine == "PROOFER"]
    for t in sorted(set(p.start)):
        if ((p.start <= t) & (p.end > t)).sum() > machines.capacity["PROOFER"]:
            problems.append(f"proofer over capacity at {clock(t)}")
    return problems


def summary(sched, jobs):
    finish = sched.groupby("job_id").end.max()
    late = (finish - jobs.due_minute).clip(lower=0)
    return {"makespan": int(finish.max()), "done_at": clock(finish.max()), "late_jobs": int((late > 0).sum()),
            "minutes_late": int(late.sum()), "late": {j: int(v) for j, v in late.items() if v > 0}}
```

`dispatch_fifo` is the head baker's method: take the jobs in list order, start each step as soon as its machine is free. `check_schedule` is an independent checker, as on Day 32. Then `day34/run_shop.py`:

```python
# ---------- 34.1 The data, and the head baker's plan ----------
print(jobs.to_string())
print(ops[ops.job_id.isin(["J01", "J08", "J15", "J17"])].to_string(index=False))
print(machines.to_string())
fifo = dispatch_fifo(jobs, ops, machines)
for p in report("head baker's plan (list order, as soon as possible)", fifo)[:4]:
    print("   ", p)
```

```text
          sku      category  due_minute
job_id
J01     BK-SD         bread         420
J02     BK-WW         bread         420
J03     BK-03         bread         420
J04     BK-04         bread         420
J05     BK-05         bread         420
J06     BK-SD         bread         420
J07     BK-WW         bread         420
J08     BK-CR  viennoiserie         420
J09     BK-12  viennoiserie         420
J10     BK-13  viennoiserie         420
J11     BK-CR  viennoiserie         420
J12     BK-14  viennoiserie         420
J13     BK-20          cake         480
J14     BK-22          cake         540
J15     BK-23          cake         480
J16     BK-25        cookie         480
J17     BK-26        cookie         540
J18     BK-27        cookie         480
job_id  step operation machine_group  minutes
   J01     1       mix         mixer       18
   J01     2     proof       proofer       90
   J01     3      bake          oven       45
   J01     4      pack       packing       23
   J08     1  laminate     laminator       30
   J08     2     proof       proofer       60
   J08     3      bake          oven       21
   J08     4      pack       packing       13
   J15     1       mix         mixer       23
   J15     2      bake          oven       49
   J15     3      cool          none       40
   J15     4  decorate    decorating       30
   J15     5      pack       packing       11
   J17     1       mix         mixer       13
   J17     2      bake          oven       17
   J17     3      pack       packing       11
           machine_group  capacity
machine_id
MIX-1              mixer         1
MIX-2              mixer         1
LAM-1          laminator         1
PROOFER          proofer         4
OVEN-1              oven         1
OVEN-2              oven         1
PACK-1           packing         1
DECO-1        decorating         1
head baker's plan (list order, as soon as possible): done at 06:04; 3 late jobs, 137 minutes late; 8 rule breaks
    J03: waits 21 min after proofing (over-proofed)
    J04: waits 26 min after proofing (over-proofed)
    J07: waits 32 min after proofing (over-proofed)
    J08: waits 63 min after proofing (over-proofed)
```

The head baker's plan finishes at 06:04 with 3 late jobs, and **8 doughs wait too long after proofing**: the baker's simple rule doesn't see that an oven busy with a cake makes the proofed bread wait. Over-proofed dough collapses in the oven.

## 34.2 Interval variables and the job-shop model

An **interval variable** is a start, a length and an end tied together: `new_interval_var(start, minutes, end)`. Three constraints build the whole model:

- `add_no_overlap(intervals)`: a machine does one operation at a time.
- `add_cumulative(intervals, demands, capacity)`: the proofer holds at most 4 racks at once.
- For a step that can use either of two machines (two mixers, two ovens), one **optional interval** per machine, each with a yes/no "present" variable, and `add_exactly_one` of them.

Precedence (`start[next] >= end[previous]`) keeps each job's steps in order, and `start[bake] <= end[proof] + 15` stops dough over-proofing.

```python
# ---------- 34.2 The job-shop model ----------
m = build_schedule(jobs, ops, machines, rules)
solver, status, sched = solve_schedule(m, ops)
print(solver.status_name(status))
report("model, finish as early as possible", sched, solver)
```

```text
FEASIBLE
model, finish as early as possible: done at 03:14 (no plan can finish before 03:07); 0 late jobs, 0 minutes late; 0 rule breaks
```

**Done at 03:14, nothing late, no rule broken**: almost three hours earlier than the head baker's plan, with the same machines and people. CP-SAT reports `FEASIBLE`, not `OPTIMAL`: in the work allowed (`max_deterministic_time=10`, about 20 seconds) it proved only that no plan finishes before 03:07. A 7-minute gap is plenty for a decision; Day 26's lesson applies to CP-SAT too.

`max_deterministic_time` measures the solver's work in units that don't depend on the computer, so with `interleave_search` every run gives the same schedule, unlike a limit in seconds.

## 34.3 What the rules cost

```python
# ---------- 34.3 What the rules cost ----------
for name, kw in [("no proofer limit, no maximum wait", dict(proofer_limit=False, max_wait=False)),
                 ("proofer limit only", dict(max_wait=False))]:
    s2, st2, sc2 = solve_schedule(build_schedule(jobs, ops, machines, rules, **kw), ops)
    print(f"{name}: done at {summary(sc2, jobs)['done_at']}")
```

```text
no proofer limit, no maximum wait: done at 03:09
proofer limit only: done at 03:12
```

Respecting the proofer and the 15-minute limit costs about 5 minutes against a plan that ignores them. The rules are cheap when the schedule is built around them; they are expensive when they're checked after the fact, as in the baker's plan.

![Tonight's run](figures/day34_gantt.png)

Read the Gantt chart: the two ovens are the busiest machines; cakes and cookies go first (they don't need the proofer); breads are mixed early so their 90-minute proof finishes while the ovens are free.

> **Decision it supports:** in what order each machine processes the night's jobs.
>
> **Other decisions you can support the same way:** print shops · machine shops · hospital operating rooms · software build pipelines · laboratory instruments.

## 34.4 A short-staffed night

With only one mixer operator and one oven crew, not everything can be ready on time. Now the objective matters.

```python
# ---------- 34.4 A short-staffed night: one mixer, one oven ----------
for objective in ("makespan", "lateness"):
    s3, st3, sc3 = solve_schedule(build_schedule(jobs, ops, machines, rules, one_of_each=True, objective=objective), ops)
    report(f"one mixer and one oven, objective {objective}", sc3)
    print("    late:", {j: f"{v} min" for j, v in summary(sc3, jobs)["late"].items()})
```

```text
one mixer and one oven, objective makespan: done at 08:04; 7 late jobs, 1102 minutes late; 0 rule breaks
    late: {'J03': '174 min', 'J04': '128 min', 'J05': '221 min', 'J10': '67 min', 'J11': '234 min', 'J13': '94 min', 'J16': '184 min'}
one mixer and one oven, objective lateness: done at 09:14; 6 late jobs, 564 minutes late; 0 rule breaks
    late: {'J04': '122 min', 'J06': '37 min', 'J07': '197 min', 'J14': '194 min', 'J16': '1 min', 'J17': '13 min'}
```

- **"Finish everything as early as possible"** (makespan, proven optimal: 08:04) leaves 7 jobs late by 1,102 minutes in total.
- **"Be as little late as possible"** finishes later (09:14) but has 6 jobs late by 564 minutes: half the lateness. On a bad night, choose the objective that matches what customers feel.

### The planted bug

`day34/bug_precedence.py` is a colleague's quick version of the model: "each step starts after the step before". Run it.

```python
for job, g in ops.groupby("job_id"):
    steps = sorted(g.step)
    for a, b in zip(steps, steps[1:]):
        m.add(start[job, b] >= start[job, a])                 # each step after the one before
```

<details><summary>What happens, and why</summary>

```text
OPTIMAL done at 07:36
57 rule breaks, e.g.:
    J01: proof starts before mix ends
    J01: bake starts before proof ends
    J01: pack starts before bake ends
    J02: proof starts before mix ends
```

"Optimal", and 57 rule breaks: proofing starts before mixing ends, baking before proofing ends. `start[b] >= start[a]` says the next step *starts* after the previous one *starts*; it must be `start[b] >= end[a]`. With interval variables the end is always there: use it. The checker, written separately, found what the solver couldn't.

</details>

## Mini-project: can the night crew start later?

The night crew would like to start at 21:30, 22:00 or 22:30 instead of 21:00 (the shift premium is lower before midnight, and it's easier to staff). The 04:00 loading and the cake deadlines don't move. How late can the run start with nothing late?

<details><summary>Solution</summary>

```python
"""Day 34 mini-project: the night crew asks to start later. How late can tonight's run start
with nothing late at 04:00 loading?

Run from the repository root (after day06/load_tables.py):  python day34/mini_project_start.py   (about 1.5 minutes)
"""
import sys

import duckdb
import pandas as pd

sys.path.insert(0, "day34")
from bakery_shop import build_schedule, check_schedule, clock, load_shop, solve_schedule, summary

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
jobs, ops, machines, rules = load_shop(con)
rows = []
for delay in (0, 30, 60, 90):
    later = jobs.assign(due_minute=jobs.due_minute - delay)       # same clock deadlines, less time to meet them
    m = build_schedule(later, ops, machines, rules, objective="lateness")
    solver, status, sched = solve_schedule(m, ops)
    s = summary(sched, later)
    rows.append((clock(delay), clock(s["makespan"] + delay), s["late_jobs"], s["minutes_late"],
                 ", ".join(f"{j} {v}" for j, v in s["late"].items()) or "-",
                 int(solver.best_objective_bound // 1000),                  # proven: no plan is less late than this
                 len(check_schedule(sched, later, machines, rules))))
print(pd.DataFrame(rows, columns=["start", "done_at", "late_jobs", "minutes_late", "late (minutes)", "at_least_late",
                                  "rule_breaks"])
      .to_string(index=False))
```

```text
start done_at  late_jobs  minutes_late late (minutes)  at_least_late  rule_breaks
21:00   03:18          0             0              -              0            0
21:30   03:43          0             0              -              0            0
22:00   04:41          0             0              -              0            0
22:30   05:52          1            14         J06 14              2            0
```

- Starting at **21:30 or 22:00, nothing is late**. At 22:00 the run finishes at 04:41, after loading starts, but the last jobs are cakes and cookies due at 05:00 or later.
- At **22:30, at least 2 minutes of lateness can't be avoided**: CP-SAT's bound proves it. The plan it found is 14 minutes late on one sourdough job.

*Note to the bakery manager:* "The night run can start at 22:00 instead of 21:00 with every deadline met. Note that the run then ends at 04:41, so the vans would load the last breads at about 04:00 and the cakes later. Day 37 checks what a later dispatch costs on the road."

</details>

## Commit

```bash
git add .
git commit -m "Day 34: job-shop scheduling of the bakery run"
git push
```

### Day 34 checklist

- [ ] I can model a job shop with interval variables, `NoOverlap` and precedence.
- [ ] I can model a machine choice with optional intervals and a shared capacity with `Cumulative`.
- [ ] I can add maximum waits between steps.
- [ ] I compare makespan and lateness objectives and say which fits the situation.
- [ ] I use deterministic work limits and report the bound with the plan.
- [ ] I check schedules with an independent checker.

---

# DAY 35 — Mini-project: A 4-Week Warehouse Roster

December at the Montreal DC: four weeks, the last one a +10% peak. Days 32 and 33 built one week. Today's mini-project builds a month: rest rules that cross from one week into the next, Day 33's requests, fairness measured over the month, and a question from Day 33's memo: should two pickers be certified as forklift drivers so that everyone gets every other weekend off?

| Time | Block |
|---|---|
| 0:00 – 0:30 | 35.1 The brief and the model |
| 0:30 – 1:15 | 35.2 Four normal weeks in one model |
| 1:15 – 1:45 | 35.3 December in one model |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 35.4 December, one week at a time |
| 2:45 – 3:45 | 35.5 Fairness policies |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | 35.6 The memo |
| 4:45 – 5:00 | Commit with Git + checklist |

## 35.1 The brief and the model

- **Horizon:** 4 weeks, 28 days; week 4 needs 10% more people on every shift (rounded up), with agency pickers at $36 an hour.
- **Rules:** Day 32's, in every week; the 11-hour rest rule runs across week boundaries too.
- **Requests:** Day 33's 47 requests for week 1, at $50 each.
- **Fairness over the month** (full-timers): nights worked, and weekends worked (a weekend counts if any shift falls on Saturday or Sunday).
- **Options:** two weekend teams that alternate (every other weekend off); certify E13 and E14 as forklift drivers (`forklift_certification`: $1,800 each for the course, $1.50 an hour more).

Create `day35/four_weeks.py`:

```python
"""Day 35 mini-project: a 4-week roster for the Montreal DC warehouse (December: the 4th week is the peak).

Import it:  from four_weeks import build_four_weeks, solve_four_weeks
"""
import math

import pandas as pd
from ortools.sat.python import cp_model

from roster import DAYS, SHIFTS, TOO_CLOSE

ROLES = ["picker", "forklift"]


def calendar(weeks):
    return [(w, d) for w in range(1, weeks + 1) for d in DAYS]


def build_four_weeks(staff, shifts, coverage, rules, last_sunday, weeks=4, peak=None, requests=None, weight=50.0,
                     max_nights=None, max_weekends=None, agency_rate=36.0, new_drivers=(), weekend_teams=None,
                     first_week=1, nights_so_far=None, raise_per_hour=0.0):
    """work[e, (w, d), s, r]. peak: {week: extra share of people}. new_drivers: pickers certified for forklifts.
    Fairness over the whole horizon: at most max_nights nights and max_weekends weekends worked per full-timer.
    Pay is in half-cents (Day 32); a broken week-1 request costs `weight` dollars."""
    m = cp_model.CpModel()
    E, cal = list(staff.index), calendar(weeks)
    drives = {e: staff.role[e] == "forklift" or e in new_drivers for e in E}
    work = {(e, c, s, r): m.new_bool_var(f"w_{e}_{c[0]}{c[1]}_{s}_{r}")
            for e in E for c in cal for s in SHIFTS for r in ROLES if r == "picker" or drives[e]}
    on = {(e, c, s): sum(work[e, c, s, r] for r in ROLES if (e, c, s, r) in work) for e in E for c in cal for s in SHIFTS}
    need = {}
    for (w, d) in cal:
        extra = (peak or {}).get(w, 0.0)
        for s in SHIFTS:
            for r in ROLES:
                need[(w, d), s, r] = math.ceil(coverage[d, s, r] * (1 + extra))
    agency = {(c, s): m.new_int_var(0, 20, f"agency_{c[0]}{c[1]}_{s}") for c in cal for s in SHIFTS}
    for (c, s, r), people in need.items():
        m.add(sum(work[e, c, s, r] for e in E if (e, c, s, r) in work) + (agency[c, s] if r == "picker" else 0) >= people)
    hours = int(rules["hours_per_shift"])
    cost = []
    for e in E:
        wage = int(round(200 * (staff.hourly_wage[e] + (raise_per_hour if e in new_drivers else 0.0))))
        for c in cal:
            m.add(sum(on[e, c, s] for s in SHIFTS) <= 1)
        days = [None] + cal
        for a, b in zip(days, days[1:]):                         # 11 hours off, across weeks too
            for s1, s2 in TOO_CLOSE:
                if a is None:
                    if last_sunday[e] == s1:
                        m.add(on[e, b, s2] == 0)
                else:
                    m.add(on[e, a, s1] + on[e, b, s2] <= 1)
        for w in range(1, weeks + 1):
            n = sum(on[e, (w, d), s] for d in DAYS for s in SHIFTS)
            limit = int(rules["part_time_max_shifts"]) if staff.contract[e] == "part-time" else int(rules["max_hours_per_week"]) // hours
            m.add(n <= limit)
            paid = n
            if staff.contract[e] == "full-time":
                paid = m.new_int_var(int(rules["full_time_min_shifts"]), 7, f"paid_{e}_{w}")
                m.add(paid >= n)
            ot = m.new_int_var(0, 7 * hours, f"ot_{e}_{w}")
            m.add(ot >= hours * n - int(rules["overtime_after_hours"]))
            cost.append(wage * hours * paid + (wage // 2) * ot)
        cost += [int(round(200 * shifts.premium_per_hour[s])) * hours * on[e, c, s] for c in cal for s in SHIFTS]
    cost.append(int(round(200 * agency_rate)) * hours * sum(agency.values()))
    m.pay = sum(cost)
    # fairness over the horizon (full-timers)
    m.nights, m.weekends = {}, {}
    for e in staff.index[staff.contract == "full-time"]:
        m.nights[e] = m.new_int_var(0, 7 * weeks, f"nights_{e}")
        m.add(m.nights[e] == sum(on[e, c, "N"] for c in cal))
        worked = []
        for w in range(1, weeks + 1):
            ww = m.new_bool_var(f"weekend_{e}_{w}")
            for d in ("Sat", "Sun"):
                for s in SHIFTS:
                    m.add(ww >= on[e, (w, d), s])
            worked.append(ww)
        m.weekends[e] = m.new_int_var(0, weeks, f"weekends_{e}")
        m.add(m.weekends[e] == sum(worked))
        m.weekend_worked = getattr(m, "weekend_worked", {})
        for w, ww in enumerate(worked, start=1):
            m.weekend_worked[e, w] = ww
        if weekend_teams is not None:                            # team A works odd weekends, team B even ones
            for w, ww in enumerate(worked, start=first_week):
                if (w % 2 == 1) != (weekend_teams[e] == "A"):
                    m.add(ww == 0)
        if max_nights is not None:
            m.add(m.nights[e] + (nights_so_far or {}).get(e, 0) <= max_nights)
        if max_weekends is not None:
            m.add(m.weekends[e] <= max_weekends)
    # a redundant row that helps the search: each weekend day needs enough full-timers working that weekend
    full = list(staff.index[staff.contract == "full-time"])
    part = [e for e in E if e not in full]
    for w in range(1, weeks + 1):
        for d in ("Sat", "Sun"):
            people = sum(need[(w, d), s, r] for s in SHIFTS for r in ROLES)
            m.add(sum(m.weekend_worked[e, w] for e in full)
                  + sum(on[e, (w, d), s] for e in part for s in SHIFTS)
                  + sum(agency[(w, d), s] for s in SHIFTS) >= people)
    broken = []
    if requests is not None:                                     # Day 33's requests, for week 1
        for r in requests.itertuples():
            b = m.new_bool_var(f"broken_{r.Index}")
            if r.request == "day_off":
                xs = [on[r.employee_id, (1, r.day), s] for s in SHIFTS]
            elif r.request == "no_nights":
                xs = [on[r.employee_id, (1, d), "N"] for d in DAYS]
            else:
                xs = [on[r.employee_id, (1, d), s] for d in DAYS for s in SHIFTS if s != r.shift]
            for x in xs:
                m.add(b >= x)
            broken.append((int(r.priority), b))
    m.broken = broken
    m.minimize(m.pay + int(round(200 * weight)) * sum(p * b for p, b in broken))
    m.agency, m.work = agency, work
    return m


def teams(staff):
    """Every other weekend off: full-timers alternate, by employee number."""
    full = staff.index[staff.contract == "full-time"]
    return {e: "A" if k % 2 == 0 else "B" for k, e in enumerate(full)}


def rolling(staff, shifts, coverage, rules, last_sunday, weeks=4, peak=None, requests=None, max_nights=None,
            weekend_teams=None, work=10.0, **kw):
    """Plan one week at a time: carry last Sunday's shifts and the nights worked so far into the next week."""
    last, nights, total, rosters, info = dict(last_sunday), {}, {}, [], []
    for w in range(1, weeks + 1):
        m = build_four_weeks(staff, shifts, coverage, rules, last, weeks=1, peak={1: (peak or {}).get(w, 0.0)},
                             requests=requests if w == 1 else None, max_nights=max_nights,
                             weekend_teams=weekend_teams, first_week=w, nights_so_far=nights, **kw)
        out, r = solve_four_weeks(m, work)
        if "pay" not in out:
            return {"status": f"week {w}: {out['status']}"}, None
        r["week"] = w
        rosters.append(r)
        sunday = r[r.day == "Sun"].set_index("employee_id")["shift"].to_dict()
        last = {e: sunday.get(e, "off") for e in staff.index}
        for e, n in r[r["shift"] == "N"].groupby("employee_id").size().items():
            nights[e] = nights.get(e, 0) + n
        info.append(out)
    roster = pd.concat(rosters, ignore_index=True)
    full = staff.index[staff.contract == "full-time"]
    weekends = roster[roster.day.isin(["Sat", "Sun"]) & roster.employee_id.isin(full)].groupby("employee_id").week.nunique()
    night_counts = [nights.get(e, 0) for e in full]
    return {"status": "rolling, " + ", ".join(o["status"].lower() for o in info),
            "pay": round(sum(o["pay"] for o in info), 2), "bound": None,
            "agency_shifts": sum(o["agency_shifts"] for o in info), "broken": sum(o["broken"] for o in info),
            "most_nights": max(night_counts), "fewest_nights": min(night_counts),
            "most_weekends": int(weekends.reindex(full, fill_value=0).max()),
            "all_four_weekends": int((weekends == 4).sum())}, roster


def solve_four_weeks(m, work=40.0):
    s = cp_model.CpSolver()
    s.parameters.max_deterministic_time = work
    s.parameters.num_workers = 8
    s.parameters.interleave_search = True
    status = s.solve(m)
    ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    roster = pd.DataFrame([(e, c[0], c[1], sh, r) for (e, c, sh, r), v in m.work.items() if ok and s.value(v)],
                          columns=["employee_id", "week", "day", "shift", "role"])
    out = {"status": s.status_name(status)}
    if ok:
        nights = [s.value(v) for v in m.nights.values()]
        weekends = [s.value(v) for v in m.weekends.values()]
        out.update(pay=round(s.value(m.pay) / 200, 2), bound=round(s.best_objective_bound / 200, 2),
                   agency_shifts=sum(s.value(v) for v in m.agency.values()),
                   broken=sum(s.value(b) for _, b in m.broken), most_nights=max(nights), fewest_nights=min(nights),
                   most_weekends=max(weekends), all_four_weekends=sum(x == 4 for x in weekends))
    return out, roster
```

Two parts are new. `rolling` plans one week at a time, carrying last Sunday's shifts and the nights worked so far into the next week. The **redundant row** ("each weekend day needs enough full-timers working that weekend") adds nothing a whole-number plan doesn't already satisfy, but it tightens the bound, as on Day 27.

## 35.2 Four normal weeks in one model

A known-answer check first: four normal weeks with no requests should cost four times Day 32's week.

```python
# ---------- 35.2 Four normal weeks in one model ----------
out, roster = solve_four_weeks(build_four_weeks(staff, shifts, coverage, rules, last_sunday), work=40)
rows = [line("4 normal weeks, one model", out)]
print(f"4 x Day 32's week = ${4 * 28_876.48:,.2f}")
problems = []
prev = dict(last_sunday)
for w, g in roster.groupby("week"):
    problems += check_roster(g, staff, coverage, rules, prev)       # Day 32's checker, one week at a time
    sunday = g[g.day == "Sun"].set_index("employee_id")["shift"].to_dict()
    prev = {e: sunday.get(e, "off") for e in staff.index}
print(f"Day 32's checker, week by week: {len(problems)} rule breaks")
```

```text
4 normal weeks, one model: OPTIMAL
4 x Day 32's week = $115,505.92
Day 32's checker, week by week: 0 rule breaks
```

$115,505.92, exactly 4 × $28,876.48, proven optimal, and Day 32's checker finds nothing wrong in any week.

## 35.3 December in one model, and 35.4 one week at a time

```python
# ---------- 35.3 December in one model ----------
out, roster = solve_four_weeks(build_four_weeks(staff, shifts, coverage, rules, last_sunday, **DECEMBER), work=40)
rows.append(line("December, one model", out))

# ---------- 35.4 December, one week at a time ----------
out, roster = rolling(staff, shifts, coverage, rules, last_sunday, **DECEMBER)
rows.append(line("December, week by week", out))
print(pd.DataFrame(rows).to_string(index=False))
```

```text
December, one model: FEASIBLE
December, week by week: rolling, optimal, optimal, optimal, optimal
                     plan       pay     bound  agency_shifts  broken  most_nights  fewest_nights  most_weekends  all_four_weekends
4 normal weeks, one model 115505.92 115505.92              0       0            9              0              4                 26
      December, one model 125861.40 125859.68              8       0           10              0              4                 26
   December, week by week 125861.40       NaN              8       0            9              0              4                 21
```

- **One model** for the whole month finds $125,861.40 and proves no plan costs less than $125,859.68: within $2.
- **Week by week**, each week solved to optimality, gives the same $125,861.40. Here the weeks hardly interact (only Sunday-to-Monday rest), so planning them one at a time loses nothing. When weeks interact more (monthly fairness, hours averaged over two weeks), week-by-week planning can lose money or even fail: see 35.5 and Day 39.
- **The month is unfair**: some full-timers work 9 nights, others none; 21 of 26 work all four weekends.

## 35.5 Fairness policies

```python
# ---------- 35.5 Fairness policies ----------
rows = []
team = teams(staff)
drivers = tuple(training.index)
for name, kw in [("today's rules", {}),
                 ("every other weekend", {"weekend_teams": team}),
                 ("every other weekend, 2 new drivers", {"weekend_teams": team, "new_drivers": drivers,
                                                         "raise_per_hour": float(training.raise_per_hour.iloc[0])}),
                 ("... and at most 6 nights", {"weekend_teams": team, "new_drivers": drivers, "max_nights": 6,
                                               "raise_per_hour": float(training.raise_per_hour.iloc[0])})]:
    out, roster = rolling(staff, shifts, coverage, rules, last_sunday, **DECEMBER, **kw)
    rows.append(line(name, out))
print(pd.DataFrame(rows).drop(columns="bound").to_string(index=False))
print(f"training {len(drivers)} drivers: ${training.course_cost.sum():,} once")
```

```text
today's rules: rolling, optimal, optimal, optimal, optimal
every other weekend: week 4: INFEASIBLE
every other weekend, 2 new drivers: rolling, optimal, optimal, optimal, optimal
... and at most 6 nights: rolling, optimal, optimal, optimal, optimal
                              plan       pay  agency_shifts  broken  most_nights  fewest_nights  most_weekends  all_four_weekends
                     today's rules 125861.40            8.0     0.0          9.0            0.0            4.0               21.0
               every other weekend       NaN            NaN     NaN          NaN            NaN            NaN                NaN
every other weekend, 2 new drivers 126768.04           18.0     7.0         17.0            0.0            2.0                0.0
          ... and at most 6 nights 128075.88           20.0     7.0          6.0            0.0            2.0                0.0
training 2 drivers: $3,600 once
```

- **Every other weekend off, with today's drivers, is impossible in week 4**: the peak weekend needs 16 forklift shifts, and with half the full-time drivers off each weekend the drivers can cover only 14 (Day 33 found the same limit). Agency staff can't drive forklifts.
- **With E13 and E14 certified**, every-other-weekend works: $126,768.04 for the month, **$906.64 more** than today's rules (including the drivers' raise), plus **$3,600 once** for the courses. Nobody works more than 2 weekends; 7 week-1 requests are broken.
- But nights become *less* fair (one person works 17 of 28): teams fix the weekends and push nights onto fewer people. **Adding "at most 6 nights"** costs another $1,307.84: $128,075.88.

## 35.6 The memo

<details><summary>A model memo</summary>

*To: the DC manager. Re: December roster and the weekend policy.*

"**December** can be covered legally for $125,861 over the four weeks, including 8 agency picker shifts in the peak week. All 47 week-1 requests can be granted.

**Every other weekend off** is not possible in the peak week with today's forklift drivers: the peak weekend needs 16 forklift shifts and alternating teams can cover 14. Certifying **two pickers (E13, E14) as forklift drivers** ($3,600 once, $1.50 an hour more for them) makes it possible, for about $900 more over December.

Fair weekends alone push night shifts onto a few people; adding **a limit of 6 nights a month** brings nights in line for another $1,300.

**Recommendation:** certify the two drivers in January; adopt every-other-weekend and the 6-night limit from January, when there is no peak; review the cost after three months. Total for December if adopted now: $128,076 against $125,861 (+1.8%)."

</details>

## Commit

```bash
git add .
git commit -m "Day 35: 4-week roster mini-project"
git push
```

### Day 35 checklist

- [ ] I extended a weekly model to four weeks, with rest rules across week boundaries.
- [ ] I checked the month against a known answer (4 × one week).
- [ ] I compared one model for the month with planning week by week.
- [ ] I measured fairness over the month (nights, weekends).
- [ ] I found why a policy is infeasible, and priced the fix (two new drivers).
- [ ] I wrote a recommendation with costs, one-time costs and side effects.

---

# DAY 36 — Vehicle Routing I: TSP and CVRP

Every morning the bakery's breads leave the dock for 10 Montreal and Laval stores and 24 café customers: 161 bread racks. Which vehicle visits which stops, in which order? Today, the two classic problems underneath: the **travelling salesman problem** (TSP: one tour through every stop) and the **capacitated vehicle routing problem** (CVRP: several routes, each within a vehicle's capacity). You compare simple heuristics, the Clarke and Wright savings method, OR-Tools' routing solver and an exact CP-SAT model.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 36.1 The stops; 36.2 one tour (TSP) |
| 0:45 – 1:45 | 36.3 Routes for vans (CVRP) |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 36.4 More search, better routes |
| 2:45 – 3:45 | OR-Tools routing, step by step; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: vans or trucks? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 36.1 The stops

`delivery_stops` lists the depot (the bakery dock beside DC-MTL), the stores and the cafés, with their racks; `delivery_matrix` the road kilometres and minutes between every pair (straight-line distance × 1.35 for the road network, at 30 km/h in the early morning). Create `day36/routing.py`:

```python
"""Day 36: the morning bread run. Tours and routes with simple heuristics, OR-Tools routing and CP-SAT.

Import it:  from routing import load_run, tour_km, nearest_neighbour, two_opt, tsp_exact, sweep, savings, solve_cvrp
"""
import math

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model


def load_run(con):
    stops = con.sql("SELECT * FROM delivery_stops").df().set_index("stop_id")
    ids = list(stops.index)                                  # DEPOT first
    mat = con.sql("SELECT * FROM delivery_matrix").df()
    km = mat.pivot(index="from_stop", columns="to_stop", values="km").loc[ids, ids].to_numpy()
    minutes = mat.pivot(index="from_stop", columns="to_stop", values="minutes").loc[ids, ids].to_numpy()
    return stops, ids, km, minutes


def tour_km(tour, km):
    """Length of a closed tour or route that starts and ends at the depot (node 0)."""
    path = [0] + [n for n in tour if n != 0] + [0]
    return float(sum(km[a][b] for a, b in zip(path, path[1:])))


def nearest_neighbour(km, nodes=None):
    nodes = set(range(1, len(km))) if nodes is None else set(nodes)
    tour, here = [], 0
    while nodes:
        here = min(nodes, key=lambda n: (km[here][n], n))
        tour.append(here)
        nodes.remove(here)
    return tour


def two_opt(tour, km):
    """Reverse any piece of the tour that makes it shorter, until no reversal helps."""
    best = [0] + tour + [0]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                a, b, c, d = best[i - 1], best[i], best[j], best[j + 1]
                if km[a][c] + km[b][d] < km[a][b] + km[c][d] - 1e-9:
                    best[i:j + 1] = reversed(best[i:j + 1])
                    improved = True
    return best[1:-1]


def tsp_exact(km, seconds=60):
    """The shortest tour, proven, with CP-SAT's circuit constraint (distances in metres)."""
    n = len(km)
    m = cp_model.CpModel()
    arcs, x = [], {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = m.new_bool_var(f"x_{i}_{j}")
                arcs.append((i, j, x[i, j]))
    m.add_circuit(arcs)
    m.minimize(sum(int(round(1000 * km[i][j])) * v for (i, j), v in x.items()))
    s = cp_model.CpSolver()
    s.parameters.num_workers = 8
    s.parameters.interleave_search = True
    s.parameters.max_time_in_seconds = seconds
    status = s.solve(m)
    nxt = {i: j for (i, j), v in x.items() if s.value(v)}
    tour, here = [], nxt[0]
    while here != 0:
        tour.append(here)
        here = nxt[here]
    return tour, s.status_name(status), s.best_objective_bound / 1000


def sweep(stops, ids, demand, capacity):
    """The dispatcher's method: sort customers by compass angle around the depot, fill a van, start the next."""
    lat0, lon0 = stops.latitude[ids[0]], stops.longitude[ids[0]]
    angle = {n: math.atan2(stops.latitude[ids[n]] - lat0, (stops.longitude[ids[n]] - lon0) * math.cos(math.radians(lat0)))
             for n in range(1, len(ids))}
    routes, load = [[]], 0
    for n in sorted(angle, key=angle.get):
        if load + demand[n] > capacity:
            routes.append([])
            load = 0
        routes[-1].append(n)
        load += demand[n]
    return routes


def savings(km, demand, capacity):
    """Clarke and Wright (1964): start with one route per customer, then merge the two routes whose
    joining saves the most distance (km[0,i] + km[0,j] - km[i,j]), as long as the van has room."""
    n = len(km)
    route_of = {i: [i] for i in range(1, n)}
    load = {id(r): demand[r[0]] for r in route_of.values()}
    pairs = sorted(((km[0][i] + km[0][j] - km[i][j], i, j) for i in range(1, n) for j in range(i + 1, n)), reverse=True)
    for save, i, j in pairs:
        if save <= 0:
            break
        ri, rj = route_of[i], route_of[j]
        if ri is rj or load[id(ri)] + load[id(rj)] > capacity:
            continue
        if ri[-1] != i:                                  # join only at route ends: [... i] + [j ...]
            ri.reverse()
        if rj[0] != j:
            rj.reverse()
        if ri[-1] != i or rj[0] != j:
            continue                                     # i or j is inside its route: can't join there
        merged = ri + rj
        load[id(merged)] = load.pop(id(ri)) + load.pop(id(rj))
        for k in merged:
            route_of[k] = merged
    unique = {id(r): r for r in route_of.values()}
    return sorted(unique.values(), key=lambda r: r[0])


def solve_cvrp(km, demand, capacities, solutions=200, fixed_cost=None, with_vehicle=False):
    """OR-Tools routing: vehicles with capacities; minimize km (plus a fixed cost per vehicle used, if given).
    Guided local search stopped after a number of improving solutions, so every run gives the same routes."""
    n = len(km)
    manager = pywrapcp.RoutingIndexManager(n, len(capacities), 0)
    model = pywrapcp.RoutingModel(manager)
    metres = [[int(round(1000 * km[i][j])) for j in range(n)] for i in range(n)]

    def distance(from_index, to_index):
        return metres[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit = model.RegisterTransitCallback(distance)
    model.SetArcCostEvaluatorOfAllVehicles(transit)
    load = model.RegisterUnaryTransitCallback(lambda index: int(demand[manager.IndexToNode(index)]))
    model.AddDimensionWithVehicleCapacity(load, 0, list(capacities), True, "racks")
    if fixed_cost is not None:
        model.SetFixedCostOfAllVehicles(int(fixed_cost))
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.solution_limit = solutions
    solution = model.SolveWithParameters(params)
    routes = []
    for v in range(len(capacities)):
        index, route = model.Start(v), []
        while not model.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                route.append(node)
            index = solution.Value(model.NextVar(index))
        if route:
            routes.append((v, route) if with_vehicle else route)
    return routes
```

## 36.2 One tour through every stop (TSP)

Suppose one vehicle could carry everything: what is the shortest tour from the dock through all 34 stops and back? Create `day36/run_routes.py`:

```python
# ---------- 36.1 The stops ----------
print(stops.groupby("kind").agg(stops=("name", "size"), racks=("racks", "sum")).to_string())
print(f"farthest pair {km.max():.1f} km; a van takes {van} racks")

# ---------- 36.2 One tour through every stop (TSP) ----------
nn = nearest_neighbour(km)
better = two_opt(nn, km)
exact, status, bound = tsp_exact(km)
ortools = solve_cvrp(km, [0] * len(ids), [10 ** 6])[0]
for name, tour in [("nearest neighbour", nn), ("nearest neighbour + 2-opt", better), ("OR-Tools routing", ortools),
                   ("CP-SAT circuit", exact)]:
    print(f"{name:<26} {tour_km(tour, km):7.2f} km")
print(f"CP-SAT: {status}, no tour shorter than {bound:.2f} km")
```

```text
       stops  racks
kind
cafe      24     59
depot      1      0
store     10    102
farthest pair 33.8 km; a van takes 18 racks
nearest neighbour           147.68 km
nearest neighbour + 2-opt   131.45 km
OR-Tools routing            128.15 km
CP-SAT circuit              128.15 km
CP-SAT: OPTIMAL, no tour shorter than 128.15 km
```

- **Nearest neighbour** (always drive to the closest stop not yet visited) is quick and 15% too long: the last stops are left scattered far apart.
- **2-opt** (reverse any piece of the tour that makes it shorter) brings it within 2.6% of the best.
- **OR-Tools' routing solver** and **CP-SAT's `add_circuit`** both find 128.15 km, and CP-SAT proves nothing shorter exists.

The TSP is the textbook hard problem, but at this size exact methods are quick. They stop being quick with hundreds of stops, or with capacities and time windows. That is where heuristics earn their keep.

## 36.3 Routes for vans (CVRP)

A van carries 18 racks, so the run needs at least $$\lceil 161/18 \rceil = 9$$ vans.

```python
# ---------- 36.3 Routes for vans (CVRP) ----------
print(f"{sum(racks)} racks / {van} per van: at least {math.ceil(sum(racks) / van)} vans")
results = {}
for name, routes in [("sweep (the dispatcher's method)", sweep(stops, ids, racks, van)),
                     ("savings (Clarke and Wright)", savings(km, racks, van)),
                     ("OR-Tools, guided local search", solve_cvrp(km, racks, [van] * 15, solutions=1000))]:
    results[name] = routes
    loads = [sum(racks[n] for n in r) for r in routes]
    print(f"{name:<32} {len(routes)} vans, {sum(tour_km(r, km) for r in routes):7.2f} km, fullest {max(loads)}, emptiest {min(loads)}")
```

```text
161 racks / 18 per van: at least 9 vans
sweep (the dispatcher's method)  12 vans,  344.29 km, fullest 18, emptiest 7
savings (Clarke and Wright)      10 vans,  266.93 km, fullest 18, emptiest 12
OR-Tools, guided local search    10 vans,  258.30 km, fullest 18, emptiest 7
```

- **Sweep** (the dispatcher's method: sort stops by compass direction around the depot, fill a van, start the next) needs 12 vans and 344 km.
- **Savings** (Clarke and Wright, 1964): start with one van per stop, then repeatedly join the two routes whose joining saves the most kilometres, while the van has room. 10 vans, 267 km. A method from 1964, in 30 lines, is a strong baseline.
- **OR-Tools** with guided local search: 10 vans, **258 km**.

None of these proves anything about the best possible plan: 9 vans would be enough by capacity alone, but no method found a 9-van plan, and none can say whether one exists.

## 36.4 More search, better routes

OR-Tools' routing solver starts from a quick first solution (`PATH_CHEAPEST_ARC`: extend each route to the nearest stop), then improves it with **guided local search**: moves stops between and within routes, and penalizes features of the plan it keeps returning to, to escape local optima. It never proves optimality; it stops when told.

```python
# ---------- 36.4 More search, better routes ----------
for n in (50, 200, 1000):
    routes = solve_cvrp(km, racks, [van] * 15, solutions=n)
    print(f"OR-Tools stopped after {n:>4} solutions: {len(routes)} vans, {sum(tour_km(r, km) for r in routes):.2f} km")
```

```text
OR-Tools stopped after   50 solutions: 10 vans, 284.61 km
OR-Tools stopped after  200 solutions: 10 vans, 272.11 km
OR-Tools stopped after 1000 solutions: 10 vans, 258.30 km
```

After 200 solutions OR-Tools is still behind savings; after 1,000 it is 3% ahead. We stop after a number of **solutions**, not seconds, so every run gives the same routes. A time limit gives better routes on faster computers and different ones on each run.

![Van routes](figures/day36_routes.png)

The OR-Tools routing pattern, which every routing model follows:

1. A `RoutingIndexManager` (stops, vehicles, depot) and a `RoutingModel`.
2. **Callbacks** that give the cost or time between two *indexes*: always convert with `manager.IndexToNode`.
3. **Dimensions** for quantities that accumulate along a route: racks (capacity), time (Day 37).
4. Search parameters: first solution strategy, metaheuristic, limit.

> **Decision it supports:** which vehicle visits which customers, in what order.
>
> **Other decisions you can support the same way:** parcel delivery · waste collection · home-care visits · school buses · field technicians.

### The planted bug

`day36/bug_index.py`: a colleague's distance callback crashed with an `IndexError`, so they "fixed" it. Run it.

```python
metres = [[int(round(1000 * km[i][j])) for j in range(len(ids))] for i in range(len(ids))]
cost = model.RegisterTransitCallback(lambda i, j: metres[i % len(ids)][j % len(ids)])     # i and j: the stops
model.SetArcCostEvaluatorOfAllVehicles(cost)
load = model.RegisterUnaryTransitCallback(lambda i: racks[manager.IndexToNode(i)])
model.AddDimensionWithVehicleCapacity(load, 0, [18] * vans, True, "racks")
```

<details><summary>What happens, and why</summary>

```text
the solver's objective: 131.67 km
10 vans; the routes really drive 310.05 km
```

The solver thinks its routes drive 131.67 km; they really drive **310.05 km**. OR-Tools passes *indexes* to callbacks, not stop numbers: each vehicle has its own start and end index, numbered after the stops. `metres[i][j]` crashed on those, and `% len(ids)` hid the crash by mapping them to the wrong stops. The fix is `manager.IndexToNode(i)`. A crash is a gift; never silence it without understanding it. The independent check (`tour_km` on the routes) shows the gap at once.

</details>

## Mini-project: vans or trucks?

The bakery has 6 vans (18 racks, $90 a day, $0.40 a km) and 3 trucks (36 racks, $150 a day, $0.65 a km). Ignoring delivery windows for now, what would the run cost with vans only, with trucks only (any number), and with today's fleet? Is the plan with the fewest kilometres the cheapest?

<details><summary>Solution</summary>

```python
"""Day 36 mini-project: vans or trucks? Ignoring time windows for now, what does the morning run cost
with vans only, with trucks only (unlimited numbers), and with today's fleet of 6 vans and 3 trucks?

Run from the repository root (after day06/load_tables.py):  python day36/mini_project_fleet.py   (about 1 minute)
"""
import sys

import duckdb
import pandas as pd

sys.path.insert(0, "day36")
from routing import load_run, solve_cvrp, tour_km

con = duckdb.connect("northbound.duckdb", read_only=True)
stops, ids, km, minutes = load_run(con)
racks = [int(stops.racks[s]) for s in ids]
fleet = con.sql("SELECT * FROM delivery_fleet").df().set_index("vehicle_type")
rows = []
for name, types in [("vans only", ["van"] * 15), ("trucks only", ["truck"] * 8),
                    ("6 vans + 3 trucks, shortest km", ["van"] * 6 + ["truck"] * 3)]:
    found = solve_cvrp(km, racks, [int(fleet.racks[t]) for t in types], solutions=1000, with_vehicle=True)
    used = [types[v] for v, _ in found]
    routes = [r for _, r in found]
    day = sum(fleet.cost_per_day[t] for t in used)
    drive = sum(fleet.cost_per_km[t] * tour_km(r, km) for t, r in zip(used, routes))
    rows.append((name, used.count("van"), used.count("truck"), round(sum(tour_km(r, km) for r in routes), 2),
                 day, round(drive, 2), round(day + drive, 2)))
print(pd.DataFrame(rows, columns=["fleet", "vans", "trucks", "km", "day_costs", "km_costs", "total"]).to_string(index=False))
```

```text
                         fleet  vans  trucks     km  day_costs  km_costs   total
                     vans only    10       0 258.30        900    103.32 1003.32
                   trucks only     0       5 170.10        750    110.56  860.56
6 vans + 3 trucks, shortest km     4       3 184.19        810    107.86  917.86
```

- **Trucks only is cheapest** ($860.56 a day): 5 trucks, 170 km. A truck costs 1.67 times a van but carries twice as much.
- **Vans only costs $1,003**: 10 day charges.
- The **fewest-kilometres plan with today's fleet costs $917.86**, more than trucks only: minimizing kilometres is not minimizing cost when vehicles have day charges. Day 37 puts the costs into the objective.
- Before buying trucks: the windows and downtown streets of Day 37 change this answer.

</details>

## Commit

```bash
git add .
git commit -m "Day 36: TSP and CVRP"
git push
```

### Day 36 checklist

- [ ] I can build tours with nearest neighbour and 2-opt, and prove a short tour optimal with `add_circuit`.
- [ ] I can implement the savings method.
- [ ] I can set up an OR-Tools routing model: manager, callbacks, a capacity dimension, search parameters.
- [ ] I stop the search by solution count when I need repeatable routes.
- [ ] I always convert indexes with `IndexToNode`, and check routes independently.
- [ ] I know shortest is not cheapest when vehicles have fixed costs.

---

# DAY 37 — Vehicle Routing II: Time Windows and Mixed Fleets

Cafés need their bread before they open; stores receive between 04:30 and 07:00; downtown streets are too narrow for trucks. Today you add **time windows**, **service times**, a **mixed fleet** with day and kilometre costs, and **vehicle restrictions**, check every plan with an independent simulator, and answer Day 34's question: what does a later dispatch cost?

| Time | Block |
|---|---|
| 0:00 – 0:45 | 37.1 Delivery windows and the time dimension |
| 0:45 – 1:45 | 37.2 Today's fleet |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 37.3 What each rule costs |
| 2:45 – 3:45 | Heuristics and noise; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: dispatch time and café windows |
| 4:45 – 5:00 | Commit with Git + checklist |

## 37.1 Delivery windows and the time dimension

Time is a second **dimension**, like racks: it accumulates along a route (driving time plus the time spent at each stop), and each stop's arrival time must fall inside its window. Waiting is allowed (the "slack", up to 2 hours here). Create `day37/vrptw.py`:

```python
"""Day 37: routes with time windows, service times and a mixed fleet (OR-Tools routing).

Import it:  from vrptw import load_fleet, solve_vrptw, simulate
"""
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def load_fleet(con):
    fleet = con.sql("SELECT * FROM delivery_fleet").df()
    vehicles = []
    for r in fleet.itertuples():
        for k in range(1, int(r.vehicles) + 1):
            vehicles.append({"name": f"{r.vehicle_type}-{k}", "type": r.vehicle_type, "racks": int(r.racks),
                             "cost_per_day": float(r.cost_per_day), "cost_per_km": float(r.cost_per_km)})
    return vehicles


def hhmm(minute):
    return f"{int(minute) // 60:02d}:{int(minute) % 60:02d}"


def solve_vrptw(stops, ids, km, minutes, vehicles, windows=True, vans_only_downtown=True, max_route_minutes=None,
                count_service=True, solutions=300, window_shift=0, dispatch_at=None,
                starts=("PATH_CHEAPEST_ARC", "SAVINGS", "PARALLEL_CHEAPEST_INSERTION")):
    """Minimize the day's cost (vehicle day costs + cost per km), in cents. Times are minutes after midnight.
    window_shift: minutes added to every cafe's latest delivery time; dispatch_at: earliest departure (what-ifs).
    starts: first-solution strategies; the search runs once from each, and the cheapest plan wins (multi-start)."""
    best = None
    for start in starts:
        plan = _solve_once(stops, ids, km, minutes, vehicles, windows, vans_only_downtown, max_route_minutes,
                           count_service, solutions, window_shift, dispatch_at, start)
        if plan is not None and (best is None or plan[0]["objective"] < best[0]["objective"]):
            best = plan
    return best


def _solve_once(stops, ids, km, minutes, vehicles, windows, vans_only_downtown, max_route_minutes,
                count_service, solutions, window_shift, dispatch_at, start):
    n, V = len(ids), len(vehicles)
    manager = pywrapcp.RoutingIndexManager(n, V, 0)
    model = pywrapcp.RoutingModel(manager)
    cost_of_rate = {}
    for v, veh in enumerate(vehicles):                    # one cost callback per cost rate, shared by vehicles
        rate = veh["cost_per_km"]
        if rate not in cost_of_rate:
            cents = [[int(round(100 * rate * km[a][b])) for b in range(n)] for a in range(n)]
            cost_of_rate[rate] = model.RegisterTransitCallback(
                lambda i, j, c=cents: c[manager.IndexToNode(i)][manager.IndexToNode(j)])
        model.SetArcCostEvaluatorOfVehicle(cost_of_rate[rate], v)
        model.SetFixedCostOfVehicle(int(100 * veh["cost_per_day"]), v)
    racks = model.RegisterUnaryTransitCallback(lambda i: int(stops.racks[ids[manager.IndexToNode(i)]]))
    model.AddDimensionWithVehicleCapacity(racks, 0, [veh["racks"] for veh in vehicles], True, "racks")
    service = [int(stops.service_minutes[s]) if count_service else 0 for s in ids]

    travel = [[int(minutes[a][b]) + service[a] for b in range(n)] for a in range(n)]   # drive + time at the stop we leave
    time = model.RegisterTransitCallback(lambda i, j: travel[manager.IndexToNode(i)][manager.IndexToNode(j)])
    model.AddDimension(time, 120, 24 * 60, False, "time")        # up to 2 hours waiting allowed
    clock = model.GetDimensionOrDie("time")
    for node, s in enumerate(ids):
        if node == 0:
            continue
        close = int(stops.window_close[s]) + (window_shift if stops.kind[s] == "cafe" else 0)
        if windows:
            clock.CumulVar(manager.NodeToIndex(node)).SetRange(int(stops.window_open[s]), close)
        if vans_only_downtown and stops.downtown[s]:                  # narrow streets: no trucks
            trucks = [v for v, veh in enumerate(vehicles) if veh["type"] != "van"]
            if trucks:
                model.VehicleVar(manager.NodeToIndex(node)).RemoveValues(trucks)
    depot_open, depot_close = int(stops.window_open[ids[0]]), int(stops.window_close[ids[0]])
    if dispatch_at is not None:
        depot_open = dispatch_at
    for v in range(V):
        clock.CumulVar(model.Start(v)).SetRange(depot_open, depot_close)
        clock.CumulVar(model.End(v)).SetRange(depot_open, depot_close)
        model.AddVariableMaximizedByFinalizer(clock.CumulVar(model.Start(v)))       # leave as late as possible
        model.AddVariableMinimizedByFinalizer(clock.CumulVar(model.End(v)))
    if max_route_minutes is not None:
        for v in range(V):
            model.solver().Add(clock.CumulVar(model.End(v)) - clock.CumulVar(model.Start(v)) <= max_route_minutes)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, start)
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.solution_limit = solutions                  # stop after this many solutions: the same routes every run
    params.time_limit.seconds = 300                    # a safety net only
    solution = model.SolveWithParameters(params)
    if solution is None:
        return None
    plan = []
    for v, veh in enumerate(vehicles):
        index, route = model.Start(v), []
        leaves = solution.Min(clock.CumulVar(index))
        while not model.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                route.append((node, solution.Min(clock.CumulVar(index))))
            index = solution.Value(model.NextVar(index))
        if route:
            plan.append({"vehicle": veh["name"], "type": veh["type"], "leaves": leaves,
                         "back": solution.Min(clock.CumulVar(index)), "stops": route,
                         "objective": solution.ObjectiveValue()})
    return plan


def simulate(plan, stops, ids, km, minutes, vehicles, window_shift=0, dispatch_at=None):
    """Drive each route independently of the solver: leave when the dock opens (or at dispatch_at), wait if early,
    serve, drive on. Returns (cost, km, problems)."""
    depart = int(stops.window_open[ids[0]]) if dispatch_at is None else dispatch_at
    by_name = {veh["name"]: veh for veh in vehicles}
    cost, total_km, problems = 0.0, 0.0, []
    for r in plan:
        veh = by_name[r["vehicle"]]
        t, here, racks = depart, 0, 0
        route_km = 0.0
        for node, _ in r["stops"]:
            s = ids[node]
            t += minutes[here][node]
            route_km += km[here][node]
            close = stops.window_close[s] + (window_shift if stops.kind[s] == "cafe" else 0)
            if t > close:
                problems.append(f"{r['vehicle']} reaches {s} at {hhmm(t)}, after {hhmm(close)}")
            t = max(t, stops.window_open[s]) + stops.service_minutes[s]
            racks += stops.racks[s]
            if stops.downtown[s] and veh["type"] != "van":
                problems.append(f"{r['vehicle']} can't enter downtown street of {s}")
            here = node
        route_km += km[here][0]
        t += minutes[here][0]
        if racks > veh["racks"]:
            problems.append(f"{r['vehicle']} carries {racks} racks, room for {veh['racks']}")
        if t > stops.window_close[ids[0]]:
            problems.append(f"{r['vehicle']} back at {hhmm(t)}, after {hhmm(stops.window_close[ids[0]])}")
        cost += veh["cost_per_day"] + veh["cost_per_km"] * route_km
        total_km += route_km
    return round(cost, 2), round(total_km, 2), problems
```

Points to notice:

- **Costs per vehicle type**: `SetArcCostEvaluatorOfVehicle` with a cents-per-km callback, and `SetFixedCostOfVehicle` for the day charge. The objective is the day's cost.
- **Service time** is added to the travel time *from* a stop: arriving at 05:00 with 6 minutes of unloading, the van can leave at 05:06.
- **Downtown stops**: `VehicleVar(index).RemoveValues(trucks)` takes trucks out of the choices for that stop.
- **Multi-start**: the search runs from three first solutions and keeps the cheapest. Each run stops after 300 solutions, so results repeat.
- `simulate` drives every route from the dock's opening, waits when early, and reports late arrivals, overloads and trucks downtown. It shares no code with the model.

Then `day37/run_vrptw.py`:

```python
# ---------- 37.1 Delivery windows ----------
print(stops[stops.kind != "depot"].groupby(["kind", "window_open", "window_close"]).size().rename("stops").to_string())
print(f"loading dock: {hhmm(stops.window_open['DEPOT'])}-{hhmm(stops.window_close['DEPOT'])}")
vans = [{"name": f"van-{k}", "type": "van", "racks": 18, "cost_per_day": 90.0, "cost_per_km": 0.40} for k in range(1, 16)]
for name, kw in [("vans only, no windows (Day 36)", dict(windows=False, vans_only_downtown=False)),
                 ("vans only, with windows", {})]:
    show(name, solve_vrptw(stops, ids, km, minutes, vans, **kw), vans)
```

```text
kind   window_open  window_close
cafe   300          375              7
                    405              9
                    435              8
store  270          420             10
loading dock: 04:00-08:00
vans only, no windows (Day 36): {'van': 9}, 272.92 km, $919.17; longest route 154 min; 2 problems
    van-5 reaches C20 at 06:28, after 06:15
    van-6 reaches C15 at 06:58, after 06:45
vans only, with windows: {'van': 9}, 264.04 km, $915.62; longest route 157 min; 0 problems
```

Day 36's van routes, driven against the windows, arrive late twice. With windows in the model, 9 vans keep every window.

## 37.2 Today's fleet

```python
# ---------- 37.2 Today's fleet: 6 vans, 3 trucks, vans only downtown ----------
print(pd.DataFrame(fleet).groupby("type").first()[["racks", "cost_per_day", "cost_per_km"]].to_string())
plan = solve_vrptw(stops, ids, km, minutes, fleet)
best = show("today's fleet, windows, downtown rule", plan, fleet)
rows = []
for r in plan:
    route = [n for n, _ in r["stops"]]
    rows.append((r["vehicle"], hhmm(r["leaves"]), hhmm(r["back"]), len(route), sum(int(stops.racks[ids[n]]) for n in route),
                 round(tour_km(route, km), 2), " ".join(f"{ids[n]}@{hhmm(t)}" for n, t in r["stops"])))
print(pd.DataFrame(rows, columns=["vehicle", "leaves", "back", "stops", "racks", "km", "arrivals"]).to_string(index=False))
```

```text
       racks  cost_per_day  cost_per_km
type
truck     36         150.0         0.65
van       18          90.0         0.40
today's fleet, windows, downtown rule: {'van': 5, 'truck': 2}, 229.91 km, $864.50; longest route 200 min; 0 problems
vehicle leaves  back  stops  racks    km                                                              arrivals
  van-1  04:51 06:59      7     18 34.80 C09@05:05 C07@05:30 C15@05:48 C23@06:01 C24@06:16 C03@06:34 C18@06:45
  van-3  05:38 07:44      4     18 32.03                               S32@06:08 S31@06:36 C17@07:00 C01@07:15
  van-4  05:21 06:30      3     18 14.25                                         C11@05:29 S27@05:48 C13@06:15
  van-5  05:19 07:37      6     18 37.09           C06@05:34 C12@05:51 C02@06:02 C05@06:15 S28@06:36 C19@07:11
  van-6  05:04 06:27      3     18 21.61                                         C21@05:15 S33@05:36 C20@06:15
truck-1  04:40 08:00      4     36 62.03                               S30@04:54 C10@05:22 S35@06:16 S34@06:55
truck-2  05:18 07:40      7     35 28.10 C22@05:28 C08@05:42 S29@05:57 C16@06:21 C14@06:35 C04@06:45 S26@06:58
```

5 vans and 2 trucks, **$864.50 a day**, every window kept, no truck downtown. The table is the dispatcher's sheet: departure, return and every arrival time. Vehicles leave as late as their first window allows (the finalizer maximizes the start time), so drivers don't wait at the first stop.

## 37.3 What each rule costs

```python
# ---------- 37.3 What each rule costs ----------
for name, kw in [("no windows (checked against them)", dict(windows=False)),
                 ("trucks allowed downtown", dict(vans_only_downtown=False)),
                 ("drivers back within 2.5 hours", dict(max_route_minutes=150))]:
    cost = show(name, solve_vrptw(stops, ids, km, minutes, fleet, **kw), fleet)
    print(f"    difference: ${cost - best:+,.2f} a day")
```

```text
no windows (checked against them): {'van': 3, 'truck': 3}, 206.25 km, $836.20; longest route 225 min; 4 problems
    truck-3 reaches C07 at 07:41, after 07:15
    truck-3 reaches C15 at 07:59, after 06:45
    truck-3 reaches C23 at 08:12, after 06:45
    difference: $-28.30 a day
trucks allowed downtown: {'van': 3, 'truck': 3}, 239.32 km, $837.95; longest route 181 min; 3 problems
    truck-1 can't enter downtown street of C21
    truck-1 can't enter downtown street of C06
    truck-1 can't enter downtown street of C09
    difference: $-26.55 a day
drivers back within 2.5 hours: {'van': 6, 'truck': 2}, 245.83 km, $949.16; longest route 150 min; 0 problems
    difference: $+84.66 a day
```

- **Without windows** the plan is about $28 a day cheaper, and 4 deliveries arrive late.
- **Trucks downtown** would save about $27 a day, but trucks can't use those streets.
- **Drivers back within 2.5 hours** (a union proposal) costs **$85 a day**: one more van, about $22,000 a year.

### Heuristics and noise

Routing searches are heuristics. Change a limit, a start or a small detail of the data, and the routes and cost move by a few dollars, sometimes in the "wrong" direction: in 37.1, vans with windows came out $3.55 cheaper than vans without them, which a proven-optimal plan could never do. Treat differences of a few dollars as noise. Multi-start, more search and comparing several scenarios together reduce it. Report decisions only when the difference is clearly larger than the noise.

### The planted bug

`day37/bug_service.py` builds the routes with a time callback that "only needs the driving time". Run it.

```python
plan = solve_vrptw(stops, ids, km, minutes, fleet, count_service=False)
print(f"the solver's plan: {len(plan)} vehicles, every stop inside its window")
cost, total_km, problems = simulate(plan, stops, ids, km, minutes, fleet)
print(f"driven for real: ${cost:,.2f}, {total_km:.2f} km, {len(problems)} late arrivals:")
for p in problems:
    print("   ", p)
```

<details><summary>What happens, and why</summary>

```text
the solver's plan: 6 vehicles, every stop inside its window
driven for real: $833.67, 227.92 km, 3 late arrivals:
    van-3 reaches C24 at 06:51, after 06:45
    truck-1 reaches C05 at 06:31, after 06:15
    truck-3 reaches C20 at 06:26, after 06:15
```

The solver's plan is cheaper (6 vehicles, $833.67) and, by its own count, keeps every window. Driven for real, **3 deliveries are late**: the unloading time at each stop (6 minutes per café, 20 per store) is missing, and it adds up along a route. The simulator, which serves each stop for its real service time, catches it. Every number a model leaves out is a number reality puts back in.

</details>

## Mini-project: dispatch time and café windows

Day 34 found the bakery could start at 22:00 and still be ready around 04:45. What would dispatching at 04:30 or 05:00 instead of 04:00 cost on the road? And if cafés accepted deliveries up to 30 minutes later (most open at 07:00 or 07:30), what would that save?

<details><summary>Solution</summary>

```python
"""Day 37 mini-project: Day 34 found the bakery can start at 22:00 instead of 21:00 and still be ready by
about 04:45. What does a later dispatch cost on the road? And what if cafes took delivery up to 30 minutes later?

Run from the repository root (after day06/load_tables.py):  python day37/mini_project_dispatch.py   (about 2 minutes)
"""
import sys

import duckdb
import pandas as pd

for folder in ("day36", "day37"):
    sys.path.insert(0, folder)
from routing import load_run
from vrptw import hhmm, load_fleet, simulate, solve_vrptw

con = duckdb.connect("northbound.duckdb", read_only=True)
stops, ids, km, minutes = load_run(con)
fleet = load_fleet(con)
rows = []
for dispatch in (240, 270, 300):
    for shift in (0, 30):
        plan = solve_vrptw(stops, ids, km, minutes, fleet, dispatch_at=dispatch, window_shift=shift)
        if plan is None:
            rows.append((hhmm(dispatch), f"+{shift} min", "no plan", None, None, None))
            continue
        cost, total_km, problems = simulate(plan, stops, ids, km, minutes, fleet, window_shift=shift, dispatch_at=dispatch)
        types = pd.Series([r["type"] for r in plan]).value_counts().to_dict()
        rows.append((hhmm(dispatch), f"+{shift} min", f"{types.get('van', 0)} vans, {types.get('truck', 0)} trucks",
                     round(total_km, 2), cost, len(problems)))
print(pd.DataFrame(rows, columns=["dispatch", "cafe_windows", "vehicles", "km", "cost", "late"]).to_string(index=False))
```

```text
dispatch cafe_windows         vehicles     km   cost  late
   04:00       +0 min 5 vans, 2 trucks 229.91 864.50     0
   04:00      +30 min 3 vans, 3 trucks 243.03 840.35     0
   04:30       +0 min 5 vans, 2 trucks 258.95 867.36     0
   04:30      +30 min 3 vans, 3 trucks 243.80 840.85     0
   05:00       +0 min 4 vans, 3 trucks 213.98 925.42     0
   05:00      +30 min 5 vans, 2 trucks 243.52 861.14     0
```

- **Dispatching at 04:30 costs nothing measurable** ($867.36 against $864.50: within the search's noise).
- **Dispatching at 05:00 costs about $61 a day**: everything must happen in a shorter morning, so more vehicles run at once.
- **Café windows 30 minutes later save about $24 a day** at a 04:00 or 04:30 dispatch, with 3 vans and 3 trucks instead of 5 vans and 2 trucks. At a 05:00 dispatch they almost cancel the cost of the late start.

*Memo to the operations manager:* "Start the night run at 21:30 rather than 22:00: the run then finishes at 03:43 (Day 34) and the vans can leave at 04:00–04:30 at no extra cost. Starting at 22:00 pushes dispatch towards 05:00, which costs about $61 a day ($16,000 a year) on the road, unless the cafés accept deliveries 30 minutes later. Ask the cafés: later windows save about $24 a day ($6,000 a year) at today's dispatch time."

</details>

## Commit

```bash
git add .
git commit -m "Day 37: time windows, service times, mixed fleet"
git push
```

Then run the tests: `tests/test_shop_and_routes.py` checks Days 34–37 (the short-staffed night against the checker, four normal weeks, the proven tour and savings routes, and every window kept). Add `"day34"` to `"day37"` to the folders in `conftest.py`. The suite now takes about a minute.

```python
"""Module 4, Days 34-37: the bakery job shop, the 4-week roster, tours and routes."""
import pytest
from bakery_shop import build_schedule, check_schedule, dispatch_fifo, load_shop, solve_schedule, summary
from four_weeks import build_four_weeks, solve_four_weeks
from roster import load_roster_data
from routing import load_run, savings, tour_km, tsp_exact, two_opt, nearest_neighbour
from vrptw import load_fleet, simulate, solve_vrptw


def test_short_staffed_night_is_proven_and_legal(con):
    jobs, ops, machines, rules = load_shop(con)
    solver, status, sched = solve_schedule(build_schedule(jobs, ops, machines, rules, one_of_each=True), ops)
    assert solver.status_name(status) == "OPTIMAL" and summary(sched, jobs)["makespan"] == 664
    assert check_schedule(sched, jobs, machines, rules) == []
    assert check_schedule(dispatch_fifo(jobs, ops, machines), jobs, machines, rules)       # the hand plan over-proofs


def test_four_normal_weeks_cost_four_day32_weeks(con):
    out, roster = solve_four_weeks(build_four_weeks(*load_roster_data(con)), work=40)
    assert out["status"] == "OPTIMAL" and out["pay"] == pytest.approx(4 * 28_876.48, abs=0.01)


def test_tours_and_savings(con):
    stops, ids, km, minutes = load_run(con)
    tour, status, bound = tsp_exact(km)
    assert status == "OPTIMAL" and tour_km(tour, km) == pytest.approx(128.15, abs=0.01)
    assert tour_km(two_opt(nearest_neighbour(km), km), km) >= tour_km(tour, km) - 1e-9
    racks = [int(stops.racks[s]) for s in ids]
    routes = savings(km, racks, 18)
    assert sorted(n for r in routes for n in r) == list(range(1, len(ids)))
    assert all(sum(racks[n] for n in r) <= 18 for r in routes)


def test_fleet_routes_keep_every_window(con):
    stops, ids, km, minutes = load_run(con)
    fleet = load_fleet(con)
    plan = solve_vrptw(stops, ids, km, minutes, fleet)
    cost, total_km, problems = simulate(plan, stops, ids, km, minutes, fleet)
    assert problems == [] and sum(len(r["stops"]) for r in plan) == len(ids) - 1
```

```text
57 passed
```

### Day 37 checklist

- [ ] I can add a time dimension with windows and service times.
- [ ] I can give each vehicle type its own costs and restrict which vehicles may visit a stop.
- [ ] I check routes with an independent simulator.
- [ ] I price each rule (windows, street access, route length) by removing it.
- [ ] I treat small differences between heuristic plans as noise.
- [ ] I connected two plans: the bakery's finish time and the vans' dispatch time.
