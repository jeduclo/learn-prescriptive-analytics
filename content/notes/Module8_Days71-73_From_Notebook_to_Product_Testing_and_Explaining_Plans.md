# DAY 71 — From Notebook to Product

Seventy days of models, and every one of them runs as a script that a student starts by hand. That's how analysis begins, but it isn't how a planner runs a bakery every Sunday night. Module 8 turns a model into a **product**: something other people run, trust and come back to.

Today's work is the first step for the weekly bakery plan. It becomes `nbplan`, a small Python package with:

- data contracts checked before every run;
- strict settings in a file;
- a command line;
- run records that make every plan reproducible.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 71.1 What changes, and the same answer as the notebook |
| 0:45 – 1:45 | 71.2 Data contracts |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 71.3 Settings and the command line |
| 2:45 – 3:45 | 71.4 Reproducible runs; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: a batch of scenarios |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). No new tables for Days 71–73.

## 71.1 What changes, and the same answer as the notebook

| A notebook-style script | A product |
|---|---|
| Trusts its inputs | Checks them against a contract, and says what is wrong |
| Settings edited in the code | Settings in a file, checked strictly |
| Run by its author | Run by anyone, from a command line or a service |
| Prints a result | Keeps a record: what went in, what came out, which version |
| Tested by looking | Tested automatically (Day 72) |
| Explained by its author | Explains itself (Day 73) |

The package lives next to the course code:

```text
northbound/
  nbplan/
    __init__.py     version
    config.py       Settings, strict checks
    contracts.py    what each input table must satisfy
    data.py         PlanInputs: read, check, apply settings, fingerprint
    model.py        build and solve; a plain Result
    runs.py         run records
    cli.py          the command line
    __main__.py     python -m nbplan
  pyproject.toml    makes it installable: pip install -e .
```

Create `pyproject.toml` and install the package in your virtual environment. `-e` (editable) means your edits take effect without reinstalling:

```python
[build-system]
requires = ["setuptools>=80"]
build-backend = "setuptools.build_meta"

[project]
name = "nbplan"
version = "0.1.0"
description = "Northbound's weekly bakery planner (Prescriptive Analytics course, Module 8)"
requires-python = ">=3.13"
dependencies = ["duckdb", "pyomo", "highspy"]

[project.scripts]
nbplan = "nbplan.cli:main"

[tool.setuptools]
packages = ["nbplan"]
```

```bash
pip install -e .
nbplan --version
```

Add `*.egg-info/` and `runs/` to `.gitignore`.

Create the package files, starting with `nbplan/__init__.py`:

```python
"""nbplan: Northbound's weekly bakery planner, as a product. Days 71-73 build it from Day 6's notebook-style code:
validated inputs, a strict configuration, a command line, run records, an independent checker and explanations.
"""
__version__ = "0.1.0"
```

`nbplan/model.py`: Day 6's model, returning a plain `Result` (plan, duals, reduced costs) instead of a Pyomo model, so nothing downstream needs Pyomo:

```python
"""The weekly plan: Day 6's LP, built from PlanInputs, solved with HiGHS, returned as a plain Result."""
from dataclasses import dataclass

import pyomo.environ as pyo


@dataclass
class Result:
    status: str                 # "optimal" or "infeasible" (or another solver status)
    profit: float | None
    plan: dict                  # sku -> batches
    duals: dict                 # resource -> value of one more unit
    reduced_costs: dict         # sku -> profit change per batch forced in
    cap_duals: dict             # sku -> value of one more batch of demand


def build(inputs):
    m = pyo.ConcreteModel(name="nbplan")
    m.P, m.R = pyo.Set(initialize=inputs.products), pyo.Set(initialize=inputs.resources)
    m.x = pyo.Var(m.P, within=pyo.NonNegativeReals)
    m.demand = pyo.Constraint(m.P, rule=lambda m, p: m.x[p] <= inputs.cap[p])
    m.capacity_limit = pyo.Constraint(m.R, rule=lambda m, r: sum(inputs.use.get((r, p), 0) * m.x[p] for p in m.P)
                                      <= inputs.capacity[r])
    m.minimum = pyo.Constraint([p for p in inputs.min_batches], rule=lambda m, p: m.x[p] >= inputs.min_batches[p])
    m.total_profit = pyo.Objective(expr=sum(inputs.profit[p] * m.x[p] for p in m.P), sense=pyo.maximize)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def solve(inputs, time_limit=60.0):
    m = build(inputs)
    res = pyo.SolverFactory("highs").solve(m, load_solutions=False, options={"time_limit": time_limit})
    tc = res.solver.termination_condition
    if tc != pyo.TerminationCondition.optimal:
        return Result(str(tc), None, {}, {}, {}, {})
    m.solutions.load_from(res)
    clean = lambda v: round(v, 9) + 0.0
    return Result("optimal", clean(pyo.value(m.total_profit)), {p: clean(pyo.value(m.x[p])) for p in m.P},
                  {r: clean(m.dual[m.capacity_limit[r]]) for r in m.R}, {p: clean(m.rc[m.x[p]]) for p in m.P},
                  {p: clean(m.dual[m.demand[p]]) for p in m.P})
```

The first test of any rewrite: **the same answer as before**. `day71/run_product.py`:

```python
# ---------- 71.1 The same answer as the notebook ----------
print("--- same answer")
m = build_model(load_data(con))
solve_day6(m)
r = solve(load_inputs(con, Settings()))
print(f"Day 6 code: ${m.total_profit():,.2f}; nbplan: ${r.profit:,.2f}; same plan: "
      f"{all(abs(m.x[p]() - r.plan[p]) < 1e-6 for p in r.plan)}")
```

```text
--- same answer
Day 6 code: $108,351.28; nbplan: $108,351.28; same plan: True
```

## 71.2 Data contracts

A **data contract** states what the model needs from each table:

- columns and their types;
- unique keys;
- rules that must hold (prices above costs, capacities not negative);
- links between tables (every recipe row is for a known product).

Day 6's `load_data` had some checks; a contract makes them complete and readable. `nbplan/contracts.py`:

```python
"""Data contracts: what the planner expects of each input table. Checked before every run."""

CONTRACTS = {
    "bakery_range": {"key": ["sku"], "columns": {"sku": "VARCHAR", "category": "VARCHAR", "units_per_batch": "number",
                                                 "transfer_price": "number", "ingredient_cost": "number"},
                     "rules": ["transfer_price > ingredient_cost", "units_per_batch > 0"]},
    "bakery_range_recipe": {"key": ["sku", "resource"], "columns": {"sku": "VARCHAR", "resource": "VARCHAR",
                                                                    "use_per_batch": "number"},
                            "rules": ["use_per_batch >= 0"]},
    "bakery_range_resources": {"key": ["resource"], "columns": {"resource": "VARCHAR", "weekly_capacity": "number"},
                               "rules": ["weekly_capacity >= 0"]},
    "bakery_range_forecast": {"key": ["sku"], "columns": {"sku": "VARCHAR", "demand_cap_batches": "number"},
                              "rules": ["demand_cap_batches >= 0"]},
}
NUMBER = ("DOUBLE", "FLOAT", "BIGINT", "INTEGER", "DECIMAL", "HUGEINT", "SMALLINT")


def validate(con):
    """Every broken promise in the input tables, as readable sentences. An empty list means the data can be used."""
    problems = []
    tables = {r[0] for r in con.sql("SELECT table_name FROM information_schema.tables").fetchall()}
    for table, c in CONTRACTS.items():
        if table not in tables:
            problems.append(f"{table}: table missing")
            continue
        cols = dict(con.sql(f"SELECT column_name, data_type FROM information_schema.columns "
                            f"WHERE table_name = '{table}'").fetchall())
        for col, kind in c["columns"].items():
            if col not in cols:
                problems.append(f"{table}: column {col} missing")
            elif kind == "number" and not cols[col].startswith(NUMBER):
                problems.append(f"{table}.{col}: expected a number, found {cols[col]}")
        if any(p.startswith(f"{table}:") or p.startswith(f"{table}.") for p in problems):
            continue
        key = ", ".join(c["key"])
        dup = con.sql(f"SELECT {key}, count(*) n FROM {table} GROUP BY ALL HAVING n > 1").fetchall()
        for row in dup:
            problems.append(f"{table}: duplicate key {row[:-1]}")
        for rule in c["rules"]:
            n = con.sql(f"SELECT count(*) FROM {table} WHERE NOT ({rule}) OR ({rule}) IS NULL").fetchall()[0][0]
            if n:
                problems.append(f"{table}: {n} row(s) break the rule '{rule}'")
    if not any("missing" in p or "expected a number" in p for p in problems):      # columns are there: check links
        orphans = con.sql("""SELECT DISTINCT r.sku FROM bakery_range_recipe r
                             LEFT JOIN bakery_range p USING (sku) WHERE p.sku IS NULL""").fetchall()
        problems += [f"bakery_range_recipe: product {o[0]} is not in bakery_range" for o in orphans]
        missing = con.sql("""SELECT p.sku FROM bakery_range p LEFT JOIN bakery_range_forecast f USING (sku)
                             WHERE f.sku IS NULL""").fetchall()
        problems += [f"bakery_range_forecast: no forecast for {m[0]}" for m in missing]
    return problems
```

`nbplan/data.py` reads the tables only after they pass, applies the settings, and **fingerprints** what the model will see:

```python
"""Inputs for one planning run: read the tables (after validation), apply the settings, fingerprint the result."""
import hashlib
import json
from dataclasses import dataclass

from .contracts import validate


class DataError(ValueError):
    pass


@dataclass(frozen=True)
class PlanInputs:
    products: tuple
    resources: tuple
    category: dict
    profit: dict          # per batch
    capacity: dict
    use: dict             # (resource, sku) -> per batch
    cap: dict             # demand cap, batches
    min_batches: dict

    def fingerprint(self):
        """A short hash of everything the model will see: same fingerprint, same plan."""
        payload = json.dumps({k: (sorted((str(a), b) for a, b in v.items()) if isinstance(v, dict) else v)
                              for k, v in self.__dict__.items()}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_inputs(con, settings):
    problems = validate(con)
    if problems:
        raise DataError("; ".join(problems))
    p = con.sql("""SELECT sku, category, units_per_batch * (transfer_price - ingredient_cost) AS profit
                   FROM bakery_range ORDER BY sku""").fetchall()
    capacity = dict(con.sql("SELECT resource, weekly_capacity FROM bakery_range_resources ORDER BY resource").fetchall())
    capacity.update(settings.capacity)
    use = {(r, s): u for s, r, u in con.sql("SELECT sku, resource, use_per_batch FROM bakery_range_recipe").fetchall()}
    cap = {s: c * settings.demand_multiplier for s, c in con.sql("SELECT * FROM bakery_range_forecast").fetchall()}
    unknown = sorted(set(settings.min_batches) - {r[0] for r in p})
    if unknown:
        raise DataError(f"min_batches for unknown product(s): {', '.join(unknown)}")
    return PlanInputs(products=tuple(r[0] for r in p), resources=tuple(sorted(capacity)),
                      category={r[0]: r[1] for r in p}, profit={r[0]: r[2] for r in p}, capacity=capacity,
                      use=use, cap=cap, min_batches=dict(settings.min_batches))
```

```python
# ---------- 71.2 Data contracts ----------
print("--- contracts")
print(f"today's tables: {validate(con) or 'no problems'}")
bad = duckdb.connect()                                               # a broken copy, in memory
for t in ("bakery_range", "bakery_range_recipe", "bakery_range_resources", "bakery_range_forecast"):
    bad.register(f"src_{t}", con.sql(f"SELECT * FROM {t}").df())
    bad.sql(f"CREATE TABLE {t} AS SELECT * FROM src_{t}")
bad.sql("INSERT INTO bakery_range SELECT * FROM bakery_range WHERE sku = 'BK-CR'")                 # a duplicate
bad.sql("UPDATE bakery_range_resources SET weekly_capacity = -600 WHERE resource = 'oven_hours'")   # a sign error
bad.sql("INSERT INTO bakery_range_recipe VALUES ('BK-99', 'oven_hours', 1.0)")                     # an unknown product
bad.sql("UPDATE bakery_range SET transfer_price = 1.00 WHERE sku = 'BK-12'")                       # sold below cost
for p in validate(bad):
    print(f"problem: {p}")
```

```text
--- contracts
today's tables: no problems
problem: bakery_range: duplicate key ('BK-CR',)
problem: bakery_range: 1 row(s) break the rule 'transfer_price > ingredient_cost'
problem: bakery_range_resources: 1 row(s) break the rule 'weekly_capacity >= 0'
problem: bakery_range_recipe: product BK-99 is not in bakery_range
```

Four mistakes, four plain sentences. Without the contract, the duplicate croissant row would double its demand, the negative oven capacity would make the model infeasible, and the product sold below cost would simply never be made. Each would be found late, if at all.

## 71.3 Settings and the command line

Settings live in a **TOML** file. TOML is a simple settings format; Python reads it with `tomllib`. A peak week, for example, is `day71/scenarios/peak_week.toml`:

```python
# December peak: 15% more demand, and the bakery rents oven time to 650 hours
demand_multiplier = 1.15

[capacity]
oven_hours = 650
```

`nbplan/config.py` checks them **strictly**. An unknown setting or resource is an error, never silently ignored:

```python
"""Settings for a planning run, read from a TOML file. Strict: unknown keys are errors, not silently ignored."""
import tomllib
from dataclasses import asdict, dataclass, field, fields

RESOURCES = ("baker_hours", "oven_hours", "packing_hours", "sourdough_starter")


@dataclass(frozen=True)
class Settings:
    database: str = "northbound.duckdb"
    demand_multiplier: float = 1.0                     # scale every product's demand cap
    capacity: dict = field(default_factory=dict)       # resource -> weekly capacity, replacing the table's
    min_batches: dict = field(default_factory=dict)    # sku -> at least this many batches
    time_limit: float = 60.0                           # seconds for the solver

    def as_dict(self):
        return asdict(self)


class ConfigError(ValueError):
    pass


def check_settings(raw):
    """Turn a dictionary into Settings, refusing anything unknown or out of range."""
    known = {f.name for f in fields(Settings)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown setting(s): {', '.join(unknown)}")
    bad = sorted(set(raw.get("capacity", {})) - set(RESOURCES))
    if bad:
        raise ConfigError(f"unknown resource(s) in [capacity]: {', '.join(bad)} (known: {', '.join(RESOURCES)})")
    if any(v < 0 for v in raw.get("capacity", {}).values()) or raw.get("demand_multiplier", 1.0) <= 0:
        raise ConfigError("capacities must be >= 0 and the demand multiplier > 0")
    return Settings(**raw)


def load_settings(path=None):
    if path is None:
        return Settings()
    with open(path, "rb") as f:
        return check_settings(tomllib.load(f))
```

`nbplan/cli.py` and `nbplan/__main__.py` give it a command line, built with the standard library's `argparse`:

```python
"""The command line: python -m nbplan validate | plan | whatif."""
import argparse
import sys

import duckdb

from . import __version__
from .config import ConfigError, load_settings
from .data import DataError, load_inputs
from .model import solve
from .runs import record


def cmd_validate(args, settings):
    from .contracts import validate
    con = duckdb.connect(settings.database, read_only=True)
    problems = validate(con)
    for p in problems:
        print(f"problem: {p}")
    print("data OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


def cmd_plan(args, settings):
    con = duckdb.connect(settings.database, read_only=True)
    inputs = load_inputs(con, settings)
    result = solve(inputs, settings.time_limit)
    if result.status != "optimal":
        print(f"no plan: {result.status}")
        return 2
    rid = record(result, inputs, settings, args.runs) if args.runs else "-"
    made = {p: x for p, x in result.plan.items() if x > 1e-6}
    print(f"run {rid}: profit ${result.profit:,.2f} a week; {len(made)} of {len(result.plan)} products made "
          f"(inputs {inputs.fingerprint()})")
    if args.show:
        for p, x in sorted(made.items(), key=lambda kv: -kv[1])[: args.show]:
            print(f"  {p}: {x:.2f} batches")
    binding = [r for r, v in result.duals.items() if abs(v) > 1e-6]
    print("bottlenecks: " + ", ".join(f"{r} (${result.duals[r]:,.2f} a unit)" for r in binding))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nbplan", description="Northbound's weekly bakery planner")
    parser.add_argument("--version", action="version", version=f"nbplan {__version__}")
    parser.add_argument("--config", help="a TOML settings file")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="check the input tables against their contracts")
    p = sub.add_parser("plan", help="solve this week's plan")
    p.add_argument("--show", type=int, default=0, help="print the largest N products")
    p.add_argument("--runs", default=None, help="folder to keep a record of the run")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        return {"validate": cmd_validate, "plan": cmd_plan}[args.command](args, settings)
    except (ConfigError, DataError) as e:
        print(f"error: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
```

```python
import sys

from .cli import main

sys.exit(main())
```

```python
# ---------- 71.3 Settings and the command line ----------
print("--- command line")
print(cli("validate"))
print(cli("plan", "--show", "3"))
print(cli("--config", "day71/scenarios/peak_week.toml", "plan"))
with tempfile.TemporaryDirectory() as tmp:
    typo = Path(tmp) / "typo.toml"
    typo.write_text("[capacity]\noven_hour = 700\n")
    print(cli("--config", str(typo), "plan").replace(str(typo), "typo.toml"))
try:
    check_settings({"demand_multiplyer": 1.2})
except ConfigError as e:
    print(f"check_settings: {e}")
```

```text
--- command line
$ nbplan validate
data OK
(exit code 0)
$ nbplan plan --show 3
run -: profit $108,351.28 a week; 22 of 30 products made (inputs 94f8d590e531)
  BK-WW: 45.00 batches
  BK-SD: 35.00 batches
  BK-CR: 31.00 batches
bottlenecks: oven_hours ($99.34 a unit), packing_hours ($94.46 a unit), sourdough_starter ($54.09 a unit)
(exit code 0)
$ nbplan --config day71/scenarios/peak_week.toml plan
run -: profit $113,200.98 a week; 20 of 30 products made (inputs 37241d1498ea)
bottlenecks: baker_hours ($58.13 a unit), packing_hours ($155.85 a unit), sourdough_starter ($115.36 a unit)
(exit code 0)
$ nbplan --config typo.toml plan
error: unknown resource(s) in [capacity]: oven_hour (known: baker_hours, oven_hours, packing_hours, sourdough_starter)
(exit code 3)
check_settings: unknown setting(s): demand_multiplyer
```

- The command line prints the plan's profit, how many products are made, the inputs' fingerprint and the bottlenecks. It ends with an **exit code**: 0 for success, other numbers for different failures, so scripts and schedulers can react.
- The peak week (+15% demand, 650 oven hours) earns $113,201, and the bottlenecks move to bakers, packing and starter.
- A typo (`oven_hour`) **stops the run** and names the valid resources.

## 71.4 Reproducible runs

`nbplan/runs.py` keeps a record of every run:

- the inputs' fingerprint, the settings and the package version;
- the result.

The run id is a hash of the inputs and settings, so the same question always gets the same id.

```python
"""Run records: every plan is kept with what produced it (inputs fingerprint, settings, version), as JSON lines."""
import hashlib
import json
from pathlib import Path

from . import __version__


def run_id(fingerprint, settings):
    return hashlib.sha256((fingerprint + json.dumps(settings.as_dict(), sort_keys=True)).encode()).hexdigest()[:10]


def record(result, inputs, settings, folder="runs"):
    rid = run_id(inputs.fingerprint(), settings)
    Path(folder).mkdir(exist_ok=True)
    row = {"run_id": rid, "version": __version__, "inputs": inputs.fingerprint(), "settings": settings.as_dict(),
           "status": result.status, "profit": result.profit, "plan": result.plan}
    with open(Path(folder) / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return rid
```

```python
# ---------- 71.4 Reproducible runs ----------
print("--- runs")
with tempfile.TemporaryDirectory() as tmp:
    for args in (["plan"], ["plan"], ["--config", "day71/scenarios/lease_ovens.toml", "plan"]):
        out = subprocess.run([sys.executable, "-m", "nbplan", *args, "--runs", tmp], capture_output=True, text=True)
        print(out.stdout.splitlines()[0])
    rows = [json.loads(line) for line in (Path(tmp) / "runs.jsonl").read_text().splitlines()]
    print(f"{len(rows)} runs recorded; distinct run ids: {len({r['run_id'] for r in rows})}; "
          f"each keeps: {', '.join(sorted(rows[0]))}")
```

```text
--- runs
run c206699294: profit $108,351.28 a week; 22 of 30 products made (inputs 94f8d590e531)
run c206699294: profit $108,351.28 a week; 22 of 30 products made (inputs 94f8d590e531)
run 3b81b5348e: profit $109,434.90 a week; 23 of 30 products made (inputs 59dab16d3afb)
3 runs recorded; distinct run ids: 2; each keeps: inputs, plan, profit, run_id, settings, status, version
```

Running the same plan twice gives the same run id; changing the settings gives a new one. Months later, anyone can answer "which data and settings produced the plan we baked on March 3?", and re-run it.

> **Decision it supports:** every decision the model supports, made repeatable, checkable and runnable by people other than its author.
>
> **Other places the same practices apply:** any model that leaves the analyst's laptop: forecasts, pricing tools, schedulers, risk models.

### The planted bug

`day71/bug_lenient_config.py` is a colleague's settings loader that "doesn't crash on small mistakes". Planning asked for the peak week **with** the leased ovens. Run it.

```python
request = """
demand_multiplier = 1.15
[capacity]
oven_hour = 700          # lease the extra ovens
"""
raw = tomllib.loads(request)
settings = Settings(demand_multiplier=raw.get("demand_multiplier", 1.0),
                    capacity={k: v for k, v in raw.get("capacity", {}).items() if k in RESOURCES})
con = duckdb.connect("northbound.duckdb", read_only=True)
r = solve(load_inputs(con, settings))
print(f"peak week with leased ovens: profit ${r.profit:,.2f}; settings used: {settings.capacity or 'no capacity change'}")
```

<details><summary>What happens, and why</summary>

```text
peak week with leased ovens: profit $110,646.03; settings used: no capacity change
```

The plan earns $110,646, the peak week **without** the leased ovens. The settings file says `oven_hour`, and the lenient loader quietly dropped it. Planning believes they've planned with the lease and orders accordingly. A crash would have cost a minute; this costs a week of wrong plans. In a product, **unknown settings are errors**: `check_settings` refuses them and says which keys are valid.

</details>

## Mini-project: a batch of scenarios

Planning keeps scenario files in `day71/scenarios/`. Write a script that:

- runs every one through the `nbplan` package (not the command line);
- refuses the whole batch if any file is invalid;
- prints one comparison against the base plan: profit, change, bottlenecks, and the products whose batches change most.

<details><summary>Solution</summary>

```python
"""Day 71 mini-project: Planning keeps scenario files in day71/scenarios/. Write a script that runs every one of them
through the nbplan package (not the command line), refuses the whole batch if any file is invalid, and prints one
comparison table against the base plan: profit, change, bottlenecks and the products whose batches change most.

Run from the repository root (after day06/load_tables.py):  python day71/mini_project_scenarios.py
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, ".")
from nbplan.config import Settings, load_settings
from nbplan.data import load_inputs
from nbplan.model import solve

pd.set_option("display.width", 200)
con = duckdb.connect("northbound.duckdb", read_only=True)
files = sorted(Path("day71/scenarios").glob("*.toml"))
settings = {f.stem: load_settings(f) for f in files}                # any invalid file stops here, before solving
base = solve(load_inputs(con, Settings()))
rows = []
for name, s in settings.items():
    r = solve(load_inputs(con, s), s.time_limit)
    if r.status != "optimal":
        rows.append((name, None, None, r.status, ""))
        continue
    moves = sorted(((p, r.plan[p] - base.plan[p]) for p in r.plan), key=lambda kv: -abs(kv[1]))[:3]
    rows.append((name, round(r.profit, 2), round(r.profit - base.profit, 2),
                 ", ".join(k for k, v in r.duals.items() if abs(v) > 1e-6),
                 ", ".join(f"{p} {d:+.1f}" for p, d in moves if abs(d) > 0.05)))
print(f"base plan: ${base.profit:,.2f}")
print(pd.DataFrame(rows, columns=["scenario", "profit", "change", "bottlenecks", "biggest_moves"]).to_string(index=False))
```

```text
base plan: $108,351.28
    scenario    profit  change                                   bottlenecks                       biggest_moves
    festival       NaN     NaN                                    infeasible
 lease_ovens 109434.90 1083.62 baker_hours, packing_hours, sourdough_starter  BK-30 -8.5, BK-15 +8.0, BK-18 +5.0
   peak_week 113200.98 4849.70 baker_hours, packing_hours, sourdough_starter BK-30 -20.0, BK-18 +9.5, BK-23 -9.4
starter_tank 108956.52  605.24  oven_hours, packing_hours, sourdough_starter BK-07 +12.0, BK-18 -7.2, BK-26 -1.6
```

- All files are checked **before** any solving: a typo in the fourth file shouldn't waste three solves and then fail.
- The festival scenario is **infeasible**: four festival breads at 18 batches need more starter than exists. Day 7 found the same conflict. The table reports it instead of crashing, and the next step is the starter tank (`starter_tank.toml` shows it's worth $605 a week on its own).
- The lease and peak-week results match Days 13 and 21, a quick check that the product reproduces the course's earlier answers.

</details>

## Commit

```bash
git add .
git commit -m "Day 71: nbplan, from notebook to product"
git push
```

### Day 71 checklist

- [ ] My product gives the same answer as the code it replaces.
- [ ] Inputs are checked against a written contract, with readable messages.
- [ ] Settings are in a file and checked strictly; unknown keys are errors.
- [ ] The command line returns meaningful exit codes.
- [ ] Every run is recorded with its inputs' fingerprint, settings and version.

---

# DAY 72 — Testing Optimization Models

An optimization model can be wrong in ways that don't crash:

- a constraint left out;
- a sign flipped;
- a unit mixed up.

It still returns "optimal", with a plausible number. Every planted bug in this course was such a mistake. Today you build the tests that catch them automatically, every time the code changes:

- an **independent checker**;
- **known answers**;
- **properties** that must hold for any input;
- **regression** tests that compare the right thing;
- a CI workflow that runs it all.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 72.1 Check the plan independently |
| 0:45 – 1:45 | 72.2 Known answers; 72.3 properties on random instances |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 72.4 Alternative optima: compare objectives, not plans |
| 2:45 – 3:45 | 72.5 The test suite and CI; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: property tests for the delivery model |
| 4:45 – 5:00 | Commit with Git + checklist |

## 72.1 Check the plan independently

A checker re-reads the plan against the **inputs**, not against the model, and recomputes the objective. It is written separately, ideally by someone else, so a modelling mistake isn't copied into it. `nbplan/check.py`:

```python
"""An independent check of a plan: written separately from the model, so a modelling mistake isn't repeated here."""


def check_plan(inputs, plan, tol=1e-6):
    """Every constraint the plan breaks, and the profit recomputed from the plan itself."""
    broken = []
    for p in inputs.products:
        x = plan.get(p, 0.0)
        if x < -tol:
            broken.append(f"{p}: negative batches {x:.4f}")
        if x > inputs.cap[p] + tol:
            broken.append(f"{p}: {x:.2f} batches above demand {inputs.cap[p]:.2f}")
        if x < inputs.min_batches.get(p, 0.0) - tol:
            broken.append(f"{p}: {x:.2f} batches below the minimum {inputs.min_batches[p]}")
    for r in inputs.resources:
        used = sum(inputs.use.get((r, p), 0.0) * plan.get(p, 0.0) for p in inputs.products)
        if used > inputs.capacity[r] + tol:
            broken.append(f"{r}: uses {used:.2f} of {inputs.capacity[r]:.2f}")
    for p in plan:
        if p not in inputs.products:
            broken.append(f"{p}: not a product")
    return broken, sum(inputs.profit[p] * plan.get(p, 0.0) for p in inputs.products)
```

Then `day72/run_testing.py`:

```python
# ---------- 72.1 Check the plan independently ----------
print("--- checker")
broken, profit = check_plan(inputs, r.plan)
print(f"the plan breaks {len(broken)} constraints; recomputed profit ${profit:,.2f} (model ${r.profit:,.2f})")
wrong = {**r.plan, "BK-CR": r.plan["BK-CR"] + 2}
print("plus 2 croissant batches: " + "; ".join(check_plan(inputs, wrong)[0]))
```

```text
--- checker
the plan breaks 0 constraints; recomputed profit $108,351.28 (model $108,351.28)
plus 2 croissant batches: BK-CR: 33.00 batches above demand 31.00; oven_hours: uses 603.00 of 600.00; packing_hours: uses 301.20 of 300.00
```

The real plan passes and the profit matches. A plan with two extra croissant batches breaks three constraints, each named. A model that forgot the packing constraint would produce plans that fail here.

## 72.2 Known answers

Tiny instances you can solve by hand pin down the model's logic exactly:

```python
# ---------- 72.2 Known answers ----------
print("--- known answers")
tiny = PlanInputs(products=("A", "B"), resources=("oven",), category={"A": "x", "B": "x"}, profit={"A": 300, "B": 200},
                  capacity={"oven": 100}, use={("oven", "A"): 2, ("oven", "B"): 1}, cap={"A": 40, "B": 30}, min_batches={})
t = solve(tiny)
print(f"by hand: B first (200 per oven hour vs 150), 30 B + 35 A = $16,500; model: ${t.profit:,.2f} with {t.plan}, "
      f"oven worth ${t.duals['oven']:.2f} an hour (A's 300 / 2)")
```

```text
--- known answers
by hand: B first (200 per oven hour vs 150), 30 B + 35 A = $16,500; model: $16,500.00 with {'A': 35.0, 'B': 30.0}, oven worth $150.00 an hour (A's 300 / 2)
```

## 72.3 Properties on random instances

Some things must be true for **every** input. Generate many random instances and check them. Testers call these **metamorphic** properties: change the input in a known way and you know how the answer must change. `day72/random_instances.py`:

```python
"""Day 72: small random planning problems for property tests (no data files needed)."""
import numpy as np

from nbplan.data import PlanInputs


def random_inputs(rng, n_products=6, n_resources=3):
    products = tuple(f"P{i}" for i in range(n_products))
    resources = tuple(f"R{j}" for j in range(n_resources))
    use = {(r, p): float(rng.choice([0.0, rng.uniform(0.5, 3.0)], p=[0.3, 0.7])) for r in resources for p in products}
    return PlanInputs(products=products, resources=resources, category={p: "x" for p in products},
                      profit={p: float(rng.uniform(50, 300)) for p in products},
                      capacity={r: float(rng.uniform(20, 80)) for r in resources}, use=use,
                      cap={p: float(rng.uniform(5, 30)) for p in products}, min_batches={})
```

```python
# ---------- 72.3 Properties on random instances ----------
print("--- properties")
rng = np.random.default_rng(72)
fails = {"plan is feasible": 0, "more capacity never lowers profit": 0, "profits x2 doubles the optimum": 0,
         "a useless product changes nothing": 0, "duals price capacity correctly (small change)": 0}
N = 200
for _ in range(N):
    inp = random_inputs(rng)
    base = solve(inp)
    fails["plan is feasible"] += bool(check_plan(inp, base.plan)[0])
    r0 = inp.resources[0]
    more = solve(dataclasses.replace(inp, capacity={**inp.capacity, r0: inp.capacity[r0] + 5}))
    fails["more capacity never lowers profit"] += more.profit < base.profit - 1e-6
    double = solve(dataclasses.replace(inp, profit={p: 2 * v for p, v in inp.profit.items()}))
    fails["profits x2 doubles the optimum"] += abs(double.profit - 2 * base.profit) > 1e-6 * max(1, base.profit)
    useless = dataclasses.replace(inp, products=inp.products + ("Z",), profit={**inp.profit, "Z": 0.0},
                                  cap={**inp.cap, "Z": 10.0}, use={**inp.use, **{(r, "Z"): 1.0 for r in inp.resources}})
    fails["a useless product changes nothing"] += abs(solve(useless).profit - base.profit) > 1e-6
    tiny_step = solve(dataclasses.replace(inp, capacity={**inp.capacity, r0: inp.capacity[r0] + 1e-3}))
    fails["duals price capacity correctly (small change)"] += abs((tiny_step.profit - base.profit) / 1e-3 - base.duals[r0]) > 1e-3 * max(1, base.duals[r0]) + 1e-6
for k, v in fails.items():
    print(f"{k}: {N - v}/{N} instances pass")
```

```text
--- properties
plan is feasible: 200/200 instances pass
more capacity never lowers profit: 200/200 instances pass
profits x2 doubles the optimum: 200/200 instances pass
a useless product changes nothing: 200/200 instances pass
duals price capacity correctly (small change): 200/200 instances pass
```

The properties checked:

- More capacity never lowers the optimum.
- Doubling every profit doubles it.
- A product that earns nothing changes nothing.
- For a small step, the dual predicts the change (Day 11).

A sign error in a constraint, a dropped term or a wrong dual would fail one of these on some random instance, without anyone having to think of the failing case.

## 72.4 Alternative optima: compare objectives, not plans

```python
# ---------- 72.4 Alternative optima: compare objectives, not plans ----------
print("--- alternative optima")
P0 = "BK-26"                                                         # made in part: 1.6 of its demand cap
twin = dataclasses.replace(inputs, products=inputs.products + ("BK-26B",), category={**inputs.category, "BK-26B": inputs.category[P0]},
                           profit={**inputs.profit, "BK-26B": inputs.profit[P0]}, cap={**inputs.cap, "BK-26B": inputs.cap[P0]},
                           use={**inputs.use, **{(rr, "BK-26B"): inputs.use[(rr, P0)] for rr in inputs.resources
                                                 if (rr, P0) in inputs.use}})
a = solve(twin)
b = solve(dataclasses.replace(twin, products=tuple(reversed(twin.products))))
print(f"a twin of {P0} (same profit, recipe and demand): profit ${a.profit:,.2f} vs ${b.profit:,.2f} "
      f"with the products listed in reverse")
print(f"plans equal? {all(abs(a.plan[p] - b.plan[p]) < 1e-6 for p in a.plan)}; "
      f"{P0} / twin: {a.plan[P0]:.2f} / {a.plan['BK-26B']:.2f} vs {b.plan[P0]:.2f} / {b.plan['BK-26B']:.2f}")
```

```text
--- alternative optima
a twin of BK-26 (same profit, recipe and demand): profit $108,351.28 vs $108,351.28 with the products listed in reverse
plans equal? False; BK-26 / twin: 0.00 / 1.60 vs 1.60 / 0.00
```

Add a twin of a partly-made product (same profit, recipe and demand), and the solver can put the 1.6 batches on either. Which one it picks depends on the **order** of the products. Both plans are optimal, with the same profit. **Many LPs have several optimal plans.** A test that expects one particular plan fails for reasons that don't matter: a new solver version, a reordered table, an extra product. Test the **objective**, the **feasibility** of the plan (the checker), and properties. Pin exact quantities only where the optimum is unique.

## 72.5 The test suite and CI

`tests/test_nbplan.py` puts it together. `check_settings` raises on unknown keys, the tiny instance has a known answer, an infeasible run is reported (not raised), properties are checked on 20 seeds, and the course data must still give the golden $108,351.28:

```python
"""nbplan's test suite (Day 72): unit, known-answer, property and regression tests for the product."""
import dataclasses
import sys

import numpy as np
import pytest

sys.path[:0] = ["."]
from nbplan.check import check_plan
from nbplan.config import ConfigError, Settings, check_settings
from nbplan.contracts import validate
from nbplan.data import PlanInputs, load_inputs
from nbplan.model import solve
from random_instances import random_inputs

TINY = PlanInputs(products=("A", "B"), resources=("oven",), category={"A": "x", "B": "x"}, profit={"A": 300, "B": 200},
                  capacity={"oven": 100}, use={("oven", "A"): 2, ("oven", "B"): 1}, cap={"A": 40, "B": 30}, min_batches={})


def test_settings_are_strict():
    with pytest.raises(ConfigError):
        check_settings({"capacity": {"oven_hour": 700}})
    with pytest.raises(ConfigError):
        check_settings({"demand_multiplyer": 1.2})
    assert check_settings({"capacity": {"oven_hours": 700}}).capacity == {"oven_hours": 700}


def test_known_answer():
    r = solve(TINY)
    assert r.profit == pytest.approx(16_500) and r.plan == pytest.approx({"A": 35, "B": 30})
    assert r.duals["oven"] == pytest.approx(150)


def test_infeasible_is_reported_not_raised():
    r = solve(dataclasses.replace(TINY, min_batches={"A": 60}, cap={"A": 80, "B": 30}))
    assert r.status != "optimal" and r.profit is None


@pytest.mark.parametrize("seed", range(20))
def test_properties(seed):
    inp = random_inputs(np.random.default_rng(seed))
    base = solve(inp)
    assert check_plan(inp, base.plan)[0] == []
    r0 = inp.resources[0]
    assert solve(dataclasses.replace(inp, capacity={**inp.capacity, r0: inp.capacity[r0] + 5})).profit >= base.profit - 1e-6
    assert solve(dataclasses.replace(inp, profit={p: 2 * v for p, v in inp.profit.items()})).profit == pytest.approx(2 * base.profit)


def test_regression_on_the_course_data(con):
    """The golden number: today's plan earns $108,351.28 (Day 6). Compare the objective, not the batches."""
    assert validate(con) == []
    r = solve(load_inputs(con, Settings()))
    assert r.profit == pytest.approx(108_351.28, abs=0.01)
    assert check_plan(load_inputs(con, Settings()), r.plan)[0] == []


def test_explanations(con):
    from nbplan.explain import value_curve, why_not
    inputs = load_inputs(con, Settings())
    r = solve(inputs)
    first = value_curve(inputs, "oven_hours", steps=(1,))[0][1]
    assert first == pytest.approx(r.duals["oven_hours"], rel=1e-6)            # the dual holds for a small step
    assert why_not(inputs, "BK-CR", 5)[0] == "above forecast"                   # made to its forecast already
    cost, pushed, _ = why_not(inputs, "BK-08", 5)
    assert cost < 0 and pushed                                                  # insisting costs money, pushes others out
```

A **continuous integration** workflow runs the suite on every push. On GitHub this is `.github/workflows/tests.yml`; the **Actions** tab shows each run and a red cross on any failing commit:

```python
# Day 72: run the test suite on every push and pull request (GitHub Actions).
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
          cache: pip
      - run: pip install -r requirements.txt && pip install -e .
      - run: python make_data.py && python day01/ladder.py && python day06/load_tables.py
      - run: python -m nbplan validate
      - run: python -m pytest -q
```

Add `"day71"` to `"day73"` to the folders in `conftest.py`, then run the whole suite:

```text
126 passed
```

> **Decision it supports:** trusting a model enough to let others use it, and changing it without fear.
>
> **Other places the same practices apply:** any model in production; every change to a solver, a library or the data.

### The planted bug

`day72/bug_exact_plan.py`: a colleague's regression test saves last month's plan and checks that the new run gives exactly the same batches. This month a café asked for BK-26 under its own label (BK-26B: same recipe, same price, a separate demand). Run the two regression tests.

```python
golden = solve(inputs)                                              # last month's plan, saved
new = dataclasses.replace(inputs, products=inputs.products + ("BK-26B",),
                          category={**inputs.category, "BK-26B": inputs.category["BK-26"]},
                          profit={**inputs.profit, "BK-26B": inputs.profit["BK-26"]}, cap={**inputs.cap, "BK-26B": inputs.cap["BK-26"]},
                          use={**inputs.use, **{(r, "BK-26B"): inputs.use[(r, "BK-26")] for r in inputs.resources
                                                if (r, "BK-26") in inputs.use}})
r = solve(new)
same_batches = all(abs(r.plan.get(p, 0) - golden.plan[p]) < 1e-6 for p in golden.plan)
print(f"test_plan_unchanged: {'passed' if same_batches else 'FAILED'} "
      f"(BK-26 {golden.plan['BK-26']:.2f} -> {r.plan['BK-26']:.2f}, BK-26B {r.plan['BK-26B']:.2f})")
print(f"test_profit_unchanged: {'passed' if abs(r.profit - golden.profit) < 0.01 else 'FAILED'} "
      f"(${golden.profit:,.2f} -> ${r.profit:,.2f})")
```

<details><summary>What happens, and why</summary>

```text
test_plan_unchanged: FAILED (BK-26 1.60 -> 0.00, BK-26B 1.60)
test_profit_unchanged: passed ($108,351.28 -> $108,351.28)
```

The plan test **fails**: BK-26's 1.6 batches moved to its twin. The profit test passes, and it's right: the new plan is just as good. The failing test would send someone hunting for a bug that doesn't exist, and after a few false alarms people stop trusting the suite. Regression tests on optimization models should compare objectives (with a tolerance) and feasibility, not the batches of an optimum that may not be unique.

</details>

## Mini-project: property tests for the delivery model

Write property tests for Day 66's delivery model, and run them:

1. a tighter CO2 cap never makes the cheapest plan cheaper;
2. a looser stock-out cap never makes it dearer;
3. the model's cost, CO2 and service expressions agree with an independent evaluation of the chosen plan;
4. more charger capacity never makes it dearer.

<details><summary>Solution</summary>

```python
"""Day 72 mini-project: write property tests for Day 66's delivery model, and run them:
1. a tighter CO2 cap never makes the cheapest plan cheaper;
2. a looser stock-out cap never makes it dearer;
3. the model's cost, CO2 and service expressions agree with an independent evaluation of the chosen plan;
4. more charger capacity never makes it dearer.

Run from the repository root (after day06/load_tables.py):  python day72/mini_project_delivery_tests.py
"""
import sys

import duckdb
import pyomo.environ as pyo

for folder in ("day06", "day66"):
    sys.path.insert(0, folder)
from bakery import solve
from delivery_mo import build_delivery, chosen, evaluate, load_delivery

con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_delivery(con)


def cheapest(co2=None, service=None, data=d):
    m = build_delivery(data)
    if co2 is not None:
        m.c = pyo.Constraint(expr=m.co2 <= co2)
    if service is not None:
        m.s = pyo.Constraint(expr=m.service <= service + 1e-9)
    m.obj = pyo.Objective(expr=m.cost)
    solve(m)
    return m


results = {}
caps = [18.0, 15.0, 12.0, 10.0, 9.0]
costs = [pyo.value(cheapest(co2=c, service=2.6).cost) for c in caps]
results["tighter CO2 cap never cheaper"] = all(b >= a - 1e-6 for a, b in zip(costs, costs[1:]))
services = [2.4, 2.6, 2.8, 3.0, 3.4]
costs = [pyo.value(cheapest(co2=12.0, service=s).cost) for s in services]
results["looser stock-out cap never dearer"] = all(b <= a + 1e-6 for a, b in zip(costs, costs[1:]))
m = cheapest(co2=12.0, service=2.6)
e = evaluate(d, chosen(m))
results["expressions match an independent evaluation"] = (abs(e["cost"] - pyo.value(m.cost)) < 1e-6
                                                          and abs(e["co2"] - pyo.value(m.co2)) < 1e-9
                                                          and abs(e["service"] - pyo.value(m.service)) < 1e-9)
more = {**d, "chargers": {k: v + 20 for k, v in d["chargers"].items()}}
results["more chargers never dearer"] = pyo.value(cheapest(co2=10.0, service=2.6, data=more).cost) <= pyo.value(cheapest(co2=10.0, service=2.6).cost) + 1e-6
for k, ok in results.items():
    print(f"{'pass' if ok else 'FAIL'}: {k}")
```

```text
pass: tighter CO2 cap never cheaper
pass: looser stock-out cap never dearer
pass: expressions match an independent evaluation
pass: more chargers never dearer
```

All four pass. Property 3 is the delivery model's checker: `evaluate` computes cost and CO2 from the plan with plain arithmetic, independently of the Pyomo expressions. Properties 1, 2 and 4 are the monotonicity every capped model must show. A violation would mean a constraint isn't doing what it says. To make them part of CI, move them into `tests/` as `test_` functions.

</details>

## Commit

```bash
git add .
git commit -m "Day 72: testing optimization models"
git push
```

### Day 72 checklist

- [ ] I check every plan with an independent checker that recomputes the objective.
- [ ] I keep known-answer tests on tiny instances.
- [ ] I test properties (monotonicity, scaling, invariance) on random instances.
- [ ] My regression tests compare objectives and feasibility, not plans with possible ties.
- [ ] The suite runs automatically in CI on every push.

---

# DAY 73 — Explaining a Plan

"The model says so" doesn't get a plan accepted. The head baker wants to know:

- why croissants are capped at 31;
- why the olive sourdough isn't made at all;
- what it would cost to make more of it anyway.

The finance team wants to know why profit fell this week. Today `nbplan` learns to answer such questions in plain language. Every number comes from the model or from a re-solve, never from intuition.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 73.1 What limits the plan |
| 0:45 – 1:45 | 73.2 How long a shadow price lasts |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 73.3 "Why not more of this?" |
| 2:45 – 3:45 | 73.4 What changed since last week; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: why does my store get three deliveries? |
| 4:45 – 5:00 | Commit with Git + checklist |

## 73.1 What limits the plan

Three kinds of sentence explain an LP plan, all from Module 2:

- **Bottlenecks** (shadow prices): what is full, and what one more unit is worth.
- **Forecast-limited products** (duals of the demand caps): made to their forecast, and what more demand would be worth.
- **Products left out** (reduced costs): what they earn per batch, what the capacity they'd use is worth, and how much their profit would have to rise.

`nbplan/explain.py`:

```python
"""Explanations of a plan in plain language: what limits it, why a product is or isn't made, what insisting on
something would cost, and what changed between two runs. Every number comes from the model or a re-solve."""
import dataclasses

from .model import solve


def value_curve(inputs, resource, steps=(1, 10, 25, 50, 100)):
    """Average value of each extra unit of a resource, for bigger and bigger additions (the dual only holds near 0)."""
    base = solve(inputs).profit
    out = []
    for k in steps:
        r = solve(dataclasses.replace(inputs, capacity={**inputs.capacity, resource: inputs.capacity[resource] + k}))
        out.append((k, (r.profit - base) / k))
    return out


def explain_plan(inputs, result, names=None, top=3):
    names = names or {}
    label = lambda p: f"{names.get(p, p)} ({p})" if p in names else p
    lines = []
    full = sorted((r for r, v in result.duals.items() if v > 1e-6), key=lambda r: -result.duals[r])
    for r in full:
        lines.append(f"{r.replace('_', ' ')} is fully used: one more unit would add ${result.duals[r]:,.2f} a week.")
    capped = sorted((p for p, v in result.cap_duals.items() if v > 1e-6), key=lambda p: -result.cap_duals[p])[:top]
    for p in capped:
        lines.append(f"{label(p)} is made up to its forecast ({inputs.cap[p]:.0f} batches): each extra batch of demand "
                     f"would add ${result.cap_duals[p]:,.2f}.")
    skipped = sorted((p for p, x in result.plan.items() if x < 1e-6 and result.reduced_costs[p] < -1e-6),
                     key=lambda p: result.reduced_costs[p], reverse=True)[:top]
    for p in skipped:
        need = -result.reduced_costs[p]
        lines.append(f"{label(p)} is not made: each batch would earn ${inputs.profit[p]:,.2f} but use capacity worth "
                     f"${inputs.profit[p] + need:,.2f}; it would be made if its profit rose by ${need:,.2f} a batch.")
    return lines


def why_not(inputs, sku, extra):
    """What insisting on `extra` more batches of sku would cost, and what it would push out of the plan."""
    base = solve(inputs)
    want = base.plan[sku] + extra
    if want > inputs.cap[sku] + 1e-9:
        return "above forecast", [], base                         # more than the stores are expected to sell
    forced = solve(dataclasses.replace(inputs, min_batches={**inputs.min_batches, sku: want}))
    if forced.status != "optimal":
        return "impossible", [], base
    moves = sorted(((p, forced.plan[p] - base.plan[p]) for p in base.plan if p != sku), key=lambda kv: kv[1])
    return forced.profit - base.profit, [(p, d) for p, d in moves if d < -1e-6], base


def explain_change(inputs_a, a, inputs_b, b, top=4):
    """What differs in the inputs, and the products whose plan moved most."""
    changes = []
    for r in inputs_a.resources:
        if abs(inputs_a.capacity[r] - inputs_b.capacity[r]) > 1e-9:
            changes.append(f"{r}: {inputs_a.capacity[r]:g} -> {inputs_b.capacity[r]:g}")
    for p in inputs_a.products:
        if abs(inputs_a.profit[p] - inputs_b.profit[p]) > 1e-9:
            changes.append(f"{p} profit a batch: ${inputs_a.profit[p]:,.2f} -> ${inputs_b.profit[p]:,.2f}")
        if abs(inputs_a.cap[p] - inputs_b.cap[p]) > 1e-9:
            changes.append(f"{p} demand: {inputs_a.cap[p]:g} -> {inputs_b.cap[p]:g}")
    moves = sorted(((p, b.plan[p] - a.plan[p]) for p in a.plan), key=lambda kv: -abs(kv[1]))[:top]
    return changes, [(p, d) for p, d in moves if abs(d) > 1e-6], b.profit - a.profit
```

Then `day73/run_explain.py`:

```python
# ---------- 73.1 What limits the plan ----------
print("--- the plan")
print(f"This week's plan earns ${r.profit:,.2f}.")
for line in explain_plan(inputs, r, names):
    print("- " + line)
```

```text
--- the plan
This week's plan earns $108,351.28.
- oven hours is fully used: one more unit would add $99.34 a week.
- packing hours is fully used: one more unit would add $94.46 a week.
- sourdough starter is fully used: one more unit would add $54.09 a week.
- Raisin bread (BK-10) is made up to its forecast (20 batches): each extra batch of demand would add $134.38.
- Multigrain loaf (BK-04) is made up to its forecast (18 batches): each extra batch of demand would add $107.21.
- Apple turnover 4-pack (BK-17) is made up to its forecast (9 batches): each extra batch of demand would add $73.87.
- Olive sourdough (BK-07) is not made: each batch would earn $227.00 but use capacity worth $229.97; it would be made if its profit rose by $2.97 a batch.
- Cheesecake (BK-22) is not made: each batch would earn $245.40 but use capacity worth $253.89; it would be made if its profit rose by $8.49 a batch.
- Danish 4-pack (BK-15) is not made: each batch would earn $208.00 but use capacity worth $225.07; it would be made if its profit rose by $17.07 a batch.
```

"Olive sourdough is not made: each batch would earn $227.00 but use capacity worth $229.97; it would be made if its profit rose by $2.97 a batch." A baker can act on that sentence (a price rise of a few cents a loaf would bring it in) without knowing what a reduced cost is.

## 73.2 How long a shadow price lasts

A shadow price holds only **near** today's capacity (Day 12's ranging). Before quoting "an oven hour is worth $99.34", check how far it lasts:

```python
# ---------- 73.2 How long a shadow price lasts ----------
print("--- value of more oven hours")
for k, v in value_curve(inputs, "oven_hours"):
    print(f"{k:>4} more oven hours: ${v:,.2f} an hour on average (total ${k * v:,.2f})")
```

```text
--- value of more oven hours
   1 more oven hours: $99.34 an hour on average (total $99.34)
  10 more oven hours: $83.07 an hour on average (total $830.73)
  25 more oven hours: $43.34 an hour on average (total $1,083.62)
  50 more oven hours: $21.67 an hour on average (total $1,083.62)
 100 more oven hours: $10.84 an hour on average (total $1,083.62)
```

The first hour is worth $99.34. On average, the first 10 are worth $83, and after about 25 hours **extra ovens are worth nothing**: other resources (packing, starter, bakers) become the bottleneck. Total gain: $1,083.62, Day 13's number for leasing ovens. An explanation must say where its number stops being true.

## 73.3 "Why not more of this?"

The most common question from the floor. Answer it by **insisting** and re-solving: force the extra batches in, and show what the plan gives up.

```python
# ---------- 73.3 "Why not more croissants?" ----------
print("--- why not")
for sku, extra in (("BK-CR", 5), ("BK-08", 5), ("BK-26", 3)):
    cost, pushed, base = why_not(inputs, sku, extra)
    if isinstance(cost, str):
        print(f"{extra} more batches of {names[sku]} ({sku}): {cost} - the plan already makes "
              f"{base.plan[sku]:.0f} of the {inputs.cap[sku]:.0f} batches the stores are expected to sell")
        continue
    print(f"{extra} more batches of {names[sku]} ({sku}) would cost ${-cost:,.2f} a week, pushing out "
          + ", ".join(f"{names[p]} {d:.1f}" for p, d in pushed[:3]))
```

```text
--- why not
5 more batches of Croissant 6-pack (BK-CR): above forecast - the plan already makes 31 of the 31 batches the stores are expected to sell
5 more batches of Seeded sourdough (BK-08) would cost $257.56 a week, pushing out Sourdough loaf -5.0, Chocolate chip cookies 12-pack -1.2
3 more batches of Chocolate chip cookies 12-pack (BK-26) would cost $43.69 a week, pushing out Brownies 6-pack -2.7
```

- **Croissants** are made to their forecast: more would not sell. Don't re-solve; say so.
- Five batches of **seeded sourdough** would cost $258 a week, pushing out five batches of sourdough loaf (they compete for starter) and some cookies.
- Three more batches of **chocolate chip cookies** would cost $44 a week, pushing out brownies.

Counterfactuals ("if we did X, we'd lose Y and give up Z") are the explanations people find most convincing, because they answer the question actually asked.

## 73.4 What changed since last week

Profit falls, and someone asks why. Compare the two runs' **inputs** and their **plans**:

```python
# ---------- 73.4 What changed since last week ----------
print("--- what changed")
butter = {p for p, c in inputs.category.items() if c in ("viennoiserie", "cake")}     # Day 13's butter scenario
units = dict(con.sql("SELECT sku, units_per_batch * ingredient_cost FROM bakery_range").fetchall())
week2 = dataclasses.replace(inputs, profit={p: v - (0.2 * units[p] if p in butter else 0) for p, v in inputs.profit.items()},
                            capacity={**inputs.capacity, "oven_hours": 620})
r2 = solve(week2)
changes, moves, delta = explain_change(inputs, r, week2, r2)
print(f"Profit {'rose' if delta > 0 else 'fell'} by ${abs(delta):,.2f}. What changed in the inputs: "
      f"{len(changes)} items, e.g. " + "; ".join(changes[:3]))
print("What the plan did: " + "; ".join(f"{names[p]} {d:+.1f} batches" for p, d in moves))
```

```text
--- what changed
Profit fell by $3,109.31. What changed in the inputs: 15 items, e.g. oven_hours: 600 -> 620; BK-12 profit a batch: $282.00 -> $253.90; BK-13 profit a batch: $262.50 -> $241.00
What the plan did: Brownies 6-pack -8.5 batches; Danish 4-pack +8.0 batches; Butter tart 6-pack +5.0 batches; Cheesecake +2.0 batches
```

Butter went up 20% (14 products' profits fell) while 20 more oven hours arrived. Profit fell $3,109. The plan shifted from brownies to Danish and butter tarts. The explanation lists the input changes first, then what the plan did about them. To attribute the profit change to each cause, re-solve with one change at a time: the order matters when causes interact.

> **Decision it supports:** getting a plan accepted, and getting the questions about it answered in the language of the people who run it.
>
> **Other places the same practices apply:** schedules (why am I on nights?), routes (why does my van go there first?), prices (why did this go up?), credit decisions.

### The planted bug

`day73/bug_extrapolate.py` is a colleague's explanation for the operations meeting. Should Northbound lease the ovens (100 hours for $900 a week, Day 21) on this argument?

```python
r = solve(load_inputs(con, Settings()))
gain = 100 * r.duals["oven_hours"]
print(f"Each oven hour is worth ${r.duals['oven_hours']:,.2f}, so leasing 100 hours adds ${gain:,.2f} a week "
      f"for $900: a clear yes.")
```

<details><summary>What happens, and why</summary>

```text
Each oven hour is worth $99.34, so leasing 100 hours adds $9,934.00 a week for $900: a clear yes.
```

The argument multiplies the shadow price by 100 hours: $9,934 a week. 73.2 showed the real gain is **$1,083.62**: the dual holds only for the first few hours. The lease is still worth it ($1,083.62 against $900, Day 21), but the explanation overstated its value nine times. Next time the decision might be close and the error decisive. Quote shadow prices with their range, or re-solve the actual change.

</details>

## Mini-project: why does my store get three deliveries?

A store manager asks "why does my store get only 3 deliveries a week under option B?" (Day 67: the 40% CO2 target at today's service). Explain one such store's choice with counterfactuals:

- Fix that store to each other frequency (same vehicle) and re-solve the whole plan.
- Report what each alternative would cost the chain.
- Report what the store's own deliveries would add in cost and CO2.

<details><summary>Solution</summary>

```python
"""Day 73 mini-project: a store manager asks "why does my store get only 3 deliveries a week under option B?"
(Day 67: the 40% CO2 target at today's service). Explain one such store's choice with counterfactuals: fix that store
to each other frequency (same vehicle) and re-solve the whole plan; report what each alternative would cost the
chain, and what else would have to change to stay within the CO2 and service targets.

Run from the repository root (after day06/load_tables.py):  python day73/mini_project_store.py
"""
import sys

import duckdb
import pyomo.environ as pyo

for folder in ("day06", "day66"):
    sys.path.insert(0, folder)
from bakery import solve
from delivery_mo import build_delivery, chosen, evaluate, load_delivery, today_plan

con = duckdb.connect("northbound.duckdb", read_only=True)
d = load_delivery(con)
today = evaluate(d, today_plan(d))
target = (1 - float(d["settings"]["target_2030_cut"])) * today["co2"]


def plan_b(fix=None):
    m = build_delivery(d)
    m.target = pyo.Constraint(expr=m.co2 <= target)
    m.keep = pyo.Constraint(expr=m.service <= today["service"] + 1e-9)
    if fix:
        m.fix_store = pyo.Constraint(expr=sum(m.y[k] for k in m.K if k[0] == fix[0] and k[1:] == fix[1:]) == 1)
    m.obj = pyo.Objective(expr=m.cost)
    solve(m)
    return m


b = plan_b()
picks = {s: (v, f) for s, v, f in chosen(b)}
store = sorted(s for s, (v, f) in picks.items() if f == 3)[0]
v, f = picks[store]
st = d["stores"].loc[store]
print(f"{store} ({st.dc_id}, {st.km:.0f} km away, {st.pallets_per_week:.0f} pallets a week): {v}, {f} deliveries a week; "
      f"plan B costs ${b.cost():,.2f} a week")
for g in sorted(d["freq"]):
    if g == f:
        continue
    m = plan_b(fix=(store, v, g))
    own = evaluate(d, [(store, v, g)])
    mine = evaluate(d, [(store, v, f)])
    print(f"  {g} deliveries a week: trips {d['trips'][store, f]} -> {d['trips'][store, g]}; this store's deliveries "
          f"${own['cost'] - mine['cost']:+,.2f} and {own['co2'] - mine['co2']:+.2f} t CO2 a week; the whole plan, "
          f"re-solved to stay on target: ${m.cost() - b.cost():+,.2f} a week")
```

```text
S22 (DC-OTT, 201 km away, 51 pallets a week): diesel, 3 deliveries a week; plan B costs $59,273.00 a week
  4 deliveries a week: trips 3 -> 4; this store's deliveries $+621.98 and +0.38 t CO2 a week; the whole plan, re-solved to stay on target: $+448.24 a week
  5 deliveries a week: trips 3 -> 5; this store's deliveries $+1,243.96 and +0.76 t CO2 a week; the whole plan, re-solved to stay on target: $+993.70 a week
  6 deliveries a week: trips 3 -> 6; this store's deliveries $+1,865.94 and +1.15 t CO2 a week; the whole plan, re-solved to stay on target: $+1,858.25 a week
```

S22 is far from its DC (201 km) and small (51 pallets a week). Each extra delivery is a 400 km round trip:

- +$622 a week and 0.38 t of CO2 for this store alone;
- for the chain, after re-planning to stay on target, +$448 a week for a fourth delivery and +$1,858 for a sixth.

The chain's cost rises less than the store's own: better service at S22 lets the plan deliver slightly less often elsewhere and still keep the chain's stock-out rate.

*For the manager:* "Your store is far from Ottawa and its orders fill a truck in three trips. A fourth delivery would cost the chain about $450 a week and add 0.4 tonnes of CO2 that we'd have to cut elsewhere. If your shelves run short, tell us. We can test four deliveries for a month and measure the difference."

</details>

## Commit

```bash
git add .
git commit -m "Day 73: explaining a plan"
git push
```

### Day 73 checklist

- [ ] I can turn duals and reduced costs into plain sentences people can act on.
- [ ] I state how far a shadow price holds before quoting it.
- [ ] I answer "why not more X?" by insisting and re-solving, or by pointing to the forecast.
- [ ] I can explain a change between two runs: inputs first, then the plan's response.
- [ ] I never extrapolate a dual beyond its range.
