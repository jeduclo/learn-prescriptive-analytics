# DAY 74 — A Decision Service with FastAPI

`nbplan` runs from a command line, so a planner with a terminal can use it. Most people who need a plan don't have one: the store-ordering spreadsheet, the operations dashboard, a colleague's script. Today the model becomes a **service**: a small web API that any program can call with settings and get back a plan, its bottlenecks and plain-language explanations.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 74.1 The service and its contract |
| 0:45 – 1:45 | 74.2 Plans on request; 74.3 bad requests |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 74.4 Why not? Running it for real |
| 2:45 – 3:45 | A planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: API tests |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data and packages

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). No new tables for Days 74–77.

Four new packages for this week (also in `requirements.txt`):

```bash
pip install fastapi==0.165.0 uvicorn==0.54.0 httpx2==2.16.2 streamlit==1.67.0
pip install -e .
```

- **FastAPI** builds the web API; **uvicorn** serves it.
- **httpx2** is what FastAPI's `TestClient` uses to send requests in tests.
- **Streamlit** is for tomorrow's app.

## 74.1 The service and its contract

A service is a promise to other programs. They must know what they may send, what they will get back, and what happens when they get it wrong. FastAPI turns that promise into code with **pydantic models**:

- each field has a type and limits (`gt=0`, `le=3`);
- `extra="forbid"` refuses unknown fields, the same rule as Day 71's settings files;
- the models generate the API's documentation at `/docs`.

```python
"""Day 74: nbplan as a web service (FastAPI). Start it with:  uvicorn nbplan.service:app --reload
Interactive documentation is generated at /docs. Every request reads the database afresh (no stale data)."""
import os

import duckdb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .config import ConfigError, check_settings
from .contracts import validate
from .data import DataError, load_inputs
from .explain import explain_plan, why_not
from .model import solve

app = FastAPI(title="nbplan", version=__version__, description="Northbound's weekly bakery planner")


def database():
    return os.environ.get("NBPLAN_DATABASE", "northbound.duckdb")


class PlanRequest(BaseModel):
    """What a client may ask for. Unknown fields are refused (the API version of Day 71's strict settings)."""
    model_config = ConfigDict(extra="forbid")
    demand_multiplier: float = Field(1.0, gt=0, le=3)
    capacity: dict[str, float] = {}
    min_batches: dict[str, float] = {}
    time_limit: float = Field(60.0, gt=0, le=600)


class PlanResponse(BaseModel):
    run_inputs: str
    status: str
    profit: float | None
    plan: dict[str, float]
    bottlenecks: dict[str, float]
    explanation: list[str]


class WhyNotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sku: str
    extra: float = Field(gt=0, le=100)
    settings: PlanRequest = PlanRequest()


def _inputs(req):
    try:
        settings = check_settings({**req.model_dump(), "database": database()})
        con = duckdb.connect(settings.database, read_only=True)
        return load_inputs(con, settings), settings
    except ConfigError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except DataError as e:
        raise HTTPException(status_code=503, detail=f"input data failed its contract: {e}")


@app.get("/health")
def health():
    con = duckdb.connect(database(), read_only=True)
    problems = validate(con)
    return {"status": "ok" if not problems else "degraded", "version": __version__, "data_problems": len(problems)}


@app.post("/plan", response_model=PlanResponse)
def plan(req: PlanRequest):
    inputs, settings = _inputs(req)
    r = solve(inputs, settings.time_limit)
    if r.status != "optimal":
        return PlanResponse(run_inputs=inputs.fingerprint(), status=r.status, profit=None, plan={}, bottlenecks={},
                            explanation=["No plan meets these settings: relax a minimum or add capacity."])
    names = dict(duckdb.connect(settings.database, read_only=True).sql("SELECT sku, product_name FROM bakery_range").fetchall())
    return PlanResponse(run_inputs=inputs.fingerprint(), status=r.status, profit=round(r.profit, 2),
                        plan={p: round(x, 2) for p, x in r.plan.items() if x > 1e-6},
                        bottlenecks={k: round(v, 2) for k, v in r.duals.items() if v > 1e-6},
                        explanation=explain_plan(inputs, r, names))


@app.post("/why-not")
def why(req: WhyNotRequest):
    inputs, _ = _inputs(req.settings)
    if req.sku not in inputs.products:
        raise HTTPException(status_code=404, detail=f"unknown product {req.sku}")
    cost, pushed, base = why_not(inputs, req.sku, req.extra)
    if isinstance(cost, str):
        return {"sku": req.sku, "answer": cost, "planned": round(base.plan[req.sku], 2), "forecast": inputs.cap[req.sku]}
    return {"sku": req.sku, "answer": "possible", "weekly_cost": round(-cost, 2),
            "pushed_out": {p: round(d, 2) for p, d in pushed}}
```

Design choices worth copying:

- **No logic in the service.** It checks the request, calls `load_inputs`, `solve` and `explain_plan`, and shapes the answer. Everything else is already tested (Day 72).
- **The data is read on every request**, from the database named by `NBPLAN_DATABASE`. A plan solves in well under a second; a stale plan costs a week.
- **Status codes mean something:** 422 for a bad request (the caller must fix it), 404 for an unknown product, 503 when the data fails its contract (operations must fix it). An infeasible scenario is a valid question with a valid answer, so it returns 200 with `status: "infeasible"`.

The walkthrough calls the service with `TestClient`: the same requests a real caller sends, without starting a server.

```python
# ---------- 74.1 Health and the contract ----------
print("--- health")
show("GET", "/health")
spec = client.get("/openapi.json").json()
print(f"OpenAPI: {spec['info']['title']} {spec['info']['version']}; paths {sorted(spec['paths'])}")
```

```text
--- health
GET /health
  -> 200 {"status": "ok", "version": "0.1.0", "data_problems": 0}
OpenAPI: nbplan 0.1.0; paths ['/health', '/plan', '/why-not']
```

## 74.2 Plans on request

```python
# ---------- 74.2 Plans on request ----------
print("--- plan")
show("POST", "/plan", {})
show("POST", "/plan", {"demand_multiplier": 1.15, "capacity": {"oven_hours": 650}})
show("POST", "/plan", {"min_batches": {"BK-SD": 18, "BK-03": 18, "BK-07": 18, "BK-08": 18}})
```

```text
--- plan
POST /plan {}
  -> 200 {"run_inputs": "94f8d590e531", "status": "optimal", "profit": 108351.28, "plan": "22 products", "bottlenecks": {"oven_hours": 99.34, "packing_hours": 94.46, "sourdough_starter": 54.09}, "explanation": ["oven hours is fully used: one more unit would add $99.34 a week.", "packing hours is fully used: one more unit would add $94.46 a week.", "..."]}
POST /plan {"demand_multiplier": 1.15, "capacity": {"oven_hours": 650}}
  -> 200 {"run_inputs": "d2adf8628693", "status": "optimal", "profit": 113200.98, "plan": "20 products", "bottlenecks": {"baker_hours": 58.13, "packing_hours": 155.85, "sourdough_starter": 115.36}, "explanation": ["packing hours is fully used: one more unit would add $155.85 a week.", "sourdough starter is fully used: one more unit would add $115.36 a week.", "..."]}
POST /plan {"min_batches": {"BK-SD": 18, "BK-03": 18, "BK-07": 18, "BK-08": 18}}
  -> 200 {"run_inputs": "4cd5886c6b2b", "status": "infeasible", "profit": null, "plan": "0 products", "bottlenecks": {}, "explanation": ["No plan meets these settings: relax a minimum or add capacity.", "..."]}
```

- The default plan earns **$108,351.28**, the same as `nbplan plan` and every test since Day 71.
- The peak week with 650 oven hours earns $113,200.98. Ovens stop being a bottleneck; baker hours become one.
- The festival minimums are infeasible, as on Day 71. The caller gets a sentence it can show a person.
- `run_inputs` is the Day 71 fingerprint: two callers with the same request can check they got the same plan.

## 74.3 Bad requests get clear answers

```python
# ---------- 74.3 Bad requests get clear answers ----------
print("--- bad requests")
show("POST", "/plan", {"capacity": {"oven_hour": 700}})
show("POST", "/plan", {"demand_multiplyer": 1.2})
show("POST", "/plan", {"demand_multiplier": -1})
```

```text
--- bad requests
POST /plan {"capacity": {"oven_hour": 700}}
  -> 422 {"detail": "unknown resource(s) in [capacity]: oven_hour (known: baker_hours, oven_hours, packing_hours, sourdough_starter)"}
POST /plan {"demand_multiplyer": 1.2}
  -> 422 {"detail": [{"type": "extra_forbidden", "loc": ["body", "demand_multiplyer"], "msg": "Extra inputs are not permitted", "input": 1.2}]}
POST /plan {"demand_multiplier": -1}
  -> 422 {"detail": [{"type": "greater_than", "loc": ["body", "demand_multiplier"], "msg": "Input should be greater than 0", "input": -1, "ctx": {"gt": 0.0}}]}
```

`oven_hour` is caught by `check_settings` and the message lists the valid names. `demand_multiplyer` and `-1` are caught by pydantic before any of our code runs.

## 74.4 Why not?

The most common question about a plan is "why don't we make more X?". Day 73's `why_not` answers it; the service exposes it.

```python
# ---------- 74.4 Why not? ----------
print("--- why not")
show("POST", "/why-not", {"sku": "BK-08", "extra": 5})
show("POST", "/why-not", {"sku": "BK-CR", "extra": 5})
show("POST", "/why-not", {"sku": "BK-99", "extra": 5})
```

```text
--- why not
POST /why-not {"sku": "BK-08", "extra": 5}
  -> 200 {"sku": "BK-08", "answer": "possible", "weekly_cost": 257.56, "pushed_out": {"BK-SD": -5.0, "BK-26": -1.2}}
POST /why-not {"sku": "BK-CR", "extra": 5}
  -> 200 {"sku": "BK-CR", "answer": "above forecast", "planned": 31.0, "forecast": 31.0}
POST /why-not {"sku": "BK-99", "extra": 5}
  -> 404 {"detail": "unknown product BK-99"}
```

Five more batches of seeded sourdough cost $257.56 a week: they push out 5 sourdough loaves and 1.2 cookie packs. Croissants are already at their forecast: the answer is "the forecast", not "the bottleneck".

### Running it for real

From the repository root:

```bash
uvicorn nbplan.service:app --reload
```

Open <http://127.0.0.1:8000/docs>. FastAPI's interactive page lists the three endpoints with their fields and limits. **Try it out** on `/plan` with `{"capacity": {"oven_hours": 700}}`: $109,434.90. On Windows, set the database with `set NBPLAN_DATABASE=northbound.duckdb` (PowerShell: `$env:NBPLAN_DATABASE="northbound.duckdb"`) if you run from another folder.

> **Decision it supports:** any decision where the people or programs who need the answer are not the people who built the model: store ordering, operations dashboards, other teams' tools.
>
> **Other places the same practices apply:** pricing engines, credit decisions, scheduling and routing services, any model called by other software.

### The planted bug

`day74/bug_startup_cache.py` is a colleague's version of the service that reads the inputs once, when it starts: "solving is fast, but reading the database every time is wasteful". On Thursday the forecasting team raises the croissant forecast from 31 to 40 batches. Run it.

```python
app = FastAPI()
INPUTS = load_inputs(duckdb.connect(str(db), read_only=True), Settings(database=str(db)))   # read once, at start-up


@app.post("/plan")
def plan():
    r = solve(INPUTS)
    return {"profit": round(r.profit, 2), "croissants": r.plan["BK-CR"]}


client = TestClient(app)
print(f"Monday:   {client.post('/plan').json()}")
with duckdb.connect(str(db)) as con:                                # Thursday: the forecast is updated
    con.sql("UPDATE bakery_range_forecast SET demand_cap_batches = 40 WHERE sku = 'BK-CR'")
print(f"Thursday: {client.post('/plan').json()}")
fresh = solve(load_inputs(duckdb.connect(str(db), read_only=True), Settings(database=str(db))))
print("a fresh read of the database:", {"profit": round(fresh.profit, 2), "croissants": fresh.plan["BK-CR"]})
shutil.rmtree(tmp)
```

<details><summary>What happens, and why</summary>

```text
Monday:   {'profit': 108351.28, 'croissants': 31.0}
Thursday: {'profit': 108351.28, 'croissants': 31.0}
a fresh read of the database: {'profit': 108750.11, 'croissants': 40.0}
```

Thursday's plan is Monday's plan: 31 batches of croissants, $108,351.28. The database already says 40, and a fresh read plans 40 for $108,750.11. Nothing fails, nothing is logged, and the service keeps serving last week's forecast until someone restarts it.

Read the data on every request, as `nbplan.service` does. If reading ever becomes slow, cache **keyed on the data's version** (a fingerprint or a timestamp), never "until restart".

</details>

## Mini-project: API tests

Write a script that checks the service's contract: one line per check, pass or fail, and a total. Cover at least:

- the default plan;
- the same request twice gives the same fingerprint;
- an unknown field, an unknown resource, a negative capacity and a time limit above 10 minutes are refused;
- a minimum for an unknown product is refused (which status code should that be?);
- impossible minimums give a 200 with `status: "infeasible"`;
- why-not for an unknown product is a 404.

<details><summary>Solution</summary>

```python
"""Day 74 mini-project: the planners' spreadsheet will call the service, so its contract must not change by accident.
Write API tests and run them: the happy path, every kind of bad request (unknown field, unknown resource, negative
value, unknown product, impossible minimums), and the response's shape. Print one line per check.

Run from the repository root (after day06/load_tables.py):  python day74/mini_project_api_tests.py
"""
import sys

sys.path.insert(0, ".")
from fastapi.testclient import TestClient

from nbplan.service import app

client = TestClient(app)
checks = []


def check(name, response, status, condition=lambda d: True):
    ok = response.status_code == status and condition(response.json())
    checks.append(ok)
    print(f"{'pass' if ok else 'FAIL'}: {name} ({response.status_code})")


check("default plan", client.post("/plan", json={}), 200,
      lambda d: d["status"] == "optimal" and abs(d["profit"] - 108_351.28) < 0.01
      and set(d) == {"run_inputs", "status", "profit", "plan", "bottlenecks", "explanation"})
check("the same request gives the same inputs fingerprint", client.post("/plan", json={}), 200,
      lambda d: d["run_inputs"] == client.post("/plan", json={}).json()["run_inputs"])
check("unknown field refused", client.post("/plan", json={"oven": 700}), 422)
check("unknown resource refused", client.post("/plan", json={"capacity": {"ovens": 700}}), 422,
      lambda d: "known:" in d["detail"])
check("negative capacity refused", client.post("/plan", json={"capacity": {"oven_hours": -5}}), 422)
check("time limit above 10 minutes refused", client.post("/plan", json={"time_limit": 3600}), 422)
check("minimum for an unknown product refused", client.post("/plan", json={"min_batches": {"BK-99": 3}}), 422)
check("impossible minimums: a 200 with status infeasible", client.post("/plan", json={"min_batches": {"BK-08": 500}}), 200,
      lambda d: d["status"] != "optimal" and d["profit"] is None)
check("why-not for an unknown product: 404", client.post("/why-not", json={"sku": "BK-99", "extra": 1}), 404)
check("why-not with settings", client.post("/why-not", json={"sku": "BK-08", "extra": 5,
                                                             "settings": {"capacity": {"sourdough_starter": 72}}}), 200,
      lambda d: d["answer"] == "possible")
print(f"{sum(checks)} of {len(checks)} checks pass")
```

```text
pass: default plan (200)
pass: the same request gives the same inputs fingerprint (200)
pass: unknown field refused (422)
pass: unknown resource refused (422)
pass: negative capacity refused (422)
pass: time limit above 10 minutes refused (422)
pass: minimum for an unknown product refused (422)
pass: impossible minimums: a 200 with status infeasible (200)
pass: why-not for an unknown product: 404 (404)
pass: why-not with settings (200)
10 of 10 checks pass
```

A minimum for an unknown product is the **caller's** mistake, so it's a 422, not a 503. A first version of `load_inputs` raised a `DataError` for it, and the service answered 503 ("our data is broken"). That would have sent operations looking for a problem that doesn't exist. Writing this test found it; `nbplan/data.py` now raises `ConfigError`.

</details>

## Commit

```bash
git add .
git commit -m "Day 74: nbplan as a FastAPI service"
git push
```

### Day 74 checklist

- [ ] My service has a typed contract that refuses unknown fields.
- [ ] Status codes say who must act: the caller (4xx) or us (5xx).
- [ ] An infeasible scenario is an answer, not an error.
- [ ] The service reads current data, or caches it keyed on its version.
- [ ] The service has no model logic of its own: it calls the tested package.

---

# DAY 75 — Mini-project: A Streamlit Scenario App

The service answers programs. Planners and managers want something else: a page where they move a slider and see the plan change. Today is a mini-project day. You build that page with **Streamlit**, a Python library that turns a script into a web app, and test it without a browser.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 75.1 The brief and the design |
| 0:45 – 1:45 | Build: settings, metrics, explanations |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | Build: value curve and why-not |
| 2:45 – 3:45 | 75.2 – 75.3 Test the app with AppTest |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Show it to a "planner" (a classmate); fix what confused them |
| 4:45 – 5:00 | Commit with Git + checklist |

## 75.1 The brief

The planning team asked for:

- a **sidebar** with the week's settings: demand (a slider), capacity of each resource, and one minimum for a product;
- at the top, three **numbers**: profit a week (with the change from today's settings), products made, bottlenecks;
- the **explanations** from Day 73, in sentences;
- the **plan** as a table: product, batches, forecast;
- the **value curve** for a resource they choose: what each extra hour is worth, and where that stops;
- a **why-not** box: "what would 5 more batches of X cost?";
- a clear message, not a crash, when a scenario is impossible.

Rules for the build:

- the app calls `nbplan`; it contains no model of its own (the same rule as the service);
- everything slow and unchanging (product names, today's capacity and profit) is cached with `@st.cache_data`, **keyed on the database's modification time**, so a new forecast clears the cache (Day 74's bug, avoided);
- every widget has a `key`, so tests can find it.

<details><summary>Solution</summary>

```python
"""Day 75 mini-project: a Streamlit scenario app for the weekly bakery plan.
Run it from the repository root:  streamlit run apps/planner_app.py"""
import os
import sys

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, os.getcwd())                                     # finds nbplan (or: pip install -e .)
from nbplan.config import RESOURCES, ConfigError, check_settings
from nbplan.data import load_inputs
from nbplan.explain import explain_plan, value_curve, why_not
from nbplan.model import solve

DB = os.environ.get("NBPLAN_DATABASE", "northbound.duckdb")
st.set_page_config(page_title="Northbound bakery planner", layout="wide")


@st.cache_data
def reference():
    """The table values and today's plan: read once per data version (the cache key is the database file's mtime)."""
    con = duckdb.connect(DB, read_only=True)
    names = dict(con.sql("SELECT sku, product_name FROM bakery_range").fetchall())
    base_inputs = load_inputs(con, check_settings({"database": DB}))
    return names, base_inputs.capacity, solve(base_inputs).profit, os.path.getmtime(DB)


names, table_capacity, base_profit, _ = reference()

st.sidebar.header("Scenario")
demand = st.sidebar.slider("Demand (x forecast)", 0.80, 1.30, 1.00, 0.05, key="demand")
capacity = {r: st.sidebar.number_input(r.replace("_", " "), min_value=0.0, value=float(table_capacity[r]), step=10.0,
                                       key=f"cap_{r}") for r in RESOURCES}
minimum_sku = st.sidebar.selectbox("Insist on a minimum for", ["(none)"] + sorted(names), key="min_sku")
minimum = st.sidebar.number_input("... batches", min_value=0.0, value=0.0, step=1.0, key="min_value")

try:
    settings = check_settings({"database": DB, "demand_multiplier": demand,
                               "capacity": {r: v for r, v in capacity.items() if v != table_capacity[r]},
                               "min_batches": {minimum_sku: minimum} if minimum_sku != "(none)" and minimum > 0 else {}})
except ConfigError as e:
    st.error(str(e))
    st.stop()
inputs = load_inputs(duckdb.connect(DB, read_only=True), settings)
result = solve(inputs, settings.time_limit)

st.title("Weekly bakery plan")
if result.status != "optimal":
    st.error("No plan meets this scenario. Lower the minimum or add capacity.")
    st.stop()
c1, c2, c3 = st.columns(3)
c1.metric("Profit a week", f"${result.profit:,.0f}", f"{result.profit - base_profit:+,.0f} vs today's settings")
c2.metric("Products made", sum(x > 1e-6 for x in result.plan.values()), f"of {len(result.plan)}")
c3.metric("Bottlenecks", sum(v > 1e-6 for v in result.duals.values()))

left, right = st.columns([3, 2])
with left:
    st.subheader("Why the plan looks like this")
    for line in explain_plan(inputs, result, names):
        st.markdown(f"- {line}")
    plan = pd.DataFrame({"product": [names[p] for p in result.plan], "batches": list(result.plan.values()),
                         "forecast": [inputs.cap[p] for p in result.plan]}, index=list(result.plan))
    st.subheader("The plan")
    st.dataframe(plan[plan.batches > 1e-6].sort_values("batches", ascending=False).round(2), key="plan_table")
with right:
    st.subheader("What is more capacity worth?")
    resource = st.selectbox("Resource", list(RESOURCES), index=list(RESOURCES).index("oven_hours"), key="curve_resource")
    curve = value_curve(inputs, resource, steps=(1, 5, 10, 25, 50, 100))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([str(k) for k, _ in curve], [v for _, v in curve], color="#3b82f6")
    ax.set(xlabel=f"extra {resource.replace('_', ' ')}", ylabel="$ a unit, on average", ylim=(0, None))
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    st.subheader("Why not more of ...?")
    sku = st.selectbox("Product", sorted(names), index=sorted(names).index("BK-08"), key="why_sku")
    extra = st.number_input("extra batches", min_value=1.0, value=5.0, step=1.0, key="why_extra")
    cost, pushed, base = why_not(inputs, sku, extra)
    if isinstance(cost, str):
        st.info(f"{names[sku]}: {cost} (planned {base.plan[sku]:.1f} of a forecast of {inputs.cap[sku]:.1f}).")
    else:
        st.info(f"{extra:.0f} more batches of {names[sku]} would cost ${-cost:,.2f} a week, pushing out "
                + ", ".join(f"{names[p]} {d:.1f}" for p, d in pushed[:3]) + ".")
```

Run it from the repository root:

```bash
streamlit run apps/planner_app.py
```

Streamlit opens <http://localhost:8501>. Every change to a widget reruns the script from the top, which is why the cached `reference()` matters.

![The scenario app with ovens at 700 hours](figures/day75_app.png)

*The figure is a screenshot of the app running; it isn't produced by a script, so yours will differ in detail.*

</details>

## 75.2 Test the app as it opens

Streamlit's `AppTest` runs the script headless, lets a test set widgets and reads what the page shows. The test file finds the app relative to itself.

```python
APP = "../apps/planner_app.py"                                    # relative to this file


def summary(at):
    m = {x.label: (x.value, x.delta) for x in at.metric}
    return (f"profit {m['Profit a week'][0]} ({m['Profit a week'][1]}), products {m['Products made'][0]}, "
            f"bottlenecks {m['Bottlenecks'][0]}; why-not: {at.info[0].value if at.info else '-'}")


# ---------- 75.2 The app as it opens ----------
print("--- opening")
at = AppTest.from_file(APP, default_timeout=60).run()
print(f"exceptions: {len(at.exception)}; errors: {len(at.error)}")
print(summary(at))
print("first explanation: " + at.markdown[0].value)
print(f"plan table: {at.dataframe[0].value.shape[0]} rows, columns {list(at.dataframe[0].value.columns)}")
```

```text
--- opening
exceptions: 0; errors: 0
profit $108,351 (+0 vs today's settings), products 22, bottlenecks 3; why-not: 5 more batches of Seeded sourdough would cost $257.56 a week, pushing out Sourdough loaf -5.0, Chocolate chip cookies 12-pack -1.2.
first explanation: - oven hours is fully used: one more unit would add $99.34 a week.
plan table: 22 rows, columns ['product', 'batches', 'forecast']
```

The opening page matches the service and the command line: $108,351, 22 products, 3 bottlenecks, and the same why-not answer for seeded sourdough.

## 75.3 Scenarios through the widgets

```python
# ---------- 75.3 Scenarios through the widgets ----------
print("--- scenarios")
at.number_input(key="cap_oven_hours").set_value(700.0).run()
print("ovens 700:      " + summary(at))
at.number_input(key="cap_oven_hours").set_value(600.0)
at.slider(key="demand").set_value(1.15).run()
print("demand x1.15:   " + summary(at))
at.slider(key="demand").set_value(1.0)
at.selectbox(key="why_sku").set_value("BK-CR").run()
print("why-not BK-CR:  " + summary(at))
at.selectbox(key="min_sku").set_value("BK-08")
at.number_input(key="min_value").set_value(500.0).run()
print(f"impossible:     errors shown: {[e.value for e in at.error]}")
```

```text
--- scenarios
ovens 700:      profit $109,435 (+1,084 vs today's settings), products 23, bottlenecks 3; why-not: 5 more batches of Seeded sourdough would cost $79.62 a week, pushing out Rye loaf -5.0, Brownies 6-pack -0.2.
demand x1.15:   profit $110,646 (+2,295 vs today's settings), products 20, bottlenecks 3; why-not: 5 more batches of Seeded sourdough would cost $261.24 a week, pushing out Sourdough loaf -5.0, Brownies 6-pack -1.2.
why-not BK-CR:  profit $108,351 (+0 vs today's settings), products 22, bottlenecks 3; why-not: Croissant 6-pack: above forecast (planned 31.0 of a forecast of 31.0).
impossible:     errors shown: ['No plan meets this scenario. Lower the minimum or add capacity.']
```

- Ovens at 700 hours: +$1,084 (Day 13's lease value) and one more product. With ovens looser, 5 more seeded sourdough cost only $79.62 a week.
- Demand ×1.15: +$2,295 with 20 products: the bakery drops low-margin products to make more of the best ones.
- Croissants are at forecast, so why-not says so.
- 500 batches of BK-08 is impossible: the page shows the error message, and no exception.

AppTest prints a "missing ScriptRunContext" warning to the terminal. It's harmless: it only means no browser is attached.

> **Decision it supports:** scenario questions from planners and managers ("what if demand is up 15%?", "is the lease worth it?"), answered in minutes and in the room, by the people who ask them.
>
> **Other places the same practices apply:** what-if tools for budgets, staffing, pricing and capacity; any model whose users aren't programmers.

## Commit

```bash
git add .
git commit -m "Day 75: Streamlit scenario app with AppTest"
git push
```

### Day 75 checklist

- [ ] My app calls the tested package and has no model of its own.
- [ ] Slow, unchanging data is cached, keyed on the data's version.
- [ ] Every widget has a key, and the app has automated tests.
- [ ] An impossible scenario shows a message, not a stack trace.
- [ ] A real user tried it, and I fixed what confused them.

---

# DAY 76 — Re-optimization and Plan Stability

Every Sunday the plan is re-solved with the new forecast. Each solve is optimal for its week. But the bakery doesn't experience a series of optimal plans; it experiences the **changes** between them:

- batches moved between products;
- products switched on and off;
- staff rosters, flour orders and store displays redone.

A plan that changes a lot for $20 of profit is a bad plan. Today you measure that nervousness and put a price on change.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 76.1 Nervous plans |
| 0:45 – 1:45 | 76.2 A price on change |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 76.3 Re-planning on Wednesday |
| 2:45 – 3:45 | A planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: an oven breaks down |
| 4:45 – 5:00 | Commit with Git + checklist |

## 76.1 Nervous plans

Twelve weeks of re-planning. Each week's forecast scales the product caps by that week's historical forecast error (Day 42's error table: bread by the sourdough and whole-wheat average, everything else by croissants).

```python
# ---------- 76.1 Nervous plans ----------
print("--- re-planning from scratch")
free = rolling(0.0)
print(free.round(2).to_string(index=False))
print(f"12 weeks: profit ${free.profit.sum():,.2f}; {free.batches_moved.sum():,.1f} batches moved; "
      f"{int(free.products_switched.sum())} products switched on or off")
```

```text
--- re-planning from scratch
      week    profit  batches_moved  products_switched
2026-01-26 108447.58           0.00                  0
2026-02-02 108207.46          11.90                  1
2026-02-09 108271.66           8.95                  1
2026-02-16 107984.07          14.64                  1
2026-02-23 108713.69          26.00                  2
2026-03-02 108591.42          17.37                  1
2026-03-09 108041.70          26.39                  2
2026-03-16 108122.54          22.44                  2
2026-03-23 108120.58           6.62                  1
2026-03-30 108468.46          18.20                  1
2026-04-06 108097.52          17.40                  1
2026-04-13 108622.77          20.29                  2
12 weeks: profit $1,299,689.44; 190.2 batches moved; 15 products switched on or off
```

Profit barely moves: $107,984 to $108,714. The plan moves a lot more: 190 batches over 11 changes, and **15 products switched on or off**. Each switch means a new line on the bake sheet, a roster change and a gap on a store shelf. The model doesn't know any of that, because nothing in it says change costs anything.

## 76.2 A price on change

`nbplan.model` now takes the previous plan and a **change cost** in dollars per batch moved. Moving is measured with two non-negative variables per product:

- `x[p] − previous[p] = up[p] − down[p]`;
- the objective subtracts `change_cost × Σ (up + down)`.

The solver never sets both `up` and `down` for one product (that costs more for nothing), so `up + down` is exactly |x − previous|. `Result.profit` stays the production profit, without the penalty, so it's comparable with every earlier run.

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
    changes: float = 0.0        # batches changed against the previous plan (Day 76)


def build(inputs, previous=None, change_cost=0.0):
    m = pyo.ConcreteModel(name="nbplan")
    m.P, m.R = pyo.Set(initialize=inputs.products), pyo.Set(initialize=inputs.resources)
    m.x = pyo.Var(m.P, within=pyo.NonNegativeReals)
    m.demand = pyo.Constraint(m.P, rule=lambda m, p: m.x[p] <= inputs.cap[p])
    m.capacity_limit = pyo.Constraint(m.R, rule=lambda m, r: sum(inputs.use.get((r, p), 0) * m.x[p] for p in m.P)
                                      <= inputs.capacity[r])
    m.minimum = pyo.Constraint([p for p in inputs.min_batches], rule=lambda m, p: m.x[p] >= inputs.min_batches[p])
    m.production_profit = pyo.Expression(expr=sum(inputs.profit[p] * m.x[p] for p in m.P))
    if previous is None:
        m.total_profit = pyo.Objective(expr=m.production_profit, sense=pyo.maximize)
    else:
        # Day 76: every batch moved up or down against the previous plan costs change_cost
        m.up = pyo.Var(m.P, within=pyo.NonNegativeReals)
        m.down = pyo.Var(m.P, within=pyo.NonNegativeReals)
        m.moved = pyo.Constraint(m.P, rule=lambda m, p: m.x[p] - previous.get(p, 0.0) == m.up[p] - m.down[p])
        m.total_profit = pyo.Objective(expr=m.production_profit - change_cost * sum(m.up[p] + m.down[p] for p in m.P),
                                       sense=pyo.maximize)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def solve(inputs, time_limit=60.0, previous=None, change_cost=0.0):
    """previous: last plan (sku -> batches); change_cost: $ per batch changed against it (0: re-plan freely)."""
    m = build(inputs, previous, change_cost)
    res = pyo.SolverFactory("highs").solve(m, load_solutions=False, options={"time_limit": time_limit})
    tc = res.solver.termination_condition
    if tc != pyo.TerminationCondition.optimal:
        return Result(str(tc), None, {}, {}, {}, {})
    m.solutions.load_from(res)
    clean = lambda v: round(v, 9) + 0.0
    plan = {p: clean(pyo.value(m.x[p])) for p in m.P}
    changes = sum(abs(plan[p] - previous.get(p, 0.0)) for p in m.P) if previous is not None else 0.0
    return Result("optimal", clean(pyo.value(m.production_profit)), plan,
                  {r: clean(m.dual[m.capacity_limit[r]]) for r in m.R}, {p: clean(m.rc[m.x[p]]) for p in m.P},
                  {p: clean(m.dual[m.demand[p]]) for p in m.P}, clean(changes))
```

```python
# ---------- 76.2 A price on change ----------
print("--- change costs")
rows = []
for c in (0, 5, 10, 25, 50, 100):
    t = rolling(c)
    rows.append((c, round(t.profit.sum(), 2), round(free.profit.sum() - t.profit.sum(), 2), round(t.batches_moved.sum(), 1),
                 int(t.products_switched.sum())))
curve = pd.DataFrame(rows, columns=["change_cost", "profit_12_weeks", "profit_given_up", "batches_moved", "products_switched"])
print(curve.to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(curve.batches_moved, curve.profit_given_up / 1000, "o-", color="#3b82f6")
for r in curve.itertuples():
    ax.annotate(f"${r.change_cost}/batch", (r.batches_moved, r.profit_given_up / 1000), textcoords="offset points",
                xytext=(5, 5), fontsize=8, color="#6b7280")
ax.set(xlabel="batches moved between weekly plans (12 weeks)", ylabel="profit given up, $ thousands",
       title="Stability costs a little profit; nervousness moves a lot of batches")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/day76_stability.png", dpi=150)
print("saved figures/day76_stability.png")
```

```text
--- change costs
 change_cost  profit_12_weeks  profit_given_up  batches_moved  products_switched
           0       1299689.44             0.00          190.2                 15
           5       1299672.79            16.65          180.2                 10
          10       1299588.02           101.42          163.6                  3
          25       1298552.14          1137.30           99.7                  3
          50       1297454.40          2235.04           73.3                  4
         100       1296041.06          3648.38           52.7                  4
saved figures/day76_stability.png
```

![Profit given up against batches moved](figures/day76_stability.png)

- **$10 a batch** gives up $101 over 12 weeks and cuts switches from 15 to 3.
- **$25 a batch** halves the batches moved (190 to 100) for $1,137, about $95 a week.
- Beyond that, each extra batch kept still costs more and more.

Who picks the price? Not the analyst. Show the planners this curve and ask what a moved batch costs them in rework and confusion. That number is the change cost. Switches also fall faster than batches: a product turned off and on again is the most expensive kind of nervousness, and the penalty removes it first.

## 76.3 Re-planning on Wednesday

A plan can also change **during** the week. On Wednesday morning a storm forecast arrives (the January 2025 storm week): demand will be far below Sunday's forecast. Monday's and Tuesday's batches (2/6 of the week) are already baked.

```python
# ---------- 76.3 Re-planning on Wednesday ----------
print("--- wednesday")
monday = solve(forecast_week(WEEKS[0]))
storm = [i for i, w in enumerate(err.index) if str(w).startswith("2025-01-06")][0]
update = forecast_week(storm)                                          # Wednesday: a storm, demand far below forecast
baked = {p: x * 2 / 6 for p, x in monday.plan.items() if x > 1e-6}     # Monday and Tuesday are already baked
scratch = solve(update)
below = {p: round(baked[p] - scratch.plan[p], 2) for p in baked if scratch.plan[p] < baked[p] - 1e-6}
print(f"from scratch: ${scratch.profit:,.2f}, but {len(below)} products below what is already baked: {below}")
fenced = solve(dataclasses.replace(update, min_batches={p: min(v, update.cap[p]) for p, v in baked.items()}))
print(f"with the baked batches fixed: ${fenced.profit:,.2f}; moved {sum(abs(fenced.plan[p] - monday.plan[p]) for p in fenced.plan):.1f} batches")
calm = solve(dataclasses.replace(update, min_batches={p: min(v, update.cap[p]) for p, v in baked.items()}),
             previous=monday.plan, change_cost=25)
print(f"fixed + $25 a changed batch: ${calm.profit:,.2f}; moved {calm.changes:.1f} batches")
```

```text
--- wednesday
from scratch: $95,682.47, but 1 products below what is already baked: {'BK-07': 0.13}
with the baked batches fixed: $95,680.62; moved 200.1 batches
fixed + $25 a changed batch: $95,676.91; moved 199.6 batches
```

- Re-solving **from scratch** plans 0.13 batches of BK-07 fewer than are already on the shelves. It's a small error here, but a plan that undoes what's already been baked is not a plan.
- **Fixing the baked batches** (as minimums) makes the plan feasible for $1.85.
- The change cost barely matters this time: the storm forces about 200 batches of cuts whatever the penalty. A change cost is for **noise**; a real shock should move the plan. If the penalty were high enough to block these cuts, the bakery would bake for customers who aren't coming.

> **Decision it supports:** weekly and mid-week re-planning where changes have costs the model doesn't see: production, staffing, inventory, transport.
>
> **Other places the same practices apply:** airline crew and aircraft recovery, hospital rosters, supply plans with frozen horizons, model predictive control.

### The planted bug

`day76/bug_one_sided.py` is a colleague's hand-built change penalty, at $25 a batch. Run it over the same 12 weeks.

```python
prev, moved, profit = None, 0.0, 0.0
for w in range(60, 72):
    e = err.iloc[w]
    f = {"bread": (e["BK-SD"] + e["BK-WW"]) / 2}
    inputs = dataclasses.replace(base, cap={p: c * f.get(base.category[p], e["BK-CR"]) for p, c in base.cap.items()})
    m = build(inputs)
    if prev is not None:
        m.change = pyo.Var(m.P, within=pyo.NonNegativeReals)
        m.count = pyo.Constraint(m.P, rule=lambda m, p: m.change[p] >= m.x[p] - prev[p])      # "batches changed"
        m.total_profit.deactivate()
        m.with_penalty = pyo.Objective(expr=m.production_profit - 25 * sum(m.change[p] for p in m.P), sense=pyo.maximize)
    pyo.SolverFactory("highs").solve(m)
    plan = {p: pyo.value(m.x[p]) for p in m.P}
    if prev is not None:
        moved += sum(abs(plan[p] - prev[p]) for p in plan)
    profit += pyo.value(m.production_profit)
    prev = plan
print(f"$25 a changed batch: 12 weeks, profit ${profit:,.2f}, {moved:.1f} batches moved")
```

<details><summary>What happens, and why</summary>

```text
$25 a changed batch: 12 weeks, profit $1,299,485.68, 149.8 batches moved
```

The correct $25 penalty moves 99.7 batches. This one moves 149.8. The constraint `change >= x - prev` counts only **increases**. A cut gives `x - prev < 0`, so `change` can be 0, and cuts are free. The plan still swings, by cutting one product and then raising another later.

Add the other side, `change >= prev - x`, or use `up` and `down` as `nbplan.model` does. The property test in `tests/test_delivery.py` (a huge change cost must give zero changes) catches this bug.

</details>

## Mini-project: an oven breaks down

Wednesday morning an oven fails for the rest of the week. Wednesday to Saturday run at 70% of their oven hours (Day 44's breakdown, from `bakery_disruptions`). Monday's and Tuesday's batches are baked.

Re-plan three ways and compare profit and batches moved against Monday's plan:

1. from scratch;
2. with the baked batches fixed;
3. fixed, with a $25 change cost.

<details><summary>Solution</summary>

```python
"""Day 76 mini-project: Wednesday morning, an oven fails for the rest of the week: the four remaining days run at 70% of
their oven hours (Day 44's breakdown). Monday's and Tuesday's batches are baked. Re-plan the rest of the week three ways
and compare profit and batches moved: from scratch (ignoring what is baked), with the baked batches fixed, and fixed
with a $25 change cost against Monday's plan.

Run from the repository root (after day06/load_tables.py):  python day76/mini_project_breakdown.py
"""
import dataclasses
import sys

import duckdb
import pandas as pd

sys.path.insert(0, ".")
from nbplan.check import check_plan
from nbplan.config import Settings
from nbplan.data import load_inputs
from nbplan.model import solve

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
week = load_inputs(con, Settings())
left = con.sql("SELECT capacity_left FROM bakery_disruptions WHERE event = 'oven_breakdown'").fetchall()[0][0]
monday = solve(week)
baked = {p: x * 2 / 6 for p, x in monday.plan.items() if x > 1e-6}
oven_used = sum(week.use.get(("oven_hours", p), 0) * x for p, x in baked.items())
broken = dataclasses.replace(week, capacity={**week.capacity, "oven_hours": oven_used + week.capacity["oven_hours"] * 4 / 6 * left})
print(f"oven hours for the week: {broken.capacity['oven_hours']:.1f} (Monday-Tuesday used {oven_used:.1f}; "
      f"Wednesday-Saturday at {left:.0%})")
rows = []
scratch = solve(broken)
short = [p for p in baked if scratch.plan[p] < baked[p] - 1e-6]
rows.append(("from scratch", scratch.profit, sum(abs(scratch.plan[p] - monday.plan[p]) for p in scratch.plan),
             f"{len(short)} products below what is already baked"))
fenced_inputs = dataclasses.replace(broken, min_batches=baked)
fenced = solve(fenced_inputs)
rows.append(("baked batches fixed", fenced.profit, sum(abs(fenced.plan[p] - monday.plan[p]) for p in fenced.plan),
             f"{len(check_plan(fenced_inputs, fenced.plan)[0])} constraints broken"))
calm = solve(fenced_inputs, previous=monday.plan, change_cost=25)
rows.append(("fixed + $25 a changed batch", calm.profit, calm.changes, f"{len(check_plan(fenced_inputs, calm.plan)[0])} constraints broken"))
t = pd.DataFrame(rows, columns=["re-plan", "profit", "batches_moved", "note"])
t["vs_monday_plan"] = (t.profit - monday.profit).round(2)
print(t.round(2).to_string(index=False))
```

```text
oven hours for the week: 480.0 (Monday-Tuesday used 200.0; Wednesday-Saturday at 70%)
                    re-plan   profit  batches_moved                                   note  vs_monday_plan
               from scratch 92569.69         126.26 3 products below what is already baked       -15781.59
        baked batches fixed 91711.67         118.11                   0 constraints broken       -16639.61
fixed + $25 a changed batch 90679.95          67.33                   0 constraints broken       -17671.33
```

- The oven has 480 hours left for the week: 200 already used, and 280 for Wednesday to Saturday.
- **From scratch** looks best ($92,570), but it plans 3 products below what is already baked: it's not a real option.
- **Fixed** is the honest re-plan: $91,712, moving 118 batches.
- **Fixed + $25** keeps 51 more batches where Monday put them (67 moved instead of 118) for $1,032. Whether that's worth it is the planners' call: 51 fewer changes on a day the team is already dealing with a broken oven.

</details>

## Commit

```bash
git add .
git commit -m "Day 76: re-optimization and plan stability"
git push
```

### Day 76 checklist

- [ ] I measure plan changes (batches moved, products switched), not just profit.
- [ ] I penalize change in both directions, with a price the planners choose.
- [ ] I report production profit separately from the penalty.
- [ ] Mid-week re-plans fix what's already done.
- [ ] I don't let a change cost block the response to a real shock.

---

# DAY 77 — Monitoring Plans and Models

A model that was right in testing can go wrong in use, quietly:

- the forecast feeding it gets worse;
- the oven times it assumes drift;
- the plans it produces start to look different;
- the service stops running.

None of these raise an error. Today you build the monitors that notice.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 77.1 Inputs: control limits on forecast errors |
| 0:45 – 1:45 | 77.2 A drift the limits miss: CUSUM |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 77.3 Outputs: plan KPIs; 77.4 the service's own health |
| 2:45 – 3:45 | A planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: monitoring the ovens |
| 4:45 – 5:00 | Commit with Git + checklist |

The monitoring functions go into the package:

```python
"""Day 77: monitoring. Control limits, a CUSUM drift detector, forecast errors from the sales history, and a health
check of the run records."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def forecast_errors(con):
    """Chain-level weekly sales of the core products, the 'same as last week' forecast (Day 1) and its % error."""
    s = con.sql("""SELECT week_start, sku, sum(units_sold) AS units FROM bakery_sales_weekly GROUP BY ALL
                   ORDER BY week_start""").df().pivot(index="week_start", columns="sku", values="units")
    return (s / s.shift(1) - 1).dropna()


def control_limits(series, baseline):
    """Mean +/- 3 standard deviations, estimated on the first `baseline` points only (a period known to be normal)."""
    ref = np.asarray(series[:baseline], dtype=float)
    mu, sd = ref.mean(), ref.std(ddof=1)
    return mu, mu - 3 * sd, mu + 3 * sd


def cusum(series, baseline, k=0.5, h=5.0):
    """Two-sided tabular CUSUM on standardized values: accumulates small persistent shifts that a 3-sigma rule misses.
    k: allowance (in sd), h: decision limit (in sd). Returns the upper and lower sums and the index of each alarm."""
    x = np.asarray(series, dtype=float)
    mu, sd = x[:baseline].mean(), x[:baseline].std(ddof=1)
    z = (x - mu) / sd
    hi, lo, alarms = np.zeros(len(z)), np.zeros(len(z)), []
    for i in range(len(z)):
        hi[i] = max(0.0, (hi[i - 1] if i else 0) + z[i] - k)
        lo[i] = max(0.0, (lo[i - 1] if i else 0) - z[i] - k)
        if hi[i] > h or lo[i] > h:
            alarms.append(i)
            hi[i] = lo[i] = 0.0                                   # restart after an alarm
    return hi, lo, alarms


def run_health(folder):
    """What the run records say about the service itself: runs, failures, repeated inputs, weeks without a run."""
    rows = [json.loads(line) for line in (Path(folder) / "runs.jsonl").read_text().splitlines()]
    df = pd.DataFrame(rows)
    return {"runs": len(df), "not_optimal": int((df.status != "optimal").sum()),
            "distinct_inputs": df.inputs.nunique(), "duplicate_runs": int(df.run_id.duplicated().sum())}
```

Two rules sit behind them:

- **Control limits come from a period known to be normal**, and are then held fixed. The question is "is this week unlike the normal weeks?", not "is this week unlike recent weeks?".
- **Two detectors for two kinds of trouble.** A 3-sigma limit catches a single extreme week. CUSUM (a cumulative sum of small deviations) catches a small shift that persists.

## 77.1 Inputs: is the forecast still as good as it was?

The core products' forecast is Day 1's "same as last week". Its error each week is `sales / last week's sales − 1`, chain-wide. The first 26 weeks set the limits.

```python
# ---------- 77.1 Inputs: is the forecast still as good as it was? ----------
print("--- forecast errors")
e = forecast_errors(con)
BASE = 26                                                            # the first half-year sets the limits
for sku in e.columns:
    mu, lo, hi = control_limits(e[sku], BASE)
    out = e.index[(e[sku] < lo) | (e[sku] > hi)]
    print(f"{sku}: limits {lo:+.1%} to {hi:+.1%}; {len(out)} weeks outside: "
          + ", ".join(str(w)[:10] for w in out))
```

```text
--- forecast errors
BK-CR: limits -37.9% to +38.8%; 2 weeks outside: 2024-12-16, 2025-12-15
BK-SD: limits -15.4% to +15.1%; 2 weeks outside: 2025-09-01, 2025-12-15
BK-WW: limits -14.1% to +14.1%; 3 weeks outside: 2025-01-06, 2025-09-01, 2026-07-06
```

Every alarm is a week someone could explain:

- the weeks of 2024-12-16 and 2025-12-15 are the run-up to Christmas;
- 2025-01-06 is the storm week (Day 76);
- 2025-09-01 is Labour Day week;
- 2026-07-06 shows up again in 77.3.

An alarm is a question to ask, not a conclusion. Log the answer next to the alarm, so next year's monitor knows the holiday weeks.

## 77.2 A drift the limits miss

From week 70 a new forecasting system goes live and over-forecasts whole wheat by 8%. Every week's error drops by about 8 points. That's well inside the ±14% limits.

```python
# ---------- 77.2 A drift the limits miss ----------
print("--- drift")
ww = e["BK-WW"].to_numpy()
start = 70                                                           # a new forecasting system goes live ...
drift = ww.copy()
drift[start:] = (1 + ww[start:]) / 1.08 - 1                          # ... and over-forecasts whole wheat by 8%
for name, series in (("history as it was", ww), ("with the drift", drift)):
    mu, lo, hi = control_limits(series, BASE)
    shewhart = [i for i in range(BASE, len(series)) if not lo <= series[i] <= hi]
    _, _, alarms = cusum(series, BASE)
    after = [a for a in alarms if a >= start]
    print(f"{name}: 3-sigma alarms at weeks {shewhart}; CUSUM alarms at {alarms}"
          + (f" -> drift found {after[0] - start} weeks after it began" if name != "history as it was" and after else ""))
hi_s, lo_s, alarms = cusum(drift, BASE)
fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
weeks = np.arange(len(drift))
mu, lo, hi = control_limits(drift, BASE)
axes[0].plot(weeks, drift * 100, color="#3b82f6", lw=1)
axes[0].axhline(lo * 100, ls="--", color="#e5484d", lw=1)
axes[0].axhline(hi * 100, ls="--", color="#e5484d", lw=1)
axes[0].axvline(start, color="#9aa4b2", lw=1)
axes[0].set(ylabel="whole-wheat forecast\nerror (%)", title="An 8% bias stays inside the 3-sigma limits; CUSUM finds it")
axes[1].plot(weeks, lo_s, color="#22a06b", label="CUSUM (downward)")
axes[1].axhline(5, ls="--", color="#e5484d", lw=1, label="decision limit")
axes[1].axvline(start, color="#9aa4b2", lw=1)
for a in alarms:
    axes[1].plot(a, 5, "v", color="#e5484d")
axes[1].set(xlabel="week", ylabel="CUSUM (sd)")
axes[1].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/day77_monitor.png", dpi=150)
print("saved figures/day77_monitor.png")
```

```text
--- drift
history as it was: 3-sigma alarms at weeks [39, 83]; CUSUM alarms at []
with the drift: 3-sigma alarms at weeks [39, 83]; CUSUM alarms at [73, 78, 83, 89, 97] -> drift found 3 weeks after it began
saved figures/day77_monitor.png
```

![An 8% bias stays inside the 3-sigma limits; CUSUM finds it](figures/day77_monitor.png)

- The 3-sigma rule flags the same two weeks with or without the bias. It can't see the bias at all.
- CUSUM finds it at week 73, **3 weeks after it began**, and keeps alarming (it restarts after each alarm) until someone fixes the forecast.
- On the real history, CUSUM raises no alarm.

An 8% over-forecast means 8% too much whole-wheat bread baked every week, much of it waste. Without CUSUM it would show up months later, in the waste report.

## 77.3 Outputs: what the weekly plans say

Monitor what the model produces, too. The script runs `nbplan` on 26 weeks of forecasts, records every run (Day 71), and watches:

- profit, with limits from the first 13 weeks;
- the value of oven hours;
- products made;
- whether the **set of bottlenecks** changed.

```python
# ---------- 77.3 Outputs: what the weekly plans say ----------
print("--- plan kpis")
base = load_inputs(con, Settings())
err = error_table(con)
folder = tempfile.mkdtemp()
rows, prev_bottle = [], None
for w in range(60, 86):
    x = err.iloc[w]
    f = {"bread": (x["BK-SD"] + x["BK-WW"]) / 2}
    inputs = dataclasses.replace(base, cap={p: c * f.get(base.category[p], x["BK-CR"]) for p, c in base.cap.items()})
    r = solve(inputs)
    record(r, inputs, Settings(), folder)
    bottle = tuple(k for k, v in r.duals.items() if v > 1e-6)
    rows.append((str(err.index[w])[:10], round(r.profit), round(r.duals["oven_hours"], 2), sum(v > 1e-6 for v in r.plan.values()),
                 "CHANGED: " + ", ".join(bottle) if prev_bottle is not None and bottle != prev_bottle else ""))
    prev_bottle = bottle
k = pd.DataFrame(rows, columns=["week", "profit", "oven_value", "products_made", "bottleneck_alert"])
mu, lo, hi = control_limits(k.profit, 13)
k["profit_alert"] = np.where((k.profit < lo) | (k.profit > hi), "OUTSIDE", "")
print(k.to_string(index=False))
```

```text
--- plan kpis
      week  profit  oven_value  products_made                                                   bottleneck_alert profit_alert
2026-01-26  108448       89.15             22
2026-02-02  108207       99.34             23
2026-02-09  108272       99.34             22
2026-02-16  107984       99.34             23
2026-02-23  108714       89.15             21
2026-03-02  108591       99.34             22
2026-03-09  108042       89.15             22
2026-03-16  108123       99.34             22
2026-03-23  108121       99.34             23
2026-03-30  108468       89.15             22
2026-04-06  108098       99.34             23
2026-04-13  108623       89.15             21
2026-04-20  108302       99.34             22
2026-04-27  107921       88.72             24
2026-05-04  108215       99.34             22
2026-05-11  108292       99.34             23
2026-05-18  108315       99.34             22
2026-05-25  108410       99.34             22
2026-06-01  107837       88.72             24
2026-06-08  108540       89.15             22
2026-06-15  108073       99.34             23
2026-06-22  108507       99.34             22
2026-06-29  108209       99.34             22
2026-07-06  106355       27.71             26 CHANGED: baker_hours, oven_hours, packing_hours, sourdough_starter      OUTSIDE
2026-07-13  108547       99.34             22              CHANGED: oven_hours, packing_hours, sourdough_starter
2026-07-20  108171       99.34             23
```

For 23 weeks the plan is boring: profit within about $900, ovens worth $89 to $99 an hour. In the week of 2026-07-06 everything moves at once:

- profit falls outside the limits ($106,355);
- the oven's value drops to $27.71;
- 26 products are made;
- baker hours join the bottlenecks.

That's the week the forecast monitor (77.1) also flagged for whole wheat. The forecast fell sharply, the plan responded, and both monitors agree. The next week the bottlenecks change back. A planner should look at that week **before** it's baked: is the drop real, or a data problem?

## 77.4 The service's own health

```python
# ---------- 77.4 The service's own health ----------
print("--- run health")
print(run_health(folder))
```

```text
--- run health
{'runs': 26, 'not_optimal': 0, 'distinct_inputs': 26, 'duplicate_runs': 0}
```

The run records answer the operational questions:

- did it run every week (26 runs);
- did any solve fail (0 not optimal);
- did the inputs actually change each week (26 distinct fingerprints: a repeated fingerprint would mean the forecast wasn't updated)?

In production, add one more: alarm if no run has been recorded by Sunday 10 p.m.

> **Decision it supports:** whether to keep trusting a model's plans this week, and when to fix its inputs or re-fit it.
>
> **Other places the same practices apply:** statistical process control on production lines, model-risk monitoring in banks, drift detection for machine-learning models, service-level monitoring.

### The planted bug

`day77/bug_limits.py` is a colleague's monitor that keeps its limits "up to date": each week, the limits and CUSUM use the last 26 weeks. Run it on the whole-wheat errors with the 8% bias.

```python
last = range(len(drift) - 15, len(drift))
rolling_alarms = []
for week in last:                                                   # the colleague's monitor, week by week
    window = drift[week - 26: week]
    mu, sd = window.mean(), window.std(ddof=1)
    _, _, a = cusum(np.append(window, drift[week]), baseline=26)
    if a and a[-1] == 26:
        rolling_alarms.append(week)
_, _, fixed = cusum(drift, baseline=26)
print(f"last 15 weeks ({last.start}-{last.stop - 1}): rolling-limit monitor alarms {rolling_alarms or 'none'}; "
      f"fixed-limit CUSUM alarms {[a for a in fixed if a in last] or 'none'}")
print(f"the 'normal' level the rolling monitor has learned by the end: {drift[-26:].mean():+.1%} "
      f"(before the bias: {ww[:70].mean():+.1%})")
```

<details><summary>What happens, and why</summary>

```text
last 15 weeks (88-102): rolling-limit monitor alarms none; fixed-limit CUSUM alarms [89, 97]
the 'normal' level the rolling monitor has learned by the end: -7.0% (before the bias: +0.2%)
```

In the last 15 weeks, the rolling monitor raises no alarm; fixed limits alarm twice. By then the rolling window holds only biased weeks, and its "normal" is −7.0%: the bias **is** the new normal. A rolling baseline learns whatever is happening, including the problem it was meant to catch.

Set limits on a period known to be good, hold them fixed, and re-set them only on purpose (after a fix, with a note saying why).

</details>

## Mini-project: monitoring the ovens

Day 57's `bakery_oven_log` records actual and standard oven hours. Monitor the week's ratio, actual / standard, for the whole chain:

- set limits on the first 13 weeks;
- check the real log with the 3-sigma rule and CUSUM;
- test the monitor: from week 18, one oven in four runs 15% slow (a failing thermostat). How soon is it found?

<details><summary>Solution</summary>

```python
"""Day 77 mini-project: Day 57's oven log records actual against standard oven hours. Monitor the whole week's oven-time
ratio: set limits on the first 13 weeks, run a CUSUM, and test the monitor on a simulated problem: from week 18 an oven
runs 15% slow (a failing thermostat). When is it found? Is there any alarm on the real log?

Run from the repository root (after day06/load_tables.py):  python day77/mini_project_ovens.py
"""
import sys

import duckdb
import numpy as np

sys.path.insert(0, ".")
from nbplan.monitor import control_limits, cusum

con = duckdb.connect("northbound.duckdb", read_only=True)
week = con.sql("""SELECT week_start, sum(actual_oven_hours) / sum(standard_oven_hours) AS ratio
                  FROM bakery_oven_log GROUP BY ALL ORDER BY week_start""").df()
ratio = week.ratio.to_numpy()
slow = ratio.copy()
slow[18:] += 0.15 * 0.25                                             # one of four ovens 15% slower: +3.75% on the week
for name, series in (("the real log", ratio), ("one oven 15% slow from week 18", slow)):
    mu, lo, hi = control_limits(series, 13)
    shewhart = [i for i in range(13, len(series)) if not lo <= series[i] <= hi]
    _, _, alarms = cusum(series, 13)
    print(f"{name}: limits {lo:.3f}-{hi:.3f}; 3-sigma alarms {shewhart or 'none'}; CUSUM alarms {alarms or 'none'}")
print(f"weekly ratio: mean {ratio[:13].mean():.4f}, sd {ratio[:13].std(ddof=1):.4f} (first 13 weeks)")
```

```text
the real log: limits 0.926-1.084; 3-sigma alarms none; CUSUM alarms none
one oven 15% slow from week 18: limits 0.926-1.084; 3-sigma alarms [23]; CUSUM alarms [21, 24]
weekly ratio: mean 1.0048, sd 0.0264 (first 13 weeks)
```

- The real log is clean: no alarms either way. The ovens run at 1.005 of standard time, with a weekly spread of 0.026.
- One oven 15% slow adds 3.75% to the chain's ratio, less than 1.5 standard deviations a week. CUSUM finds it at week 21, three weeks in; the 3-sigma rule first fires at week 23.
- Three weeks of a slow oven is three weeks of plans that assume oven hours that don't exist. At the oven's shadow price ($99.34 an hour, 74.2), every missing hour costs real profit.

</details>

## Commit

```bash
git add .
git commit -m "Day 77: monitoring plans and models"
git push
```

### Day 77 checklist

- [ ] I monitor the model's inputs, its outputs and its own running.
- [ ] My control limits come from a known-good period and are held fixed.
- [ ] I use CUSUM (or similar) for small, persistent shifts.
- [ ] I test each monitor by injecting the problem it should catch.
- [ ] Every alarm leads to a question and a logged answer.
