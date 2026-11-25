# DAY 78 — Human in the Loop and Governance

A plan is a recommendation until a person releases it. Planners know things the model doesn't: a festival, a tank being cleaned, a café's extra order. Sometimes they are right and the model is wrong. Sometimes they are wrong, and every override quietly costs money.

Today you build the loop between the two:

- an override is priced before it's accepted;
- the right person approves it;
- it's recorded with who and why;
- it's checked later against what actually happened.

Then you write down who owns what: the model card.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 78.1 Pricing override requests |
| 0:45 – 1:45 | 78.2 Approving and recording; 78.3 were they right? |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 78.4 The model card and decision rights |
| 2:45 – 3:45 | A planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: an approval budget |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). Days 78–80 add five small tables:

- `override_requests.csv`: this week's six requests from planners and managers;
- `approval_rules.csv`: who approves an override, by its cost a week;
- `flood_scenarios.csv`, `recovery_options.csv`, `capstone_settings.csv`: for Day 80.

No new packages.

## 78.1 Pricing override requests

Rule one: **an override is a constraint, never a number typed over the plan.**

- "At least 8 batches of olive bread" becomes a minimum.
- "At most 25 batches of sourdough" lowers that product's cap.

The model is re-solved, so the rest of the plan adjusts around the override and still fits the ovens. The profit given up is the override's price.

```python
"""Day 78: planner overrides. An override is a constraint added to the model and re-solved, never a number typed
over the plan; its cost is the profit it gives up, and the cost decides who must approve it."""
import dataclasses
from dataclasses import dataclass

from .model import solve


@dataclass(frozen=True)
class Override:
    request_id: str
    requested_by: str
    sku: str
    kind: str                # "at_least" or "at_most"
    batches: float
    reason: str

    def as_dict(self):
        return dataclasses.asdict(self)


def load_overrides(con):
    rows = con.sql("SELECT request_id, requested_by, sku, kind, batches, reason FROM override_requests ORDER BY 1").fetchall()
    return [Override(*r) for r in rows]


def apply_overrides(inputs, overrides):
    """The inputs with the overrides as bounds: at_least raises a minimum, at_most lowers the product's cap."""
    unknown = sorted({o.sku for o in overrides} - set(inputs.products))
    if unknown:
        raise ValueError(f"override for unknown product(s): {', '.join(unknown)}")
    mins, caps = dict(inputs.min_batches), dict(inputs.cap)
    for o in overrides:
        if o.kind == "at_least":
            mins[o.sku] = max(mins.get(o.sku, 0.0), o.batches)
        elif o.kind == "at_most":
            caps[o.sku] = min(caps[o.sku], o.batches)
        else:
            raise ValueError(f"{o.request_id}: kind must be at_least or at_most, not {o.kind!r}")
    return dataclasses.replace(inputs, min_batches=mins, cap=caps)


def override_cost(inputs, overrides, base=None):
    """(status, profit given up a week, the re-solved result). Infeasible overrides have no cost: they can't be done."""
    base = base or solve(inputs)
    r = solve(apply_overrides(inputs, overrides))
    if r.status != "optimal":
        return r.status, None, r
    return r.status, base.profit - r.profit, r


def why_infeasible(inputs, o):
    """A sentence for the usual reasons a single override can't be met."""
    if o.kind == "at_least" and o.batches > inputs.cap[o.sku] + 1e-9:
        return f"{o.batches:g} batches is above the forecast ({inputs.cap[o.sku]:g}): change the forecast, not the plan"
    if o.kind == "at_most" and o.batches < inputs.min_batches.get(o.sku, 0) - 1e-9:
        return f"{o.batches:g} batches is below the agreed minimum ({inputs.min_batches[o.sku]:g})"
    return "the capacities can't make this together with the other minimums"


def approver(cost, rules):
    """The first rule whose limit covers the cost. rules: [(level, max_cost_per_week, approver)] in increasing order."""
    for level, limit, who in rules:
        if cost <= limit + 1e-9:
            return level, who
    return rules[-1][0], rules[-1][2]
```

```python
# ---------- 78.1 What does each request cost, and who must say yes? ----------
print("--- requests")
rows, approved = [], []
for o in requests:
    status, cost, r = override_cost(inputs, [o], base)
    if status != "optimal":
        rows.append((o.request_id, o.sku, f"{o.kind} {o.batches:g}", f"{base.plan[o.sku]:.1f}", None, "rejected",
                     why_infeasible(inputs, o)))
        continue
    level, who = approver(cost, rules)
    moved = sorted(((p, r.plan[p] - base.plan[p]) for p in r.plan if abs(r.plan[p] - base.plan[p]) > 0.05 and p != o.sku),
                   key=lambda kv: -abs(kv[1]))[:2]
    rows.append((o.request_id, o.sku, f"{o.kind} {o.batches:g}", f"{base.plan[o.sku]:.1f}", round(cost, 2), level,
                 "other changes: " + ", ".join(f"{p} {d:+.1f}" for p, d in moved) if moved else "no other product moves"))
    approved.append(o)
print(pd.DataFrame(rows, columns=["request", "sku", "asks", "model", "cost_a_week", "approval", "effect"]).to_string(index=False))
```

```text
--- requests
request   sku        asks model  cost_a_week      approval                                                                   effect
     R1 BK-CR at_least 34  31.0          NaN      rejected 34 batches is above the forecast (31): change the forecast, not the plan
     R2 BK-07  at_least 8   0.0       122.51     automatic                                    other changes: BK-SD -8.0, BK-18 +5.0
     R3 BK-WW at_least 55  45.0          NaN      rejected 55 batches is above the forecast (45): change the forecast, not the plan
     R4 BK-SD  at_most 25  35.0       179.97     automatic                                   other changes: BK-07 +10.0, BK-30 -6.7
     R5 BK-19  at_least 4   0.0       495.77 planning lead                                    other changes: BK-18 -5.2, BK-26 -0.4
     R6 BK-12  at_most 10  15.0       187.77     automatic                                    other changes: BK-18 +4.5, BK-26 +0.9
```

- **R1 and R3 are rejected**, but not because they're wrong. Both ask for more than the forecast. A plan can't sell more croissants than the demand cap says exist, so these are **forecast questions**. Send them to Forecasting with the planner's evidence ("we run out by 10 am on Saturdays").
- **R2, R4 and R6 cost under $250 a week**, so they're approved automatically. The table shows what each one moves. R2's 8 olive bread batches come out of sourdough; R4's sourdough cut lets 10 batches of olive bread in.
- **R5 costs $495.77 a week** (four batches of a product the model drops), so the Head of Planning decides. A launch promised to the board may well be worth $496 a week, but a person has to say so.

## 78.2 Approve, re-solve together, record

```python
# ---------- 78.2 Approve, re-solve together, record ----------
print("--- approved together")
together = [o for o in approved if o.request_id in ("R2", "R4", "R6")]                        # R5 waits for the planning lead
status, cost, r = override_cost(inputs, together, base)
print(f"R2 + R4 + R6: ${r.profit:,.2f} (gives up ${cost:,.2f}; one at a time "
      f"${sum(row[4] for row in rows if row[0] in ('R2', 'R4', 'R6')):,.2f})")
used = apply_overrides(inputs, together)
broken, _ = check_plan(used, r.plan)
print(f"independent check of the overridden plan: {len(broken)} constraints broken")
folder = tempfile.mkdtemp()
rid_model = record(base, inputs, Settings(), folder)
rid_final = record(r, used, Settings(), folder, overrides=together)
last = json.loads((Path(folder) / "runs.jsonl").read_text().splitlines()[-1])
print(f"model run {rid_model}, released run {rid_final}; recorded overrides:")
for o in last["overrides"]:
    print(f"  {o['request_id']} {o['requested_by']}: {o['sku']} {o['kind']} {o['batches']:g} - {o['reason']}")
```

```text
--- approved together
R2 + R4 + R6: $107,916.02 (gives up $435.26; one at a time $490.25)
independent check of the overridden plan: 0 constraints broken
model run c206699294, released run 36c983fa11; recorded overrides:
  R2 Raj Mehta: BK-07 at_least 8 - Italian festival near S12-S14 this weekend
  R4 Marc Dube: BK-SD at_most 25 - starter tank cleaning on Thursday
  R6 Tom Leblanc: BK-12 at_most 10 - slow sales in the last two weeks
```

- **Overrides interact.** Together, R2, R4 and R6 cost $435.26, not the $490.25 you get by adding them. Always price the final set together.
- The released plan passes the independent check (Day 72).
- It gets **its own run id**: its inputs differ from the model's. The record keeps each override with **who asked and why** (`nbplan/runs.py` now takes `overrides`). Six months from now, "why did we bake olive bread that week?" has an answer.

## 78.3 Were the overrides right?

Each week's overrides are a bet against the model. Score them. Day 54's `evaluate_plan` gives a plan's expected profit over the 103 historical forecast-error weeks, with Day 53's top-ups and day-old sales. Compare the model's plan with each overridden plan.

```python
# ---------- 78.3 Were the overrides right? ----------
print("--- audit against 103 historical weeks")
from saa import evaluate_plan
from two_stage import load_two_stage

d, econ, st, demand, weeks = load_two_stage(con)
model_value = evaluate_plan(d, econ, st, demand, weeks, base.plan)
audit = []
r1 = next(o for o in requests if o.request_id == "R1")                # rejected as an override: tried as a forecast change
as_forecast = dataclasses.replace(inputs, cap={**inputs.cap, r1.sku: r1.batches})
for o, inp in [(r1, as_forecast)] + [(o, inputs) for o in approved]:
    _, cost, ro = override_cost(inp, [o], base)
    value = evaluate_plan(d, econ, st, demand, weeks, ro.plan)
    at_forecast = None if o is r1 else round(cost, 2)               # R1's plan can't be costed at the old forecast
    audit.append((o.request_id, o.sku, at_forecast, round(model_value - value, 2),
                  None if at_forecast is None else round(cost - (model_value - value), 2)))
a = pd.DataFrame(audit, columns=["request", "sku", "cost_at_forecast", "cost_over_history", "model_missed"])
print(f"model plan over history: ${model_value:,.2f} a week")
print(a.to_string(index=False))
```

```text
--- audit against 103 historical weeks
model plan over history: $111,778.48 a week
request   sku  cost_at_forecast  cost_over_history  model_missed
     R1 BK-CR               NaN             633.05           NaN
     R2 BK-07            122.51             -75.30        197.81
     R4 BK-SD            179.97             -67.54        247.51
     R5 BK-19            495.77             462.65         33.12
     R6 BK-12            187.77             166.11         21.66
```

Read `cost_over_history`: positive means the override lost money over real demand, negative means it made money.

- **R2 (olive bread) and R4 (less sourdough) cost money at the forecast but make money over history.** These are the same moves Day 53's two-stage plan made: BK-07 +7.79, sourdough −4.16. The planners' instinct matches the stochastic model. The forecast-only model is missing about $200–250 a week of value on each.
- **R1 as a forecast change (croissants to 34) loses $633 a week over history.** Day 51 found 33 batches better than 31 when croissants are planned alone. Here, 34 batches push out other products, and over history that costs more than the extra croissants earn. "We run out on Saturdays" is a real observation, but the answer is in the Saturday stores' allocation, not in a higher chain-wide forecast.
- **R5 and R6 cost about what the model said.** The model wasn't missing anything there.

This is what the loop is for. When one kind of override keeps winning, change the model, not the planners: here, plan with the two-stage model (Day 53) instead of the forecast alone. When one keeps losing, show the requester this table.

## 78.4 The model card and decision rights

A model in use needs a one-page answer to: what is it for, what does it assume, how is it tested, who owns it, and when should you not trust it? `nbplan/governance.py` writes it from the package itself, so it can't drift from the code. The script saves it as `MODEL_CARD.md` in the repository.

```python
"""Day 78: the model card: what the model is for, what it assumes, how it was tested, who owns it and when not to
trust it, written from the package itself so it can't drift from the code."""
from . import __version__
from .contracts import CONTRACTS


def model_card(inputs, result, tests, owners, limits):
    lines = [
        f"# Model card: nbplan {__version__}",
        "",
        "**Decision supported:** the weekly bakery production plan (batches of each product) for Northbound Fresh.",
        f"**Objective:** maximize weekly contribution (transfer price - ingredients) within {len(inputs.resources)} "
        f"capacities, demand forecasts and agreed minimums.",
        f"**Inputs:** {', '.join(sorted(CONTRACTS))} (checked against data contracts before every run); "
        f"this week's fingerprint {inputs.fingerprint()}.",
        f"**This week:** {result.status}, ${result.profit:,.2f}, {sum(v > 1e-6 for v in result.plan.values())} products.",
        f"**Tested by:** {tests}.",
        "",
        "**Owners**",
    ]
    lines += [f"- {role}: {who}" for role, who in owners]
    lines += ["", "**Do not use it when**"]
    lines += [f"- {x}" for x in limits]
    return "\n".join(lines)
```

```text
--- model card
# Model card: nbplan 0.1.0

**Decision supported:** the weekly bakery production plan (batches of each product) for Northbound Fresh.
**Objective:** maximize weekly contribution (transfer price - ingredients) within 4 capacities, demand forecasts and agreed minimums.
**Inputs:** bakery_range, bakery_range_forecast, bakery_range_recipe, bakery_range_resources (checked against data contracts before every run); this week's fingerprint 94f8d590e531.
**This week:** optimal, $108,351.28, 22 products.
**Tested by:** 130 automated tests (known answers, properties, contracts, the service, monitors) run on every push.

**Owners**
- Model owner: Analytics (Northbound Fresh decision science)
- Decision owner: Head of Planning: releases the plan every Sunday
- Data owners: Bakery operations (recipes, capacities), Forecasting (demand)

**Do not use it when**
- a capacity changes during the week (re-plan with the baked batches fixed, Day 76)
- the forecast monitor or CUSUM is in alarm (Day 77): check the forecast first
- a new product has no recipe or forecast rows (the contracts will refuse the data)
- the question is about prices or the product range: the model takes both as given
saved MODEL_CARD.md
```

**Decision rights** are part of governance too: who may change what.

| Change | Who decides | Recorded where |
|---|---|---|
| A product's batches this week (override) | the requester, under the approval rules by cost | run record: `overrides` |
| The forecast | Forecasting, with the planner's evidence | forecast table version (fingerprint) |
| Capacities, recipes | Bakery operations | data tables (contracts) |
| The approval limits, change cost (Day 76) | Head of Planning | `approval_rules`, settings file |
| The model itself | Analytics, after the tests and the release checks (Day 79) | Git history, version, model card |

> **Decision it supports:** when to accept a person's change to a model's plan, who approves it, and whether the model or the person should learn from it.
>
> **Other places the same practices apply:** credit decisions with manual overrides, clinical decision support, pricing tools with manager discretion, any recommendation system with a human sign-off.

### The planted bug

`day78/bug_hand_edit.py` is a colleague's "quick override" button. It types R1's and R2's batches over the model's plan and saves the run. Run it.

```python
edited = dataclasses.replace(result, plan={**result.plan, "BK-CR": 34.0, "BK-07": 8.0})     # the "override"
released = record(edited, inputs, Settings(), folder)

row = json.loads((Path(folder) / "runs.jsonl").read_text().splitlines()[-1])
print(f"model run {model_run}, released run {released}; released record says profit ${row['profit']:,.2f}")
broken, profit = check_plan(inputs, edited.plan)
print(f"independent check of the released plan: profit ${profit:,.2f}; broken:")
for b in broken:
    print("  " + b)
```

<details><summary>What happens, and why</summary>

```text
model run c206699294, released run c206699294; released record says profit $108,351.28
independent check of the released plan: profit $110,917.28; broken:
  BK-CR: 34.00 batches above demand 31.00
  oven_hours: uses 614.10 of 600.00
  packing_hours: uses 306.60 of 300.00
  sourdough_starter: uses 68.00 of 60.00
```

Everything about the released plan is wrong:

- it breaks the croissant forecast, the ovens, packing and starter;
- its record says $108,351.28, but the plan claims $110,917.28 (a profit the bakery can't make);
- it has the **same run id** as the model's plan, so the audit trail can't tell them apart.

A typed number doesn't re-plan anything else. Overrides go in as constraints (`apply_overrides`), the model re-solves, the independent check runs, and the record gets the overrides.

</details>

## Mini-project: an approval budget

Next week the Head of Planning has a budget of $600 a week for overrides. The requests are:

- the four feasible ones from today (R2, R4, R5, R6);
- R7: at least 5 batches of BK-22 for a school fundraiser;
- R8: at least 35 batches of sourdough "on every shelf on Saturday".

Price every combination by re-solving and find the set with the most requests that fits the budget. Does "approve the cheapest first" find it?

<details><summary>Solution</summary>

```python
"""Day 78 mini-project: next week the planning lead receives the four feasible requests that weren't automatic plus
two new ones, and can approve at most $600 a week of overrides in total. Overrides interact: price every combination
by re-solving, then pick the set the lead should approve. Does approving the cheapest ones first find it?

Run from the repository root (after day06/load_tables.py):  python day78/mini_project_approvals.py
"""
import itertools
import sys

import duckdb

sys.path.insert(0, ".")
from nbplan.config import Settings
from nbplan.data import load_inputs
from nbplan.model import solve
from nbplan.overrides import Override, load_overrides, override_cost

con = duckdb.connect("northbound.duckdb", read_only=True)
inputs = load_inputs(con, Settings())
base = solve(inputs)
asks = [o for o in load_overrides(con) if o.request_id in ("R2", "R4", "R5", "R6")]
asks += [Override("R7", "Raj Mehta", "BK-22", "at_least", 5, "school fundraiser order"),
         Override("R8", "Amelie Roy", "BK-SD", "at_least", 35, "keep sourdough on every shelf on Saturday")]
BUDGET = 600
single = {o.request_id: override_cost(inputs, [o], base)[1] for o in asks}
print("one at a time: " + ", ".join(f"{k} ${v:,.2f}" if v is not None else f"{k} infeasible" for k, v in single.items()))

best, cheapest = None, None
for n in range(len(asks), 0, -1):
    for combo in itertools.combinations(asks, n):
        status, cost, _ = override_cost(inputs, list(combo), base)
        if status != "optimal" or cost > BUDGET + 1e-9:
            continue
        key = (n, -cost)
        if best is None or key > best[0]:
            best = (key, [o.request_id for o in combo], cost)
print(f"most requests within ${BUDGET}: {best[1]} costing ${best[2]:,.2f} together "
      f"(sum one at a time ${sum(single[k] for k in best[1]):,.2f})")

picked, total = [], 0.0
for k in sorted((k for k in single if single[k] is not None), key=single.get):
    status, cost, _ = override_cost(inputs, [o for o in asks if o.request_id in picked + [k]], base)
    if status == "optimal" and cost <= BUDGET + 1e-9:
        picked, total = picked + [k], cost
print(f"cheapest first: {picked} costing ${total:,.2f}")
status, cost, _ = override_cost(inputs, [o for o in asks if o.request_id in ("R4", "R8")], base)
print(f"R4 and R8 together: {status}")
```

```text
one at a time: R2 $122.51, R4 $179.97, R5 $495.77, R6 $187.77, R7 $43.92, R8 $0.00
most requests within $600: ['R2', 'R4', 'R6', 'R7'] costing $435.26 together (sum one at a time $534.17)
cheapest first: ['R8', 'R7', 'R2', 'R6'] costing $550.23
R4 and R8 together: infeasible
```

- **Best: R2, R4, R6 and R7 for $435.26** together. R7 costs nothing once R2, R4 and R6 are in, so the set costs less than the requests added one at a time ($534.17).
- **Cheapest first also approves four, but costs $550.23.** It starts with R8, which is free because sourdough is already at 35.
- **R4 and R8 contradict each other**: at most 25 against at least 35. No re-solve can settle that. Two people need to talk, and the system should say so before either is approved.

</details>

## Commit

```bash
git add .
git commit -m "Day 78: overrides, approvals and the model card"
git push
```

### Day 78 checklist

- [ ] Overrides go into the model as constraints and are priced by re-solving.
- [ ] Approval depends on the cost; requests above the forecast go to Forecasting.
- [ ] The final set of overrides is priced and checked together.
- [ ] Every override is recorded with who, why, and its own run id.
- [ ] I audit overrides against real demand and change the model when they keep winning.
- [ ] The model card and decision rights are written down.

---

# DAY 79 — Production Readiness

"It works on my laptop" isn't a release. Before the planning team depends on `nbplan` every Sunday, it must be able to answer five questions:

- Can someone find out what it did? (logs)
- What happens when the solver fails? (fallback)
- Does a new version still give the known answers? (release checks)
- Does it run the same way everywhere? (a container)
- Does the real server work, not just the tests? (a smoke test)

| Time | Block |
|---|---|
| 0:00 – 0:45 | 79.1 Logs a machine can search |
| 0:45 – 1:45 | 79.2 When the solver fails |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 79.3 Release checks; 79.4 a container |
| 2:45 – 3:45 | A planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: a smoke test |
| 4:45 – 5:00 | Commit with Git + checklist |

## 79.1 Logs a machine can search

`nbplan/service.py` now writes **one JSON line per event**:

- every request: an id, the path, the status and the duration;
- every plan: the request id, the inputs fingerprint, the status, the profit and the version.

The request id is the caller's `X-Request-ID` if it sent one (so a planner's complaint can be traced through every system), otherwise a new one. It's sent back in the response.

```python
log = logging.getLogger("nbplan")                    # Day 79: one JSON line per event, for the log search tool


def event(**fields):
    log.info(json.dumps(fields, sort_keys=True))


@app.middleware("http")
async def access_log(request: Request, call_next):
    """Every request gets an id (the caller's X-Request-ID if it sent one), a status and a duration."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.rid = rid
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    event(event="request", request_id=rid, method=request.method, path=request.url.path, status=response.status_code,
          ms=round(1000 * (time.perf_counter() - start), 1))
    return response
```

```python
# ---------- 79.1 Logs a machine can search ----------
print("--- logs")
lines = []
handler = logging.Handler()
handler.emit = lambda rec: lines.append(json.loads(rec.getMessage()))
log = logging.getLogger("nbplan")
log.addHandler(handler)
log.setLevel(logging.INFO)
client = TestClient(app)
client.get("/health", headers={"X-Request-ID": "r-001"})
client.post("/plan", json={}, headers={"X-Request-ID": "r-002"})
client.post("/plan", json={"min_batches": {"BK-SD": 50}}, headers={"X-Request-ID": "r-003"})
client.post("/plan", json={"capacity": {"oven_hour": 700}}, headers={"X-Request-ID": "r-004"})
for e in lines:
    e.pop("ms", None)                                                # durations differ from machine to machine
    print(json.dumps(e, sort_keys=True))
print(f"every event has a request id: {all('request_id' in e for e in lines)}")
```

```text
--- logs
{"event": "request", "method": "GET", "path": "/health", "request_id": "r-001", "status": 200}
{"event": "plan", "inputs": "94f8d590e531", "profit": 108351.28, "request_id": "r-002", "status": "optimal", "version": "0.1.0"}
{"event": "request", "method": "POST", "path": "/plan", "request_id": "r-002", "status": 200}
{"event": "plan", "inputs": "3019f7aded2c", "profit": null, "request_id": "r-003", "status": "infeasible", "version": "0.1.0"}
{"event": "request", "method": "POST", "path": "/plan", "request_id": "r-003", "status": 200}
{"event": "request", "method": "POST", "path": "/plan", "request_id": "r-004", "status": 422}
every event has a request id: True
```

The durations (`ms`) are removed from the printout because they differ on every machine. From these lines you can answer "what did the planner get at 9:14?" (r-002: $108,351.28 from inputs 94f8d590e531) and "why did their app show an error?" (r-004: a 422). Don't log whole plans or anything personal; the run records (Day 71) keep the plans.

## 79.2 When the solver fails

Two different things can go wrong:

- **Infeasible** is an answer: the settings ask for something impossible. Say so (Day 74).
- **A failure** (a crash, a time limit with no plan, a licence error) is not an answer. The bakery still needs a plan by Sunday night.

`nbplan/fallback.py` falls back to the last good plan from the run records, **but checks it against this week's inputs first**. If it no longer fits, it scales the plan down until every capacity holds, and always flags it for a person.

```python
"""Day 79: what the service does when the solver fails. Infeasible is an answer, not a failure. A crash or a time
limit without a plan is a failure: fall back to the last good plan, but only after checking it against this week's
inputs; if it no longer fits, scale it down until it does, and flag it for a person."""
import json
from pathlib import Path

from .check import check_plan
from .model import Result, solve


def last_good(folder):
    path = Path(folder) / "runs.jsonl"
    if not path.exists():
        return None
    good = [r for r in map(json.loads, path.read_text().splitlines()) if r["status"] == "optimal"]
    return good[-1] if good else None


def fit(inputs, plan):
    """The plan within this week's demand caps, scaled down by one factor until every capacity holds."""
    x = {p: min(plan.get(p, 0.0), inputs.cap[p]) for p in inputs.products}
    factor = 1.0
    for r in inputs.resources:
        used = sum(inputs.use.get((r, p), 0.0) * x[p] for p in inputs.products)
        if used > inputs.capacity[r]:
            factor = min(factor, inputs.capacity[r] / used)
    return {p: v * factor for p, v in x.items()}, factor


def plan_or_fallback(inputs, folder, solver=solve, **kw):
    """(result, source, needs_review)."""
    try:
        r = solver(inputs, **kw)
        if r.status in ("optimal", "infeasible"):
            return r, "solver", r.status != "optimal"
        reason = f"solver status {r.status}"
    except Exception as e:                                           # noqa: BLE001 - any failure falls back
        reason = f"solver failed: {type(e).__name__}"
    prev = last_good(folder)
    if prev is None:
        return Result("failed", None, {}, {}, {}, {}), f"{reason}; no earlier plan", True
    broken, profit = check_plan(inputs, prev["plan"])
    if not broken:
        return Result("fallback", profit, prev["plan"], {}, {}, {}), f"{reason}; last good plan {prev['run_id']}", True
    plan, factor = fit(inputs, prev["plan"])
    _, profit = check_plan(inputs, plan)
    return (Result("fallback", profit, plan, {}, {}, {}),
            f"{reason}; last good plan {prev['run_id']} broke {len(broken)} constraints, scaled to {factor:.3f}", True)
```

```python
# ---------- 79.2 When the solver fails ----------
print("--- fallback")
con = duckdb.connect("northbound.duckdb", read_only=True)
inputs = load_inputs(con, Settings())
folder = tempfile.mkdtemp()
record(solve(inputs), inputs, Settings(), folder)                    # last Sunday's plan, released and recorded


def broken_solver(inputs, **kw):
    raise RuntimeError("solver crashed")


for name, s, solver in [("normal week", Settings(), solve),
                        ("normal week, solver crashes", Settings(), broken_solver),
                        ("oven breakdown (420 h), solver crashes", Settings(capacity={"oven_hours": 420}), broken_solver),
                        ("festival minimums", Settings(min_batches={p: 18 for p in ("BK-SD", "BK-03", "BK-07", "BK-08")}), solve)]:
    week = load_inputs(con, s)
    r, source, review = plan_or_fallback(week, folder, solver=solver)
    profit = "-" if r.profit is None else f"${r.profit:,.2f}"
    print(f"{name}: {r.status} {profit} from {source}; needs review: {review}")
best = solve(load_inputs(con, Settings(capacity={"oven_hours": 420})))
print(f"oven breakdown, solved properly: ${best.profit:,.2f}")
```

```text
--- fallback
normal week: optimal $108,351.28 from solver; needs review: False
normal week, solver crashes: fallback $108,351.28 from solver failed: RuntimeError; last good plan c206699294; needs review: True
oven breakdown (420 h), solver crashes: fallback $75,845.90 from solver failed: RuntimeError; last good plan c206699294 broke 1 constraints, scaled to 0.700; needs review: True
festival minimums: infeasible - from solver; needs review: True
oven breakdown, solved properly: $83,120.93
```

- In a normal week, a crash costs nothing: last week's plan still fits and is still optimal.
- In the oven-breakdown week, last week's plan needs 600 oven hours and only 420 exist. Scaled to 70%, it fits and earns $75,845.90, against $83,120.93 for a proper re-solve.
- The fallback is a safe plan, not a good one, which is why it's flagged. Scaling everything can also push a product below an agreed minimum: the reviewer must look.
- The festival minimums are infeasible. That's not a failure, so there's no fallback, but it still needs a person.

## 79.3 Release checks

`nbplan/ready.py` runs the checks a release must pass. It's also a command: `nbplan ready` exits 0 only if all of them pass, so CI or a release script can use it.

```python
"""Day 79: a production-readiness check, run before every release (and by `nbplan ready`). Each check is a
yes/no with a reason; a release needs all of them."""
import random
import time
from pathlib import Path

from .config import ConfigError, Settings, check_settings
from .contracts import validate
from .data import load_inputs
from .model import solve

KNOWN = (108_351.28, "94f8d590e531")                                 # the default week (Days 6 and 71)


def run_checks(con, root=".", budget_s=5.0, scenarios=20, seed=79):
    root = Path(root)
    checks = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    reqs = [l.strip() for l in (root / "requirements.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    loose = [l for l in reqs if "==" not in l]
    add("dependencies pinned", not loose, f"{len(reqs)} pinned" if not loose else f"not pinned: {', '.join(loose)}")
    problems = validate(con)
    add("input data meets its contracts", not problems, "0 problems" if not problems else f"{len(problems)} problems")
    try:
        check_settings({"oven_hours": 700})
        add("unknown settings refused", False, "a misspelt setting was accepted")
    except ConfigError:
        add("unknown settings refused", True, "ConfigError")
    inputs = load_inputs(con, Settings())
    r = solve(inputs)
    add("known answer", (round(r.profit, 2), inputs.fingerprint()) == KNOWN, f"${r.profit:,.2f}, inputs {inputs.fingerprint()}")
    rng, slow, failed = random.Random(seed), 0, 0
    for _ in range(scenarios):
        s = Settings(demand_multiplier=rng.uniform(0.7, 1.4),
                     capacity={k: v * rng.uniform(0.6, 1.3) for k, v in inputs.capacity.items()})
        t = time.perf_counter()
        status = solve(load_inputs(con, s), time_limit=budget_s).status
        slow += time.perf_counter() - t > budget_s
        failed += status not in ("optimal", "infeasible")
    add("solve time budget", slow == 0 and failed == 0, f"{scenarios - slow}/{scenarios} scenarios under {budget_s:g} s, "
        f"{failed} solver failures")
    docker = root / "Dockerfile"
    text = docker.read_text() if docker.exists() else ""
    add("container image defined", "HEALTHCHECK" in text and "uvicorn" in text,
        "Dockerfile with HEALTHCHECK" if text else "no Dockerfile")
    ci = root / ".github" / "workflows" / "tests.yml"
    add("tests run on every push", ci.exists() and "pytest" in ci.read_text(), ".github/workflows/tests.yml")
    add("model card", (root / "MODEL_CARD.md").exists(), "MODEL_CARD.md" if (root / "MODEL_CARD.md").exists() else "missing")
    return checks
```

```python
# ---------- 79.3 The release checks ----------
print("--- ready")
for name, ok, detail in run_checks(con):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
```

```text
--- ready
PASS  dependencies pinned: 15 pinned
PASS  input data meets its contracts: 0 problems
PASS  unknown settings refused: ConfigError
PASS  known answer: $108,351.28, inputs 94f8d590e531
PASS  solve time budget: 20/20 scenarios under 5 s, 0 solver failures
PASS  container image defined: Dockerfile with HEALTHCHECK
PASS  tests run on every push: .github/workflows/tests.yml
PASS  model card: MODEL_CARD.md
```

The solve-time check runs 20 random scenarios with demand at 0.7–1.4 times the forecast and capacities at 0.6–1.3 times today's, and fails if any takes longer than the budget or the solver fails. The detail line may read differently on a slow machine; the budget is set with room to spare.

## 79.4 A container

A container image holds Python, the pinned packages, the code and the database, so the service runs the same on a laptop, in Codespaces or on a server. The `HEALTHCHECK` lets the platform restart it if `/health` stops answering.

```dockerfile
# Day 79: the planner as a container. Build and run (from the repository root):
#   docker build -t nbplan .
#   docker run -p 8000:8000 nbplan          -> http://localhost:8000/docs
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e . \
 && python make_data.py && python day01/ladder.py && python day06/load_tables.py \
 && nbplan validate

ENV NBPLAN_DATABASE=/app/northbound.duckdb
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "nbplan.service:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run it in Codespaces, where Docker is installed:

```bash
docker build -t nbplan .
docker run -p 8000:8000 nbplan
```

Then open the forwarded port 8000 at `/docs`. `.dockerignore` keeps your local database, data and virtual environment out of the image: the image builds its own data, exactly as CI does.

*The course build couldn't pull the base image, so the image itself wasn't built here. Each step it runs was: the clean install, the data build, `nbplan validate`, and the `uvicorn` command, which the smoke test starts.*

> **Decision it supports:** whether a model is ready to be depended on, and what the business gets when something breaks.
>
> **Other places the same practices apply:** any model behind an API: pricing, scheduling, routing, credit scoring, forecasting services.

### The planted bug

`day79/bug_unchecked_fallback.py` is a colleague's fallback: "if the solver fails, serve last week's plan, it was optimal". On Wednesday an oven breaks down (420 oven hours left) and the solver crashes.

```python
def plan_with_fallback(inputs):
    try:
        raise RuntimeError("solver crashed")                         # Wednesday
    except RuntimeError:
        last = json.loads((Path(folder) / "runs.jsonl").read_text().splitlines()[-1])
        return last["plan"], last["profit"]


week = load_inputs(con, Settings(capacity={"oven_hours": 420}))
plan, profit = plan_with_fallback(week)
print(f"served: last week's plan, 'profit ${profit:,.2f}'")
broken, real = check_plan(week, plan)
print(f"checked against this week's inputs: {len(broken)} broken: {'; '.join(broken)}")
```

<details><summary>What happens, and why</summary>

```text
served: last week's plan, 'profit $108,351.28'
checked against this week's inputs: 1 broken: oven_hours: uses 600.00 of 420.00
```

The bakery gets a plan that needs 600 oven hours with 420 available, and a profit it can't make. "Optimal last week" says nothing about this week. Check a fallback against the current inputs (`check_plan`), make it fit (`fit`), and flag it for review, as `nbplan.fallback` does.

</details>

## Mini-project: a smoke test

The tests use `TestClient`, which never starts a real server. Write a release smoke test that:

- starts the real server with the same `uvicorn` command the container runs, on a spare port;
- waits until `/health` answers;
- checks a plan, a bad request, why-not and the request-id header;
- stops the server and prints whether the release can go.

Use only the standard library for the HTTP calls.

<details><summary>Solution</summary>

```python
"""Day 79 mini-project: a smoke test for a release. Start the real server (the same command the container runs, on
a spare port), wait until /health answers, check a plan, a bad request and the request-id header, then stop it.

Run from the repository root (after day06/load_tables.py):  python day79/mini_project_smoke_test.py
"""
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

PORT = 8079
URL = f"http://127.0.0.1:{PORT}"


def call(path, body=None, headers=None):
    req = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)


server = subprocess.Popen([sys.executable, "-m", "uvicorn", "nbplan.service:app", "--host", "127.0.0.1", "--port", str(PORT)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
checks = []
try:
    for _ in range(60):                                              # wait up to 30 s for the server to start
        try:
            status, body, _ = call("/health")
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.5)
    else:
        raise SystemExit("FAIL  the server did not start")
    checks.append(("server starts and is healthy", status == 200 and body["status"] == "ok"))
    status, body, headers = call("/plan", {}, {"X-Request-ID": "smoke-1"})
    checks.append(("default plan is the known answer", status == 200 and body["profit"] == 108351.28))
    checks.append(("request id comes back", headers.get("x-request-id") == "smoke-1"))
    status, body, _ = call("/plan", {"capacity": {"oven_hour": 700}})
    checks.append(("a misspelt resource is refused", status == 422))
    status, body, _ = call("/why-not", {"sku": "BK-08", "extra": 5})
    checks.append(("why-not answers", status == 200 and body["weekly_cost"] == 257.56))
finally:
    server.terminate()
    server.wait(timeout=30)
for name, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
print(f"release {'can go' if all(ok for _, ok in checks) else 'is blocked'}: {sum(ok for _, ok in checks)}/{len(checks)} checks")
```

```text
PASS  server starts and is healthy
PASS  default plan is the known answer
PASS  request id comes back
PASS  a misspelt resource is refused
PASS  why-not answers
release can go: 5/5 checks
```

Run it after every deployment, against the real address instead of 127.0.0.1. It's the cheapest way to find a missing environment variable or database file, which no unit test can.

</details>

## Commit

```bash
git add .
git commit -m "Day 79: logs, fallback, release checks, container, smoke test"
git push
```

### Day 79 checklist

- [ ] Every request and plan writes a searchable log line with a request id.
- [ ] Solver failures fall back to a checked plan that's flagged for review; infeasible is an answer, not a failure.
- [ ] `nbplan ready` passes before every release: pinned packages, contracts, strict config, known answer, time budget.
- [ ] The service runs from a container with a health check.
- [ ] A smoke test hits the real server after every deployment.

---

# DAY 80 — Final Capstone: The Decision Simulation

Sunday night, 29 November 2026. The Montreal DC floods. The inspection report will say in a week how long it stays closed:

- 2 weeks (probability 0.30);
- 3 weeks (0.45);
- 4 weeks (0.25).

In its first week back it runs at half capacity. These are the four busiest weeks of the year, Day 18's December forecast. On the same day, the board doubles the internal carbon price from $650 to $1,300 a tonne.

You have until Monday noon to recommend what to do, defend it to the executive team, and leave a plan the system can run. Everything you need is already built:

- the four-week distribution model (Day 18);
- Day 23's Laval site and its lanes;
- decision analysis and the value of information (Days 61–63);
- re-planning with what's already happened fixed (Day 76);
- the delivery model and carbon price (Days 66–67);
- the memo checker (Day 69).

| Time | Block |
|---|---|
| 0:00 – 0:45 | 80.1 What the flood costs |
| 0:45 – 1:45 | 80.2 Decide now or wait; 80.3 stress the decision |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 80.4 The carbon price doubles |
| 2:45 – 3:45 | 80.5 The memo; your defence |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Final mini-project: Monday of week 2 |
| 4:45 – 5:00 | Commit, tag v8.0, and the course checklist |

The recovery options (course assumptions, in `recovery_options.csv`):

| Option | Extra pallets a week | Cost | Available |
|---|---|---|---|
| Temporary warehouse in Laval (a cross-dock: no storage) | 900 in and out, 200 chilled | $150,000 set-up + $18,000 a week | one week after it's ordered |
| Weekend shifts at Mississauga | 350 in and out, 80 chilled | $12,000 a week | at once |
| Weekend shifts at Ottawa | 150 in and out, 40 chilled | $6,000 a week | at once |

A pallet a store needs and doesn't get costs $120 in lost margin (Day 18).

## 80.1 What the flood costs

`day80/flood.py` wraps Day 18's model:

- the flooded DC loses its capacity for the closed weeks and gets half in its first week back;
- its stock is lost;
- options add capacity from the week they start;
- Laval is a new "DC" with Day 23's NEW-LAV lane costs.

`run_flood` can also cancel an option after a given week, or fix week 1 to what already happened, for the final mini-project.

```python
"""Day 80: the DC-MTL flood. The Day 18 four-week distribution model with the flooded DC's capacity removed for the
weeks it is closed (and half capacity in its first week back), plus recovery options: weekend shifts at the other
DCs and a temporary warehouse in Laval (Day 23's NEW-LAV lanes, a cross-dock: no storage).

Import it:  from flood import load_flood, run_flood
"""
import itertools

import pandas as pd
import pyomo.environ as pyo

from bakery import solve
from flows import build_weeks, load_weeks

TEMP = "TMP-LAV"


def load_flood(con):
    d = load_weeks(con)
    s = dict(con.sql("SELECT * FROM capstone_settings").fetchall())
    opts = con.sql("SELECT * FROM recovery_options ORDER BY option").df().set_index("option")
    for st, c in con.sql("SELECT store_id, cost_per_pallet FROM site_store_lanes WHERE site_id = 'NEW-LAV'").fetchall():
        d["lane"]["ambient", TEMP, st] = c
        d["lane"]["chilled", TEMP, st] = round(c * d["costs"]["chilled_lane_factor"], 4)
    zero = pd.DataFrame([{"dc_id": TEMP, "week": t, **{c: 0 for c in d["dc_weeks"].columns}} for t in d["weeks"]])
    d["dc_weeks"] = pd.concat([d["dc_weeks"], zero.set_index(["dc_id", "week"])]).sort_index()
    d["dcs"] = d["dcs"] + [TEMP]
    return {"d": d, "flood_dc": s["flood_dc"], "back": float(s["first_week_back_capacity"]),
            "report_week": int(s["inspection_report_week"]), "options": opts,
            "scenarios": dict(con.sql("SELECT * FROM flood_scenarios ORDER BY 1").fetchall())}


def available_from(f, option, decided_week):
    """The first week an option works: its own lead time counted from the week it is decided."""
    return decided_week + int(f["options"].first_week[option]) - 1


def run_flood(f, weeks_closed, bought=(), decided_week=1, lost_sale=None, last_week=None, fixed=None, keep=False):
    """Four weeks with the flood lasting weeks_closed weeks and the options bought in decided_week (or a dict
    {option: week decided}). Returns a dict of costs (option costs are paid from the week an option starts to the
    end of week 4, or to last_week[option] if it is cancelled). fixed: values of week-1 variables that already happened
    (from an earlier run with keep=True, which returns them as 'week1')."""
    when = dict(bought) if isinstance(bought, dict) else {o: decided_week for o in bought}
    d = dict(f["d"])
    w = d["dc_weeks"].copy()
    cols = ["receiving_pallets", "outbound_pallets", "chilled_zone_pallets", "ambient_storage_pallets"]
    w[cols] = w[cols].astype(float)
    for t in d["weeks"]:
        factor = 1.0 if not weeks_closed else 0.0 if t <= weeks_closed else f["back"] if t == weeks_closed + 1 else 1.0
        w.loc[(f["flood_dc"], t), cols] *= factor
    if weeks_closed:
        w.loc[(f["flood_dc"], 1), "opening_ambient_stock"] = 0             # the stock in the warehouse is lost
    option_cost = 0.0
    for o, week in when.items():
        r = f["options"].loc[o]
        start = available_from(f, o, week)
        stop = (last_week or {}).get(o, max(d["weeks"]))
        for t in d["weeks"]:
            if start <= t <= stop:
                w.loc[(r.dc_id, t), "receiving_pallets"] += r.extra_receiving
                w.loc[(r.dc_id, t), "outbound_pallets"] += r.extra_outbound
                w.loc[(r.dc_id, t), "chilled_zone_pallets"] += r.extra_chilled
                option_cost += r.weekly_cost
        option_cost += r.setup_cost if start <= max(d["weeks"]) else 0
    d["dc_weeks"] = w
    if lost_sale is not None:
        d["costs"] = {**d["costs"], "lost_sale_per_pallet": lost_sale}
    m = build_weeks(d)
    for (name, idx), value in (fixed or {}).items():                    # what was done in week 1 can't be undone
        getattr(m, name)[idx].fix(value)
    solve(m)
    lost = sum(pyo.value(m.short[k, s, t]) for k in m.K for s in m.S for t in m.T)
    ops = pyo.value(m.transport + m.inbound + m.holding)
    out = {"operations": ops, "lost_pallets": lost, "lost_sales": pyo.value(m.lost), "options": option_cost,
           "total": pyo.value(m.cost) + option_cost}
    if keep:
        out["week1"] = {(v.name, i): pyo.value(v[i]) for v in (m.ship, m.receive, m.stock, m.short) for i in v
                        if i[-1] == 1}
    return out


def combos(f):
    names = list(f["options"].index)
    return [c for n in range(len(names) + 1) for c in itertools.combinations(names, n)]
```

```python
# ---------- 80.1 What the flood costs ----------
print("--- the flood")
normal = run_flood(f, 0)
print(f"four weeks without a flood: ${normal['total']:,.2f} ({normal['lost_pallets']:,.0f} pallets short)")
for wc in P:
    r = run_flood(f, wc)
    print(f"closed {wc} weeks (p = {P[wc]:.2f}), nothing done: ${r['total']:,.2f}, {r['lost_pallets']:,.0f} pallets short "
          f"(+${r['total'] - normal['total']:,.0f})")
```

```text
--- the flood
four weeks without a flood: $256,444.53 (145 pallets short)
closed 2 weeks (p = 0.30), nothing done: $639,481.80, 3,693 pallets short (+$383,037)
closed 3 weeks (p = 0.45), nothing done: $796,088.35, 5,193 pallets short (+$539,644)
closed 4 weeks (p = 0.25), nothing done: $880,190.34, 5,943 pallets short (+$623,746)
```

With no flood, the model reproduces Day 18's four-week plan: $256,444.53. Doing nothing about the flood costs $383,000 to $624,000 more, almost all of it in lost sales: 3,700 to 5,900 pallets that stores needed and didn't get. Ottawa and Mississauga can't absorb the 1,400-odd pallets a week that Montreal ships.

## 80.2 Decide now, or wait for the report?

There are eight combinations of options. Cost each one under each duration. Then compare deciding now with waiting a week for the report: with the report you know the duration, but Laval can't open before week 3.

```python
# ---------- 80.2 Decide now, or wait for the report? ----------
print("--- decide now")
table = pd.DataFrame({label(c): {wc: run_flood(f, wc, c)["total"] for wc in P} for c in combos(f)}).T
table["expected"] = sum(P[wc] * table[wc] for wc in P)
table["regret_max"] = (table[list(P)] - table[list(P)].min()).max(axis=1)
print(table.sort_values("expected").round(0).to_string())
now = table.expected.idxmin()
print("--- wait for the report")
strategies = {f"decide now: {now}": table.loc[now, "expected"]}
by_strategy = {f"decide now: {now}": [table.loc[now, wc] for wc in P]}
for pre in [(), ("ott_weekend",), ("mis_weekend", "ott_weekend")]:
    costs, picks = [], []
    for wc in P:
        best = min((run_flood(f, wc, {**{o: 1 for o in pre}, **{o: 2 for o in c}})["total"], c)
                   for c in combos(f) if not set(c) & set(pre))
        costs.append(best[0])
        picks.append(f"{wc} wk: {label(best[1])}")
    name = f"{label(pre)} now, the rest after the report"
    strategies[name] = sum(P[wc] * c for wc, c in zip(P, costs))
    by_strategy[name] = costs
    print(f"{name}: expected ${strategies[name]:,.0f} ({'; '.join(picks)})")
perfect = sum(P[wc] * table[wc].min() for wc in P)
print(f"if we knew the duration today: ${perfect:,.0f}; the report is worth at most ${table.expected.min() - perfect:,.0f}")
wait_cost = min(v for k, v in strategies.items() if not k.startswith("decide now")) - table.expected.min()
print(f"waiting for it costs ${wait_cost:,.0f}: Laval would open in week 3 instead of week 2")
```

```text
--- decide now
                                               2         3         4  expected  regret_max
laval_temp + ott_weekend                625317.0  692743.0  757117.0  688609.0     15355.0
laval_temp + mis_weekend + ott_weekend  648491.0  705720.0  761432.0  702479.0     38529.0
laval_temp                              640966.0  718298.0  785983.0  712020.0     31004.0
laval_temp + mis_weekend                659436.0  724458.0  783898.0  719811.0     49474.0
mis_weekend + ott_weekend               612311.0  748290.0  827981.0  727419.0     70864.0
ott_weekend                             609962.0  762331.0  846223.0  737594.0     89106.0
mis_weekend                             635210.0  778740.0  860987.0  756243.0    103870.0
nothing                                 639482.0  796088.0  880190.0  770132.0    123073.0
--- wait for the report
nothing now, the rest after the report: expected $737,640 (2 wk: ott_weekend; 3 wk: mis_weekend + ott_weekend; 4 wk: mis_weekend + ott_weekend)
ott_weekend now, the rest after the report: expected $729,625 (2 wk: nothing; 3 wk: mis_weekend; 4 wk: mis_weekend)
mis_weekend + ott_weekend now, the rest after the report: expected $727,419 (2 wk: nothing; 3 wk: nothing; 4 wk: nothing)
if we knew the duration today: $684,002; the report is worth at most $4,606
waiting for it costs $38,810: Laval would open in week 3 instead of week 2
```

![Expected cost of deciding now against waiting for the report](figures/day80_decision.png)

**Laval + Ottawa weekends, ordered today**, has the lowest expected cost ($688,609) and the smallest worst regret ($15,355). It isn't the best choice if the flood lasts only 2 weeks: then Ottawa weekends alone would be cheaper. But it's never far from the best.

**Waiting is the mistake an analyst is most tempted to make.** Perfect knowledge of the duration is worth at most $4,606 (Day 63's EVPI). Waiting for it costs $38,810, because every waiting strategy loses Laval's week 2. Information is worth waiting for only when it can change the decision **and** the delay is cheap. Here, neither is true.

## 80.3 Stress the decision

The two judgment calls in this analysis are the value of a lost pallet and the duration probabilities. Move both.

```python
# ---------- 80.3 Stress the decision ----------
print("--- stress")
rows = []
for lost in (80, 120, 160):
    t = {label(c): sum(P[wc] * run_flood(f, wc, c, lost_sale=lost)["total"] for wc in P) for c in combos(f)}
    best = min(t, key=t.get)
    rows.append((f"lost sale ${lost}/pallet", best, round(t[best]), round(t[now] - t[best])))
for p2 in (0.5, 0.7, 0.9):
    q = {2: p2, 3: (1 - p2) * 0.45 / 0.70, 4: (1 - p2) * 0.25 / 0.70}
    t = {c: sum(q[wc] * table.loc[c, wc] for wc in P) for c in table.index}
    best = min(t, key=t.get)
    rows.append((f"P(2 weeks) = {p2:.1f}", best, round(t[best]), round(t[now] - t[best])))
lo, hi = 0.3, 1.0
for _ in range(40):                                                  # where 'ott_weekend' alone starts to win
    mid = (lo + hi) / 2
    q = {2: mid, 3: (1 - mid) * 0.45 / 0.70, 4: (1 - mid) * 0.25 / 0.70}
    better = sum(q[wc] * table.loc["ott_weekend", wc] for wc in P) < sum(q[wc] * table.loc[now, wc] for wc in P)
    lo, hi = (lo, mid) if better else (mid, hi)
print(pd.DataFrame(rows, columns=["stress", "best", "expected", "our_choice_costs_more_by"]).to_string(index=False))
print(f"our choice stays best until P(2 weeks) > {hi:.2f} (assessed 0.30)")
```

```text
--- stress
               stress                                   best  expected  our_choice_costs_more_by
 lost sale $80/pallet                            ott_weekend    562274                     40880
lost sale $120/pallet               laval_temp + ott_weekend    688609                         0
lost sale $160/pallet laval_temp + mis_weekend + ott_weekend    743019                     28450
     P(2 weeks) = 0.5               laval_temp + ott_weekend    670525                         0
     P(2 weeks) = 0.7               laval_temp + ott_weekend    652442                         0
     P(2 weeks) = 0.9                            ott_weekend    628195                      6164
our choice stays best until P(2 weeks) > 0.83 (assessed 0.30)
```

- **The duration probabilities hardly matter.** The choice stays best until a 2-week closure is more likely than 0.83, against an assessed 0.30.
- **The value of a lost pallet does matter:**
  - at $80 (customers mostly buy the same items elsewhere in the store), Ottawa weekends alone is best, and our choice costs $40,880 too much;
  - at $160 (customers walk out), Mississauga's weekends should be added.

So the question to ask before Monday noon is not "how long will the flood last?". It is **"Store Operations: what does a missing pallet in December really cost us?"** (the question Day 70 left open).

## 80.4 The carbon price doubles

```python
# ---------- 80.4 The carbon price doubles ----------
print("--- carbon price")
s = dict(con.sql("SELECT * FROM capstone_settings").fetchall())
d = load_delivery(con)
today = evaluate(d, today_plan(d))
carbon = {}


def delivery_plan(d, price):
    m = build_delivery(d)
    m.keep_service = pyo.Constraint(expr=m.service <= today["service"] + 1e-9)
    m.obj = pyo.Objective(expr=m.cost + price * m.co2)
    solve(m)
    return m


for price in (0, float(s["carbon_price_now"]), float(s["carbon_price_new"])):
    m = delivery_plan(d, price)
    carbon[price] = (pyo.value(m.cost), pyo.value(m.co2), mix(m))
    print(f"${price:,.0f}/t: cost ${carbon[price][0]:,.0f} a week, CO2 {carbon[price][1]:.2f} t, cost with carbon "
          f"${carbon[price][0] + price * carbon[price][1]:,.0f}; {carbon[price][2]}")
(c1, e1, _), (c2, e2, _) = carbon[float(s["carbon_price_now"])], carbon[float(s["carbon_price_new"])]
print(f"doubling the price: {e1 - e2:.2f} t a week less for ${c2 - c1:,.0f} a week more (${(c2 - c1) / (e1 - e2):,.0f}/t); "
      f"the carbon charge goes from ${float(s['carbon_price_now']) * e1:,.0f} to ${float(s['carbon_price_new']) * e2:,.0f} a week")
price = float(s["carbon_price_new"])
with_carbon = c2 + price * e2
levers = {"5 points more renewable diesel (HVO) supply":
              {**d, "settings": {**d["settings"], "hvo_max_km_share": float(d["settings"]["hvo_max_km_share"]) + 0.05}},
          "a charger bank (+20 electric trips) at every DC":
              {**d, "chargers": {k: v + 20 for k, v in d["chargers"].items()}},
          "the electric lease at $270 a trip (Day 70)":
              {**d, "vehicles": d["vehicles"].assign(fixed_per_trip=d["vehicles"].fixed_per_trip.where(
                  d["vehicles"].index != "electric", 270))}}
lever = {}
for name, dd in levers.items():                                       # what limits the answer to the price?
    m = delivery_plan(dd, price)
    lever[name] = (e2 - pyo.value(m.co2), with_carbon - pyo.value(m.cost) - price * pyo.value(m.co2))
    print(f"{name}: {e2 - pyo.value(m.co2):.2f} t less, ${with_carbon - pyo.value(m.cost) - price * pyo.value(m.co2):,.0f} "
          f"a week saved with carbon at ${price:,.0f}")
```

```text
--- carbon price
$0/t: cost $57,401 a week, CO2 18.17 t, cost with carbon $57,401; diesel 40; deliveries 3x 2, 4x 6, 5x 20, 6x 12
$650/t: cost $59,549 a week, CO2 11.93 t, cost with carbon $67,301; diesel 24, electric 2, hvo 14; deliveries 3x 3, 4x 4, 5x 19, 6x 14
$1,300/t: cost $59,645 a week, CO2 11.83 t, cost with carbon $75,025; diesel 19, electric 2, hvo 19; deliveries 3x 3, 4x 3, 5x 21, 6x 13
doubling the price: 0.09 t a week less for $96 a week more ($1,011/t); the carbon charge goes from $7,752 to $15,380 a week
5 points more renewable diesel (HVO) supply: 0.72 t less, $725 a week saved with carbon at $1,300
a charger bank (+20 electric trips) at every DC: 0.00 t less, $0 a week saved with carbon at $1,300
the electric lease at $270 a trip (Day 70): 2.15 t less, $692 a week saved with carbon at $1,300
```

- At $650 a tonne the delivery plan already moves to renewable diesel (HVO) and two electric routes: 11.93 t a week, Day 66's answer.
- Doubling the price cuts only another 0.09 t, at $1,011 a tonne. The internal carbon charge doubles to $15,380 a week, but the plan barely changes.
- **The price isn't what limits the answer: supply is.** Five more points of renewable diesel supply would cut 0.72 t and save $725 a week at the new price. A cheaper electric lease (Day 70's $270 a trip) would cut 2.15 t and save $692. More chargers would change nothing.

A carbon price works only where there's something left to switch to. The useful response to the board is two negotiations: a fuel supply contract and a lease.

## 80.5 The memo

```python
# ---------- 80.5 The memo ----------
print("--- memo")
fig, ax = plt.subplots(figsize=(8, 4))
names = list(by_strategy)
for i, n in enumerate(names):
    ax.bar(i, strategies[n] / 1000, color="#3b82f6" if i == 0 else "#9aa4b2", width=0.6)
    ax.scatter([i] * len(P), np.array(by_strategy[n]) / 1000, color="#1f2937", s=14, zorder=3)
ax.set_xticks(range(len(names)), [n.replace(", the rest", ",\nthe rest").replace(": ", ":\n") for n in names], fontsize=7)
ax.set_ylabel("four-week cost ($k)")
ax.set_title("Expected cost (bars) and cost if the flood lasts 2, 3 or 4 weeks (dots)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/day80_decision.png", dpi=150)
nums = {"flood_cost": f"${table.loc['nothing', 'expected'] - normal['total']:,.0f}",
        "choice": "the temporary Laval warehouse and weekend shifts in Ottawa",
        "choice_cost": f"${table.loc[now, 'options'] if 'options' in table else 0:,.0f}",
        "saving": f"${table.loc['nothing', 'expected'] - table.loc[now, 'expected']:,.0f}",
        "worst": f"${table.loc[now, 4] - normal['total']:,.0f}",
        "wait": f"${wait_cost:,.0f}", "report": f"${table.expected.min() - perfect:,.0f}",
        "p2": f"{hi:.2f}", "price_new": f"${float(s['carbon_price_new']):,.0f}", "price_now": f"${float(s['carbon_price_now']):,.0f}",
        "tonnes": f"{e1 - e2:.2f}", "carbon_cost": f"${c2 - c1:,.0f}",
        "charge": f"${float(s['carbon_price_new']) * e2:,.0f}",
        "hvo_t": f"{lever['5 points more renewable diesel (HVO) supply'][0]:.2f}",
        "hvo_gain": f"${lever['5 points more renewable diesel (HVO) supply'][1]:,.0f}",
        "ev_t": f"{lever['the electric lease at $270 a trip (Day 70)'][0]:.2f}",
        "ev_gain": f"${lever['the electric lease at $270 a trip (Day 70)'][1]:,.0f}"}
nums.pop("choice_cost")
template = ("To: the executive team. Re: the Montreal DC flood and the new carbon price.\n"
            "The flood will cost us about {flood_cost} over the next four weeks in lost sales and longer deliveries if we "
            "do nothing. We recommend opening {choice} today. Against doing nothing it saves {saving} in expectation; "
            "if the DC stays closed for all four weeks, the flood still costs us {worst} more than a normal December. "
            "We do not recommend waiting for the inspection report: knowing the duration is worth at most {report}, "
            "and waiting costs {wait}, because the warehouse would open a week later. The decision changes only if the "
            "chance of a two-week closure is above {p2}.\n"
            "The internal carbon price rises from {price_now} to {price_new} a tonne. Our delivery plan answers by cutting "
            "only {tonnes} t of CO2 a week, at {carbon_cost} a week in delivery cost, so the internal carbon charge doubles to "
            "{charge} a week. The price is not what limits us. More renewable diesel supply (5 points) would cut "
            "{hvo_t} t and save {hvo_gain} a week at the new price; a cheaper electric lease would cut {ev_t} t and save "
            "{ev_gain} a week. More chargers would change nothing. We will bring both negotiations to the next meeting.")
memo = fill_memo(template, nums)
print(memo)
print(f"numbers that can't be traced to the results: {check_memo(memo, nums, allowed=('5',)) or 'none'}")
print("saved figures/day80_decision.png")
```

```text
--- memo
To: the executive team. Re: the Montreal DC flood and the new carbon price.
The flood will cost us about $513,687 over the next four weeks in lost sales and longer deliveries if we do nothing. We recommend opening the temporary Laval warehouse and weekend shifts in Ottawa today. Against doing nothing it saves $81,523 in expectation; if the DC stays closed for all four weeks, the flood still costs us $500,673 more than a normal December. We do not recommend waiting for the inspection report: knowing the duration is worth at most $4,606, and waiting costs $38,810, because the warehouse would open a week later. The decision changes only if the chance of a two-week closure is above 0.83.
The internal carbon price rises from $650 to $1,300 a tonne. Our delivery plan answers by cutting only 0.09 t of CO2 a week, at $96 a week in delivery cost, so the internal carbon charge doubles to $15,380 a week. The price is not what limits us. More renewable diesel supply (5 points) would cut 0.72 t and save $725 a week at the new price; a cheaper electric lease would cut 2.15 t and save $692 a week. More chargers would change nothing. We will bring both negotiations to the next meeting.
numbers that can't be traced to the results: none
saved figures/day80_decision.png
```

Every number comes from the run, and the checker confirms it (the "5 points" is a stated assumption). The memo:

- leads with the decision and what it saves;
- gives the bad case (four weeks closed);
- answers the question someone will ask ("why not wait?");
- says what would change its mind;
- ends with actions, not a lecture about carbon prices.

### Your defence

The executive team will ask. Answer each question with a number from today's runs and one sentence.

1. "Why spend $150,000 on a warehouse for a flood that may last two weeks?"
2. "Can't we just wait a week for the report?"
3. "What if you've got the lost-sale value wrong?"
4. "We doubled the carbon price. Why isn't CO2 falling?"

<details><summary>Model answers</summary>

1. Against Ottawa's weekends alone, the warehouse costs $15,355 more if the flood lasts two weeks, and saves $69,588 or $89,106 if it lasts three or four. We're insuring a loss of up to $89,000 for a premium of about $15,000.
2. Waiting costs $38,810, because Laval would open in week 3, not week 2. The most the report could be worth is $4,606.
3. Our choice is best at $120 a pallet. At $80 it costs $40,880 too much; at $160 we should add Mississauga's weekends. We've asked Store Operations for the December figure by Wednesday, and we can add or drop weekend shifts then.
4. Because the plan was already doing everything the price can buy: renewable diesel supply and the electric lease are the limits. A 5-point fuel contract saves $725 a week at the new price, and the $270 lease saves $692.

</details>

## Final mini-project: Monday of week 2

Laval and the Ottawa weekends were ordered last Monday. Week 1 has happened as planned, and its plan assumed the likeliest case, three weeks. Now the report arrives. For each possible report, re-plan weeks 2–4 with week 1 fixed (Day 76). You may:

- close Laval after week 3 (one week's notice), or keep it to week 4;
- end Ottawa's weekends after any week.

What do you do? And what does Laval's $150,000 set-up have to do with it?

<details><summary>Solution</summary>

```python
"""Day 80 final mini-project: Monday of week 2. Laval and the Ottawa weekend shifts were bought last Monday, and week 1
has happened (its shipments are fixed). The inspection report arrives. For each possible report (2, 3 or 4 weeks),
re-plan weeks 2-4: keep both options, or cancel Ottawa's weekends and/or close Laval after week 3 (one week's notice).
What do you do, and what did the set-up cost have to do with it?

Run from the repository root (after day06/load_tables.py):  python day80/mini_project_week2.py
"""
import itertools
import sys

import duckdb

sys.path[:0] = ["day06", "day18", "day80"]
from flood import load_flood, run_flood

con = duckdb.connect("northbound.duckdb", read_only=True)
f = load_flood(con)
bought = ("laval_temp", "ott_weekend")
monday = run_flood(f, 3, bought, keep=True)                          # last Monday's plan assumed the likeliest, 3 weeks
for weeks_closed in f["scenarios"]:
    choices = {"laval_temp": (2, 3, 4), "ott_weekend": (1, 2, 3, 4)}   # last week paid (Laval: at least weeks 2-3)
    best = None
    for laval, ott in itertools.product(*choices.values()):
        r = run_flood(f, weeks_closed, bought, last_week={"laval_temp": laval, "ott_weekend": ott}, fixed=monday["week1"])
        if best is None or r["total"] < best[0]["total"]:
            best = (r, laval, ott)
    keep_all = run_flood(f, weeks_closed, bought, fixed=monday["week1"])
    r, laval, ott = best
    print(f"report says {weeks_closed} weeks: keep everything ${keep_all['total']:,.0f}; best: Laval to week {laval}, "
          f"Ottawa weekends to week {ott}: ${r['total']:,.0f} (saves ${keep_all['total'] - r['total']:,.0f})")
print(f"with a 2-week flood, Laval's set-up (${f['options'].setup_cost['laval_temp']:,.0f}) is spent either way: "
      f"only the weekly costs from now on count")
```

```text
report says 2 weeks: keep everything $625,667; best: Laval to week 3, Ottawa weekends to week 4: $621,470 (saves $4,197)
report says 3 weeks: keep everything $692,743; best: Laval to week 4, Ottawa weekends to week 4: $692,743 (saves $0)
report says 4 weeks: keep everything $761,057; best: Laval to week 4, Ottawa weekends to week 4: $761,057 (saves $0)
with a 2-week flood, Laval's set-up ($150,000) is spent either way: only the weekly costs from now on count
```

- If the report says 2 weeks, close Laval after week 3 and keep Ottawa's weekends. That saves $4,197 against keeping everything.
- If it says 3 or 4 weeks, keep everything.
- **The set-up cost plays no part in Monday's decision.** It was spent last Monday, whatever happens now. Only costs that can still change count. Keeping Laval in week 3 of a two-week flood is right because the DC is at half capacity that week, not because "we already paid for it". Sunk costs are the most common error in re-planning meetings.
- Week 1 was planned for three weeks. Its plan costs a little in the other cases ($625,667 vs $625,317 for two weeks, $761,057 vs $757,117 for four). That's the price of deciding before you know, and it is small.

</details>

## Commit and tag

```bash
git add .
git commit -m "Day 80: final capstone decision simulation"
git tag v8.0-module8
git push && git push --tags
```

### Day 80 checklist

- [ ] I costed every option under every scenario and chose by expected cost and regret.
- [ ] I compared deciding now with waiting, using the value of information and the cost of delay.
- [ ] I stress-tested the judgment calls and named the one that matters.
- [ ] I found what limits a policy's effect (supply, not price).
- [ ] My memo's numbers all trace to the run, and I can defend each in one sentence.
- [ ] I re-planned with what already happened fixed, and ignored sunk costs.

## The course in one page

Eighty days ago the bakery's plan was "same as last week" and a spreadsheet. Since then you have:

- **modelled** the decisions: LP, networks, integer choices, schedules and routes;
- **priced** constraints and options (shadow prices, scenarios, the value of information);
- **faced uncertainty** with simulation, stochastic and robust models, and risk measures;
- **traded off** cost, carbon and service, and written the memos;
- **delivered** the model as a tested, explained, monitored service with people in the loop.

The habits matter more than any one model:

- start from the decision;
- check the answer independently;
- price every number you're asked to accept;
- say what would change your mind;
- leave a record someone else can follow.
