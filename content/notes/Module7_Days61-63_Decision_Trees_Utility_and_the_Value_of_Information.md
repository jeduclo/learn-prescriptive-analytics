# DAY 61 — Decision Trees and Expected Value

Modules 1–6 optimized plans with thousands of variables. Many big decisions look different: **a few options, a few uncertainties, a lot of money**, decided once. Northbound's ten diesel trucks at DC-MTL come off lease in 2027. Should it renew them, go electric, or go half and half? Electric trucks are cheaper to run but more expensive to lease, and nobody knows how far they will go on a Montreal winter morning.

Module 7 is about such decisions and the trade-offs behind them. Today's tool is the **decision tree**: lay out the choices and the uncertainties in the order they happen, value each path, and roll back. Then ask what could change the answer.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 61.1 The decision and its uncertainties |
| 0:45 – 1:45 | 61.2 The tree, rolled back |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 61.3 Sensitivity: what could change the decision? |
| 2:45 – 3:45 | 61.4 A tree whose leaves are optimization models; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: the maintenance contract as a tree |
| 4:45 – 5:00 | Commit with Git + checklist |

### Update the course data

Copy the latest `make_data.py` from the **Course data** page and rebuild (`python make_data.py`, `python day01/ladder.py`, `python day06/load_tables.py`). New for Days 61–63 (all numbers are course assumptions for the fictional Northbound, not market data):

| Table | What it holds |
|---|---|
| `ev_fleet_options`, `ev_settings`, `ev_uncertainties` | the truck decision |
| `ev_pilot_signal` | what a winter pilot would tell us |
| `northbound_finance` | the company's size, for its risk tolerance |
| `ottawa_growth_scenarios` | growth probabilities for Day 30's network |

## 61.1 The decision and its uncertainties

A decision tree has three kinds of node:

- **decision** (a square): you choose;
- **chance** (a circle): nature chooses, with probabilities;
- **end**: a value.

Order matters: a chance node placed **before** a decision means you know its outcome when you decide.

Create `day61/trees.py`, a small library for trees as nested dictionaries:

```python
"""Day 61: decision trees. A tree is nested dictionaries: a decision node chooses among options, a chance node has
branches with probabilities, an end node has a value (here: money, more is better; costs are negative).
Rolling back: a chance node is worth the expectation of its branches (or a certainty equivalent, Day 62),
a decision node the best of its options.

Import it:  from trees import decision, chance, end, rollback, show
"""
import numpy as np


def decision(name, options):
    return {"type": "decision", "name": name, "options": options}


def chance(name, branches):
    """branches: list of (label, probability, node)."""
    total = sum(p for _, p, _ in branches)
    if abs(total - 1) > 1e-9:
        raise ValueError(f"chance node {name}: probabilities sum to {total}")
    return {"type": "chance", "name": name, "branches": branches}


def end(value):
    return {"type": "end", "value": float(value)}


def rollback(node, ce=None):
    """Value every node from the leaves back. ce: a certainty-equivalent function of (values, probabilities)
    for a risk-averse decision maker (Day 62); None = expected value. Stores 'value' (and 'best') in each node."""
    if node["type"] == "end":
        return node["value"]
    if node["type"] == "chance":
        vals = np.array([rollback(n, ce) for _, _, n in node["branches"]])
        probs = np.array([p for _, p, _ in node["branches"]])
        node["value"] = float(probs @ vals) if ce is None else float(ce(vals, probs))
        return node["value"]
    vals = {k: rollback(n, ce) for k, n in node["options"].items()}
    node["best"] = max(vals, key=vals.get)
    node["value"] = vals[node["best"]]
    return node["value"]


def show(node, label="", depth=0, money=lambda v: f"${v:,.0f}", max_depth=9):
    """Print the rolled-back tree: decisions marked with their best option, chances with probabilities."""
    pad = "  " * depth
    if node["type"] == "end":
        print(f"{pad}{label} -> {money(node['value'])}")
        return
    if node["type"] == "decision":
        print(f"{pad}{label}[decide: {node['name']}] {money(node['value'])}, best: {node['best']}")
        for k, n in node["options"].items():
            if depth < max_depth:
                show(n, f"{k}: ", depth + 1, money, max_depth)
    else:
        print(f"{pad}{label}({node['name']}) {money(node['value'])}")
        for k, p, n in node["branches"]:
            if depth < max_depth:
                show(n, f"{k} p={p:.2f}: ", depth + 1, money, max_depth)
```

And `day61/ev.py`, the truck economics: the present value of five years of leases, energy, maintenance, a **winter penalty** (extra rented trucks and drivers when e-trucks fall short of range) and chargers:

```python
"""Day 61: the electric-truck decision for DC-MTL's store-delivery fleet. Present value of five years of costs for
each option, winter range and diesel price.

Import it:  from ev import load_ev, pv_cost, ev_tree
"""
from trees import chance, decision, end


def load_ev(con):
    s = dict(con.sql("SELECT * FROM ev_settings").fetchall())
    options = con.sql("SELECT * FROM ev_fleet_options").df().set_index("option")
    unc = con.sql("SELECT * FROM ev_uncertainties").df()
    levels = {f: list(g[["level", "probability", "value"]].itertuples(index=False, name=None)) for f, g in unc.groupby("factor", sort=False)}
    return s, options, levels


def annuity(s):
    """Present value of $1 a year, paid at the end of each of the lease years."""
    r = s["discount_rate"]
    return sum(1 / (1 + r) ** t for t in range(1, int(s["years"]) + 1))


def pv_cost(s, opt, winter_penalty, diesel_per_km):
    """PV of leases, energy, maintenance and the winter penalty (extra rented trucks and drivers when e-trucks fall
    short of range), plus chargers bought now."""
    km = s["km_per_truck_year"]
    diesel = opt.diesel_trucks * (s["diesel_lease_per_year"] + km * (diesel_per_km + s["diesel_maintenance_per_km"]))
    electric = opt.electric_trucks * (s["electric_lease_per_year"] + km * (s["electricity_per_km"] + s["electric_maintenance_per_km"])
                                      + winter_penalty)
    return opt.chargers_one_time + annuity(s) * (diesel + electric)


def ev_tree(s, options, levels, range_probs=None):
    """Decision: which fleet. Chance: winter range (only matters with e-trucks), then diesel price (only with diesel).
    Values are minus the PV of cost. range_probs: replace the range probabilities (sensitivity, Bayes)."""
    rng = levels["winter_range"] if range_probs is None else [(l, range_probs[l], v) for l, _, v in levels["winter_range"]]
    tree = {}
    for name, opt in options.iterrows():
        def diesel_node(pen):
            if opt.diesel_trucks == 0:
                return end(-pv_cost(s, opt, pen, 0.0))
            return chance("diesel price", [(l, p, end(-pv_cost(s, opt, pen, v))) for l, p, v in levels["diesel_price"]])
        tree[name] = (chance("winter range", [(l, p, diesel_node(v)) for l, p, v in rng])
                      if opt.electric_trucks > 0 else diesel_node(0.0))
    return decision("fleet", tree)
```

Then `day61/run_trees.py`:

```python
# ---------- 61.1 The decision and its uncertainties ----------
print("--- the decision")
print(options.to_string())
for f, ls in levels.items():
    print(f"{f}: " + ", ".join(f"{l} p={p:.2f} ({v:,})" for l, p, v in ls))
```

```text
--- the decision
               electric_trucks  diesel_trucks  chargers_one_time
option
diesel_renew                 0             10                  0
electric_half                5              5             150000
electric_all                10              0             250000
winter_range: good p=0.45 (0.0), fair p=0.35 (7,500.0), poor p=0.20 (22,000.0)
diesel_price: low p=0.30 (0.62), base p=0.50 (0.78), high p=0.20 (0.98)
```

## 61.2 The tree, rolled back

**Rolling back**: from the leaves toward the root, a chance node is worth the probability-weighted average of its branches and a decision node the best of its options.

```python
# ---------- 61.2 The tree, rolled back ----------
print("--- tree")
tree = ev_tree(s, options, levels)
rollback(tree)
show(tree, money=cost)
values = {k: -rollback(n) for k, n in tree["options"].items()}
print("expected PV of cost: " + ", ".join(f"{k} ${v:,.0f}" for k, v in values.items()))
```

```text
--- tree
[decide: fleet] $3,607,813 cost, best: diesel_renew
  diesel_renew: (diesel price) $3,607,813 cost
    low p=0.30:  -> $3,274,022 cost
    base p=0.50:  -> $3,625,381 cost
    high p=0.20:  -> $4,064,579 cost
  electric_half: (winter range) $3,669,274 cost
    good p=0.45: (diesel price) $3,529,031 cost
      low p=0.30:  -> $3,362,135 cost
      base p=0.50:  -> $3,537,814 cost
      high p=0.20:  -> $3,757,414 cost
    fair p=0.35: (diesel price) $3,678,757 cost
      low p=0.30:  -> $3,511,862 cost
      base p=0.50:  -> $3,687,541 cost
      high p=0.20:  -> $3,907,140 cost
    poor p=0.20: (diesel price) $3,968,229 cost
      low p=0.30:  -> $3,801,333 cost
      base p=0.50:  -> $3,977,013 cost
      high p=0.20:  -> $4,196,612 cost
  electric_all: (winter range) $3,680,736 cost
    good p=0.45:  -> $3,400,248 cost
    fair p=0.35:  -> $3,699,701 cost
    poor p=0.20:  -> $4,278,644 cost
expected PV of cost: diesel_renew $3,607,813, electric_half $3,669,274, electric_all $3,680,736
```

- **Renewing diesel** has the lowest expected cost: $3.61 million against $3.68 million for all-electric. The decision is close (2%).
- With a **good** winter range, electric is the cheapest outcome of all ($3.40 million). With a **poor** one, it is the dearest ($4.28 million). Diesel's spread comes from the diesel price instead.
- Half and half is never best on expected value. It is a compromise, and Day 62 asks whether its smaller spread is worth something.

## 61.3 Sensitivity: what could change the decision?

A close decision needs a **sensitivity analysis**. Vary each input over a plausible range, one at a time, and watch the difference between the two leading options. Sorted by swing, the bars form a **tornado diagram**.

```python
# ---------- 61.3 Sensitivity ----------
print("--- sensitivity")


def gap(settings=s, opts=options, lv=levels):
    """Expected PV cost of electric_all minus diesel_renew (negative: electric is cheaper)."""
    t = ev_tree(settings, opts, lv)
    rollback(t)
    return -t["options"]["electric_all"]["value"] + t["options"]["diesel_renew"]["value"]


def with_setting(k, v):
    return gap(settings={**s, k: v})


def with_range(p_poor):
    lv = copy.deepcopy(levels)
    good, fair, poor = lv["winter_range"]
    rest = (1 - p_poor) / (good[1] + fair[1])
    lv["winter_range"] = [(good[0], good[1] * rest, good[2]), (fair[0], fair[1] * rest, fair[2]), (poor[0], p_poor, poor[2])]
    return gap(lv=lv)


def with_diesel_base(v):
    lv = copy.deepcopy(levels)
    lv["diesel_price"] = [(l, p, v if l == "base" else x) for l, p, x in lv["diesel_price"]]
    return gap(lv=lv)


base = gap()
swings = [("electric lease $52k-$64k", with_setting("electric_lease_per_year", 52_000), with_setting("electric_lease_per_year", 64_000)),
          ("P(poor winter range) 0.10-0.30", with_range(0.10), with_range(0.30)),
          ("base diesel $0.70-$0.86/km", with_diesel_base(0.86), with_diesel_base(0.70)),
          ("km a truck a year 45k-65k", with_setting("km_per_truck_year", 65_000), with_setting("km_per_truck_year", 45_000)),
          ("discount rate 5%-11%", with_setting("discount_rate", 0.05), with_setting("discount_rate", 0.11)),
          ("electricity $0.22-$0.32/km", with_setting("electricity_per_km", 0.22), with_setting("electricity_per_km", 0.32))]
print(f"base: electric_all costs ${base:,.0f} more than diesel_renew (expected PV)")
tornado = pd.DataFrame(swings, columns=["input", "favours_electric", "favours_diesel"]).round(0)
tornado["swing"] = tornado.favours_diesel - tornado.favours_electric
tornado = tornado.sort_values("swing")
print(tornado.to_string(index=False))
lo, hi = 40_000, 80_000
for _ in range(60):                                                 # bisection: the lease at which they tie
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if with_setting("electric_lease_per_year", mid) < 0 else (lo, mid)
print(f"break-even electric lease: ${lo:,.0f} a truck a year (today's quote ${s['electric_lease_per_year']:,.0f})")
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for k, r in enumerate(tornado.itertuples()):
    ax.barh(k, r.favours_diesel - base, left=base, color="#9aa4b2")
    ax.barh(k, r.favours_electric - base, left=base, color="#3b82f6")
ax.axvline(base, color="#111827", lw=1)
ax.axvline(0, color="#e5484d", ls="--", lw=1)
ax.set_yticks(range(len(tornado)), tornado.input.str.replace("$", r"\$", regex=False), fontsize=8)
ax.set(xlabel="electric_all minus diesel_renew, expected PV of cost ($)", title="Tornado: what could change the decision?")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/day61_tornado.png", dpi=150)
print("saved figures/day61_tornado.png")
```

```text
--- sensitivity
base: electric_all costs $72,923 more than diesel_renew (expected PV)
                         input  favours_electric  favours_diesel    swing
          discount rate 5%-11%           57988.0         86087.0  28099.0
P(poor winter range) 0.10-0.30           -1815.0        147662.0 149477.0
    base diesel $0.70-$0.86/km          -14916.0        160763.0 175679.0
    electricity $0.22-$0.32/km          -36876.0        182723.0 219599.0
     km a truck a year 45k-65k         -155460.0        301306.0 456766.0
      electric lease $52k-$64k         -166639.0        312486.0 479125.0
break-even electric lease: $56,174 a truck a year (today's quote $58,000)
saved figures/day61_tornado.png
```

![Tornado diagram](figures/day61_tornado.png)

- The **electric lease** and the **kilometres a truck drives** dominate. Within their plausible ranges each could flip the decision. The break-even lease is **$56,174** a truck a year, only $1,800 below today's quote.
- The probability of a poor winter range matters less than it seems: even at 10%, electric only just breaks even.
- The discount rate barely matters.
- **Action**: before deciding, negotiate the lease. A 3% lower quote would make electric the better choice.

## 61.4 A tree whose leaves are optimization models

Trees and optimization combine. Day 30's network model said: expand Ottawa in 2027. But that assumed 3% growth. Suppose growth will be 1%, 3% or 5% a year (probabilities 0.25, 0.5, 0.25), and becomes clear during 2027. The tree's question: **expand now, or wait a year?** Each leaf is the Day 30 model, re-solved with the 2027 choice fixed and the growth known.

```python
# ---------- 61.4 A tree whose leaves are optimization models ----------
print("--- ottawa")
sites, stores, lanes, net_opts, settings = load_redesign(con)
growth = con.sql("SELECT * FROM ottawa_growth_scenarios ORDER BY growth_per_year").fetchall()
solver = pyo.SolverFactory("highs")
leaf = {}
for g, _ in growth:
    for choice in ("expand Ottawa in 2027", "wait a year"):
        m = build_redesign(sites, stores, lanes, net_opts, settings, growth=g)
        m.built["EXP-OTT", 1].fix(1 if choice.startswith("expand") else 0)
        solver.solve(m)
        leaf[choice, g] = pyo.value(m.pv)
        print(f"growth {g:.0%}, {choice}: PV ${leaf[choice, g]:,.2f}")
t = decision("Ottawa in 2027", {c: chance("growth", [(f"{g:.0%}", p, end(-leaf[c, g])) for g, p in growth])
                                for c in ("expand Ottawa in 2027", "wait a year")})
rollback(t)
show(t, money=lambda v: f"${-v:,.0f}")
diff = t["options"]["wait a year"]["value"] - t["options"]["expand Ottawa in 2027"]["value"]
print(f"waiting costs ${-diff:,.0f} more in expected PV" if diff < 0 else f"waiting saves ${diff:,.0f} in expected PV")
```

```text
--- ottawa
growth 1%, expand Ottawa in 2027: PV $21,435,192.94
growth 1%, wait a year: PV $20,996,293.73
growth 3%, expand Ottawa in 2027: PV $22,194,811.21
growth 3%, wait a year: PV $22,288,606.11
growth 5%, expand Ottawa in 2027: PV $23,987,941.78
growth 5%, wait a year: PV $24,081,736.67
[decide: Ottawa in 2027] $22,413,811, best: wait a year
  expand Ottawa in 2027: (growth) $22,453,189
    1% p=0.25:  -> $21,435,193
    3% p=0.50:  -> $22,194,811
    5% p=0.25:  -> $23,987,942
  wait a year: (growth) $22,413,811
    1% p=0.25:  -> $20,996,294
    3% p=0.50:  -> $22,288,606
    5% p=0.25:  -> $24,081,737
waiting saves $39,379 in expected PV
```

- Waiting costs about $94,000 more in PV if growth is 3% or 5%, but saves $439,000 if it is only 1%.
- On expectation, **waiting saves $39,000**. This is the **value of flexibility**: the option to decide after learning is worth something, and a deterministic model at 3% can't see it.
- (Each leaf assumes perfect knowledge of growth after 2027. Day 63 asks what knowing earlier is worth.)

> **Decision it supports:** one-off decisions with a few options and uncertainties: investments, contracts, pilots, entering a market.
>
> **Other decisions you can support the same way:** whether to settle or go to court · launching a product now or after a test · R&D projects with stage gates · bidding for a contract.

### The planted bug

`day61/bug_pruned.py` keeps the tree small: "poor winter range is unlikely, so we left that branch out and rescaled the other two". Run it.

```python
kept = [(l, p) for l, p, _ in levels["winter_range"] if l != "poor"]
total = sum(p for _, p in kept)
tree = ev_tree(s, options, levels, range_probs={**{l: p / total for l, p in kept}, "poor": 0.0})
rollback(tree)
for k, n in tree["options"].items():
    print(f"{k}: expected PV of cost ${-n['value']:,.0f}")
print(f"best: {tree['best']}")
```

<details><summary>What happens, and why</summary>

```text
diesel_renew: expected PV of cost $3,607,813
electric_half: expected PV of cost $3,594,536
electric_all: expected PV of cost $3,531,259
best: electric_all
```

The answer flips: electric now looks $77,000 cheaper. The branch left out had a 20% chance and the worst outcome on the tree. Pruning it removed exactly the risk that made diesel the better choice. Unlikely branches with large consequences are the ones a tree is for. Prune only branches whose values are close to their neighbours, and check that pruning doesn't change the answer.

</details>

## Mini-project: the maintenance contract as a tree

Day 44 simulated the oven maintenance contract ($16,000 a year; the chance of a breakdown falls from 3% to 1.5% a week). Redo it as a decision tree:

- The chance node is the number of breakdowns in a year (0, 1, 2, 3, or 4 or more), binomial over 52 weeks.
- Each breakdown costs Day 44's $25,230.

Does the tree agree with the simulation?

<details><summary>Solution</summary>

```python
"""Day 61 mini-project: Day 44 simulated the oven maintenance contract ($16,000 a year; breakdown chance 3% -> 1.5%
a week). Redo it as a decision tree: the chance node is the number of breakdowns in a year (0, 1, 2, 3, 4 or more,
binomial over 52 weeks), each costing Day 44's $25,230. Does the tree agree with the simulation?

Run from the repository root (after day06/load_tables.py):  python day61/mini_project_maintenance_tree.py
"""
import sys

import duckdb
from scipy import stats

sys.path.insert(0, "day61")
from trees import chance, decision, end, rollback, show

con = duckdb.connect("northbound.duckdb", read_only=True)
loss, contract = 25_230.35, 16_000                                     # Day 44: one breakdown's lost profit
p_now = con.sql("SELECT chance_per_week FROM bakery_disruptions WHERE event = 'oven_breakdown'").fetchall()[0][0]


def breakdowns(p, fee):
    pmf = [stats.binom.pmf(k, 52, p) for k in range(4)]
    branches = [(f"{k}", pmf[k], end(-fee - k * loss)) for k in range(4)]
    rest = 1 - sum(pmf)
    mean_rest = (52 * p - sum(k * pmf[k] for k in range(4))) / rest      # average count given 4 or more
    return chance("breakdowns in a year", branches + [("4+", rest, end(-fee - mean_rest * loss))])


tree = decision("maintenance contract", {"no contract": breakdowns(p_now, 0), "contract": breakdowns(p_now / 2, contract)})
rollback(tree)
show(tree, money=lambda v: f"${abs(v):,.0f} lost")
gain = tree["options"]["contract"]["value"] - tree["options"]["no contract"]["value"]
print(f"the contract gains ${gain:,.0f} a year on average (Day 44's simulation: +$3,920)")
```

```text
[decide: maintenance contract] $35,680 lost, best: contract
  no contract: (breakdowns in a year) $39,359 lost
    0 p=0.21:  -> $0 lost
    1 p=0.33:  -> $25,230 lost
    2 p=0.26:  -> $50,461 lost
    3 p=0.13:  -> $75,691 lost
    4+ p=0.07:  -> $109,967 lost
  contract: (breakdowns in a year) $35,680 lost
    0 p=0.46:  -> $16,000 lost
    1 p=0.36:  -> $41,230 lost
    2 p=0.14:  -> $66,461 lost
    3 p=0.04:  -> $91,691 lost
    4+ p=0.01:  -> $120,969 lost
the contract gains $3,680 a year on average (Day 44's simulation: +$3,920)
```

The tree says the contract gains **$3,680 a year**, close to the simulation's $3,920. The difference is the simulation's sampling noise over 2,000 years: the tree is exact. When the uncertainty has a known distribution with few outcomes, a tree is quicker and more precise than a simulation; simulation is for everything else. The tree also shows the contract cuts the chance of a year with three or more breakdowns from 20% to 5%. Day 62 asks what that is worth to a risk-averse manager.

</details>

## Commit

```bash
git add .
git commit -m "Day 61: decision trees and expected value"
git push
```

### Day 61 checklist

- [ ] I can build a decision tree with decision, chance and end nodes, in the order events happen.
- [ ] I can roll a tree back and read the best strategy.
- [ ] I can draw a tornado diagram and find break-even values.
- [ ] I can build tree leaves from optimization models, and value the flexibility to wait.
- [ ] I don't prune unlikely branches with large consequences.

---

# DAY 62 — Risk Attitude and Utility

Day 61 chose by **expected value**. Would you take a 50-50 gamble that either wins $600,000 or loses $300,000? Its expected value is +$150,000, yet many managers would refuse, and for a small division they may be right.

**Utility theory** makes risk attitude explicit:

- a **utility function** that values outcomes;
- the **certainty equivalent** (CE): the sure amount worth as much as the gamble;
- the **risk premium**: the gap between the expected value and the CE.

The question that matters in a company is **whose** risk attitude should count.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 62.1 Certainty equivalents |
| 0:45 – 1:45 | 62.2 Whose risk tolerance? |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 62.3 The truck decision with utility |
| 2:45 – 3:45 | 62.4 Utility and CVaR; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: a diesel price hedge |
| 4:45 – 5:00 | Commit with Git + checklist |

## 62.1 Certainty equivalents

The **exponential utility** $$u(x) = 1 - e^{-x/R}$$ is the standard choice for business decisions. Its **risk tolerance** $$R$$ (in dollars) has a direct meaning: someone with risk tolerance $$R$$ is about indifferent to a 50-50 gamble of winning $$R$$ or losing $$R/2$$. Its certainty equivalent has a closed form:

$$\text{CE} = -R \,\ln \mathbb{E}\big[e^{-X/R}\big]$$

Create `day62/utility.py`:

```python
"""Day 62: risk attitude. Exponential utility u(x) = 1 - exp(-x / R), with risk tolerance R in dollars, and the
certainty equivalent (CE): the sure amount worth as much as a gamble.

Import it:  from utility import utility, certainty_equivalent, ce_rule, risk_tolerance_rules
"""
import numpy as np


def utility(x, R):
    return 1 - np.exp(-np.asarray(x, dtype=float) / R)


def certainty_equivalent(values, probs, R):
    """CE = -R ln E[exp(-X / R)]. Computed with the best outcome factored out, so large amounts don't overflow."""
    x, p = np.asarray(values, dtype=float), np.asarray(probs, dtype=float)
    top = x.max()
    return float(top - R * np.log(p @ np.exp(-(x - top) / R)))


def ce_rule(R):
    """A chance-node rule for Day 61's rollback: the certainty equivalent instead of the expected value."""
    return lambda values, probs: certainty_equivalent(values, probs, R)


def risk_tolerance_rules(finance):
    """Rules of thumb (Howard, 1988) for a company's risk tolerance, from its accounts."""
    return {"6.4% of revenue": 0.064 * finance["annual_revenue"], "1.24 x net income": 1.24 * finance["net_income"],
            "15.7% of equity": 0.157 * finance["equity"]}
```

Then `day62/run_utility.py`:

```python
# ---------- 62.1 Certainty equivalents ----------
print("--- certainty equivalents")
gamble = ([600_000, -300_000], [0.5, 0.5])
for R in (300_000, 600_000, 2_000_000, 46_000_000):
    ce = certainty_equivalent(*gamble, R)
    print(f"R ${R:>11,}: a 50-50 gamble of +$600,000 / -$300,000 (expected +$150,000) is worth ${ce:>10,.0f} for sure; "
          f"risk premium ${150_000 - ce:,.0f}")
```

```text
--- certainty equivalents
R $    300,000: a 50-50 gamble of +$600,000 / -$300,000 (expected +$150,000) is worth $  -106,632 for sure; risk premium $256,632
R $    600,000: a 50-50 gamble of +$600,000 / -$300,000 (expected +$150,000) is worth $    -4,960 for sure; risk premium $154,960
R $  2,000,000: a 50-50 gamble of +$600,000 / -$300,000 (expected +$150,000) is worth $    99,796 for sure; risk premium $50,204
R $ 46,000,000: a 50-50 gamble of +$600,000 / -$300,000 (expected +$150,000) is worth $   147,799 for sure; risk premium $2,201
```

The same gamble is worth −$107,000 to someone with $$R$$ = $300,000, about zero at $600,000, and almost its full expected value at $46 million. **Risk aversion is about size**: a gamble that is large compared with $$R$$ is worth much less than its average.

## 62.2 Whose risk tolerance?

```python
# ---------- 62.2 Whose risk tolerance? ----------
print("--- risk tolerance")
for k, v in risk_tolerance_rules(finance).items():
    print(f"Northbound, {k}: R about ${v / 1e6:,.1f} million")
vp = finance["vp_logistics_largest_accepted_gamble"]
print(f"VP Logistics: accepts a 50-50 gamble of +${vp:,.0f} / -${vp / 2:,.0f} and no larger -> R about ${vp:,.0f} "
      f"(CE of that gamble at R = {vp:,.0f}: ${certainty_equivalent([vp, -vp / 2], [0.5, 0.5], vp):,.0f})")
```

```text
--- risk tolerance
Northbound, 6.4% of revenue: R about $79.4 million
Northbound, 1.24 x net income: R about $45.9 million
Northbound, 15.7% of equity: R about $45.5 million
VP Logistics: accepts a 50-50 gamble of +$600,000 / -$300,000 and no larger -> R about $600,000 (CE of that gamble at R = 600,000: $-4,960)
```

- Howard's rules of thumb from company accounts put Northbound's risk tolerance at **about $46 million**. For a decision whose outcomes differ by a few hundred thousand dollars, the company should be almost **risk-neutral**: choose by expected value.
- The VP Logistics, asked the lottery question, would take a 50-50 gamble of +$600,000/−$300,000 but nothing larger. **R ≈ $600,000**, about 75 times more cautious than the company.
- That gap is common. Managers are judged on their division's results, not the company's. Many companies lose value because many small decisions are made too cautiously. Name the risk tolerance a decision should use.

## 62.3 The truck decision with utility

Roll Day 61's tree back with **certainty equivalents** at the chance nodes instead of expectations:

```python
# ---------- 62.3 The truck decision with utility ----------
print("--- trucks")
s, options, levels = load_ev(con)
rows, curve = [], []
for R in (46_000_000, 2_000_000, 1_000_000, vp, 300_000, 150_000):
    t = ev_tree(s, options, levels)
    rollback(t, ce=ce_rule(R))
    rows.append((f"${R:,}", *[round(-n["value"]) for n in t["options"].values()], t["best"]))
print(pd.DataFrame(rows, columns=["risk_tolerance", *[f"{k}_ce_cost" for k in options.index], "best"]).to_string(index=False))
Rs = np.geomspace(100_000, 50_000_000, 60)
for R in Rs:
    t = ev_tree(s, options, levels)
    rollback(t, ce=ce_rule(R))
    curve.append([-n["value"] for n in t["options"].values()])
c = np.array(curve)
fig, ax = plt.subplots(figsize=(7, 3.6))
for k, (name, colour) in enumerate(zip(options.index, ["#9aa4b2", "#22a06b", "#3b82f6"])):
    ax.plot(Rs, c[:, k] / 1e6, color=colour, label=name)
ax.axvline(vp, ls="--", color="#e5484d", lw=1)
ax.text(vp * 1.08, c.max() / 1e6 * 0.995, "VP Logistics", color="#e5484d", fontsize=8)
ax.set(xscale="log", xlabel="risk tolerance R ($, log scale)", ylabel="certainty-equivalent cost ($ millions)",
       title="Which fleet? It depends on whose risk attitude counts")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/day62_utility.png", dpi=150)
print("saved figures/day62_utility.png")
```

```text
--- trucks
risk_tolerance  diesel_renew_ce_cost  electric_half_ce_cost  electric_all_ce_cost          best
   $46,000,000               3608632                3669770               3681902  diesel_renew
    $2,000,000               3626943                3680879               3708818  diesel_renew
    $1,000,000               3646567                3692881               3739288  diesel_renew
      $600,000               3673154                3709440               3782564  diesel_renew
      $300,000               3738426                3752859               3891228  diesel_renew
      $150,000               3843011                3839827               4043591 electric_half
saved figures/day62_utility.png
```

![Certainty-equivalent cost against risk tolerance](figures/day62_utility.png)

- At every realistic risk tolerance, **diesel stays best**. Risk aversion makes electric look **worse**, because its spread (good versus poor range) is wider than diesel's (fuel prices).
- Only an extremely cautious decision maker ($$R$$ = $150,000) would choose **half and half**. Splitting the fleet diversifies the two risks: range affects half the trucks and fuel the other half.
- So risk attitude reinforces Day 61's choice here. That's worth knowing: the VP's caution and the company's neutrality agree.

## 62.4 Utility and CVaR

Day 58 traded expected profit against CVaR with a weight. Utility is another way to express the same preference: for each of Day 58's plans, compute the certainty equivalent of a week's profit.

```python
# ---------- 62.4 Utility and CVaR ----------
print("--- utility and cvar")
d, econ, st, demand, weeks = load_two_stage(con)
plans = {}
for w in (0.0, 0.5, 1.0):
    m = add_cvar(build_two_stage(d, econ, st, demand, weeks), 0.05, w)
    solve(m)
    plans[w] = scenario_profits(m)
probs = np.full(len(weeks), 1 / len(weeks))
rows = []
for R in (1_000_000, 100_000, 30_000, 10_000, 5_000):
    ces = {w: certainty_equivalent(p, probs, R) for w, p in plans.items()}
    rows.append((f"${R:,}", *[round(v, 2) for v in ces.values()], max(ces, key=ces.get)))
print("Day 58's plans (CVaR weight 0, 0.5, 1): certainty equivalent of a week's profit")
print(pd.DataFrame(rows, columns=["risk_tolerance", "w_0", "w_0.5", "w_1", "best_weight"]).to_string(index=False))
```

```text
--- utility and cvar
Day 58's plans (CVaR weight 0, 0.5, 1): certainty equivalent of a week's profit
risk_tolerance       w_0     w_0.5       w_1  best_weight
    $1,000,000 112589.93 111961.66 107278.06          0.0
      $100,000 112488.49 111901.82 107270.10          0.0
       $30,000 112118.80 111697.88 107247.16          0.0
       $10,000 109376.50 110418.99 107157.87          0.5
        $5,000 100142.93 105249.66 106918.67          1.0
```

- A decision maker with a risk tolerance of $30,000 or more **for a week's profit** chooses the risk-neutral plan. At $10,000 they choose the CVaR weight of 0.5; at $5,000, the most protective plan.
- A CVaR weight is a risk attitude in disguise. Utility gives it a scale people can be asked about ("would you take this gamble?").
- Note the scale: $10,000 of risk tolerance for one week of one bakery is very cautious. At Northbound's size, the risk-neutral plan is right, unless something like a cash-flow crisis makes bad weeks matter more.

> **Decision it supports:** choosing between options with different risk, using a risk attitude that is stated, assessed and agreed.
>
> **Other decisions you can support the same way:** insurance and hedging · how much of a project to take on with partners · portfolio choices · whether to settle a lawsuit.

### The planted bug

`day62/bug_utility_of_mean.py` computes the VP's certainty equivalent for each fleet option. Run it.

```python
tree = ev_tree(*load_ev(con))
for k, n in tree["options"].items():
    expected = rollback(n)                                       # expected value of the option ...
    u = utility(expected, R)                                     # ... its utility ...
    ce = -R * np.log(1 - u)                                      # ... and back to dollars
    print(f"{k}: certainty-equivalent cost ${-ce:,.0f}")
```

<details><summary>What happens, and why</summary>

```text
diesel_renew: certainty-equivalent cost $3,607,813
electric_half: certainty-equivalent cost $3,669,274
electric_all: certainty-equivalent cost $3,680,736
```

The certainty equivalents equal the expected values, as if the VP were risk-neutral. The code takes the utility **of the expected value**, $$u(\mathbb{E}[X])$$, where it should take the **expected utility**, $$\mathbb{E}[u(X)]$$. Converting $$u(\mathbb{E}[X])$$ back to dollars returns $$\mathbb{E}[X]$$, whatever the utility function. Risk lives in the spread, so the utility must be applied to each outcome before averaging (Jensen's inequality).

</details>

## Mini-project: a diesel price hedge

If the fleet stays diesel, a fuel supplier offers to fix the price at $0.80 a km of energy for five years, above the expected $0.772. Is the hedge worth it?

- Compare the expected PV of cost with the certainty-equivalent cost, for Northbound ($$R$$ ≈ $46 million) and for the VP Logistics ($$R$$ = $600,000).
- Find the highest fixed price each would accept.

<details><summary>Solution</summary>

```python
"""Day 62 mini-project: a fuel supplier offers to fix diesel at $0.80 a km for the five years (energy only), above
the expected $0.772. For the diesel fleet, is the hedge worth it? Compare the expected PV of cost and the
certainty-equivalent cost for Northbound (R about $46 million) and for the VP Logistics (R $600,000).
What premium over the expected price would each pay?

Run from the repository root (after day06/load_tables.py):  python day62/mini_project_hedge.py
"""
import sys

import duckdb
import numpy as np
import pandas as pd

for folder in ("day61", "day62"):
    sys.path.insert(0, folder)
from ev import load_ev, pv_cost
from utility import certainty_equivalent, risk_tolerance_rules

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
s, options, levels = load_ev(con)
finance = dict(con.sql("SELECT * FROM northbound_finance").fetchall())
diesel = options.loc["diesel_renew"]
prices = [(l, p, v) for l, p, v in levels["diesel_price"]]
costs = np.array([pv_cost(s, diesel, 0, v) for _, _, v in prices])
probs = np.array([p for _, p, _ in prices])
expected_price = probs @ np.array([v for _, _, v in prices])
hedged = pv_cost(s, diesel, 0, s["diesel_hedge_per_km"])
rows = []
for who, R in [("Northbound (1.24 x net income)", risk_tolerance_rules(finance)["1.24 x net income"]),
               ("VP Logistics", finance["vp_logistics_largest_accepted_gamble"])]:
    ce_cost = -certainty_equivalent(-costs, probs, R)
    lo, hi = 0.0, 2.0                                            # the fixed price at which the hedge is a tie
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pv_cost(s, diesel, 0, mid) < ce_cost else (lo, mid)
        if hi - lo < 1e-7:
            break
    rows.append((who, f"${R:,.0f}", round(probs @ costs), round(ce_cost), round(hedged), "hedge" if hedged < ce_cost else "no hedge",
                 round(lo, 4)))
print(f"expected diesel price ${expected_price:.3f}/km; the offer ${s['diesel_hedge_per_km']:.2f}/km")
print(pd.DataFrame(rows, columns=["decision_maker", "R", "expected_cost", "ce_cost", "hedged_cost", "choice",
                                  "highest_fixed_price"]).to_string(index=False))
```

```text
expected diesel price $0.772/km; the offer $0.80/km
                decision_maker           R  expected_cost  ce_cost  hedged_cost   choice  highest_fixed_price
Northbound (1.24 x net income) $45,880,000        3607813  3608634      3669301 no hedge               0.7724
                  VP Logistics    $600,000        3607813  3673154      3669301    hedge               0.8018
```

- **Northbound should not hedge**: it would pay $61,000 in expected PV to remove risk it can easily carry. Its highest acceptable price is $0.7724, essentially the expected price.
- **The VP would hedge**, and would accept up to **$0.8018** a km. A risk premium of about $0.03 a km is worth $65,000 in PV to him.
- **The same offer, two right answers**, depending on whose risk tolerance counts. At Northbound's size, the hedge is a bad deal for shareholders and a good deal for the VP's peace of mind. Policy should say which counts. If the division's results matter to it, the company can absorb the fuel risk centrally rather than pay for a hedge.

</details>

## Commit

```bash
git add .
git commit -m "Day 62: risk attitude and utility"
git push
```

### Day 62 checklist

- [ ] I can compute certainty equivalents and risk premiums with exponential utility.
- [ ] I can assess a risk tolerance with a lottery question and with rules of thumb.
- [ ] I can roll back a tree with certainty equivalents and see when risk attitude changes the answer.
- [ ] I can relate a CVaR weight to a risk tolerance.
- [ ] I apply utility to each outcome, not to the average.

---

# DAY 63 — The Value of Information

The truck decision is close, and two uncertainties drive it. Before deciding, Northbound could learn more:

- run a **winter pilot** with two e-trucks;
- buy a **diesel price forecast**.

Information costs money and time, so the question is what it is worth. Today you compute:

- the value of **perfect** information (EVPI): an upper bound;
- the value of **imperfect** information (EVSI), through Bayes' rule. This is what a real pilot gives.

The rule is simple: information is worth something only if it can **change the decision**.

| Time | Block |
|---|---|
| 0:00 – 0:45 | 63.1 Perfect information |
| 0:45 – 1:45 | 63.2 A pilot is imperfect: Bayes' rule |
| 1:45 – 2:00 | Break |
| 2:00 – 2:45 | 63.3 Should we run the pilot? |
| 2:45 – 3:45 | 63.4 Information and risk attitude; a planted bug |
| 3:45 – 4:00 | Break |
| 4:00 – 4:45 | Mini-project: a diesel forecast for sale |
| 4:45 – 5:00 | Commit with Git + checklist |

## 63.1 Perfect information

To value perfect information about an uncertainty, move its chance node **before** the decision and roll back. The improvement over deciding now is the EVPI. Create `day63/voi.py`:

```python
"""Day 63: the value of information. Perfect information (EVPI) about one uncertainty or all, and imperfect
information from a winter pilot, through Bayes' rule.

Import it:  from voi import known, posterior, pilot_tree
"""
import copy

from ev import ev_tree
from trees import chance, decision, rollback


def known(levels, factor, level):
    """The same uncertainties, except that `factor` is known to be `level` (probability 1)."""
    lv = copy.deepcopy(levels)
    lv[factor] = [(l, 1.0 if l == level else 0.0, v) for l, _, v in lv[factor]]
    return lv


def posterior(prior, likelihood, report_fine):
    """Bayes: P(range | report) = P(report | range) P(range) / P(report)."""
    joint = {r: prior[r] * (likelihood[r] if report_fine else 1 - likelihood[r]) for r in prior}
    total = sum(joint.values())
    return {r: v / total for r, v in joint.items()}, total


def pilot_tree(s, options, levels, likelihood, cost, ce=None):
    """Decide: pilot first or decide now. With a pilot: the report (fine / short) arrives, the range probabilities
    are updated, then the fleet is chosen. The pilot's cost comes off every branch."""
    prior = {l: p for l, p, _ in levels["winter_range"]}
    branches = []
    for report in (True, False):
        post, p_report = posterior(prior, likelihood, report)
        sub = ev_tree(s, options, levels, range_probs=post)
        branches.append(("reports fine" if report else "reports short", p_report, shift(sub, -cost)))
    return decision("pilot?", {"decide now": ev_tree(s, options, levels), "pilot first": chance("pilot report", branches)})


def shift(node, amount):
    """Add a fixed amount (a cost: negative) to every end of a tree."""
    if node["type"] == "end":
        return {"type": "end", "value": node["value"] + amount}
    if node["type"] == "chance":
        return {**node, "branches": [(l, p, shift(n, amount)) for l, p, n in node["branches"]]}
    return {**node, "options": {k: shift(n, amount) for k, n in node["options"].items()}}
```

Then `day63/run_voi.py`:

```python
# ---------- 63.1 Perfect information ----------
print("--- perfect information")
rows = []
for factor in ("winter_range", "diesel_price"):
    branches, choices = [], []
    for level, p, _ in levels[factor]:
        t = ev_tree(s, options, known(levels, factor, level))
        branches.append((level, p, t))
        rollback(t)
        choices.append(f"{level}: {t['best']}")
    value = rollback(chance(f"learn {factor}", branches))
    rows.append((factor, round(value - no_info), "; ".join(choices)))
both = []
for lr, pr, _ in levels["winter_range"]:
    for ld, pd_, _ in levels["diesel_price"]:
        both.append((f"{lr}/{ld}", pr * pd_, ev_tree(s, options, known(known(levels, "winter_range", lr), "diesel_price", ld))))
rows.append(("both", round(rollback(chance("learn both", both)) - no_info), ""))
print(f"without information: expected PV of cost ${-no_info:,.0f} ({base['best']})")
print(pd.DataFrame(rows, columns=["learn_first", "evpi", "best_choice_given_it"]).to_string(index=False))
```

```text
--- perfect information
without information: expected PV of cost $3,607,813 (diesel_renew)
 learn_first   evpi                                       best_choice_given_it
winter_range  93404 good: electric_all; fair: diesel_renew; poor: diesel_renew
diesel_price  76769  low: diesel_renew; base: diesel_renew; high: electric_all
        both 135986
```

- **Knowing the winter range** in advance is worth **$93,404**. If the range is good, Northbound would go electric; otherwise it would stay diesel.
- **Knowing the diesel price** is worth $76,769: electric if diesel will be high.
- Knowing both is worth $135,986, less than the sum. Some of the time both pieces of news point the same way.
- These are **ceilings**. No pilot or forecast can be worth more, so information costing more than $93,000 about the range is never worth buying.

## 63.2 A pilot is imperfect: Bayes' rule

Two e-trucks running one winter will report whether the range was **fine** or **short**. That report is imperfect: a fair-range truck might do well in a mild winter. `ev_pilot_signal` gives $$P(\text{reports fine} \mid \text{range})$$: 0.85 if good, 0.45 if fair, 0.15 if poor. **Bayes' rule** turns the report into updated probabilities:

$$P(\text{range} \mid \text{report}) = \frac{P(\text{report} \mid \text{range})\,P(\text{range})}{P(\text{report})}$$

```python
# ---------- 63.2 A pilot is imperfect: Bayes' rule ----------
print("--- bayes")
prior = {l: p for l, p, _ in levels["winter_range"]}
likelihood = dict(con.sql("SELECT * FROM ev_pilot_signal").fetchall())
for report in (True, False):
    post, p = posterior(prior, likelihood, report)
    print(f"pilot {'reports fine ' if report else 'reports short'} (probability {p:.3f}): "
          + ", ".join(f"P({r}) {prior[r]:.2f} -> {post[r]:.3f}" for r in prior))
```

```text
--- bayes
pilot reports fine  (probability 0.570): P(good) 0.45 -> 0.671, P(fair) 0.35 -> 0.276, P(poor) 0.20 -> 0.053
pilot reports short (probability 0.430): P(good) 0.45 -> 0.157, P(fair) 0.35 -> 0.448, P(poor) 0.20 -> 0.395
```

A "fine" report raises the chance of a good range from 45% to 67% and cuts poor to 5%. A "short" report makes good unlikely (16%) and poor much likelier (40%).

## 63.3 Should we run the pilot?

**Preposterior analysis**: put the pilot decision first, then the report (with its probability), then the fleet decision with the **updated** probabilities.

```python
# ---------- 63.3 Should we run the pilot? ----------
print("--- pilot")
free = pilot_tree(s, options, levels, likelihood, cost=0.0)
rollback(free)
evsi = free["options"]["pilot first"]["value"] - free["options"]["decide now"]["value"]
evpi_range = rows[0][1]
print(f"value of the pilot's information (EVSI): ${evsi:,.0f}; perfect information about the range: ${evpi_range:,.0f}; "
      f"pilot efficiency {evsi / evpi_range:.0%}")
t = pilot_tree(s, options, levels, likelihood, cost=s["pilot_cost"])
rollback(t)
show(t, money=cost, max_depth=3)
print(f"the pilot costs ${s['pilot_cost']:,.0f}: {t['best']}; net value ${evsi - s['pilot_cost']:,.0f}")
```

```text
--- pilot
value of the pilot's information (EVSI): $44,796; perfect information about the range: $93,404; pilot efficiency 48%
[decide: pilot?] $3,607,813 cost, best: decide now
  decide now: [decide: fleet] $3,607,813 cost, best: diesel_renew
    diesel_renew: (diesel price) $3,607,813 cost
      low p=0.30:  -> $3,274,022 cost
      base p=0.50:  -> $3,625,381 cost
      high p=0.20:  -> $4,064,579 cost
    electric_half: (winter range) $3,669,274 cost
      good p=0.45: (diesel price) $3,529,031 cost
      fair p=0.35: (diesel price) $3,678,757 cost
      poor p=0.20: (diesel price) $3,968,229 cost
    electric_all: (winter range) $3,680,736 cost
      good p=0.45:  -> $3,400,248 cost
      fair p=0.35:  -> $3,699,701 cost
      poor p=0.20:  -> $4,278,644 cost
  pilot first: (pilot report) $3,623,017 cost
    reports fine p=0.57: [decide: fleet] $3,589,223 cost, best: electric_all
      diesel_renew: (diesel price) $3,667,813 cost
      electric_half: (winter range) $3,653,518 cost
      electric_all: (winter range) $3,589,223 cost
    reports short p=0.43: [decide: fleet] $3,667,813 cost, best: diesel_renew
      diesel_renew: (diesel price) $3,667,813 cost
      electric_half: (winter range) $3,829,696 cost
      electric_all: (winter range) $3,941,579 cost
the pilot costs $60,000: decide now; net value $-15,204
```

- After a "fine" report Northbound goes electric; after a "short" one it stays diesel. The pilot can change the decision, so it has value: **EVSI = $44,796**, 48% of the perfect-information ceiling.
- The pilot costs **$60,000**. On these numbers, **don't run it**: decide now (diesel) and save $15,000 in expected value.
- But the gap is small, and the pilot's design is a choice: more trucks, longer, or a cheaper version. Day 65's mini-project designs the pilot. Day 64 shows how to update with the pilot's actual data, not just a fine/short report.

## 63.4 Information and risk attitude

```python
# ---------- 63.4 Information and risk attitude ----------
print("--- risk attitude")
vp = dict(con.sql("SELECT * FROM northbound_finance").fetchall())["vp_logistics_largest_accepted_gamble"]
for who, ce in [("risk-neutral", None), (f"VP Logistics (R ${vp:,.0f})", ce_rule(vp))]:
    t0 = pilot_tree(s, options, levels, likelihood, cost=0.0)
    rollback(t0, ce=ce)
    v = t0["options"]["pilot first"]["value"] - t0["options"]["decide now"]["value"]
    after = {lbl: n["best"] for lbl, _, n in t0["options"]["pilot first"]["branches"]}
    print(f"{who}: the pilot is worth ${v:,.0f} (certainty-equivalent); after it: {after}")
```

```text
--- risk attitude
risk-neutral: the pilot is worth $44,796 (certainty-equivalent); after it: {'reports fine': 'electric_all', 'reports short': 'diesel_renew'}
VP Logistics (R $600,000): the pilot is worth $50,265 (certainty-equivalent); after it: {'reports fine': 'electric_all', 'reports short': 'diesel_renew'}
```

To the risk-averse VP, the pilot is worth a little more ($50,265): besides improving the decision, it removes risk he dislikes. Still less than $60,000.

> **Decision it supports:** whether to buy information (pilots, tests, surveys, forecasts, consultants) before deciding, and how much to pay.
>
> **Other decisions you can support the same way:** market tests before a launch · medical tests before treatment · geological surveys before drilling · A/B tests before a redesign.

### The planted bug

`day63/bug_no_update.py` values the pilot with a tree. Run it.

```python
p_fine = sum(prior[r] * likelihood[r] for r in prior)
after = lambda: ev_tree(s, options, levels)                           # the fleet decision after the report ...
tree = decision("pilot?", {"decide now": ev_tree(s, options, levels),
                           "pilot first (free)": chance("report", [("fine", p_fine, after()), ("short", 1 - p_fine, after())])})
rollback(tree)
print(f"value of the pilot's information: ${tree['options']['pilot first (free)']['value'] - tree['options']['decide now']['value']:,.0f}")
```

<details><summary>What happens, and why</summary>

```text
value of the pilot's information: $0
```

The pilot is worth $0. After each report, the tree decides with the **prior** probabilities, so it makes the same choice whatever the report says. Information that doesn't update beliefs can't change decisions, and so has no value. After every report branch, use the posterior. If a "valuable" source never changes your mind, check that you're actually updating.

</details>

## Mini-project: a diesel forecast for sale

A consultant sells a five-year diesel price forecast ("low", "base" or "high") for $15,000. With accuracy $$a$$, it names the right level with probability $$a$$, and each wrong level with $$(1-a)/2$$. How accurate must it be to be worth buying? Compute its value for $$a$$ = 0.40 to 1.00.

<details><summary>Solution</summary>

```python
"""Day 63 mini-project: a consultant sells a five-year diesel price forecast ("low", "base" or "high") for $15,000.
Its accuracy is a: it names the right level with probability a, and each wrong level with (1 - a) / 2. How accurate
must it be to be worth buying? Compute its value (EVSI) for a = 0.40 to 1.00.

Run from the repository root (after day06/load_tables.py):  python day63/mini_project_forecast.py
"""
import copy
import sys

import duckdb
import pandas as pd

for folder in ("day61", "day63"):
    sys.path.insert(0, folder)
from ev import ev_tree, load_ev
from trees import chance, rollback

pd.set_option("display.width", 160)
con = duckdb.connect("northbound.duckdb", read_only=True)
s, options, levels = load_ev(con)
price = 15_000
prior = {l: p for l, p, _ in levels["diesel_price"]}
no_info = rollback(ev_tree(s, options, levels))
rows = []
for a in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
    branches = []
    for said in prior:
        joint = {true: prior[true] * (a if said == true else (1 - a) / 2) for true in prior}
        p_said = sum(joint.values())
        lv = copy.deepcopy(levels)
        lv["diesel_price"] = [(l, joint[l] / p_said, v) for l, _, v in levels["diesel_price"]]
        branches.append((said, p_said, ev_tree(s, options, lv)))
    tree = chance("forecast says", branches)
    value = rollback(tree) - no_info
    choices = ", ".join(f"says {said}: {n['best']}" for said, _, n in tree["branches"])
    rows.append((a, round(value), "buy" if value > price else "don't buy", choices))
print(pd.DataFrame(rows, columns=["accuracy", "value_of_forecast", f"at ${price:,}", "decision_after_forecast"])
      .to_string(index=False))
```

```text
 accuracy  value_of_forecast at $15,000                                                  decision_after_forecast
      0.4                  0  don't buy says low: diesel_renew, says base: diesel_renew, says high: diesel_renew
      0.5                961  don't buy says low: diesel_renew, says base: diesel_renew, says high: electric_all
      0.6              16123        buy says low: diesel_renew, says base: diesel_renew, says high: electric_all
      0.7              31284        buy says low: diesel_renew, says base: diesel_renew, says high: electric_all
      0.8              46446        buy says low: diesel_renew, says base: diesel_renew, says high: electric_all
      0.9              61607        buy says low: diesel_renew, says base: diesel_renew, says high: electric_all
      1.0              76769        buy says low: diesel_renew, says base: diesel_renew, says high: electric_all
```

- A forecast with accuracy 0.40 is **worthless**: never acting on it is no worse than acting on it.
- At 0.60 it is worth $16,000 (just above its price). At 1.00 it reaches the EVPI of $76,769.
- The value grows linearly with accuracy here, because the only useful message is "high" (then go electric).
- **Ask the consultant for their track record**: below about 60% accuracy, decline. And note what the forecast is for: it matters only if it can say "high" convincingly.

</details>

## Commit

```bash
git add .
git commit -m "Day 63: the value of information"
git push
```

Then run the tests: `tests/test_decisions.py` checks Days 61–63:

- rolling back a small tree, and refusing probabilities that don't sum to 1;
- certainty equivalents at the limits;
- Bayes on a known case;
- 0 ≤ EVSI ≤ EVPI on the truck decision.

Add `"day61"`, `"day62"` and `"day63"` to the folders in `conftest.py`.

```python
"""Known answers for Days 61-63: rolling back trees, certainty equivalents, Bayes and the value of information."""
import numpy as np
import pytest

from ev import ev_tree, load_ev
from trees import chance, decision, end, rollback
from utility import certainty_equivalent
from voi import known, pilot_tree, posterior


def test_rollback():
    t = decision("d", {"sure": end(50), "gamble": chance("c", [("win", 0.6, end(100)), ("lose", 0.4, end(0))])})
    assert rollback(t) == pytest.approx(60) and t["best"] == "gamble"
    with pytest.raises(ValueError):
        chance("bad", [("a", 0.5, end(1)), ("b", 0.4, end(0))])


def test_certainty_equivalent():
    x, p = [100.0, -50.0], [0.5, 0.5]
    assert certainty_equivalent(x, p, 1e9) == pytest.approx(25.0, abs=1e-3)        # nearly risk-neutral: the mean
    assert certainty_equivalent(x, p, 100.0) < 25.0                                   # risk-averse: less than the mean
    assert certainty_equivalent([7.0], [1.0], 10.0) == pytest.approx(7.0)             # no risk, no premium


def test_bayes():
    post, p = posterior({"a": 0.5, "b": 0.5}, {"a": 0.8, "b": 0.2}, True)
    assert p == pytest.approx(0.5) and post["a"] == pytest.approx(0.8)


def test_information_values(con):
    s, options, levels = load_ev(con)
    no_info = rollback(ev_tree(s, options, levels))
    perfect = rollback(chance("learn", [(l, p, ev_tree(s, options, known(levels, "winter_range", l)))
                                        for l, p, _ in levels["winter_range"]]))
    likelihood = dict(con.sql("SELECT * FROM ev_pilot_signal").fetchall())
    t = pilot_tree(s, options, levels, likelihood, cost=0.0)
    rollback(t)
    sample = t["options"]["pilot first"]["value"]
    assert no_info - 1e-6 <= sample <= perfect + 1e-6                 # 0 <= EVSI <= EVPI
```

```text
93 passed
```

### Day 63 checklist

- [ ] I can compute EVPI by moving a chance node before the decision.
- [ ] I can update probabilities with Bayes' rule from an imperfect signal.
- [ ] I can compute EVSI with a preposterior tree and compare it with the information's cost.
- [ ] I know information is valuable only if it can change the decision.
- [ ] I use posteriors after every signal branch.
