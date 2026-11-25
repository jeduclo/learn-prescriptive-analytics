"""Northbound Fresh course data generator.

Standard library only. Seed 2026: every student gets identical files.
Run from the repository root:  python make_data.py
Writes CSV files into ./data
"""
import csv
import datetime
import math
import os
import random
from datetime import date, timedelta

SEED = 2026
OUT = "data"

DCS = [
    # dc_id, name, city, province, opened
    ("DC-MIS", "Mississauga DC", "Mississauga", "ON", "2009-04-01"),
    ("DC-OTT", "Ottawa DC", "Ottawa", "ON", "2014-09-15"),
    ("DC-MTL", "Montreal DC", "Montreal", "QC", "2011-06-01"),
]

# city, province, number of stores, DC
CITIES = [
    ("Toronto", "ON", 8, "DC-MIS"),
    ("Mississauga", "ON", 3, "DC-MIS"),
    ("Brampton", "ON", 2, "DC-MIS"),
    ("Hamilton", "ON", 2, "DC-MIS"),
    ("Ottawa", "ON", 6, "DC-OTT"),
    ("Kingston", "ON", 2, "DC-OTT"),
    ("Gatineau", "QC", 2, "DC-OTT"),
    ("Montreal", "QC", 8, "DC-MTL"),
    ("Laval", "QC", 2, "DC-MTL"),
    ("Quebec City", "QC", 3, "DC-MTL"),
    ("Sherbrooke", "QC", 2, "DC-MTL"),
]

# Private-label bakery (Module 1). One batch = units_per_batch units.
PRODUCTS = [
    # sku, name, unit, units_per_batch, transfer_price, ingredient_cost
    ("BK-SD", "Sourdough loaf", "loaf", 100, 4.50, 1.50),
    ("BK-WW", "Whole-wheat loaf", "loaf", 100, 3.20, 1.20),
    ("BK-CR", "Croissant 6-pack", "pack", 50, 7.00, 2.00),
]

# What each batch uses of each resource (hours, or starter batches)
RECIPE = {
    "BK-SD": {"oven_hours": 2.0, "baker_hours": 1.0, "sourdough_starter": 1.0},
    "BK-WW": {"oven_hours": 1.0, "baker_hours": 1.0},
    "BK-CR": {"oven_hours": 1.5, "baker_hours": 1.5},
}

RESOURCES = [
    # resource, unit, weekly_capacity, note
    ("oven_hours", "hours", 100, "2 deck ovens x 50 usable hours"),
    ("baker_hours", "hours", 80, "2 bakers x 40 hours"),
    ("sourdough_starter", "batches", 40, "starter culture can feed at most 40 batches a week"),
]

# Weekly chain-level demand at the end of the history, in units
BASE_UNITS = {"BK-SD": 3404, "BK-WW": 4272, "BK-CR": 1498}

FIRST_WEEK = date(2024, 11, 25)   # a Monday
N_WEEKS = 104                     # through the week of 2026-11-16
LATE_STORE = ("S40", date(2025, 9, 1))


def write(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{path}: {len(rows):,} rows")


def season(sku, week_start):
    """Multiplier for the time of year."""
    doy = week_start.timetuple().tm_yday
    m = 1.0 + 0.06 * math.cos(2 * math.pi * (doy - 15) / 365.25)   # a bit more bread in winter
    # Christmas and New Year weeks
    md = (week_start.month, week_start.day)
    if (12, 15) <= md <= (12, 31):
        m *= 1.45 if sku == "BK-CR" else 1.15
    if md >= (12, 22) or md <= (1, 4):
        m *= 1.10 if sku == "BK-CR" else 1.0
    # summer: less bread, cottage season
    if week_start.month in (7, 8):
        m *= 0.90
    return m


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)

    write("dcs.csv", ["dc_id", "dc_name", "city", "province", "opened_date"], DCS)

    stores = []
    n = 0
    for city, prov, k, dc in CITIES:
        for i in range(k):
            n += 1
            sid = f"S{n:02d}"
            size = rng.choice(["Small", "Medium", "Medium", "Large"])
            opened = LATE_STORE[1].isoformat() if sid == LATE_STORE[0] else f"{rng.randint(1998, 2022)}-{rng.randint(1, 12):02d}-01"
            stores.append((sid, f"Northbound {city}{'' if k == 1 else ' ' + str(i + 1)}", city, prov, dc, size, opened))
    write("stores.csv", ["store_id", "store_name", "city", "province", "dc_id", "size_band", "opened_date"], stores)

    write("bakery_products.csv",
          ["sku", "product_name", "unit", "units_per_batch", "transfer_price", "ingredient_cost"],
          PRODUCTS)
    write("bakery_recipe.csv", ["sku", "resource", "use_per_batch"],
          [(sku, r, h) for sku, d in RECIPE.items() for r, h in d.items()])
    write("bakery_resources.csv", ["resource", "unit", "weekly_capacity", "note"], RESOURCES)

    # The plan the bakery planner has used every week since September 2025
    write("bakery_current_plan.csv", ["sku", "batches_per_week"],
          [("BK-SD", 25), ("BK-WW", 35), ("BK-CR", 10)])

    # Weekly store sales of the bakery range
    size_w = {"Small": 0.7, "Medium": 1.0, "Large": 1.4}
    weights = {s[0]: size_w[s[5]] * rng.uniform(0.85, 1.15) for s in stores}
    total_w = sum(weights.values())
    rows = []
    for wk in range(N_WEEKS):
        ws = FIRST_WEEK + timedelta(weeks=wk)
        trend = 1.0 + 0.04 * (wk - (N_WEEKS - 1)) / 52   # +4% a year, 1.0 at the end
        for s in stores:
            sid = s[0]
            if sid == LATE_STORE[0] and ws < LATE_STORE[1]:
                continue
            for sku, *_ in PRODUCTS:
                mean = BASE_UNITS[sku] * weights[sid] / total_w * trend * season(sku, ws)
                units = max(0, round(rng.gauss(mean, 0.12 * mean)))
                rows.append((ws.isoformat(), sid, sku, units))
    write("bakery_sales_weekly.csv", ["week_start", "store_id", "sku", "units_sold"], rows)

    make_days_05_07()
    make_days_08_10()
    make_days_14_17()
    make_days_18_20()
    make_days_21_23()
    make_days_24_27()
    make_days_28_30()
    make_days_31_33()
    make_days_34_37()
    make_days_38_40()
    make_days_41_43()
    make_days_44_47()
    make_days_48_50()
    make_days_51_53()
    make_days_54_57()
    make_days_58_60()
    make_days_61_63()
    make_days_64_67()
    make_days_68_70()
    make_days_78_80()


# ---------------------------------------------------------------------------
# Days 5-7: flour blending, meal kits and the full 30-product bakery range.
# Own random stream, so the files above stay byte-identical.
# ---------------------------------------------------------------------------
FLOUR_SUPPLIERS = [
    # supplier_id, supplier, province, price_per_tonne, protein_pct, ash_pct, moisture_pct, max_tonnes_per_week
    ("FL-PHR", "Prairie Hard Red", "SK", 520, 14.5, 0.62, 13.0, 30),
    ("FL-OSW", "Ontario Soft White", "ON", 410, 10.0, 0.48, 14.2, 30),
    ("FL-QCO", "Quebec Organic", "QC", 640, 13.0, 0.55, 13.5, 8),
    ("FL-DUR", "Imported Durum", "IMP", 590, 15.5, 0.80, 12.5, 30),
]
FLOUR_SPEC = [
    # property, min, max
    ("protein_pct", 12.5, ""),
    ("ash_pct", "", 0.60),
    ("moisture_pct", "", 14.0),
]

# Approximate values per 100 g (raw or as sold). Prices are Northbound's buying prices.
MEALKIT_INGREDIENTS = [
    # ingredient, group, cost_per_100g, kcal, protein_g, fat_g, carbs_g, fibre_g, sodium_mg, max_g
    ("Whole-wheat pasta", "grain", 0.55, 350, 13.0, 2.5, 72.0, 9.0, 5, 800),
    ("Brown rice", "grain", 0.45, 365, 7.5, 2.7, 76.0, 3.5, 5, 800),
    ("Lean ground beef", "protein", 1.65, 215, 19.0, 15.0, 0.0, 0.0, 70, 600),
    ("Chicken thigh", "protein", 1.35, 180, 19.5, 11.0, 0.0, 0.0, 85, 600),
    ("Dry lentils", "protein", 0.60, 350, 25.0, 1.1, 60.0, 11.0, 6, 400),
    ("Canned chickpeas", "protein", 0.40, 140, 7.0, 2.6, 22.0, 6.0, 240, 600),
    ("Cheddar", "dairy", 2.10, 400, 25.0, 33.0, 1.3, 0.0, 650, 200),
    ("Tomato sauce", "vegetable", 0.35, 30, 1.3, 0.2, 7.0, 1.5, 430, 800),
    ("Spinach", "vegetable", 0.90, 23, 2.9, 0.4, 3.6, 2.2, 79, 400),
    ("Broccoli", "vegetable", 0.65, 34, 2.8, 0.4, 6.6, 2.6, 33, 600),
    ("Carrots", "vegetable", 0.25, 41, 0.9, 0.2, 9.6, 2.8, 69, 600),
    ("Onions", "vegetable", 0.22, 40, 1.1, 0.1, 9.3, 1.7, 4, 400),
    ("Olive oil", "fat", 1.20, 884, 0.0, 100.0, 0.0, 0.0, 2, 80),
]
MEALKIT_RULES = [
    # nutrient, min_per_kit, max_per_kit   (one kit = 4 servings)
    ("kcal", 2400, 3200),
    ("protein_g", 120, ""),
    ("fat_g", "", 110),
    ("fibre_g", 40, ""),
    ("sodium_mg", "", 2400),
    ("weight_g", 1800, 2600),
]

RANGE_NAMES = {
    "bread": ["Sourdough loaf", "Whole-wheat loaf", "Rye loaf", "Multigrain loaf", "Baguette 2-pack",
              "Country white loaf", "Olive sourdough", "Seeded sourdough", "Ciabatta 4-pack", "Raisin bread"],
    "viennoiserie": ["Croissant 6-pack", "Pain au chocolat 4-pack", "Almond croissant 4-pack", "Cinnamon roll 4-pack",
                     "Danish 4-pack", "Brioche loaf", "Apple turnover 4-pack", "Butter tart 6-pack"],
    "cake": ["Carrot cake", "Lemon loaf cake", "Black forest cake", "Cheesecake", "Banana bread", "Maple layer cake"],
    "cookie": ["Oatmeal cookies 12-pack", "Chocolate chip cookies 12-pack", "Nanaimo bars 6-pack",
               "Date squares 6-pack", "Shortbread 12-pack", "Brownies 6-pack"],
}
SOURDOUGH = {"Sourdough loaf", "Olive sourdough", "Seeded sourdough", "Rye loaf"}
RANGE_RESOURCES = [
    ("oven_hours", "hours", 600, "12 deck ovens x 50 usable hours"),
    ("baker_hours", "hours", 560, "14 bakers x 40 hours"),
    ("packing_hours", "hours", 300, "2 packing lines x 150 hours"),
    ("sourdough_starter", "batches", 60, "starter culture can feed at most 60 batches a week"),
]


def make_days_05_07():
    rng = random.Random(2004)
    write("flour_suppliers.csv",
          ["supplier_id", "supplier", "province", "price_per_tonne", "protein_pct", "ash_pct", "moisture_pct", "max_tonnes_per_week"],
          FLOUR_SUPPLIERS)
    write("flour_spec.csv", ["property", "min_value", "max_value"], FLOUR_SPEC)
    write("mealkit_ingredients.csv",
          ["ingredient", "food_group", "cost_per_100g", "kcal", "protein_g", "fat_g", "carbs_g", "fibre_g", "sodium_mg", "max_g"],
          MEALKIT_INGREDIENTS)
    write("mealkit_rules.csv", ["nutrient", "min_per_kit", "max_per_kit"], MEALKIT_RULES)

    products, recipe, forecast = [], [], []
    k = 0
    for cat, names in RANGE_NAMES.items():
        for name in names:
            k += 1
            # the three Module 1 products keep their numbers
            fixed = {"Sourdough loaf": ("BK-SD", 100, 4.50, 1.50, 2.0, 1.0, 0.5, 35),
                     "Whole-wheat loaf": ("BK-WW", 100, 3.20, 1.20, 1.0, 1.0, 0.5, 45),
                     "Croissant 6-pack": ("BK-CR", 50, 7.00, 2.00, 1.5, 1.5, 0.6, 31)}.get(name)
            if fixed:
                sku, upb, price, cost, oven, baker, pack, cap = fixed
            else:
                sku = f"BK-{k:02d}"
                if cat == "bread":
                    upb, price, cf = 100, rng.uniform(3.0, 5.5), rng.uniform(0.30, 0.40)
                    oven, baker, pack = rng.uniform(1.0, 2.2), rng.uniform(0.8, 1.3), rng.uniform(0.4, 0.7)
                elif cat == "viennoiserie":
                    upb, price, cf = 50, rng.uniform(6.0, 9.0), rng.uniform(0.28, 0.38)
                    oven, baker, pack = rng.uniform(1.2, 1.8), rng.uniform(1.4, 2.2), rng.uniform(0.6, 0.9)
                elif cat == "cake":
                    upb, price, cf = 20, rng.uniform(14.0, 22.0), rng.uniform(0.35, 0.45)
                    oven, baker, pack = rng.uniform(1.5, 2.5), rng.uniform(2.0, 3.0), rng.uniform(0.8, 1.2)
                else:
                    upb, price, cf = 60, rng.uniform(4.0, 6.5), rng.uniform(0.30, 0.40)
                    oven, baker, pack = rng.uniform(0.8, 1.2), rng.uniform(0.6, 1.0), rng.uniform(0.8, 1.2)
                price = round(price * 20) / 20
                cost = round(price * cf, 2)
                oven, baker, pack = round(oven, 1), round(baker, 1), round(pack, 1)
                cap = rng.randint(6, 30)
            products.append((sku, name, cat, upb, f"{price:.2f}", f"{cost:.2f}"))
            recipe += [(sku, "oven_hours", oven), (sku, "baker_hours", baker), (sku, "packing_hours", pack)]
            if name in SOURDOUGH:
                recipe.append((sku, "sourdough_starter", 1.0))
            forecast.append((sku, cap))
    write("bakery_range.csv", ["sku", "product_name", "category", "units_per_batch", "transfer_price", "ingredient_cost"], products)
    write("bakery_range_recipe.csv", ["sku", "resource", "use_per_batch"], recipe)
    write("bakery_range_resources.csv", ["resource", "unit", "weekly_capacity", "note"], RANGE_RESOURCES)
    write("bakery_range_forecast.csv", ["sku", "demand_cap_batches"], forecast)
    # Marketing's shelf rule and the new cafe contract (Day 7)
    write("bakery_range_minimums.csv", ["sku", "min_batches", "reason"],
          [(p[0], 2, "shelf presence") for p in products]
          + [("BK-WW", 50, "Cafe Laurier contract")]
          + [(sku, 18, "sourdough festival") for sku in ("BK-SD", "BK-03", "BK-07", "BK-08")])



# ---------------------------------------------------------------------------
# Days 8-10: twelve weeks of demand, holiday capacity, storage and overtime.
# Own random stream again.
# ---------------------------------------------------------------------------
PLAN_START = date(2026, 11, 30)   # Monday
PLAN_WEEKS = 12
HOLIDAY_FACTOR = {   # share of a normal week's ovens and bakers
    date(2026, 12, 21): (0.6, "Christmas Day and Boxing Day"),
    date(2026, 12, 28): (0.8, "New Year's Day"),
    date(2027, 2, 15): (0.8, "Family Day (Ontario)"),
}
PEAK = {  # demand multiplier for the two weeks before Christmas, by category
    "bread": 1.15, "viennoiserie": 1.45, "cake": 1.6, "cookie": 1.8,
}
STORAGE = [
    # category, shelf_life_weeks, holding_cost_per_batch_week, freezer_slots_per_batch, note
    ("bread", 0, 0, 0, "baked and sold the same week"),
    ("viennoiserie", 3, 8.0, 1.0, "frozen unbaked, baked off in stores"),
    ("cake", 2, 12.0, 1.5, "frozen finished cakes"),
    ("cookie", 4, 3.0, 0.5, "packed, shelf-stable"),
]


def make_days_08_10():
    rng = random.Random(2008)
    weeks = [PLAN_START + timedelta(weeks=k) for k in range(PLAN_WEEKS)]
    write("plan_weeks.csv", ["week", "week_start", "capacity_factor", "note"],
          [(k + 1, w.isoformat(), HOLIDAY_FACTOR.get(w, (1.0, ""))[0], HOLIDAY_FACTOR.get(w, (1.0, ""))[1])
           for k, w in enumerate(weeks)])

    cats = {}
    for cat, names in RANGE_NAMES.items():
        for name in names:
            cats[name] = cat
    with open(os.path.join(OUT, "bakery_range.csv"), encoding="utf-8") as f:
        range_rows = list(csv.DictReader(f))
    with open(os.path.join(OUT, "bakery_range_forecast.csv"), encoding="utf-8") as f:
        base = {r["sku"]: int(r["demand_cap_batches"]) for r in csv.DictReader(f)}

    rows = []
    for k, w in enumerate(weeks):
        for r in range_rows:
            cat = r["category"]
            m = 1.0
            if date(2026, 12, 14) <= w <= date(2026, 12, 21):
                m = PEAK[cat]
            elif date(2026, 12, 28) <= w <= date(2027, 1, 11):
                m = 0.9            # January: resolutions and smaller baskets
            mean = base[r["sku"]] * m * (1 + 0.002 * k)
            rows.append((k + 1, r["sku"], round(max(0.0, rng.gauss(mean, 0.04 * mean)), 1)))
    write("plan_demand.csv", ["week", "sku", "demand_batches"], rows)
    write("bakery_storage.csv",
          ["category", "shelf_life_weeks", "holding_cost_per_batch_week", "freezer_slots_per_batch", "note"], STORAGE)
    write("plan_settings.csv", ["setting", "value", "note"], [
        ("freezer_slots_module1", 15, "freezer space for the 3-product bakery (Day 8)"),
        ("freezer_slots", 120, "freezer and dry-store space for the full range"),
        ("overtime_baker_hours_max", 80, "per week"),
        ("overtime_baker_cost", 45, "$ per hour"),
        ("overtime_packing_hours_max", 40, "per week (a third shift on one line)"),
        ("overtime_packing_cost", 60, "$ per hour"),
        ("cafe_ww_batches", 50, "Cafe Laurier contract, whole-wheat batches per week, on top of store demand"),
        ("shelf_minimum_batches", 2, "every product sold at least this much each week"),
    ])



# ---------------------------------------------------------------------------
# Days 14-17: the distribution network. Coordinates are approximate city centres.
# ---------------------------------------------------------------------------
PLACES = {
    # place: (latitude, longitude, province)
    "Toronto": (43.65, -79.38, "ON"), "Mississauga": (43.59, -79.64, "ON"), "Brampton": (43.73, -79.76, "ON"),
    "Hamilton": (43.26, -79.87, "ON"), "Peterborough": (44.30, -78.32, "ON"), "Belleville": (44.16, -77.38, "ON"),
    "Kingston": (44.23, -76.49, "ON"), "Cornwall": (45.02, -74.73, "ON"), "Ottawa": (45.42, -75.70, "ON"),
    "Gatineau": (45.48, -75.70, "QC"), "Montreal": (45.50, -73.57, "QC"), "Laval": (45.61, -73.71, "QC"),
    "Saint-Hyacinthe": (45.63, -72.96, "QC"), "Drummondville": (45.88, -72.48, "QC"),
    "Trois-Rivieres": (46.35, -72.55, "QC"), "Sherbrooke": (45.40, -71.89, "QC"), "Quebec City": (46.81, -71.21, "QC"),
}
ROADS = [
    # from, to, km, road, km/h, truck slots a week (eastbound carrier contracts, Day 17)
    ("Mississauga", "Toronto", 28, "401", 50, 60), ("Brampton", "Toronto", 40, "410/401", 50, 30),
    ("Mississauga", "Brampton", 20, "410", 50, 30), ("Hamilton", "Mississauga", 55, "QEW", 80, 30),
    ("Hamilton", "Toronto", 70, "QEW", 80, 30), ("Toronto", "Peterborough", 140, "115", 85, 10),
    ("Toronto", "Belleville", 190, "401", 95, 25), ("Peterborough", "Belleville", 85, "62", 75, 8),
    ("Belleville", "Kingston", 80, "401", 95, 30), ("Peterborough", "Ottawa", 270, "7", 70, 6),
    ("Kingston", "Ottawa", 195, "15/416", 85, 12), ("Kingston", "Cornwall", 170, "401", 95, 15),
    ("Cornwall", "Ottawa", 110, "138/417", 80, 8), ("Cornwall", "Montreal", 125, "401/20", 90, 16),
    ("Ottawa", "Gatineau", 8, "bridge", 40, 20), ("Ottawa", "Montreal", 200, "417/40", 95, 14),
    ("Gatineau", "Montreal", 210, "50", 85, 6), ("Montreal", "Laval", 20, "15", 50, 30),
    ("Montreal", "Saint-Hyacinthe", 60, "20", 90, 20), ("Saint-Hyacinthe", "Drummondville", 55, "20", 95, 18),
    ("Drummondville", "Sherbrooke", 75, "55", 90, 10), ("Montreal", "Sherbrooke", 150, "10", 95, 8),
    ("Drummondville", "Trois-Rivieres", 70, "55", 90, 6), ("Montreal", "Trois-Rivieres", 140, "40", 95, 10),
    ("Laval", "Trois-Rivieres", 130, "40", 95, 8), ("Trois-Rivieres", "Quebec City", 130, "40", 95, 30),
    ("Drummondville", "Quebec City", 150, "20", 90, 20), ("Sherbrooke", "Quebec City", 240, "73", 80, 8),
]
DC_CAPACITY = {"DC-MIS": 1600, "DC-OTT": 700, "DC-MTL": 1500}   # pallets a week
DC_CITY = {"DC-MIS": "Mississauga", "DC-OTT": "Ottawa", "DC-MTL": "Montreal"}
PALLETS = {"Small": 55, "Medium": 80, "Large": 115}
HANDLING_PER_PALLET = 4.0       # $ at a DC
TRUCK_PER_PALLET_KM = 0.11      # $ for a store delivery (26-pallet truck at about $2.85 a km)
LINEHAUL_PER_PALLET_KM = 0.08   # $ for a full truck between plants, DCs and cross-docks


def haversine_km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def road_km(start):
    """Shortest road distance from start to every place (Dijkstra's algorithm, standard library only)."""
    import heapq
    graph = {}
    for a, b, km, *_ in ROADS:
        graph.setdefault(a, []).append((b, km))
        graph.setdefault(b, []).append((a, km))
    dist, todo = {start: 0}, [(0, start)]
    while todo:
        d, u = heapq.heappop(todo)
        if d > dist[u]:
            continue
        for v, km in graph[u]:
            if d + km < dist.get(v, float("inf")):
                dist[v] = d + km
                heapq.heappush(todo, (d + km, v))
    return dist


def make_days_14_17():
    rng = random.Random(2014)
    write("places.csv", ["place", "latitude", "longitude", "province"],
          [(p, la, lo, pr) for p, (la, lo, pr) in PLACES.items()])
    write("road_segments.csv", ["from_place", "to_place", "km", "road", "minutes", "truck_slots_per_week"],
          [(a, b, km, road, round(km / kmh * 60), slots) for a, b, km, road, kmh, slots in ROADS])

    with open(os.path.join(OUT, "stores.csv"), encoding="utf-8") as f:
        stores = list(csv.DictReader(f))
    rows, lanes = [], []
    for s in stores:
        la, lo, _ = PLACES[s["city"]]
        slat, slon = la + rng.uniform(-0.06, 0.06), lo + rng.uniform(-0.08, 0.08)
        pallets = round(PALLETS[s["size_band"]] * rng.uniform(0.9, 1.1))
        chilled = round(pallets * rng.uniform(0.22, 0.30))
        rows.append((s["store_id"], round(slat, 4), round(slon, 4), pallets, chilled))
    write("store_logistics.csv", ["store_id", "latitude", "longitude", "pallets_per_week", "chilled_pallets_per_week"], rows)

    write("dc_logistics.csv", ["dc_id", "city", "pallets_per_week_capacity", "ships_french_labels"],
          [(dc, DC_CITY[dc], DC_CAPACITY[dc], int(dc != "DC-MIS")) for dc in DC_CAPACITY])

    city_of = {s["store_id"]: s["city"] for s in stores}
    loc = {r[0]: (r[1], r[2]) for r in rows}
    for dc, dc_city in DC_CITY.items():
        km_from = road_km(dc_city)
        for sid in city_of:
            local = max(3.0, 1.3 * haversine_km(PLACES[city_of[sid]], loc[sid]))
            km = km_from[city_of[sid]] + local
            minutes = km / 80 * 60 + 20          # average 80 km/h plus 20 minutes in town
            lanes.append((dc, sid, round(km, 1), round(minutes), round(HANDLING_PER_PALLET + TRUCK_PER_PALLET_KM * km, 2)))
    write("dc_store_lanes.csv", ["dc_id", "store_id", "km", "minutes", "cost_per_pallet"], lanes)

    # Day 14: eight drivers, eight Montreal routes, minutes each driver needs for each route
    routes = [f"R{k}" for k in range(1, 9)]
    base = {r: rng.randint(300, 400) for r in routes}
    write("driver_route_minutes.csv", ["driver", "route", "minutes"],
          [(f"D{d}", r, base[r] + rng.randint(-40, 60)) for d in range(1, 9) for r in routes])

    # Day 16: the chilled network: two supplier plants, three DCs, two cross-docks, 40 stores
    nodes = [("PL-SHY", "plant", "Saint-Hyacinthe", 0, 950), ("PL-BRA", "plant", "Brampton", 0, 900)]
    nodes += [(dc, "dc", DC_CITY[dc], 0, {"DC-MIS": 480, "DC-OTT": 200, "DC-MTL": 460}[dc]) for dc in DC_CITY]
    nodes += [("XD-KIN", "cross_dock", "Kingston", 0, 160), ("XD-TRV", "cross_dock", "Trois-Rivieres", 0, 220)]
    chilled = {r[0]: r[4] for r in rows}
    nodes += [(sid, "store", city_of[sid], chilled[sid], "") for sid in city_of]
    write("chilled_nodes.csv", ["node", "kind", "place", "demand_pallets", "throughput_capacity"], nodes)

    arcs = []
    def linehaul(a_node, a_place, b_node, b_place, trucks):
        km = road_km(a_place)[b_place]
        arcs.append((a_node, b_node, round(km, 1), round(2.5 + LINEHAUL_PER_PALLET_KM * km, 2), trucks * 26))
    for plant, place in (("PL-SHY", "Saint-Hyacinthe"), ("PL-BRA", "Brampton")):
        for dc, dc_city in DC_CITY.items():
            linehaul(plant, place, dc, dc_city, 30)
    linehaul("PL-SHY", "Saint-Hyacinthe", "XD-TRV", "Trois-Rivieres", 6)
    linehaul("DC-MIS", "Mississauga", "XD-KIN", "Kingston", 6)
    linehaul("DC-OTT", "Ottawa", "XD-KIN", "Kingston", 3)
    linehaul("DC-MTL", "Montreal", "XD-TRV", "Trois-Rivieres", 8)
    linehaul("DC-MTL", "Montreal", "XD-KIN", "Kingston", 3)
    for dc, sid, km, minutes, cost in lanes:
        arcs.append((dc, sid, km, round(cost * 1.15, 2), ""))      # chilled trucks cost 15% more
    for xd, place in (("XD-KIN", "Kingston"), ("XD-TRV", "Trois-Rivieres")):
        km_from = road_km(place)
        for sid, city in city_of.items():
            km = km_from[city] + max(3.0, 1.3 * haversine_km(PLACES[city], loc[sid]))
            if km <= 260:
                arcs.append((xd, sid, round(km, 1), round((TRUCK_PER_PALLET_KM * km) * 1.15, 2), ""))
    write("chilled_arcs.csv", ["from_node", "to_node", "km", "cost_per_pallet", "capacity_pallets"], arcs)



# ---------------------------------------------------------------------------
# Days 18-20: four December weeks of distribution, two commodities, forecasts and actuals.
# ---------------------------------------------------------------------------
HISTORY_START = date(2025, 11, 24)     # 53 weeks of history, to the week of 2026-11-23
DC_WEEKLY = {
    # dc: (receiving pallets a week, outbound pallets a week, chilled zone pallets a week, ambient storage, opening stock)
    "DC-MIS": (1650, 1600, 400, 900, 250),
    "DC-OTT": (760, 700, 220, 250, 80),
    "DC-MTL": (1500, 1500, 420, 700, 200),
}
PEAK_EXTRA_SHIFT = 1.15      # outbound capacity with the extra December shifts (weeks 3 and 4)


def pallet_season(ws):
    md = (ws.month, ws.day)
    if (12, 14) <= md <= (12, 27):
        return 1.20
    if (12, 7) <= md <= (12, 13):
        return 1.08
    if md >= (12, 28) or ws.month == 1:
        return 0.90
    if ws.month in (7, 8):
        return 0.95
    return 1.0


def make_days_18_20():
    rng = random.Random(2018)
    with open(os.path.join(OUT, "store_logistics.csv"), encoding="utf-8") as f:
        stores = list(csv.DictReader(f))
    weeks = [HISTORY_START + timedelta(weeks=k) for k in range(53 + 4)]
    history, actual = [], []
    last = weeks[52]
    for s in stores:
        base = int(s["pallets_per_week"])
        share = int(s["chilled_pallets_per_week"]) / base
        for w in weeks:
            trend = 1 + 0.03 * (w - last).days / 365.25
            total = max(1, round(base * pallet_season(w) * trend * rng.gauss(1, 0.06)))
            if w <= last:
                history.append((w.isoformat(), s["store_id"], total))
            else:
                chilled = round(total * share * rng.gauss(1, 0.04))
                actual.append(((w - last).days // 7, s["store_id"], total - chilled, chilled))
    write("store_pallets_history.csv", ["week_start", "store_id", "pallets"], history)
    write("store_week_actual.csv", ["week", "store_id", "ambient_pallets", "chilled_pallets"], actual)

    # The planning team's forecast: the same week last year, plus 3% growth, split by each store's chilled share
    hist = {(w, sid): p for w, sid, p in history}
    forecast = []
    for s in stores:
        share = int(s["chilled_pallets_per_week"]) / int(s["pallets_per_week"])
        for k in range(1, 5):
            same_week_last_year = (last + timedelta(weeks=k) - timedelta(weeks=52)).isoformat()
            total = round(hist[same_week_last_year, s["store_id"]] * 1.03)
            chilled = round(total * share)
            forecast.append((k, s["store_id"], total - chilled, chilled))
    write("store_week_forecast.csv", ["week", "store_id", "ambient_pallets", "chilled_pallets"], forecast)

    rows = []
    for dc, (recv, out, chill, storage, opening) in DC_WEEKLY.items():
        for k in range(1, 5):
            extra = PEAK_EXTRA_SHIFT if k in (3, 4) else 1.0
            rows.append((dc, k, recv, round(out * extra), chill, storage, opening if k == 1 else ""))
    write("dc_weeks.csv", ["dc_id", "week", "receiving_pallets", "outbound_pallets", "chilled_zone_pallets",
                           "ambient_storage_pallets", "opening_ambient_stock"], rows)
    write("distribution_costs.csv", ["item", "value", "note"], [
        ("inbound_per_pallet", 3.0, "$ to receive a pallet from a supplier at a DC"),
        ("holding_per_pallet_week", 2.5, "$ to keep an ambient pallet at a DC for a week"),
        ("chilled_lane_factor", 1.15, "chilled lane cost = ambient lane cost x this"),
        ("lost_sale_per_pallet", 120.0, "$ margin lost when a store does not get a pallet it needed"),
        ("emergency_premium_per_pallet", 25.0, "$ extra for a same-week emergency delivery (Day 19)"),
    ])



# ---------------------------------------------------------------------------
# Days 21-23: yes/no decisions. Bakery investments, product set-ups, candidate sites.
# ---------------------------------------------------------------------------
BAKERY_INVESTMENTS = [
    # option, resource, extra capacity a week, cost a week ($)
    ("lease_ovens", "oven_hours", 100, 900),
    ("hire_baker", "baker_hours", 40, 1600),
    ("starter_tank", "sourdough_starter", 12, 250),
    ("packing_upgrade", "packing_hours", 60, 1400),
]
CANDIDATE_SITES = [
    # site_id, place, kind, fixed cost a week ($), capacity pallets a week, ships French labels.
    # Existing DCs: running cost, saved if closed. Candidates: running cost plus the lease or build cost, per week.
    ("DC-MIS", "Mississauga", "existing DC", 21000, 1600, 0),
    ("DC-OTT", "Ottawa", "existing DC", 11000, 700, 1),
    ("DC-MTL", "Montreal", "existing DC", 19000, 1500, 1),
    ("NEW-HAM", "Hamilton", "candidate", 16000, 1100, 0),
    ("NEW-BRA", "Brampton", "candidate", 23000, 1800, 0),
    ("NEW-KIN", "Kingston", "candidate", 12500, 900, 0),
    ("NEW-COR", "Cornwall", "candidate", 12000, 1000, 1),
    ("NEW-LAV", "Laval", "candidate", 20000, 1300, 1),
    ("NEW-DRU", "Drummondville", "candidate", 12500, 900, 1),
    ("NEW-QUE", "Quebec City", "candidate", 13500, 700, 1),
]


def make_days_21_23():
    rng = random.Random(2021)
    write("bakery_investments.csv", ["option", "resource", "extra_capacity", "cost_per_week"], BAKERY_INVESTMENTS)

    with open(os.path.join(OUT, "bakery_range.csv"), encoding="utf-8") as f:
        range_rows = list(csv.DictReader(f))
    setups = []
    for r in range_rows:
        # cleaning and changeover each week a product is made; bigger for cakes and viennoiserie
        base = {"bread": 220, "viennoiserie": 320, "cake": 380, "cookie": 260}[r["category"]]
        setups.append((r["sku"], round(base * rng.uniform(0.7, 1.4), -1), rng.choice([3, 4, 4, 5])))
    write("bakery_range_setup.csv", ["sku", "setup_cost_per_week", "min_batches_if_made"], setups)

    write("candidate_sites.csv", ["site_id", "place", "kind", "fixed_cost_per_week", "capacity_pallets", "ships_french_labels"],
          CANDIDATE_SITES)
    with open(os.path.join(OUT, "stores.csv"), encoding="utf-8") as f:
        city_of = {s["store_id"]: s["city"] for s in csv.DictReader(f)}
    with open(os.path.join(OUT, "store_logistics.csv"), encoding="utf-8") as f:
        loc = {r["store_id"]: (float(r["latitude"]), float(r["longitude"])) for r in csv.DictReader(f)}
    lanes = []
    for site, place, *_ in CANDIDATE_SITES:
        km_from = road_km(place)
        for sid, city in city_of.items():
            km = km_from[city] + max(3.0, 1.3 * haversine_km(PLACES[city], loc[sid]))
            lanes.append((site, sid, round(km, 1), round(km / 80 * 60 + 20),
                          round(HANDLING_PER_PALLET + TRUCK_PER_PALLET_KM * km, 2)))
    write("site_store_lanes.csv", ["site_id", "store_id", "km", "minutes", "cost_per_pallet"], lanes)


# ---------------------------------------------------------------------------
# Days 24-27: loading trucks, cutting cake slabs, and a bigger chilled network with candidate cross-docks.
# ---------------------------------------------------------------------------
TRUCKS = [("trailer", 26, 20000, 650), ("straight", 20, 12000, 480)]   # pallet positions, max kg, cost for a day ($)
CAKE_SLAB = (120, 18.0)           # a sheet-cake slab is 120 cm long and costs $18 to bake
CAKE_BARS = [("BAR-18", "Lemon bar", 18, 64), ("BAR-24", "Brownie bar", 24, 52), ("BAR-30", "Carrot bar", 30, 40),
             ("BAR-45", "Opera bar", 45, 28), ("BAR-50", "Black forest bar", 50, 22)]
CROSS_DOCKS = [   # id, place, fixed cost a week, throughput pallets a week (Kingston and Trois-Rivieres exist already)
    ("XD-KIN", "Kingston", 500, 160), ("XD-TRV", "Trois-Rivieres", 900, 220), ("XD-COR", "Cornwall", 650, 180),
    ("XD-BEL", "Belleville", 550, 140), ("XD-DRU", "Drummondville", 700, 200), ("XD-SHE", "Sherbrooke", 600, 120),
]
LINEHAUL_FROM = {"XD-KIN": ["DC-MIS", "DC-OTT", "DC-MTL"], "XD-TRV": ["PL-SHY", "DC-MTL"], "XD-COR": ["DC-OTT", "DC-MTL"],
                 "XD-BEL": ["DC-MIS", "DC-OTT"], "XD-DRU": ["PL-SHY", "DC-MTL"], "XD-SHE": ["PL-SHY", "DC-MTL"]}


def make_days_24_27():
    rng = random.Random(2024)
    write("truck_spec.csv", ["truck_type", "pallet_positions", "max_kg", "cost_per_day"], TRUCKS)
    with open(os.path.join(OUT, "stores.csv"), encoding="utf-8") as f:
        stores = list(csv.DictReader(f))
    mtl = [s["store_id"] for s in stores if s["dc_id"] == "DC-MTL"]

    # Day 24.1: Monday's rush truck (a straight truck) from Montreal: top-up orders that missed the weekend trucks
    rows = []
    for k in range(1, 17):
        pallets = rng.randint(1, 6)
        rows.append((f"RO-{k:02d}", rng.choice(mtl), pallets, int(round(pallets * rng.uniform(380, 820), -1)),
                     int(round(pallets * rng.uniform(140, 420), -1))))
    write("rush_orders.csv", ["order_id", "store_id", "pallets", "weight_kg", "value_if_delivered"], rows)

    # Day 24.2: Tuesday's store orders from Montreal, to be loaded on as few trucks as possible
    rows = []
    for k in range(1, 25):
        pallets = rng.choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        rows.append((f"TO-{k:02d}", rng.choice(mtl), pallets, int(round(pallets * rng.uniform(450, 900), -1))))
    write("truck_orders.csv", ["order_id", "store_id", "pallets", "weight_kg"], rows)

    # Days 26-27: the Tuesday before Christmas: 48 orders from Montreal, loaded under the spring-thaw style 16,000 kg limit
    xmas = random.Random(2026_12_22)
    rows = []
    for k in range(1, 49):
        pallets = xmas.choice(range(3, 14))
        rows.append((f"XO-{k:02d}", xmas.choice(mtl), pallets, int(round(pallets * xmas.uniform(450, 900), -1))))
    write("christmas_orders.csv", ["order_id", "store_id", "pallets", "weight_kg"], rows)

    # Day 24.3: cake bars cut from sheet-cake slabs
    write("cake_slab.csv", ["slab_length_cm", "cost_per_slab"], [CAKE_SLAB])
    write("cake_bars.csv", ["bar", "product", "length_cm", "bars_per_week"], CAKE_BARS)

    # Day 25: six candidate cross-docks for the chilled network; trucks on linehaul arcs
    write("crossdock_candidates.csv", ["xd_id", "place", "fixed_cost_per_week", "throughput_pallets"], CROSS_DOCKS)
    origin = {"PL-SHY": "Saint-Hyacinthe", "DC-MIS": "Mississauga", "DC-OTT": "Ottawa", "DC-MTL": "Montreal"}
    with open(os.path.join(OUT, "store_logistics.csv"), encoding="utf-8") as f:
        loc = {r["store_id"]: (float(r["latitude"]), float(r["longitude"])) for r in csv.DictReader(f)}
    arcs = []
    for xd, place, *_ in CROSS_DOCKS:
        for o in LINEHAUL_FROM[xd]:
            km = road_km(origin[o])[place]
            arcs.append((o, xd, "linehaul", round(km, 1), round(26 * (2.5 + LINEHAUL_PER_PALLET_KM * km), 2), ""))
        km_from = road_km(place)
        for s in stores:
            km = km_from[s["city"]] + max(3.0, 1.3 * haversine_km(PLACES[s["city"]], loc[s["store_id"]]))
            if km <= 260 and (s["province"] == "ON" or place in ("Trois-Rivieres", "Drummondville", "Sherbrooke", "Cornwall")):
                arcs.append((xd, s["store_id"], "delivery", round(km, 1), "", round(TRUCK_PER_PALLET_KM * km * 1.15, 2)))
    write("crossdock_arcs.csv", ["from_node", "to_node", "arc_type", "km", "cost_per_truck", "cost_per_pallet"], arcs)


# ---------------------------------------------------------------------------
# Days 28-30: Christmas week loads, flour discounts and overtime tiers, and the five-year network redesign.
# ---------------------------------------------------------------------------
FLOUR_DISCOUNTS = [   # incremental: each tonne in a band is paid at that band's price
    ("FL-PHR", 0, 10, 520), ("FL-PHR", 10, 20, 495), ("FL-PHR", 20, 30, 470),
    ("FL-OSW", 0, 15, 410), ("FL-OSW", 15, 30, 395),
    ("FL-QCO", 0, 8, 640),
    ("FL-DUR", 0, 10, 590), ("FL-DUR", 10, 30, 540),
]
FLOUR_ALL_UNITS = [   # an offer from Ontario Soft White: the price for every tonne depends on the week's total
    ("FL-OSW", 0, 15, 410), ("FL-OSW", 15, 25, 392), ("FL-OSW", 25, 30, 372),
]
FLOUR_DELIVERY = [("FL-PHR", 12, 380), ("FL-OSW", 12, 220), ("FL-QCO", 8, 260), ("FL-DUR", 12, 450)]  # t per truck, $
OVERTIME_TIERS = [("packing_hours", 1, 20, 60), ("packing_hours", 2, 20, 95), ("baker_hours", 1, 40, 45), ("baker_hours", 2, 40, 70)]
NETWORK_OPTIONS = [   # option, site, kind, extra pallets a week, one-time cost, extra weekly cost
    ("EXP-OTT", "DC-OTT", "expand", 300, 1_400_000, 2_500), ("EXP-MTL", "DC-MTL", "expand", 300, 1_800_000, 3_000),
    ("EXP-MIS", "DC-MIS", "expand", 400, 2_000_000, 3_200),
]
NETWORK_SETTINGS = [("first_year", 2027, ""), ("years", 5, ""), ("growth_per_year", 0.03, "pallets per store"),
                    ("discount_rate", 0.08, "per year"), ("weeks_per_year", 52, ""),
                    ("open_cost_new_site", 3_000_000, "one-time, fit-out and moving"),
                    ("close_cost_existing_dc", 1_500_000, "one-time, severance and lease exit")]


def make_days_28_30():
    with open(os.path.join(OUT, "stores.csv"), encoding="utf-8") as f:
        mtl = [s["store_id"] for s in csv.DictReader(f) if s["dc_id"] == "DC-MTL"]
    with open(os.path.join(OUT, "christmas_orders.csv"), encoding="utf-8") as f:
        tuesday = [tuple(r.values()) for r in csv.DictReader(f)]
    rng = random.Random(2026_12_21)
    rows = []
    for day, n in (("2026-12-21", 44), ("2026-12-22", None), ("2026-12-23", 52), ("2026-12-24", 40)):
        if n is None:
            rows += [(day, *r) for r in tuesday]
            continue
        for k in range(1, n + 1):
            pallets = rng.choice(range(3, 14))
            rows.append((day, f"X{day[-2:]}-{k:02d}", rng.choice(mtl), pallets, int(round(pallets * rng.uniform(450, 900), -1))))
    write("christmas_week_orders.csv", ["ship_date", "order_id", "store_id", "pallets", "weight_kg"], rows)
    write("flour_discounts.csv", ["supplier_id", "from_tonnes", "to_tonnes", "price_per_tonne"], FLOUR_DISCOUNTS)
    write("flour_all_units_offer.csv", ["supplier_id", "from_tonnes", "to_tonnes", "price_per_tonne"], FLOUR_ALL_UNITS)
    write("flour_delivery.csv", ["supplier_id", "tonnes_per_truck", "cost_per_truck"], FLOUR_DELIVERY)
    write("overtime_tiers.csv", ["resource", "tier", "hours", "cost_per_hour"], OVERTIME_TIERS)
    write("network_options.csv", ["option_id", "site_id", "kind", "extra_pallets", "one_time_cost", "weekly_cost"], NETWORK_OPTIONS)
    write("network_settings.csv", ["setting", "value", "note"], NETWORK_SETTINGS)


# ---------------------------------------------------------------------------
# Days 31-33: the Montreal DC's warehouse staff, shifts, coverage, last week's roster and requests.
# ---------------------------------------------------------------------------
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DC_SHIFTS = [("D", "07:00", "15:00", 0.0), ("E", "15:00", "23:00", 1.50), ("N", "23:00", "07:00", 2.75)]   # premium $/h
COVERAGE = {   # (shift, role): (Mon-Fri, Sat-Sun) people needed
    ("D", "picker"): (7, 5), ("D", "forklift"): (3, 2), ("E", "picker"): (6, 4), ("E", "forklift"): (2, 2),
    ("N", "picker"): (3, 2), ("N", "forklift"): (1, 1),
}


def make_days_31_33():
    rng = random.Random(2031)
    staff = []
    for k in range(1, 35):
        forklift = k <= 12
        contract = "part-time" if k in (5, 11, 17, 21, 25, 29, 31, 34) else "full-time"
        wage = round(rng.uniform(25.0, 29.0) if forklift else rng.uniform(21.5, 25.0), 2)
        staff.append((f"E{k:02d}", "forklift" if forklift else "picker", contract, wage, 2019 + k % 7))
    write("dc_staff.csv", ["employee_id", "role", "contract", "hourly_wage", "hired_year"], staff)
    write("dc_shifts.csv", ["shift", "starts", "ends", "premium_per_hour"], DC_SHIFTS)
    write("dc_coverage.csv", ["day", "shift", "role", "people"],
          [(d, s, r, n[0] if d not in ("Sat", "Sun") else n[1]) for d in WEEKDAYS for (s, r), n in COVERAGE.items()])
    write("dc_rules.csv", ["rule", "value", "source"], [
        ("hours_per_shift", 8, ""),
        ("max_hours_per_week", 48, "ESA: 48 hours a week without an excess-hours agreement"),
        ("overtime_after_hours", 44, "ESA: overtime pay after 44 hours in a work week"),
        ("overtime_rate", 1.5, "ESA: time and a half"),
        ("rest_hours_each_day", 11, "ESA: 11 consecutive hours off in each 24-hour period"),
        ("rest_hours_each_week", 24, "ESA: 24 consecutive hours off each work week (or 48 in two weeks)"),
        ("full_time_min_shifts", 5, "collective agreement: 40 guaranteed hours"),
        ("part_time_max_shifts", 3, "collective agreement"),
    ])
    # last week's Sunday, and the supervisor's roster for next week (made by hand, first come first served)
    last = {e[0]: rng.choice(["D", "E", "N", "off", "off"]) for e in staff}
    write("dc_last_sunday.csv", ["employee_id", "shift"], sorted(last.items()))
    count = {e[0]: 0 for e in staff}
    role = {e[0]: e[1] for e in staff}
    cap = {e[0]: (3 if e[2] == "part-time" else 6) for e in staff}
    roster = []
    for d in WEEKDAYS:
        busy = set()
        for s, _, _, _ in DC_SHIFTS:
            for r in ("forklift", "picker"):
                need = COVERAGE[(s, r)][0 if d not in ("Sat", "Sun") else 1]
                pool = sorted((e for e in count if e not in busy and count[e] < cap[e]
                               and (role[e] == "forklift" or r == "picker") and (r == "picker" or role[e] == "forklift")),
                              key=lambda e: (count[e], role[e] != r, e))
                for e in pool[:need]:
                    roster.append((e, d, s, r))
                    busy.add(e)
                    count[e] += 1
    write("dc_current_roster.csv", ["employee_id", "day", "shift", "role"], roster)
    # Day 33: requests for next week (weekends are popular)
    req = []
    for e in staff:
        for day in sorted(set(rng.choices(WEEKDAYS, weights=[1, 1, 1, 1, 3, 4, 4], k=rng.choice([0, 1, 1, 2]))),
                          key=WEEKDAYS.index):
            req.append((e[0], "day_off", day, "", rng.choice([1, 1, 2])))
        if rng.random() < 0.40:
            req.append((e[0], "no_nights", "", "N", 1))
        if rng.random() < 0.25:
            req.append((e[0], "prefer_shift", "", rng.choice(["D", "D", "E"]), 1))
    write("staff_requests.csv", ["employee_id", "request", "day", "shift", "priority"], req)


# ---------------------------------------------------------------------------
# Days 34-37: tonight's bakery production run, and tomorrow morning's bread deliveries in Montreal.
# ---------------------------------------------------------------------------
BAKERY_MACHINES = [("MIX-1", "mixer", 1), ("MIX-2", "mixer", 1), ("LAM-1", "laminator", 1), ("PROOFER", "proofer", 4),
                   ("OVEN-1", "oven", 1), ("OVEN-2", "oven", 1), ("PACK-1", "packing", 1), ("DECO-1", "decorating", 1)]
STEPS = {   # category: [(operation, machine group, base minutes)]; "cool" needs no machine
    "bread": [("mix", "mixer", 20), ("proof", "proofer", 90), ("bake", "oven", 45), ("pack", "packing", 20)],
    "viennoiserie": [("laminate", "laminator", 35), ("proof", "proofer", 60), ("bake", "oven", 20), ("pack", "packing", 15)],
    "cake": [("mix", "mixer", 25), ("bake", "oven", 50), ("cool", "none", 40), ("decorate", "decorating", 30), ("pack", "packing", 10)],
    "cookie": [("mix", "mixer", 15), ("bake", "oven", 15), ("pack", "packing", 10)],
}
NIGHT_JOBS = ["BK-SD", "BK-WW", "BK-03", "BK-04", "BK-05", "BK-SD", "BK-WW", "BK-CR", "BK-12", "BK-13", "BK-CR", "BK-14",
              "BK-20", "BK-22", "BK-23", "BK-25", "BK-26", "BK-27"]
DEPOT = (45.4930, -73.6280)          # the bakery's dispatch dock, beside DC-MTL
FLEET = [("van", 6, 18, 90, 0.40), ("truck", 3, 36, 150, 0.65)]   # type, how many, racks, $ a day, $ a km


def make_days_34_37():
    rng = random.Random(2034)
    with open(os.path.join(OUT, "bakery_range.csv"), encoding="utf-8") as f:
        category = {r["sku"]: r["category"] for r in csv.DictReader(f)}
    write("bakery_machines.csv", ["machine_id", "machine_group", "capacity"], BAKERY_MACHINES)
    jobs, ops = [], []
    for k, sku in enumerate(NIGHT_JOBS, start=1):
        cat = category[sku]
        due = 420 if cat in ("bread", "viennoiserie") else rng.choice([480, 540])     # minutes after 21:00 (04:00)
        jobs.append((f"J{k:02d}", sku, cat, due))
        for step, (op, group, base) in enumerate(STEPS[cat], start=1):
            minutes = base if op in ("proof", "cool") else int(round(base * rng.uniform(0.85, 1.25)))
            ops.append((f"J{k:02d}", step, op, group, minutes))
    write("bakery_night_jobs.csv", ["job_id", "sku", "category", "due_minute"], jobs)
    write("bakery_operations.csv", ["job_id", "step", "operation", "machine_group", "minutes"], ops)
    write("bakery_rules.csv", ["rule", "minutes"], [("start_clock", 1260), ("max_wait_proof_to_bake", 15),
                                                   ("min_cooling", 40), ("oven_2_maintenance_start", 240),
                                                   ("oven_2_maintenance_end", 360)])

    # Days 36-37: the morning bread run: 10 Montreal and Laval stores, 24 cafe customers
    with open(os.path.join(OUT, "stores.csv"), encoding="utf-8") as f:
        stores = [s for s in csv.DictReader(f) if s["city"] in ("Montreal", "Laval")]
    with open(os.path.join(OUT, "store_logistics.csv"), encoding="utf-8") as f:
        loc = {r["store_id"]: (float(r["latitude"]), float(r["longitude"])) for r in csv.DictReader(f)}
    racks = {"Large": 12, "Medium": 9, "Small": 6}
    stops = [("DEPOT", "depot", "Bakery dock", DEPOT[0], DEPOT[1], 0, 0, 240, 480, 0)]
    for s in stores:
        stops.append((s["store_id"], "store", s["store_name"], *loc[s["store_id"]], racks[s["size_band"]], 20, 270, 420, 0))
    for k in range(1, 25):
        lat, lon = round(rng.uniform(45.45, 45.58), 4), round(rng.uniform(-73.67, -73.54), 4)
        downtown = int(45.49 <= lat <= 45.535 and -73.61 <= lon <= -73.55)   # narrow streets: vans only
        opens = rng.choice([390, 420, 420, 450])                         # 06:30, 07:00 or 07:30
        stops.append((f"C{k:02d}", "cafe", f"Cafe partner {k}", lat, lon, rng.choice([1, 2, 2, 3, 4]), 6,
                      300, opens - 15, downtown))
    write("delivery_stops.csv", ["stop_id", "kind", "name", "latitude", "longitude", "racks", "service_minutes",
                                 "window_open", "window_close", "downtown"], stops)
    rows = []
    for a in stops:
        for b in stops:
            km = 0.0 if a[0] == b[0] else round(1.35 * haversine_km((a[3], a[4]), (b[3], b[4])) + 0.4, 2)
            rows.append((a[0], b[0], km, 0 if a[0] == b[0] else int(round(km / 30 * 60 + 2))))
    write("delivery_matrix.csv", ["from_stop", "to_stop", "km", "minutes"], rows)
    write("delivery_fleet.csv", ["vehicle_type", "vehicles", "racks", "cost_per_day", "cost_per_km"], FLEET)
    # Day 35: certifying pickers as forklift drivers
    write("forklift_certification.csv", ["employee_id", "course_cost", "raise_per_hour"], [("E13", 1800, 1.50), ("E14", 1800, 1.50)])


# ---------------------------------------------------------------------------
# Day 40: next week's bread run, day by day, and the delivery drivers.
# ---------------------------------------------------------------------------
RUN_DAYS = [("Mon", 0.90), ("Tue", 0.95), ("Wed", 1.00), ("Thu", 1.05), ("Fri", 1.20), ("Sat", 1.15)]


def make_days_38_40():
    rng = random.Random(2040)
    write("delivery_days.csv", ["day", "rack_factor"], RUN_DAYS)
    drivers = []
    for k in range(1, 12):
        contract = "part-time" if k >= 8 else "full-time"
        trucks = int(k in (1, 2, 3, 8))
        drivers.append((f"DR{k:02d}", contract, trucks, round(rng.uniform(24.0, 28.5) + (1.5 if trucks else 0), 2)))
    write("delivery_drivers.csv", ["driver_id", "contract", "truck_licence", "hourly_wage"], drivers)
    write("driver_rules.csv", ["rule", "value"], [("min_paid_hours_per_day", 4), ("pre_and_post_trip_minutes", 30),
                                                  ("full_time_min_days", 4), ("full_time_max_days", 5),
                                                  ("part_time_max_days", 3)])


# ---------------------------------------------------------------------------
# Days 41-43: six months of van trips, planned against actual minutes (with a few bad GPS records).
# ---------------------------------------------------------------------------
def make_days_41_43():
    rng = random.Random(2041)
    with open(os.path.join(OUT, "delivery_matrix.csv"), encoding="utf-8") as f:
        lanes = [(r["from_stop"], r["to_stop"], int(r["minutes"])) for r in csv.DictReader(f) if int(r["minutes"]) > 0]
    day = datetime.date(2026, 5, 25)
    rows, trip = [], 0
    while day <= datetime.date(2026, 11, 21):
        if day.weekday() < 6:                                    # Monday to Saturday
            month = day.month
            snow_p = {11: 0.25}.get(month, 0.0)
            rain_p = {5: 0.25, 6: 0.25, 7: 0.20, 8: 0.20, 9: 0.25, 10: 0.30, 11: 0.25}[month]
            u = rng.random()
            weather = "snow" if u < snow_p else ("rain" if u < snow_p + rain_p else "clear")
            day_factor = {"clear": 1.0, "rain": 1.08, "snow": 1.30}[weather] * rng.lognormvariate(0, 0.04)
            for _ in range(12):
                a, b, planned = rng.choice(lanes)
                actual = planned * day_factor * rng.lognormvariate(0.05, 0.12)
                if rng.random() < 0.03:                          # an incident: a crash, a closed street
                    actual += rng.uniform(10, 35)
                trip += 1
                rows.append((f"T{trip:05d}", day.isoformat(), weather, a, b, planned, round(actual, 1)))
        day += datetime.timedelta(days=1)
    for k, bad in ((40, 0.0), (311, 0.0), (902, 612.0), (1455, -3.0)):    # GPS glitches
        r = rows[k]
        rows[k] = r[:6] + (bad,)
    write("trip_log.csv", ["trip_id", "trip_date", "weather", "from_stop", "to_stop", "planned_minutes", "actual_minutes"], rows)


# ---------------------------------------------------------------------------
# Days 44-47: disruptions at the bakery, and four weeks of night receiving at DC-MTL.
# ---------------------------------------------------------------------------
DISRUPTIONS = [("oven_breakdown", 0.03, "oven_hours", 0.70), ("packing_line_failure", 0.02, "packing_hours", 0.75)]
NIGHT_RATE = [1.2, 2.0, 2.4, 2.2, 1.6, 1.3, 1.2, 1.1]       # trucks an hour, 22:00 to 06:00


def make_days_44_47():
    rng = random.Random(2044)
    write("bakery_disruptions.csv", ["event", "chance_per_week", "resource", "capacity_left"], DISRUPTIONS)
    rows, k = [], 0
    night = datetime.date(2026, 10, 26)
    while night <= datetime.date(2026, 11, 21):
        if night.weekday() < 6:
            for hour, rate in enumerate(NIGHT_RATE):
                t = 0.0
                while True:
                    t += rng.expovariate(rate)                   # Poisson arrivals within the hour
                    if t >= 1.0:
                        break
                    k += 1
                    minute = 22 * 60 + hour * 60 + t * 60
                    rows.append((f"R{k:04d}", night.isoformat(), round(minute % 1440, 1),
                                 round(rng.gammavariate(4.0, 8.0), 1), rng.choice(["PL-SHY", "PL-BRA", "supplier"])))
        night += datetime.timedelta(days=1)
    write("receiving_log.csv", ["truck_id", "night_of", "arrival_minute", "unload_minutes", "origin"], rows)
    write("receiving_settings.csv", ["setting", "value"], [
        ("doors", 3), ("night_forklift_drivers", 1), ("free_minutes_before_detention", 60),
        ("detention_per_hour", 85), ("extra_driver_cost_per_night", 245)])


# ---------------------------------------------------------------------------
# Days 48-50: shift options for a second night driver, and stress scenarios for the network.
# ---------------------------------------------------------------------------
DRIVER_SHIFTS = [("none", 0, 0, 0), ("all_night", 0, 8, 245),
                 ("4h_from_22", 0, 4, 150), ("4h_from_23", 60, 4, 150), ("4h_from_00", 120, 4, 150),
                 ("4h_from_01", 180, 4, 150), ("4h_from_02", 240, 4, 150),
                 ("6h_from_22", 0, 6, 200), ("6h_from_23", 60, 6, 200), ("6h_from_00", 120, 6, 200)]
STRESS = [("normal", "all", 1.00, "draw", 1.00, 1.00),
          ("holiday_rush", "december", 1.15, "draw", 1.00, 1.25),
          ("snowstorm_week", "all", 0.90, "snow", 1.00, 1.00),
          ("oven_breakdown", "all", 1.00, "draw", 0.70, 1.00),
          ("dock_surge", "all", 1.00, "draw", 1.00, 1.40),
          ("perfect_storm", "december", 1.15, "snow", 0.70, 1.40)]


def make_days_48_50():
    write("dock_shift_options.csv", ["option", "start_after_22h_minutes", "hours", "cost_per_night"], DRIVER_SHIFTS)
    write("stress_scenarios.csv", ["scenario", "demand_weeks", "store_demand_factor", "weather", "oven_capacity_left",
                                   "truck_arrivals_factor"], STRESS)


# ---------------------------------------------------------------------------
# Days 51-53: Christmas yule logs (a one-shot order), ambient items at DC-MTL (inventory policies), and the
# bakery's day-old shelf and mid-week top-up bake (two-stage planning).
# ---------------------------------------------------------------------------
YULE_BASE = {"Small": 25, "Medium": 40, "Large": 60}
YULE_ECONOMICS = [("price", 32.00), ("cost", 11.50), ("boxing_day_price", 16.00), ("boxing_day_sell_share", 0.60),
                  ("transfer_cost_per_log", 1.50)]
DC_ITEMS = [  # item, name, supplier, cost a case, margin a case, mean cases a day, coefficient of variation
    ("AM-01", "Pasta 500 g (12)", "SUP-A", 13.20, 4.30, 120, 0.30),
    ("AM-02", "Canned tomatoes 796 ml (24)", "SUP-A", 22.80, 6.90, 90, 0.30),
    ("AM-03", "Coffee 900 g (6)", "SUP-B", 58.80, 17.60, 26, 0.45),
    ("AM-04", "Olive oil 1 L (6)", "SUP-B", 50.40, 15.10, 18, 0.50),
    ("AM-05", "Basmati rice 2 kg (6)", "SUP-C", 25.20, 7.60, 42, 0.35),
    ("AM-06", "Oat cereal 500 g (12)", "SUP-C", 39.60, 11.90, 64, 0.30),
]
SUPPLIERS = {"SUP-A": (5, 1.0), "SUP-B": (10, 3.0), "SUP-C": (7, 1.5)}       # quoted lead time (days), spread
INVENTORY_SETTINGS = [("holding_rate_per_year", 0.25), ("days_per_year", 365), ("order_cost_per_po", 150),
                      ("target_fill_rate", 0.98), ("review_period_days", 7)]
SUPPLIER_OFFERS = [("AM-03", "SUP-D", 4, 1.0, 0.02), ("AM-04", "SUP-D", 4, 1.0, 0.02)]
BAKERY_TWO_STAGE = [("day_old_price_share", 0.30), ("day_old_sell_share", 0.40), ("topup_oven_hours", 40),
                    ("topup_cost_per_oven_hour", 55)]


def make_days_51_53():
    rng = random.Random(2051)
    with open(os.path.join(OUT, "stores.csv"), encoding="utf-8") as f:
        stores = list(csv.DictReader(f))
    rows, forecast26 = [], []
    for s in stores:
        base = YULE_BASE[s["size_band"]] * rng.uniform(0.8, 1.2)
        last = None
        for season, growth in ((2023, 1.00), (2024, 1.04), (2025, 1.07)):
            if s["opened_date"] > f"{season}-12-01":
                continue                                          # the store wasn't open that Christmas
            forecast = round(base * growth * rng.uniform(0.9, 1.1)) if last is None else round(last * 1.03)
            demand = max(0, round(base * growth * rng.lognormvariate(-0.04, 0.28)))
            rows.append((season, s["store_id"], forecast, demand))
            last = demand
        forecast26.append((2026, s["store_id"], round(last * 1.03)))
    write("yule_log_history.csv", ["season", "store_id", "forecast_logs", "demand_logs"], rows)
    write("yule_log_forecast.csv", ["season", "store_id", "forecast_logs"], forecast26)
    write("yule_log_economics.csv", ["setting", "value"], YULE_ECONOMICS)

    write("dc_items.csv", ["item_id", "item_name", "supplier_id", "cost_per_case", "margin_per_case"],
          [r[:5] for r in DC_ITEMS])
    write("dc_suppliers.csv", ["supplier_id", "quoted_lead_days"], [(k, v[0]) for k, v in SUPPLIERS.items()])
    days = [datetime.date(2026, 5, 25) + datetime.timedelta(days=k) for k in range(182)]
    demand = []
    for item, _, _, _, _, mean, cv in DC_ITEMS:
        shape = 1 / cv ** 2
        for d in days:
            demand.append((item, d.isoformat(), round(rng.gammavariate(shape, mean / shape))))
    write("dc_daily_demand.csv", ["item_id", "demand_date", "cases"], demand)
    pos, k = [], 0
    for item, _, sup, *_ in DC_ITEMS:
        quoted, spread = SUPPLIERS[sup]
        day = days[0] + datetime.timedelta(days=rng.randrange(7))
        while day <= days[-1]:
            lead = max(1, round(rng.gauss(quoted + 1, spread)))  # suppliers are a day late on average
            k += 1
            pos.append((f"PO{k:04d}", item, sup, day.isoformat(), (day + datetime.timedelta(days=lead)).isoformat()))
            day += datetime.timedelta(days=7)
    write("purchase_orders.csv", ["po_id", "item_id", "supplier_id", "ordered_on", "received_on"], pos)
    write("inventory_settings.csv", ["setting", "value"], INVENTORY_SETTINGS)
    write("dc_supplier_offers.csv", ["item_id", "supplier_id", "lead_mean_days", "lead_sd_days", "price_premium"],
          SUPPLIER_OFFERS)
    write("bakery_two_stage.csv", ["setting", "value"], BAKERY_TWO_STAGE)


# ---------------------------------------------------------------------------
# Days 54-57: how long the ovens really take (oven hours per batch vary), for robust planning.
# ---------------------------------------------------------------------------
OVEN_SPREAD = {"bread": 0.03, "viennoiserie": 0.06, "cake": 0.10, "cookie": 0.04}
ROBUST_SETTINGS = [("oven_overrun_cost_per_hour", 120)]


def make_days_54_57():
    rng = random.Random(2054)
    with open(os.path.join(OUT, "bakery_range.csv"), encoding="utf-8") as f:
        category = {r["sku"]: r["category"] for r in csv.DictReader(f)}
    with open(os.path.join(OUT, "bakery_range_recipe.csv"), encoding="utf-8") as f:
        oven = {r["sku"]: float(r["use_per_batch"]) for r in csv.DictReader(f) if r["resource"] == "oven_hours"}
    with open(os.path.join(OUT, "bakery_range_forecast.csv"), encoding="utf-8") as f:
        cap = {r["sku"]: float(r["demand_cap_batches"]) for r in csv.DictReader(f)}
    rows = []
    for k in range(26):
        week = (datetime.date(2026, 5, 25) + datetime.timedelta(weeks=k)).isoformat()
        busy = rng.gauss(0, 0.015)                                   # a slow week slows every product a little
        for sku in sorted(oven):
            batches = max(1, round(cap[sku] * rng.uniform(0.5, 1.0)))
            dev = rng.gauss(0.01, OVEN_SPREAD[category[sku]]) + busy
            if rng.random() < 0.02:
                dev += rng.uniform(0.10, 0.25)                       # an oven fault, a re-bake
            rows.append((week, sku, batches, round(oven[sku] * batches, 2), round(oven[sku] * batches * (1 + dev), 2)))
    write("bakery_oven_log.csv", ["week_start", "sku", "batches", "standard_oven_hours", "actual_oven_hours"], rows)
    write("bakery_robust.csv", ["setting", "value"], ROBUST_SETTINGS)


# ---------------------------------------------------------------------------
# Days 58-60: an overnight wholesaler for the dynamic-programming day, and stress scenarios for the capstone.
# ---------------------------------------------------------------------------
MDP_SETTINGS = [("item_id", "AM-04"), ("max_stock_cases", 600), ("order_step_cases", 20)]
REPLENISHMENT_SCENARIOS = [("as_tested", 1.00, "none", 0), ("demand_up_20", 1.20, "none", 0),
                           ("sup_b_late_3_days", 1.00, "SUP-B", 3), ("both", 1.20, "SUP-B", 3)]


def make_days_58_60():
    write("mdp_settings.csv", ["setting", "value"], MDP_SETTINGS)
    write("replenishment_scenarios.csv", ["scenario", "demand_factor", "late_supplier", "extra_lead_days"],
          REPLENISHMENT_SCENARIOS)


# ---------------------------------------------------------------------------
# Days 61-63: decision analysis. Renewing DC-MTL's ten store-delivery trucks in 2027: diesel or electric?
# (All numbers are course assumptions for the fictional Northbound, not market data.)
# ---------------------------------------------------------------------------
EV_OPTIONS = [  # option, electric trucks, diesel trucks, chargers (one-time $)
    ("diesel_renew", 0, 10, 0), ("electric_half", 5, 5, 150_000), ("electric_all", 10, 0, 250_000)]
EV_SETTINGS = [("km_per_truck_year", 55_000), ("years", 5), ("discount_rate", 0.08),
               ("diesel_lease_per_year", 38_000), ("electric_lease_per_year", 58_000),
               ("diesel_maintenance_per_km", 0.18), ("electric_maintenance_per_km", 0.11),
               ("electricity_per_km", 0.27), ("diesel_co2_kg_per_km", 0.95), ("electric_co2_kg_per_km", 0.10),
               ("pilot_cost", 60_000), ("diesel_hedge_per_km", 0.80)]
EV_UNCERTAINTIES = [  # factor, level, probability, value
    ("winter_range", "good", 0.45, 0), ("winter_range", "fair", 0.35, 7_500), ("winter_range", "poor", 0.20, 22_000),
    ("diesel_price", "low", 0.30, 0.62), ("diesel_price", "base", 0.50, 0.78), ("diesel_price", "high", 0.20, 0.98)]
PILOT_SIGNAL = [("good", 0.85), ("fair", 0.45), ("poor", 0.15)]        # P(the pilot reports "range is fine" | range)
FINANCE = [("annual_revenue", 1_240_000_000), ("net_income", 37_000_000), ("equity", 290_000_000),
           ("vp_logistics_largest_accepted_gamble", 600_000)]
OTTAWA_GROWTH = [(0.01, 0.25), (0.03, 0.50), (0.05, 0.25)]


def make_days_61_63():
    write("ev_fleet_options.csv", ["option", "electric_trucks", "diesel_trucks", "chargers_one_time"], EV_OPTIONS)
    write("ev_settings.csv", ["setting", "value"], EV_SETTINGS)
    write("ev_uncertainties.csv", ["factor", "level", "probability", "value"], EV_UNCERTAINTIES)
    write("ev_pilot_signal.csv", ["winter_range", "p_reports_fine"], PILOT_SIGNAL)
    write("northbound_finance.csv", ["setting", "value"], FINANCE)
    write("ottawa_growth_scenarios.csv", ["growth_per_year", "probability"], OTTAWA_GROWTH)


# ---------------------------------------------------------------------------
# Days 64-67: a vendor's demo e-truck last February (Bayesian updating), and the delivery network's vehicles,
# delivery frequencies and chargers for multi-objective planning (cost, CO2, service). Course assumptions.
# ---------------------------------------------------------------------------
EV_RANGE = [("theta_good", 0.95), ("theta_fair", 0.85), ("theta_poor", 0.65), ("penalty_per_point", 75_000),
            ("theta_no_penalty", 0.95), ("cold_day_below_c", -10), ("typical_cold_share", 0.55),
            ("pilot_cost_per_truck", 30_000), ("pilot_days", 90)]
VEHICLES = [("diesel", 1.10, 180, 0.95, 9999), ("hvo", 1.32, 180, 0.19, 9999), ("electric", 0.70, 300, 0.10, 240)]
FREQUENCIES = [(3, 0.048), (4, 0.034), (5, 0.026), (6, 0.022)]
CHARGERS = [("DC-MIS", 60), ("DC-OTT", 25), ("DC-MTL", 50)]
NETWORK_CARBON = [("pallets_per_truck", 26), ("current_frequency", 5), ("current_vehicle", "diesel"),
                  ("target_2030_cut", 0.40), ("hvo_max_km_share", 0.35)]


def make_days_64_67():
    rng = random.Random(2064)
    write("ev_range_prior.csv", ["setting", "value"], EV_RANGE)
    rows = []
    for k in range(30):
        day = datetime.date(2026, 2, 2) + datetime.timedelta(days=k)
        temp = round(rng.gauss(-8, 6), 1)
        ok = rng.random() < (0.72 if temp < -10 else 0.92)
        rows.append((day.isoformat(), temp, round(rng.uniform(170, 230), 1), int(ok)))
    write("ev_demo_log.csv", ["demo_date", "min_temp_c", "route_km", "completed_without_charge"], rows)
    write("fleet_vehicle_types.csv", ["vehicle", "cost_per_km", "fixed_per_trip", "co2_kg_per_km", "max_round_trip_km"],
          VEHICLES)
    write("delivery_frequency.csv", ["deliveries_per_week", "stockout_rate"], FREQUENCIES)
    write("dc_chargers.csv", ["dc_id", "electric_trips_per_week"], CHARGERS)
    write("network_carbon_settings.csv", ["setting", "value"], NETWORK_CARBON)


# ---------------------------------------------------------------------------
# Days 68-70: price elasticities (as estimated in the Applied Causal Inference course), the yule-log markdown,
# the evening croissant markdown, and the 2030 delivery network. Course assumptions.
# ---------------------------------------------------------------------------
ELASTICITIES = [("bread", -1.6, 0.25), ("viennoiserie", -2.1, 0.30), ("cake", -2.8, 0.45), ("cookie", -2.4, 0.35),
                ("yule_log", -1.8, 0.30)]
PRICE_LADDER = [(-0.10,), (-0.05,), (0.0,), (0.05,), (0.10,)]
YULE_DAYS = [("2026-12-21", 0.20), ("2026-12-22", 0.25), ("2026-12-23", 0.30), ("2026-12-24", 0.25)]
YULE_PRICES = [(32.0,), (28.0,), (24.0,), (20.0,)]
EVENING = [("evening_share_of_demand", 0.15), ("waiting_share_per_10pct", 0.05), ("croissants_left_at_6pm", 4.0)]
CARBON_2030 = [("growth_per_year", 0.03), ("years", 4), ("electric_fixed_per_trip_2030", 270),
               ("charger_trips_per_bank", 20), ("charger_bank_cost_per_week", 1_150)]
CARBON_2030_STRESS = [("as_planned", 0.03, 0.35, 270), ("hvo_scarce", 0.03, 0.20, 270),
                      ("lease_not_cut", 0.03, 0.35, 300), ("fast_growth", 0.05, 0.35, 270)]


def make_days_68_70():
    write("price_elasticities.csv", ["category", "elasticity", "std_error"], ELASTICITIES)
    write("price_ladder.csv", ["price_change"], PRICE_LADDER)
    write("yule_markdown_days.csv", ["sale_date", "share_of_season_demand"], YULE_DAYS)
    write("yule_markdown_prices.csv", ["price"], YULE_PRICES)
    write("evening_markdown.csv", ["setting", "value"], EVENING)
    write("carbon_2030_settings.csv", ["setting", "value"], CARBON_2030)
    write("carbon_2030_stress.csv", ["scenario", "growth_per_year", "hvo_max_km_share", "electric_fixed_per_trip"],
          CARBON_2030_STRESS)



# ---------------------------------------------------------------------------
# Days 78-80: this week's plan override requests and the approval rules (Day 78); the DC-MTL flood, the recovery
# options and the doubled internal carbon price (Day 80). Course assumptions.
# ---------------------------------------------------------------------------
OVERRIDE_REQUESTS = [
    ("R1", "Amelie Roy", "store operations", "BK-CR", "at_least", 34, "Montreal stores run out of croissants by 10 am on Saturdays"),
    ("R2", "Raj Mehta", "planning", "BK-07", "at_least", 8, "Italian festival near S12-S14 this weekend"),
    ("R3", "Lucie Gagnon", "cafe accounts", "BK-WW", "at_least", 55, "Cafe Laurier asked for extra whole wheat on Friday"),
    ("R4", "Marc Dube", "bakery", "BK-SD", "at_most", 25, "starter tank cleaning on Thursday"),
    ("R5", "Priya Shah", "finance", "BK-19", "at_least", 4, "launch promised to the board"),
    ("R6", "Tom Leblanc", "category management", "BK-12", "at_most", 10, "slow sales in the last two weeks"),
]
APPROVAL_RULES = [("automatic", 250, "none"), ("planning lead", 1_000, "Head of Planning"),
                  ("director", 1_000_000, "Director of Operations")]
FLOOD_SCENARIOS = [(2, 0.30), (3, 0.45), (4, 0.25)]
RECOVERY_OPTIONS = [
    ("laval_temp", "TMP-LAV", 900, 900, 200, 150_000, 18_000, 2, "temporary warehouse in Laval (NEW-LAV lanes)"),
    ("mis_weekend", "DC-MIS", 350, 350, 80, 0, 12_000, 1, "weekend shifts at Mississauga"),
    ("ott_weekend", "DC-OTT", 150, 150, 40, 0, 6_000, 1, "weekend shifts at Ottawa"),
]
CAPSTONE = [("flood_dc", "DC-MTL"), ("first_week_back_capacity", 0.5), ("inspection_report_week", 2),
            ("carbon_price_now", 650), ("carbon_price_new", 1_300)]


def make_days_78_80():
    write("override_requests.csv", ["request_id", "requested_by", "team", "sku", "kind", "batches", "reason"],
          OVERRIDE_REQUESTS)
    write("approval_rules.csv", ["level", "max_cost_per_week", "approver"], APPROVAL_RULES)
    write("flood_scenarios.csv", ["weeks_closed", "probability"], FLOOD_SCENARIOS)
    write("recovery_options.csv", ["option", "dc_id", "extra_receiving", "extra_outbound", "extra_chilled", "setup_cost",
                                   "weekly_cost", "first_week", "description"], RECOVERY_OPTIONS)
    write("capstone_settings.csv", ["setting", "value"], CAPSTONE)


if __name__ == "__main__":
    main()
