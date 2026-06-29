"""
WWTP Toolkit — Demo & Sample Data Generator
============================================
This module does two things:
  1. Generates realistic sample data so you can explore the toolkit
     without hitting the live EPA API (useful for offline dev/testing).
  2. Shows a complete end-to-end workflow demonstrating all key features.

Run:
    python demo.py

To use live EPA data instead, replace `load_sample_data()` with:
    from epa_extractor import EPAExtractor
    extractor = EPAExtractor()
    df = extractor.get_facilities(state="TX", fac_type="POTWs")
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from pathlib import Path

# Ensure output dirs exist
for d in ("output/maps", "output/lists", "output/reports"):
    Path(d).mkdir(parents=True, exist_ok=True)


# ── Sample data generator ─────────────────────────────────────────────────────

STATES = [
    "TX","CA","FL","NY","PA","OH","IL","GA","NC","MI",
    "VA","WA","AZ","MA","TN","IN","MO","MD","WI","CO",
    "MN","SC","AL","LA","KY","OR","OK","CT","IA","UT",
    "NV","AR","MS","KS","NM","NE","WV","ID","HI","NH",
    "ME","RI","MT","DE","SD","ND","AK","VT","WY","DC",
]

RECEIVING_WATERS = [
    "Mississippi River","Colorado River","Rio Grande","Hudson River",
    "Potomac River","Delaware River","Columbia River","Missouri River",
    "Ohio River","Tennessee River","Sacramento River","Snake River",
    "Chesapeake Bay","San Francisco Bay","Lake Michigan","Lake Erie",
    "Gulf of Mexico","Atlantic Ocean","Pacific Ocean","Unnamed Tributary",
    "Platte River","Arkansas River","Red River","Savannah River",
    "Cape Fear River","Neuse River","Susquehanna River","Willamette River",
]

PLANT_SUFFIXES = [
    "WWTP","Water Reclamation Facility","Wastewater Treatment Plant",
    "Regional WRF","Sanitation District","Water Pollution Control Plant",
    "Water Resource Recovery Facility","Sewage Treatment Plant",
]

STATE_CITIES = {
    "TX": ["Houston","San Antonio","Dallas","Austin","Fort Worth","El Paso","Arlington","Plano"],
    "CA": ["Los Angeles","San Diego","San Jose","San Francisco","Fresno","Sacramento","Long Beach"],
    "FL": ["Jacksonville","Miami","Tampa","Orlando","St. Petersburg","Hialeah","Tallahassee"],
    "NY": ["New York","Buffalo","Rochester","Yonkers","Syracuse","Albany","New Rochelle"],
    "OH": ["Columbus","Cleveland","Cincinnati","Toledo","Akron","Dayton","Parma"],
}

np.random.seed(42)


def load_sample_data(n: int = 2500) -> pd.DataFrame:
    """
    Generate n synthetic WWTP records that mirror the EPA ECHO schema.
    Distributions reflect real-world patterns (log-normal flows, etc.)
    """
    state_choices = np.random.choice(STATES, n, replace=True,
                                     p=_state_weights(n))

    # Log-normal flow distribution (most plants are small; a few are massive)
    flows = np.random.lognormal(mean=0.5, sigma=1.8, size=n)
    flows = np.clip(flows, 0.01, 800)

    # Violations: Poisson, higher for smaller/older plants
    base_violations = np.random.poisson(lam=1.2, size=n)
    # Larger plants have more resources for compliance
    violation_adjustment = (flows < 0.5).astype(int)
    violations = np.clip(base_violations + violation_adjustment * np.random.poisson(1, n), 0, 30)

    # Lat/lon — rough US bounds with state-level clustering
    lats = np.random.uniform(25, 48, n)
    lons = np.random.uniform(-124, -67, n)

    # 30% are designated "major" NPDES facilities
    is_major = np.random.choice(["Y", "N"], n, p=[0.3, 0.7])

    records = []
    for i in range(n):
        state = state_choices[i]
        city_pool = STATE_CITIES.get(state, [f"{state} City", f"{state} Town", f"Springfield"])
        city = np.random.choice(city_pool)
        suffix = np.random.choice(PLANT_SUFFIXES)
        name = f"{city} {suffix}"
        npdes_id = f"{state}{np.random.randint(1000000, 9999999):07d}"
        flow = flows[i]

        records.append({
            "registry_id":       f"110{i:07d}",
            "facility_name":     name,
            "address":           f"{np.random.randint(1,9999)} Industrial Blvd",
            "city":              city,
            "state":             state,
            "zip":               f"{np.random.randint(10000,99999)}",
            "latitude":          round(lats[i], 6),
            "longitude":         round(-lons[i], 6),
            "npdes_ids":         npdes_id,
            "facility_type":     "POTW",
            "active":            np.random.choice(["Y","N"], p=[0.92, 0.08]),
            "permit_status":     np.random.choice(["EFF","EXP","PND"], p=[0.88, 0.08, 0.04]),
            "design_flow_mgd":   round(flow, 3),
            "violations_3yr":    int(violations[i]),
            "is_major":          is_major[i],
            "receiving_waters":  np.random.choice(RECEIVING_WATERS),
            "dfr_url":           f"https://echo.epa.gov/detailed-facility-report?fid=110{i:07d}",
            "border_facility":   "N",
            "tribal_land":       "N",
        })

    df = pd.DataFrame(records)
    print(f"Generated {len(df):,} sample WWTP records.")
    return df


def _state_weights(n: int) -> np.ndarray:
    """Approximate state populations as weights so bigger states get more plants."""
    pop = [30,39,22,20,13,12,13,11,10,10,  # TX CA FL NY PA OH IL GA NC MI
           8,8,7,7,7,7,6,6,6,6,            # VA WA AZ MA TN IN MO MD WI CO
           6,5,5,5,5,4,4,4,4,3,            # MN SC AL LA KY OR OK CT IA UT
           3,3,3,3,1,1,1,1,1,1,            # NV AR MS KS NM NE WV ID HI NH
           1,1,1,1,1,1,1,1,1,1,]           # ME RI MT DE SD ND AK VT WY DC
    arr = np.array(pop[:len(STATES)], dtype=float)
    return arr / arr.sum()


# ── Full demo workflow ────────────────────────────────────────────────────────

def run_demo():
    from facility_filter import FacilityFilter
    from wwtp_mapper import WWTPMapper

    print("\n" + "="*60)
    print("  WWTP TOOLKIT DEMO")
    print("="*60 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("STEP 1: Loading data (sample / replace with EPAExtractor for live data)")
    # df = load_sample_data(2500)
    from epa_extractor import EPAExtractor
    extractor = EPAExtractor()
    df = extractor.get_facilities(state="CO", fac_type="POTWs")
    df.to_csv("output/all_facilities.csv", index=False)
    print(f"  ✓ {len(df):,} facilities loaded\n")

    # ── 2. Summary ────────────────────────────────────────────────────────────
    print("STEP 2: National summary")
    fc = FacilityFilter(df)
    state_summary = fc.summary_by_state()
    state_summary.to_csv("output/reports/summary_by_state.csv")
    print(f"  Top 5 states by facility count:\n{state_summary.head()}\n")

    tier_summary = fc.summary_by_flow_tier()
    tier_summary.to_csv("output/reports/summary_by_flow_tier.csv", index=False)
    print(f"  Facilities by size tier:\n{tier_summary.to_string(index=False)}\n")

    # ── 3. Filtered lists ─────────────────────────────────────────────────────
    print("STEP 3: Creating filtered lists")

    # A) Large active plants in Texas
    texas_large = fc.by_criteria({
        "state": "TX",
        "min_flow_mgd": 5.0,
        "active_only": True,
    })
    fc.export_list(texas_large, "output/lists/texas_large_plants.csv")
    print(f"  ✓ TX large plants (>5 MGD): {len(texas_large)}")

    # B) Major violators nationwide
    violators = fc.major_violators(years=3, min_violations=5)
    fc.export_list(violators, "output/lists/major_violators_nationwide.csv")
    print(f"  ✓ Major violators (5+ in 3yr): {len(violators)}")

    # C) Small plants with clean record (compliance leaders)
    clean_small = fc.by_criteria({
        "max_flow_mgd": 0.5,
        "max_violations_3yr": 0,
        "active_only": True,
    })
    fc.export_list(clean_small, "output/lists/small_plants_clean_record.csv")
    print(f"  ✓ Small clean-record plants: {len(clean_small)}")

    # D) Plants discharging to Mississippi River
    mississippi = fc.by_receiving_water("Mississippi")
    fc.export_list(mississippi, "output/lists/mississippi_river_dischargers.csv")
    print(f"  ✓ Mississippi River dischargers: {len(mississippi)}")

    # E) Gulf Coast plants (bounding box)
    gulf_coast = fc.by_criteria({
        "lat_min": 25.5, "lat_max": 31.0,
        "lon_min": -97.5, "lon_max": -80.5,
        "min_flow_mgd": 0.1,
    })
    fc.export_list(gulf_coast, "output/lists/gulf_coast_plants.csv")
    print(f"  ✓ Gulf Coast plants: {len(gulf_coast)}")

    # F) Top 25 largest plants
    top25 = fc.top_polluters(n=25, metric="design_flow_mgd")
    fc.export_list(top25, "output/lists/top25_largest_plants.csv")
    print(f"  ✓ Top 25 largest plants saved")

    print()

    # ── 4. Maps ───────────────────────────────────────────────────────────────
    print("STEP 4: Generating maps")
    mapper = WWTPMapper(df)

    # National map colored by flow tier
    mapper.national_map("output/maps/national_by_flow.html", color_by="flow", cluster=True)

    # National map colored by violation history
    mapper.national_map("output/maps/national_by_violations.html", color_by="violations", cluster=True)

    # Violation heatmap
    mapper.violation_heatmap("output/maps/violation_heatmap.html")

    # State maps for top 5 states
    for state in ["TX", "CA", "FL", "NY", "OH"]:
        mapper.state_map(state, f"output/maps/{state.lower()}_facilities.html")

    # Static choropleth
    mapper.static_choropleth("facility_count", "output/maps/choropleth_count.png")
    mapper.static_choropleth("total_flow_mgd", "output/maps/choropleth_flow.png")

    print()

    # ── 5. Summary report ─────────────────────────────────────────────────────
    print("STEP 5: Summary report")
    top_violator_states = (
        df.groupby("state")["violations_3yr"]
        .agg(["sum", "mean", "count"])
        .rename(columns={"sum": "total_violations", "mean": "avg_violations", "count": "facilities"})
        .sort_values("total_violations", ascending=False)
        .head(10)
    )
    top_violator_states.to_csv("output/reports/top_violator_states.csv")
    print(f"  Top 10 states by violations:\n{top_violator_states.to_string()}\n")

    print("="*60)
    print("  DEMO COMPLETE")
    print("="*60)
    print("\nOutput files:")
    print("  output/all_facilities.csv         — Full facility dataset")
    print("  output/lists/                      — 6 filtered lists")
    print("  output/maps/national_by_flow.html  — National interactive map")
    print("  output/maps/national_by_violations.html")
    print("  output/maps/violation_heatmap.html")
    print("  output/maps/[state]_facilities.html — 5 state maps")
    print("  output/maps/choropleth_*.png        — Static choropleth maps")
    print("  output/reports/                    — Summary statistics CSVs")
    print("\nNext steps:")
    print("  1. Replace load_sample_data() with EPAExtractor() for live data")
    print("  2. Add your Census API key for demographic cross-referencing")
    print("  3. Use FacilityFilter.by_criteria() to build any custom list")


if __name__ == "__main__":
    run_demo()
