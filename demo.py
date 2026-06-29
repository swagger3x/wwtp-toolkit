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


# ── Index page builder ───────────────────────────────────────────────────────

def _build_index(df, tier_summary):
    """Write output/index.html reflecting the current run's actual counts."""
    from datetime import datetime
    import os

    total = len(df)
    with_flow = int(df["design_flow_mgd"].notna().sum()) if "design_flow_mgd" in df.columns else 0
    with_viols = int((df["violations_3yr"] > 0).sum()) if "violations_3yr" in df.columns else 0
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Collect optional state map links
    state_map_links = ""
    for f in sorted(Path("output/maps").glob("*_facilities.html")):
        state = f.stem.replace("_facilities", "").upper()
        size = os.path.getsize(f) // 1024
        state_map_links += (
            f'<div class="card map-card"><a href="maps/{f.name}" target="_blank">'
            f'<div class="card-body"><h3>{state} State Map</h3>'
            f'<p>Interactive facility map for {state}. ({size} KB)</p>'
            f'<span class="badge">Open in browser →</span></div></a></div>\n'
        )

    # Tier summary rows
    tier_rows = ""
    if not tier_summary.empty:
        for _, row in tier_summary.iterrows():
            tier_rows += (
                f"<tr><td>{row['tier']}</td><td>{row['flow_range']}</td>"
                f"<td>{int(row['count'])}</td>"
                f"<td>{row['total_flow_mgd']:.1f}</td>"
                f"<td>{row['avg_violations']:.2f}</td></tr>\n"
            )
    else:
        tier_rows = '<tr><td colspan="5" style="text-align:center;color:#999">No flow tier data</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>WWTP Toolkit — Output</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f0f4f8; color: #1a202c; min-height: 100vh; }}
    header {{ background: #1a365d; color: white; padding: 28px 40px; box-shadow: 0 2px 8px rgba(0,0,0,.3); }}
    header h1 {{ font-size: 1.6rem; font-weight: 700; }}
    header p  {{ margin-top: 4px; opacity: .75; font-size: .88rem; }}
    main {{ max-width: 1100px; margin: 36px auto; padding: 0 24px 60px; }}
    h2 {{ font-size: .95rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;
          color: #2d3748; margin: 36px 0 14px; padding-bottom: 8px; border-bottom: 2px solid #bee3f8; }}
    .stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }}
    .stat-box {{ background: white; border-radius: 10px; padding: 18px 24px; flex: 1; min-width: 140px;
                 box-shadow: 0 1px 4px rgba(0,0,0,.1); text-align: center; }}
    .stat-box .num {{ font-size: 2rem; font-weight: 700; color: #2b6cb0; }}
    .stat-box .lbl {{ font-size: .78rem; color: #718096; margin-top: 3px; }}
    .grid {{ display: grid; gap: 16px; }}
    .grid-2 {{ grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); }}
    .card {{ background: white; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.1);
             overflow: hidden; transition: box-shadow .15s, transform .15s; }}
    .card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,.15); transform: translateY(-2px); }}
    .card-img img {{ width: 100%; display: block; border-bottom: 1px solid #e2e8f0; background: #edf2f7; }}
    .card-body {{ padding: 16px; }}
    .card-body h3 {{ font-size: .95rem; font-weight: 600; color: #2b6cb0; margin-bottom: 4px; }}
    .card-body p  {{ font-size: .8rem; color: #718096; line-height: 1.5; }}
    .card a {{ text-decoration: none; color: inherit; display: block; }}
    .map-card .card-body::before {{ content: "🗺"; font-size: 1.3rem; display: block; margin-bottom: 8px; }}
    .csv-card {{ display: flex; align-items: center; gap: 14px; padding: 14px 18px; }}
    .csv-card .icon {{ width: 38px; height: 38px; min-width: 38px; background: #ebf8ff; border-radius: 8px;
                       display: flex; align-items: center; justify-content: center; font-size: 1.1rem; }}
    .csv-card .text h3 {{ font-size: .9rem; font-weight: 600; color: #2b6cb0; }}
    .csv-card .text p  {{ font-size: .78rem; color: #718096; margin-top: 2px; }}
    .badge {{ display: inline-block; background: #ebf8ff; color: #2b6cb0; font-size: .7rem;
              font-weight: 600; padding: 2px 8px; border-radius: 999px; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px;
             overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.1); font-size: .85rem; }}
    th {{ background: #ebf8ff; color: #2b6cb0; padding: 10px 14px; text-align: left;
          font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; }}
    td {{ padding: 9px 14px; border-top: 1px solid #e2e8f0; color: #4a5568; }}
    tr:hover td {{ background: #f7fafc; }}
    footer {{ text-align: center; padding: 20px; font-size: .78rem; color: #a0aec0; }}
  </style>
</head>
<body>

<header>
  <h1>WWTP Toolkit — Output Dashboard</h1>
  <p>US Wastewater Treatment Plant data &mdash; EPA ECHO &amp; CWA live APIs &mdash; Generated {generated}</p>
</header>

<main>

  <h2>This Run</h2>
  <div class="stats">
    <div class="stat-box"><div class="num">{total:,}</div><div class="lbl">Facilities loaded</div></div>
    <div class="stat-box"><div class="num">{with_flow:,}</div><div class="lbl">With design flow data</div></div>
    <div class="stat-box"><div class="num">{with_viols:,}</div><div class="lbl">With violations (3yr)</div></div>
  </div>

  <h2>Flow Tier Breakdown</h2>
  <table>
    <thead><tr><th>Tier</th><th>Flow Range</th><th>Count</th><th>Total Flow (MGD)</th><th>Avg Violations</th></tr></thead>
    <tbody>{tier_rows}</tbody>
  </table>

  <h2>Full Dataset</h2>
  <div class="grid grid-2">
    <div class="card"><a href="all_facilities.csv" download>
      <div class="csv-card">
        <div class="icon">📋</div>
        <div class="text">
          <h3>all_facilities.csv</h3>
          <p>Complete {total:,}-row export with GPS, NPDES IDs, flow, violations, permit status.</p>
          <span class="badge">⬇ Download CSV</span>
        </div>
      </div>
    </a></div>
  </div>

  <h2>Interactive Maps</h2>
  <div class="grid grid-2">
    <div class="card map-card"><a href="maps/national_by_flow.html" target="_blank">
      <div class="card-body">
        <h3>National Map — Flow Capacity</h3>
        <p>Markers colored by size tier: blue (micro) → red (major). Click any dot for details.</p>
        <span class="badge">Open in browser →</span>
      </div>
    </a></div>
    <div class="card map-card"><a href="maps/national_by_violations.html" target="_blank">
      <div class="card-body">
        <h3>National Map — Violation History</h3>
        <p>Markers colored by 3-year compliance record: green (clean) → red (high violations).</p>
        <span class="badge">Open in browser →</span>
      </div>
    </a></div>
    <div class="card map-card"><a href="maps/violation_heatmap.html" target="_blank">
      <div class="card-body">
        <h3>Violation Density Heatmap</h3>
        <p>Heat map weighted by violation count — shows geographic clusters of non-compliance.</p>
        <span class="badge">Open in browser →</span>
      </div>
    </a></div>
    {state_map_links}
  </div>

  <h2>Static Choropleth Maps</h2>
  <div class="grid grid-2">
    <div class="card card-img"><a href="maps/choropleth_count.png" target="_blank">
      <img src="maps/choropleth_count.png" alt="Choropleth — Facility Count" />
      <div class="card-body"><h3>Facility Count by State</h3><p>Bubble size = facilities per state.</p></div>
    </a></div>
    <div class="card card-img"><a href="maps/choropleth_flow.png" target="_blank">
      <img src="maps/choropleth_flow.png" alt="Choropleth — Total Flow" />
      <div class="card-body"><h3>Total Flow Capacity by State</h3><p>Bubble size = combined MGD per state.</p></div>
    </a></div>
  </div>

  <h2>Summary Reports</h2>
  <div class="grid grid-2">
    <div class="card"><a href="reports/summary_by_state.csv" download><div class="csv-card">
      <div class="icon">📊</div>
      <div class="text"><h3>summary_by_state.csv</h3>
      <p>Facility count, total flow, violations — grouped by state.</p>
      <span class="badge">⬇ Download CSV</span></div>
    </div></a></div>
    <div class="card"><a href="reports/summary_by_flow_tier.csv" download><div class="csv-card">
      <div class="icon">📊</div>
      <div class="text"><h3>summary_by_flow_tier.csv</h3>
      <p>Count and avg violations broken out by EPA size tier.</p>
      <span class="badge">⬇ Download CSV</span></div>
    </div></a></div>
    <div class="card"><a href="reports/top_violator_states.csv" download><div class="csv-card">
      <div class="icon">⚠️</div>
      <div class="text"><h3>top_violator_states.csv</h3>
      <p>Top 10 states by total violation count over 3 years.</p>
      <span class="badge">⬇ Download CSV</span></div>
    </div></a></div>
  </div>

  <h2>Filtered Lists</h2>
  <div class="grid grid-2">
    <div class="card"><a href="lists/major_violators_nationwide.csv" download><div class="csv-card">
      <div class="icon">🚨</div>
      <div class="text"><h3>major_violators_nationwide.csv</h3>
      <p>Facilities with 5+ violations in the past 3 years.</p>
      <span class="badge">⬇ Download CSV</span></div>
    </div></a></div>
    <div class="card"><a href="lists/top25_largest_plants.csv" download><div class="csv-card">
      <div class="icon">🏭</div>
      <div class="text"><h3>top25_largest_plants.csv</h3>
      <p>Top 25 facilities by design flow capacity (MGD).</p>
      <span class="badge">⬇ Download CSV</span></div>
    </div></a></div>
    <div class="card"><a href="lists/texas_large_plants.csv" download><div class="csv-card">
      <div class="icon">💧</div>
      <div class="text"><h3>texas_large_plants.csv</h3>
      <p>Active TX facilities with design flow over 5 MGD.</p>
      <span class="badge">⬇ Download CSV</span></div>
    </div></a></div>
    <div class="card"><a href="lists/small_plants_clean_record.csv" download><div class="csv-card">
      <div class="icon">✅</div>
      <div class="text"><h3>small_plants_clean_record.csv</h3>
      <p>Small active facilities (under 0.5 MGD) with zero violations.</p>
      <span class="badge">⬇ Download CSV</span></div>
    </div></a></div>
    <div class="card"><a href="lists/mississippi_river_dischargers.csv" download><div class="csv-card">
      <div class="icon">🌊</div>
      <div class="text"><h3>mississippi_river_dischargers.csv</h3>
      <p>Facilities permitted to discharge into the Mississippi River.</p>
      <span class="badge">⬇ Download CSV</span></div>
    </div></a></div>
    <div class="card"><a href="lists/gulf_coast_plants.csv" download><div class="csv-card">
      <div class="icon">🌊</div>
      <div class="text"><h3>gulf_coast_plants.csv</h3>
      <p>Facilities within the Gulf Coast bounding box with flow over 0.1 MGD.</p>
      <span class="badge">⬇ Download CSV</span></div>
    </div></a></div>
  </div>

</main>
<footer>WWTP Toolkit &mdash; data from EPA ECHO &amp; CWA &mdash; {generated}</footer>
</body>
</html>"""

    Path("output/index.html").write_text(html, encoding="utf-8")
    print(f"  ✓ output/index.html generated")


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

    # ── 6. Generate index.html ────────────────────────────────────────────────
    _build_index(df, tier_summary)

    print("="*60)
    print("  DEMO COMPLETE")
    print("="*60)
    print("\n  → Open output/index.html in a browser (or serve the output/ folder)")
    print("    to browse all maps, reports, and filtered lists.\n")


if __name__ == "__main__":
    run_demo()
