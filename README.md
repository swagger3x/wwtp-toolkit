# WWTP Data Toolkit
## EPA Wastewater Treatment Plant Intelligence Platform

A complete Python ETL and analysis toolkit for extracting, filtering, cross-referencing, and mapping US wastewater treatment plant data from EPA databases.

---

## Architecture

```
wwtp_toolkit/
├── epa_extractor.py      # ETL: EPA ECHO + Envirofacts + FRS + Census APIs
├── facility_filter.py    # Criteria engine: build any filtered facility list
├── wwtp_mapper.py        # Maps: interactive Folium + static Matplotlib
├── demo.py               # Sample data generator + full workflow demo
└── output/
    ├── all_facilities.csv
    ├── lists/            # Filtered CSVs
    ├── maps/             # HTML interactive + PNG static maps
    └── reports/          # Aggregate statistics
```

---

## Data Sources

| Source | What it provides | Module |
|--------|-----------------|--------|
| **EPA ECHO** | Facility locations, permit status, violation counts, design flow | `EPAExtractor.get_facilities()` |
| **EPA Envirofacts** | Permit effluent limits (BOD, TSS, ammonia, etc.), DMR data | `EPAExtractor.get_permit_limits()` |
| **EPA FRS** | Canonical registry IDs, precise GPS coordinates | `EPAExtractor.enrich_coordinates()` |
| **US Census ACS** | Population, income, poverty rate by state/county | `EPAExtractor.add_census_demographics()` |

---

## Quick Start

### Installation
```bash
pip install requests pandas folium geopandas matplotlib census us openpyxl
```

### 1. Pull live EPA data

```python
from epa_extractor import EPAExtractor

extractor = EPAExtractor(census_api_key="YOUR_KEY")  # key is optional

# Get all active POTWs in Texas
df = extractor.get_facilities(
    state="TX",
    fac_type="POTWs",
    active_only=True,
)

# Get all states (national dataset, ~10-15 min)
df_national = extractor.get_all_states(
    output_dir="data/by_state",
    combined_output="data/all_facilities.csv",
)

# Enrich with permit limits for specific plants
limits = extractor.get_permit_limits(
    npdes_ids=["TX0001234", "TX0056789"],
    parameters=["BOD", "TSS", "Ammonia", "Total Nitrogen"],
)

# Pull 3 years of Discharge Monitoring Reports
dmrs = extractor.get_discharge_monitoring(
    npdes_ids=["TX0001234"],
    years=3,
)

# Add Census demographics
df = extractor.add_census_demographics(df, year=2022)
```

### 2. Filter by criteria

```python
from facility_filter import FacilityFilter

fc = FacilityFilter(df)

# ── Single filters ──────────────────────────────────────────
large_plants     = fc.large_plants(min_flow_mgd=10.0)
violators        = fc.major_violators(years=3, min_violations=5)
texas            = fc.by_state("TX")
mississippi      = fc.by_receiving_water("Mississippi River")
near_houston     = fc.near_point(lat=29.76, lon=-95.36, radius_miles=50)
small_clean      = fc.clean_record()

# ── Combined criteria dict ──────────────────────────────────
results = fc.by_criteria({
    "state":              ["TX", "LA", "MS"],   # multi-state
    "min_flow_mgd":       1.0,                  # at least 1 MGD
    "max_violations_3yr": 2,                    # low violation history
    "active_only":        True,                 # active permits only
    "is_major":           False,                # exclude major facilities
    "receiving_water":    "Gulf",               # Gulf coast dischargers
})

# ── Summary views ───────────────────────────────────────────
fc.summary_by_state()       # count, flow, violations by state
fc.summary_by_flow_tier()   # micro / small / medium / large / major
fc.top_polluters(n=25)      # 25 worst violators

# ── Export ──────────────────────────────────────────────────
fc.export_list(results, "output/lists/gulf_potws.csv", fmt="csv")
fc.export_list(results, "output/lists/gulf_potws.xlsx", fmt="excel")
```

### 3. Generate maps

```python
from wwtp_mapper import WWTPMapper

mapper = WWTPMapper(df)

# Interactive national map (open in any browser)
mapper.national_map("output/maps/national.html", color_by="flow")
mapper.national_map("output/maps/violations.html", color_by="violations")

# Violation density heatmap
mapper.violation_heatmap("output/maps/heatmap.html")

# State-level drill-down
mapper.state_map("TX", "output/maps/texas.html")
mapper.state_map("CA", "output/maps/california.html")

# Static choropleth (for reports/presentations)
mapper.static_choropleth("facility_count", "output/maps/count.png")
mapper.static_choropleth("total_flow_mgd", "output/maps/capacity.png")
mapper.static_choropleth("total_violations", "output/maps/violations.png")

# Census overlay (requires demographic enrichment first)
mapper.census_overlay_map("census_median_income", "output/maps/income_overlay.html")
```

---

## Filter Criteria Reference

| Key | Type | Description |
|-----|------|-------------|
| `state` | str or list | State abbreviation(s): `"TX"` or `["TX","LA"]` |
| `city` | str | City name substring match |
| `min_flow_mgd` | float | Minimum design flow (million gallons/day) |
| `max_flow_mgd` | float | Maximum design flow |
| `flow_tier` | str | `micro` / `small` / `medium` / `large` / `major` |
| `min_violations_3yr` | int | Minimum violations in past 3 years |
| `max_violations_3yr` | int | Maximum violations (0 = clean record) |
| `is_major` | bool | EPA-designated major NPDES facility |
| `active_only` | bool | Active permits only |
| `receiving_water` | str | Receiving water body substring |
| `facility_type` | str | POTW, INDUSTRIAL, etc. |
| `lat_min/lat_max/lon_min/lon_max` | float | Geographic bounding box |

**Flow tiers (EPA definitions):**

| Tier | Range |
|------|-------|
| Micro | < 0.1 MGD |
| Small | 0.1 – 1.0 MGD |
| Medium | 1 – 10 MGD |
| Large | 10 – 100 MGD |
| Major | > 100 MGD |

---

## Census Cross-Reference

Get a free Census API key at: https://api.census.gov/data/key_signup.html

Default ACS variables pulled:
- `B01003_001E` — Total population
- `B19013_001E` — Median household income  
- `B17001_002E` — Population below poverty level

---

## Map Types

| Map | Format | Description |
|-----|--------|-------------|
| National overview | HTML (interactive) | All 2,500+ facilities, clustered, colored by flow or violations |
| State drill-down | HTML (interactive) | Single state, toggle layers by size tier |
| Violation heatmap | HTML (interactive) | Density weighted by violation count |
| Choropleth | PNG (static) | State-level aggregates, suitable for reports |
| Census overlay | HTML (interactive) | Facilities over demographic background |

---

## Common Workflows

### Find all plants near a specific river for watershed analysis
```python
river_plants = fc.by_receiving_water("Columbia River")
mapper = WWTPMapper(river_plants)
mapper.national_map("columbia_river.html")
```

### Identify environmental justice candidates
```python
# Small plants in low-income areas with violations
ej_candidates = fc.by_criteria({
    "max_flow_mgd": 1.0,
    "min_violations_3yr": 3,
})
# If census data joined:
ej_candidates = ej_candidates[ej_candidates["census_median_income"] < 45000]
```

### State compliance report
```python
state_df = fc.by_state("FL")
state_fc = FacilityFilter(state_df)
print(state_fc.summary_by_flow_tier())
print(state_fc.top_polluters(n=10))
state_fc.export_list(
    state_fc.major_violators(min_violations=3),
    "florida_violators.csv"
)
```

### National batch pull (all 50 states)
```python
extractor = EPAExtractor()
df_all = extractor.get_all_states(
    combined_output="data/national_wwtp.csv"
)  # ~15-30 minutes
```
