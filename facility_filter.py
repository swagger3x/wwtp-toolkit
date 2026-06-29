"""
WWTP Facility Filter & Criteria Engine
=======================================
Build lists of plants meeting any combination of criteria:
  - Flow capacity ranges
  - Violation history
  - Permit status
  - Geographic (state, watershed, proximity)
  - Receiving water body
  - Facility type (POTW, industrial, etc.)
  - Census demographics

Usage:
    fc = FacilityFilter(df)
    results = fc.major_violators(years=3, min_violations=5)
    results = fc.large_plants(min_flow_mgd=10)
    results = fc.by_criteria({
        "state": ["TX", "LA"],
        "min_flow_mgd": 1.0,
        "max_violations_3yr": 2,
        "is_major": True,
    })
"""

import pandas as pd
import numpy as np
from typing import Optional, Union


class FacilityFilter:
    """
    Fluent filtering interface for WWTP facility DataFrames.

    All filter methods return a new DataFrame (non-destructive).
    Chain multiple filters using the .pipe() pattern or call individually.
    """

    # NPDES size tiers (EPA definition)
    FLOW_TIERS = {
        "micro":   (0,      0.1),    # < 0.1 MGD
        "small":   (0.1,    1.0),    # 0.1–1.0 MGD
        "medium":  (1.0,    10.0),   # 1–10 MGD
        "large":   (10.0,   100.0),  # 10–100 MGD
        "major":   (100.0,  None),   # > 100 MGD
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._ensure_columns()

    def _ensure_columns(self):
        """Add missing columns with default values."""
        defaults = {
            "design_flow_mgd":   np.nan,
            "violations_3yr":    0,
            "is_major":          False,
            "active":            True,
            "state":             "",
            "facility_type":     "",
            "receiving_waters":  "",
            "permit_status":     "",
        }
        for col, val in defaults.items():
            if col not in self.df.columns:
                self.df[col] = val

        for col in ("design_flow_mgd", "latitude", "longitude"):
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        if "violations_3yr" in self.df.columns:
            self.df["violations_3yr"] = pd.to_numeric(
                self.df["violations_3yr"], errors="coerce"
            ).fillna(0).astype(int)

    # ── Basic filters ────────────────────────────────────────────────────────

    def by_state(self, states: Union[str, list]) -> pd.DataFrame:
        """Filter by one or more state abbreviations."""
        if isinstance(states, str):
            states = [states]
        states = [s.upper() for s in states]
        return self.df[self.df["state"].str.upper().isin(states)]

    def by_city(self, city: str, exact: bool = False) -> pd.DataFrame:
        """Filter by city name (case-insensitive substring match by default)."""
        col = "city" if "city" in self.df.columns else None
        if not col:
            return self.df
        if exact:
            return self.df[self.df[col].str.upper() == city.upper()]
        return self.df[self.df[col].str.contains(city, case=False, na=False)]

    def active_only(self) -> pd.DataFrame:
        """Return only facilities with active permits."""
        if "permit_status" in self.df.columns:
            return self.df[self.df["permit_status"].str.upper().isin(["EFF", "ACT", "Y", "ACTIVE"])]
        return self.df[self.df["active"].astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"])]

    # ── Flow / size filters ──────────────────────────────────────────────────

    def by_flow_tier(self, tier: str) -> pd.DataFrame:
        """
        Filter by EPA size tier: 'micro', 'small', 'medium', 'large', 'major'.
        """
        if tier not in self.FLOW_TIERS:
            raise ValueError(f"Unknown tier '{tier}'. Choose from: {list(self.FLOW_TIERS)}")
        lo, hi = self.FLOW_TIERS[tier]
        mask = self.df["design_flow_mgd"] >= lo
        if hi is not None:
            mask &= self.df["design_flow_mgd"] < hi
        return self.df[mask]

    def large_plants(self, min_flow_mgd: float = 10.0) -> pd.DataFrame:
        """Return plants with design flow ≥ min_flow_mgd million gallons/day."""
        return self.df[self.df["design_flow_mgd"] >= min_flow_mgd]

    def small_plants(self, max_flow_mgd: float = 1.0) -> pd.DataFrame:
        """Return plants with design flow ≤ max_flow_mgd MGD."""
        return self.df[self.df["design_flow_mgd"] <= max_flow_mgd]

    def by_flow_range(self, min_mgd: float = 0, max_mgd: float = None) -> pd.DataFrame:
        """Filter by exact MGD range."""
        mask = self.df["design_flow_mgd"] >= min_mgd
        if max_mgd is not None:
            mask &= self.df["design_flow_mgd"] <= max_mgd
        return self.df[mask]

    # ── Compliance / violation filters ───────────────────────────────────────

    def major_violators(self, years: int = 3, min_violations: int = 3) -> pd.DataFrame:
        """
        Return facilities with ≥ min_violations in the past N years.
        These are candidates for enforcement actions.
        """
        col = f"violations_{years}yr" if f"violations_{years}yr" in self.df.columns else "violations_3yr"
        return self.df[self.df[col] >= min_violations].sort_values(col, ascending=False)

    def clean_record(self, years: int = 3) -> pd.DataFrame:
        """Facilities with zero violations in N years."""
        col = "violations_3yr"
        return self.df[self.df[col] == 0]

    def significant_noncompliance(self) -> pd.DataFrame:
        """Return Significant Non-Compliance (SNC) facilities."""
        if "snc_flag" in self.df.columns:
            return self.df[self.df["snc_flag"].str.upper() == "Y"]
        # Proxy: facilities with many violations
        return self.major_violators(min_violations=5)

    # ── Type / classification ────────────────────────────────────────────────

    def potw_only(self) -> pd.DataFrame:
        """Publicly Owned Treatment Works only (municipal)."""
        if "facility_type" in self.df.columns:
            return self.df[self.df["facility_type"].str.upper().str.contains("POTW|MUNI|PUBLIC", na=False)]
        return self.df

    def major_npdes(self) -> pd.DataFrame:
        """EPA-designated major NPDES facilities (high impact)."""
        return self.df[self.df["is_major"].astype(str).str.upper().isin(["Y", "YES", "TRUE", "1"])]

    def by_receiving_water(self, water_name: str) -> pd.DataFrame:
        """Filter by receiving water body (river, lake, bay, etc.)."""
        return self.df[self.df["receiving_waters"].str.contains(water_name, case=False, na=False)]

    # ── Geographic filters ───────────────────────────────────────────────────

    def by_bounding_box(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> pd.DataFrame:
        """Return facilities within a geographic bounding box."""
        mask = (
            (self.df["latitude"] >= lat_min) & (self.df["latitude"] <= lat_max) &
            (self.df["longitude"] >= lon_min) & (self.df["longitude"] <= lon_max)
        )
        return self.df[mask]

    def near_point(
        self, lat: float, lon: float, radius_miles: float
    ) -> pd.DataFrame:
        """
        Return facilities within radius_miles of a coordinate.
        Uses Haversine approximation (accurate to ~0.5% for distances < 500 mi).
        """
        R = 3958.8  # Earth radius in miles
        df = self.df.dropna(subset=["latitude", "longitude"])
        lat_r = np.radians(df["latitude"])
        lon_r = np.radians(df["longitude"])
        dlat = lat_r - np.radians(lat)
        dlon = lon_r - np.radians(lon)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(lat_r) * np.sin(dlon / 2) ** 2
        dist = 2 * R * np.arcsin(np.sqrt(a))
        return df[dist <= radius_miles].assign(distance_miles=dist[dist <= radius_miles])

    # ── Combined criteria engine ─────────────────────────────────────────────

    def by_criteria(self, criteria: dict) -> pd.DataFrame:
        """
        Apply multiple filter criteria at once via a dict.

        Supported keys
        --------------
        state               : str or list of state codes
        city                : str (substring match)
        min_flow_mgd        : float
        max_flow_mgd        : float
        flow_tier           : str (micro/small/medium/large/major)
        min_violations_3yr  : int
        max_violations_3yr  : int
        is_major            : bool
        active_only         : bool
        receiving_water     : str (substring)
        lat_min/lat_max/lon_min/lon_max : float (bounding box)
        facility_type       : str (substring)

        Returns
        -------
        Filtered DataFrame, sorted by design_flow_mgd descending.

        Example
        -------
        >>> fc.by_criteria({
        ...     "state": ["TX", "LA"],
        ...     "min_flow_mgd": 5.0,
        ...     "max_violations_3yr": 2,
        ...     "is_major": True,
        ... })
        """
        result = self.df.copy()
        fc = FacilityFilter(result)

        if "state" in criteria:
            result = fc.by_state(criteria["state"])
            fc = FacilityFilter(result)

        if "city" in criteria:
            result = fc.by_city(criteria["city"])
            fc = FacilityFilter(result)

        if "flow_tier" in criteria:
            result = fc.by_flow_tier(criteria["flow_tier"])
            fc = FacilityFilter(result)
        else:
            if "min_flow_mgd" in criteria or "max_flow_mgd" in criteria:
                result = fc.by_flow_range(
                    criteria.get("min_flow_mgd", 0),
                    criteria.get("max_flow_mgd", None),
                )
                fc = FacilityFilter(result)

        if criteria.get("active_only"):
            result = fc.active_only()
            fc = FacilityFilter(result)

        if criteria.get("is_major"):
            result = fc.major_npdes()
            fc = FacilityFilter(result)

        if "min_violations_3yr" in criteria:
            result = result[result["violations_3yr"] >= criteria["min_violations_3yr"]]
        if "max_violations_3yr" in criteria:
            result = result[result["violations_3yr"] <= criteria["max_violations_3yr"]]

        if "receiving_water" in criteria:
            result = FacilityFilter(result).by_receiving_water(criteria["receiving_water"])

        if "facility_type" in criteria:
            result = result[result["facility_type"].str.contains(
                criteria["facility_type"], case=False, na=False
            )]

        bb = {k: criteria[k] for k in ("lat_min","lat_max","lon_min","lon_max") if k in criteria}
        if len(bb) == 4:
            result = FacilityFilter(result).by_bounding_box(**bb)

        if "design_flow_mgd" in result.columns:
            result = result.sort_values("design_flow_mgd", ascending=False)

        return result.reset_index(drop=True)

    # ── Summary reports ──────────────────────────────────────────────────────

    def summary_by_state(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Aggregate statistics by state.
        Returns: count, total_flow_mgd, avg_flow_mgd, total_violations, major_count.
        """
        if df is None:
            df = self.df
        _summary_cols = ["facility_count", "total_flow_mgd", "avg_flow_mgd",
                         "total_violations_3yr", "major_facility_count"]
        if df.empty or "state" not in df.columns or "facility_name" not in df.columns:
            return pd.DataFrame(columns=_summary_cols)
        return df.groupby("state").agg(
            facility_count=("facility_name", "count"),
            total_flow_mgd=("design_flow_mgd", "sum"),
            avg_flow_mgd=("design_flow_mgd", "mean"),
            total_violations_3yr=("violations_3yr", "sum"),
            major_facility_count=("is_major", lambda x: (x.astype(str).str.upper().isin(["Y","YES","TRUE","1"])).sum()),
        ).round(2).sort_values("facility_count", ascending=False)

    def summary_by_flow_tier(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Count and total flow capacity broken out by size tier."""
        if df is None:
            df = self.df
        if df.empty or "design_flow_mgd" not in df.columns:
            return pd.DataFrame(columns=["tier", "flow_range", "count", "total_flow_mgd", "avg_violations"])
        records = []
        for tier, (lo, hi) in self.FLOW_TIERS.items():
            mask = df["design_flow_mgd"] >= lo
            if hi:
                mask &= df["design_flow_mgd"] < hi
            subset = df[mask]
            records.append({
                "tier":         tier,
                "flow_range":   f"{lo}–{hi if hi else '∞'} MGD",
                "count":        len(subset),
                "total_flow_mgd": round(subset["design_flow_mgd"].sum(), 1),
                "avg_violations": round(subset["violations_3yr"].mean(), 2) if len(subset) else 0,
            })
        return pd.DataFrame(records)

    def top_polluters(self, n: int = 25, metric: str = "violations_3yr") -> pd.DataFrame:
        """
        Return top N facilities by a given metric.
        Metric options: violations_3yr, design_flow_mgd
        """
        if metric not in self.df.columns:
            raise ValueError(f"Column '{metric}' not found.")
        return self.df.nlargest(n, metric)[[
            col for col in
            ["facility_name", "city", "state", "design_flow_mgd",
             "violations_3yr", "npdes_ids", "receiving_waters", "latitude", "longitude"]
            if col in self.df.columns
        ]]

    def export_list(
        self,
        df: pd.DataFrame,
        path: str,
        fmt: str = "csv",
        include_cols: list = None,
    ) -> str:
        """
        Export a filtered list to CSV or Excel.

        Parameters
        ----------
        df          : Filtered DataFrame to export.
        path        : Output file path.
        fmt         : 'csv' or 'excel'.
        include_cols: Subset of columns. None = all.

        Returns
        -------
        Absolute path of saved file.
        """
        out = df[include_cols] if include_cols else df
        if fmt == "excel":
            out.to_excel(path, index=False, engine="openpyxl")
        else:
            out.to_csv(path, index=False)
        print(f"Exported {len(out)} records → {path}")
        return path
