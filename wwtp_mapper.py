"""
WWTP Mapping Engine
===================
Generates interactive (Folium/Leaflet) and static (Matplotlib) maps of
wastewater treatment plant locations.

Map types:
  - National overview with state-level choropleth
  - State-level point map with facility details
  - Violation hotspot map
  - Flow-capacity heat map
  - Receiving waters overlay

Usage:
    mapper = WWTPMapper(df)
    mapper.national_map("output/maps/national.html")
    mapper.state_map("TX", "output/maps/texas.html")
    mapper.violation_map("output/maps/violations.html")
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, HeatMap, MiniMap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")


# ── Color schemes ─────────────────────────────────────────────────────────────
FLOW_COLORS = {
    "micro":  "#4575b4",   # blue
    "small":  "#74add1",   # light blue
    "medium": "#fee090",   # yellow
    "large":  "#f46d43",   # orange
    "major":  "#d73027",   # red
}

VIOLATION_COLORS = {
    "clean":    "#1a9850",  # green  (0 violations)
    "low":      "#fee08b",  # yellow (1-2)
    "moderate": "#fc8d59",  # orange (3-5)
    "high":     "#d73027",  # red    (6+)
}

STATE_CENTERS = {
    "AL":(32.8,-86.8),"AK":(64.2,-153.4),"AZ":(34.3,-111.1),"AR":(34.8,-92.2),
    "CA":(36.8,-119.4),"CO":(39.0,-105.5),"CT":(41.6,-72.7),"DE":(39.0,-75.5),
    "FL":(28.1,-82.5),"GA":(32.7,-83.2),"HI":(20.7,-156.3),"ID":(44.4,-114.0),
    "IL":(40.0,-89.2),"IN":(40.3,-86.1),"IA":(42.0,-93.2),"KS":(38.5,-98.4),
    "KY":(37.7,-85.0),"LA":(30.5,-91.2),"ME":(45.4,-69.0),"MD":(39.1,-76.8),
    "MA":(42.2,-71.5),"MI":(44.4,-85.4),"MN":(46.4,-93.1),"MS":(32.7,-89.7),
    "MO":(38.5,-92.5),"MT":(47.0,-110.3),"NE":(41.5,-99.9),"NV":(39.3,-116.6),
    "NH":(43.7,-71.6),"NJ":(40.1,-74.5),"NM":(34.3,-106.0),"NY":(42.9,-75.6),
    "NC":(35.6,-79.8),"ND":(47.5,-100.5),"OH":(40.4,-82.7),"OK":(35.6,-96.9),
    "OR":(44.1,-120.5),"PA":(40.9,-77.8),"RI":(41.7,-71.5),"SC":(33.9,-80.9),
    "SD":(44.4,-100.2),"TN":(35.9,-86.5),"TX":(31.5,-99.3),"UT":(39.3,-111.1),
    "VT":(44.1,-72.7),"VA":(37.8,-78.2),"WA":(47.4,-120.6),"WV":(38.6,-80.6),
    "WI":(44.3,-89.8),"WY":(42.8,-107.6),"DC":(38.9,-77.0),
}


def _flow_tier(flow_mgd: float) -> str:
    if pd.isna(flow_mgd) or flow_mgd <= 0:
        return "unknown"
    if flow_mgd < 0.1:  return "micro"
    if flow_mgd < 1.0:  return "small"
    if flow_mgd < 10.0: return "medium"
    if flow_mgd < 100:  return "large"
    return "major"


def _violation_tier(v: int) -> str:
    if v == 0:  return "clean"
    if v <= 2:  return "low"
    if v <= 5:  return "moderate"
    return "high"


def _radius(flow_mgd: float) -> int:
    """Circle radius proportional to log(flow)."""
    if pd.isna(flow_mgd) or flow_mgd <= 0:
        return 4
    return max(4, min(20, int(4 + 5 * np.log10(flow_mgd + 0.1))))


class WWTPMapper:
    """
    Interactive and static mapping for WWTP facility data.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prep()

    def _prep(self):
        """Normalize key columns."""
        for col in ("latitude", "longitude", "design_flow_mgd", "violations_3yr"):
            if col not in self.df.columns:
                self.df[col] = np.nan if col != "violations_3yr" else 0
            else:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.df["violations_3yr"] = self.df["violations_3yr"].fillna(0).astype(int)
        self.df["_flow_tier"] = self.df["design_flow_mgd"].apply(_flow_tier)
        self.df["_viol_tier"] = self.df["violations_3yr"].apply(_violation_tier)

        # Drop rows with no coordinates
        self.geo = self.df.dropna(subset=["latitude", "longitude"])
        self.geo = self.geo[
            (self.geo["latitude"].between(-90, 90)) &
            (self.geo["longitude"].between(-180, 0))  # Continental/US
        ]

    # ── Popup builder ────────────────────────────────────────────────────────

    def _popup(self, row: pd.Series, extra_fields: list = None) -> folium.Popup:
        name   = row.get("facility_name", "Unknown")
        city   = row.get("city", "")
        state  = row.get("state", "")
        flow   = row.get("design_flow_mgd", None)
        viols  = row.get("violations_3yr", 0)
        npdes  = row.get("npdes_ids", row.get("npdes_id", ""))
        water  = row.get("receiving_waters", "—")
        dfr    = row.get("dfr_url", "")

        flow_str = f"{flow:.2f} MGD" if pd.notna(flow) else "Unknown"
        link = f'<a href="{dfr}" target="_blank">View DFR ↗</a>' if dfr else ""

        html = f"""
        <div style="font-family:Arial,sans-serif;font-size:12px;min-width:220px">
          <b style="font-size:13px">{name}</b><br>
          <span style="color:#555">{city}, {state}</span><br>
          <hr style="margin:4px 0">
          <b>NPDES ID:</b> {npdes}<br>
          <b>Design Flow:</b> {flow_str}<br>
          <b>Violations (3yr):</b> {viols}<br>
          <b>Receiving Water:</b> {water}<br>
          {link}
        </div>"""

        if extra_fields:
            for field in extra_fields:
                val = row.get(field, "—")
                html += f"<b>{field}:</b> {val}<br>"

        return folium.Popup(html, max_width=280)

    # ── National map ─────────────────────────────────────────────────────────

    def national_map(
        self,
        output_path: str = "output/maps/national.html",
        color_by: str = "flow",          # 'flow' | 'violations' | 'type'
        cluster: bool = True,
        show_minimap: bool = True,
    ) -> str:
        """
        Generate a national interactive map of all facilities.

        Parameters
        ----------
        output_path : HTML file path for the map.
        color_by    : How to color markers: 'flow' (size tier) | 'violations'.
        cluster     : Group nearby markers into clusters (recommended for 1000+ points).
        show_minimap: Show a small overview map in the corner.

        Returns
        -------
        Path to saved HTML file.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        m = folium.Map(
            location=[38.5, -96.0],
            zoom_start=5,
            tiles="CartoDB positron",
        )

        if show_minimap:
            MiniMap(toggle_display=True, position="bottomleft").add_to(m)

        # ── Build marker layer ──
        marker_layer = MarkerCluster(name="Facilities") if cluster else folium.FeatureGroup(name="Facilities")

        for _, row in self.geo.iterrows():
            if color_by == "violations":
                color = VIOLATION_COLORS.get(row["_viol_tier"], "#888")
            else:
                color = FLOW_COLORS.get(row["_flow_tier"], "#888")

            r = _radius(row["design_flow_mgd"])
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=r,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                weight=0.5,
                popup=self._popup(row),
                tooltip=row.get("facility_name", ""),
            ).add_to(marker_layer)

        marker_layer.add_to(m)

        # ── Legend ──
        legend_items = FLOW_COLORS if color_by == "flow" else VIOLATION_COLORS
        legend_labels = {
            "micro":    "Micro  (< 0.1 MGD)",
            "small":    "Small  (0.1–1 MGD)",
            "medium":   "Medium (1–10 MGD)",
            "large":    "Large  (10–100 MGD)",
            "major":    "Major  (> 100 MGD)",
            "clean":    "Clean record (0 violations)",
            "low":      "Low (1–2 violations)",
            "moderate": "Moderate (3–5 violations)",
            "high":     "High (6+ violations)",
        }

        legend_html = """
        <div style="position:fixed;bottom:30px;right:10px;z-index:1000;
                    background:white;padding:10px;border-radius:6px;
                    box-shadow:0 2px 6px rgba(0,0,0,.3);font-size:12px;
                    font-family:Arial,sans-serif">
          <b>Legend</b><br>
        """
        for key, color in legend_items.items():
            label = legend_labels.get(key, key)
            legend_html += (
                f'<i style="background:{color};width:12px;height:12px;'
                f'display:inline-block;border-radius:50%;margin-right:6px"></i>'
                f'{label}<br>'
            )
        legend_html += "</div>"

        m.get_root().html.add_child(folium.Element(legend_html))

        # ── Title ──
        title = ("US Wastewater Treatment Plants — "
                 f"{'Flow Capacity' if color_by=='flow' else 'Violation History'}")
        title_html = f"""
        <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                    z-index:1000;background:white;padding:8px 16px;
                    border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,.3);
                    font-size:14px;font-weight:bold;font-family:Arial,sans-serif">
          {title} — {len(self.geo):,} facilities
        </div>"""
        m.get_root().html.add_child(folium.Element(title_html))

        folium.LayerControl().add_to(m)
        m.save(output_path)
        print(f"National map saved → {output_path}  ({len(self.geo):,} facilities)")
        return output_path

    # ── State map ────────────────────────────────────────────────────────────

    def state_map(
        self,
        state: str,
        output_path: str = None,
        color_by: str = "flow",
        show_receiving_waters: bool = True,
    ) -> str:
        """
        Detailed map of a single state's wastewater facilities.

        Parameters
        ----------
        state  : 2-letter state abbreviation.
        color_by: 'flow' | 'violations'.

        Returns
        -------
        Path to saved HTML file.
        """
        state = state.upper()
        df_state = self.geo[self.geo["state"].str.upper() == state]

        if df_state.empty:
            print(f"No facilities with coordinates found for state: {state}")
            return ""

        if output_path is None:
            output_path = f"output/maps/{state.lower()}_facilities.html"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        center = STATE_CENTERS.get(state, [df_state["latitude"].mean(), df_state["longitude"].mean()])
        zoom = 7 if state not in ("AK","HI") else 5

        m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")
        MiniMap(toggle_display=True, position="bottomleft").add_to(m)

        # ── By-tier feature groups for filtering ──
        tiers = df_state["_flow_tier" if color_by == "flow" else "_viol_tier"].unique()
        groups = {}
        color_map = FLOW_COLORS if color_by == "flow" else VIOLATION_COLORS

        for tier in sorted(tiers):
            groups[tier] = folium.FeatureGroup(name=f"{tier.title()} ({color_by})", show=True)

        for _, row in df_state.iterrows():
            tier = row["_flow_tier"] if color_by == "flow" else row["_viol_tier"]
            color = color_map.get(tier, "#888")
            r = _radius(row["design_flow_mgd"])

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=r,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                weight=0.5,
                popup=self._popup(row),
                tooltip=f"{row.get('facility_name','')} — {row.get('design_flow_mgd', '?')} MGD",
            ).add_to(groups.get(tier, m))

        for g in groups.values():
            g.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        title_html = f"""
        <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                    z-index:1000;background:white;padding:8px 16px;
                    border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,.3);
                    font-size:14px;font-weight:bold;font-family:Arial,sans-serif">
          {state} Wastewater Treatment Plants — {len(df_state):,} facilities
        </div>"""
        m.get_root().html.add_child(folium.Element(title_html))

        m.save(output_path)
        print(f"State map saved → {output_path}  ({len(df_state):,} facilities)")
        return output_path

    # ── Violation heatmap ────────────────────────────────────────────────────

    def violation_heatmap(
        self, output_path: str = "output/maps/violation_heatmap.html"
    ) -> str:
        """
        Generate a heat map weighted by violation count.
        Highlights geographic clusters of compliance problems.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        m = folium.Map(location=[38.5, -96.0], zoom_start=5, tiles="CartoDB dark_matter")

        heat_data = [
            [row["latitude"], row["longitude"], row["violations_3yr"]]
            for _, row in self.geo.iterrows()
            if row["violations_3yr"] > 0
        ]

        if heat_data:
            HeatMap(heat_data, radius=15, blur=10, min_opacity=0.4,
                    gradient={"0.4": "blue", "0.65": "orange", "1": "red"}).add_to(m)

        title_html = """
        <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                    z-index:1000;background:#111;color:white;padding:8px 16px;
                    border-radius:6px;font-size:14px;font-weight:bold;font-family:Arial,sans-serif">
          NPDES Violation Density Heatmap (3-Year Window)
        </div>"""
        m.get_root().html.add_child(folium.Element(title_html))

        m.save(output_path)
        print(f"Violation heatmap saved → {output_path}")
        return output_path

    # ── Static choropleth (matplotlib) ──────────────────────────────────────

    def static_choropleth(
        self,
        metric: str = "facility_count",   # or "total_flow_mgd", "violations"
        output_path: str = "output/maps/choropleth.png",
        figsize: tuple = (16, 10),
        title: str = None,
    ) -> str:
        """
        Static US choropleth map colored by state-level metrics.
        Requires geopandas + US state shapefile.

        Parameters
        ----------
        metric     : 'facility_count' | 'total_flow_mgd' | 'total_violations'
        output_path: PNG output path.

        Returns
        -------
        Output path.
        """
        try:
            import geopandas as gpd
        except ImportError:
            print("geopandas not installed. Run: pip install geopandas")
            return ""

        # Aggregate by state
        agg = self.df.groupby("state").agg(
            facility_count    = ("facility_name", "count"),
            total_flow_mgd    = ("design_flow_mgd", "sum"),
            total_violations  = ("violations_3yr", "sum"),
        ).reset_index()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot as fallback if no shapefile available
        state_data = {}
        for _, row in agg.iterrows():
            s = row["state"].upper()
            if s in STATE_CENTERS:
                state_data[s] = {
                    "lat": STATE_CENTERS[s][0],
                    "lon": STATE_CENTERS[s][1],
                    "value": row.get(metric, 0),
                    "label": row.get(metric, 0),
                }

        vals = [v["value"] for v in state_data.values()]
        if not vals:
            print(f"No state data available for choropleth ({metric}). Skipping.")
            plt.close()
            return ""
        vmin, vmax = min(vals), max(vals)
        cmap = plt.cm.YlOrRd

        for state, d in state_data.items():
            norm_val = (d["value"] - vmin) / (vmax - vmin + 1e-9)
            color = cmap(norm_val)
            ax.scatter(d["lon"], d["lat"], c=[color], s=300, zorder=3, edgecolors="white", linewidths=0.5)
            ax.text(d["lon"], d["lat"] - 0.8, state, ha="center", va="top",
                    fontsize=7, color="white", fontweight="bold", zorder=4)
            ax.text(d["lon"], d["lat"] + 0.9, str(d["label"]), ha="center", va="bottom",
                    fontsize=6.5, color="black", zorder=4)

        ax.set_xlim(-125, -65)
        ax.set_ylim(24, 50)
        ax.set_facecolor("#d4e8f5")
        ax.set_aspect("equal")
        ax.axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        plt.colorbar(sm, ax=ax, shrink=0.6, label=metric.replace("_", " ").title())

        plt.title(title or f"US Wastewater Plants — {metric.replace('_',' ').title()} by State",
                  fontsize=15, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0a1628")
        plt.close()

        print(f"Choropleth saved → {output_path}")
        return output_path

    # ── Census overlay ───────────────────────────────────────────────────────

    def census_overlay_map(
        self,
        demographic_col: str = "census_median_income",
        output_path: str = "output/maps/census_overlay.html",
    ) -> str:
        """
        Map facilities overlaid on a Census demographic variable.
        Requires census data to be joined to df (see EPAExtractor.add_census_demographics).
        """
        if demographic_col not in self.df.columns:
            print(f"Column '{demographic_col}' not found. "
                  "Run EPAExtractor.add_census_demographics() first.")
            return ""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        m = folium.Map(location=[38.5, -96.0], zoom_start=5, tiles="CartoDB positron")

        for _, row in self.geo.iterrows():
            demo_val = row.get(demographic_col, np.nan)
            color = "#e41a1c" if pd.isna(demo_val) else "#377eb8"
            tooltip = (f"{row.get('facility_name','')} | "
                       f"{demographic_col}: {demo_val:,.0f}" if pd.notna(demo_val) else "")
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=self._popup(row),
                tooltip=tooltip,
            ).add_to(m)

        m.save(output_path)
        print(f"Census overlay map saved → {output_path}")
        return output_path
