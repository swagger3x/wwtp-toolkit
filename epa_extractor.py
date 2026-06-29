"""
EPA Wastewater Treatment Plant Data Extractor
============================================
Sources:
  - EPA ECHO (Enforcement and Compliance History Online): facility permits, locations, violations
  - EPA Envirofacts (efservice): NPDES permit details, DMRs, effluent limits
  - EPA FRS (Facility Registry System): canonical facility IDs and coordinates
  - US Census API: demographics cross-reference

Usage:
  extractor = EPAExtractor()
  df = extractor.get_facilities(state="TX", fac_type="POTWs")
  limits_df = extractor.get_permit_limits(npdes_ids=df["npdes_id"].tolist())
  dmr_df = extractor.get_discharge_monitoring(npdes_ids=["TX0001234"], years=3)
"""

import io
import requests
import pandas as pd
import time
import logging
from typing import Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── API base URLs ────────────────────────────────────────────────────────────
ECHO_BASE   = "https://echodata.epa.gov/echo/echo_rest_services"
CWA_BASE    = "https://echodata.epa.gov/echo/cwa_rest_services"
ENVIRO_BASE = "https://data.epa.gov/efservice"
FRS_BASE    = "https://ofmpub.epa.gov/frs_public2/frs_rest_services"
CENSUS_BASE = "https://api.census.gov/data"

# ECHO numeric column IDs for the two-step QueryID→CSV download.
# Verified against the live ECHO REST API (2026-06).
# 1:FacName  2:FacStreet  3:FacCity  4:FacState  5:FacZip  6:RegistryID
# 16:FacNAICSCodes  17:FacLat  18:FacLong  21:NPDESIDs
# 38:CWAComplianceStatus  73:FacMajorFlag  82:FacUsMexBorderFlg
# 95:FacActiveFlag  119:CWAQtrsWithNC  137:DfrUrl
# Note: col 147 (SourceID) is the FRS RegistryID, NOT the NPDES permit ID — excluded.
ECHO_QCOLS = "1,2,3,4,5,6,16,17,18,21,38,73,82,95,119,137"

# NAICS code for Sewage Treatment Facilities (POTWs)
POTW_NAICS = "22132"

# Individual NPDES permit pattern: state abbrev + '0' + digits (e.g. CO0020052)
# Matches individual discharge permits; excludes general/stormwater (COG/COR/COX).
_NPDES_INDIVIDUAL_RE = r'\b([A-Z]{2}0\d+)\b'

# Column rename map: ECHO CSV name → toolkit canonical name
ECHO_COL_MAP = {
    "FacName":                "facility_name",
    "FacStreet":              "address",
    "FacCity":                "city",
    "FacState":               "state",
    "FacZip":                 "zip",
    "RegistryID":             "registry_id",
    "FacNAICSCodes":          "naics_codes",
    "FacLat":                 "latitude",
    "FacLong":                "longitude",
    "NPDESIDs":               "npdes_ids",
    "CWAComplianceStatus":    "permit_status",
    "FacMajorFlag":           "is_major",
    "FacUsMexBorderFlg":      "border_facility",
    "FacActiveFlag":          "active",
    "CWAQtrsWithNC":          "violations_3yr",
    "DfrUrl":                 "dfr_url",
    # From CWA download (merged on primary NPDES ID)
    "CWPTotalDesignFlowNmbr": "design_flow_mgd",
}

STATE_FIPS = {
    "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10",
    "FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20",
    "KY":"21","LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28",
    "MO":"29","MT":"30","NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36",
    "NC":"37","ND":"38","OH":"39","OK":"40","OR":"41","PA":"42","RI":"44","SC":"45",
    "SD":"46","TN":"47","TX":"48","UT":"49","VT":"50","VA":"51","WA":"53","WV":"54",
    "WI":"55","WY":"56","DC":"11","PR":"72","GU":"66","VI":"78","AS":"60","MP":"69",
}


class EPAExtractor:
    """
    Full ETL pipeline for EPA wastewater facility data.
    Handles pagination, rate-limiting, and data normalization.
    """

    def __init__(self, census_api_key: str = "", request_delay: float = 0.5):
        self.census_api_key = census_api_key
        self.delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "WWTP-Research-Tool/1.0 (environmental data analysis)",
            "Accept": "application/json",
        })

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get(self, url: str, params: dict = None, timeout: int = 30) -> dict | list:
        """GET with retry logic and rate limiting."""
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                time.sleep(self.delay)
                ct = resp.headers.get("content-type", "")
                if "json" in ct:
                    return resp.json()
                return resp.text
            except requests.HTTPError as e:
                log.warning(f"HTTP {e.response.status_code} on attempt {attempt+1}: {url}")
                if e.response.status_code in (429, 503):
                    time.sleep(10 * (attempt + 1))
                elif attempt == 2:
                    raise
            except requests.RequestException as e:
                log.warning(f"Request error attempt {attempt+1}: {e}")
                if attempt == 2:
                    raise
                time.sleep(5)

    # ── ECHO facility search ─────────────────────────────────────────────────

    def get_facilities(
        self,
        state: str = None,
        city: str = None,
        fac_type: str = "POTWs",    # POTWs | INDUSTRIAL | all
        active_only: bool = True,
        major_only: bool = False,
        min_flow_mgd: float = None,
        max_results: int = 10000,
        output_path: str = None,
    ) -> pd.DataFrame:
        """
        Query EPA ECHO for NPDES-permitted wastewater facilities.

        Uses the ECHO two-step API: get_facilities → QueryID → get_download (CSV).

        Parameters
        ----------
        state        : 2-letter state abbreviation (e.g. 'TX'). None = all states.
        city         : City name filter.
        fac_type     : 'POTWs' (municipal, NAICS 22132), 'INDUSTRIAL', or '' for all NPDES.
        active_only  : Only return currently active permits.
        major_only   : Only return major NPDES facilities.
        min_flow_mgd : Post-filter by design flow (million gallons/day).
        max_results  : Cap on total records returned.
        output_path  : If set, saves CSV to this path.

        Returns
        -------
        pd.DataFrame with columns defined in ECHO_COL_MAP.
        """
        _empty = pd.DataFrame(columns=list(ECHO_COL_MAP.values()))

        # ── Step 1: Query ECHO → get QueryID ──────────────────────────────────
        params = {
            "output":        "JSON",
            "p_permit_type": "NPD",               # NPDES permits only
            "p_act":         "Y" if active_only else "",
            "p_maj":         "Y" if major_only else "",
        }
        if state:
            params["p_st"] = state.upper()
        if city:
            params["p_city"] = city
        if fac_type and fac_type.lower() == "potws":
            params["p_naics"] = POTW_NAICS        # Sewage Treatment Facilities

        log.info(f"Querying ECHO: state={state}, fac_type={fac_type}")
        data = self._get(f"{ECHO_BASE}.get_facilities", params)
        if not isinstance(data, dict):
            log.warning("Unexpected response from ECHO.")
            return _empty

        results = data.get("Results", {})
        qid   = results.get("QueryID")
        total = int(results.get("QueryRows", 0) or 0)
        log.info(f"  QueryID={qid}, {total} facilities available")

        if not qid or total == 0:
            log.warning("No facilities found for this query.")
            return _empty

        # ── Step 2: Download ECHO CSV with selected columns ────────────────────
        log.info(f"  Downloading ECHO facility data...")
        raw_echo = self._get(
            f"{ECHO_BASE}.get_download",
            {"output": "CSV", "qid": qid, "qcolumns": ECHO_QCOLS},
            timeout=120,
        )
        if not isinstance(raw_echo, str) or not raw_echo.strip():
            log.warning("Empty ECHO download response.")
            return _empty

        df = pd.read_csv(io.StringIO(raw_echo))
        log.info(f"  ECHO download: {len(df)} rows")

        # Keep only NPDES-permitted facilities (non-empty NPDESIDs)
        if "NPDESIDs" in df.columns:
            df = df[df["NPDESIDs"].notna() & (df["NPDESIDs"].astype(str).str.strip() != "")]

        # Extract the primary individual NPDES permit ID (e.g. CO0020052) from the
        # space-separated NPDESIDs list — used as join key for the CWA flow merge.
        # Individual permits follow the pattern: 2-letter state + '0' + digits.
        if "NPDESIDs" in df.columns:
            extracted = (
                df["NPDESIDs"].astype(str)
                .str.extractall(_NPDES_INDIVIDUAL_RE)[0]
                .groupby(level=0).first()
            )
            df["_npdes_primary"] = extracted

        # ── Step 3: CWA download → design flow ────────────────────────────────
        try:
            cwa_params = {"output": "JSON", "p_act": params.get("p_act", "")}
            if state:
                cwa_params["p_st"] = state.upper()

            cwa_data = self._get(f"{CWA_BASE}.get_facilities", cwa_params)
            cwa_qid  = (cwa_data or {}).get("Results", {}).get("QueryID")
            if cwa_qid:
                log.info(f"  Downloading CWA flow data (qid={cwa_qid})...")
                raw_cwa = self._get(
                    f"{CWA_BASE}.get_download",
                    {"output": "CSV", "qid": cwa_qid},
                    timeout=120,
                )
                if isinstance(raw_cwa, str) and raw_cwa.strip():
                    df_cwa = pd.read_csv(io.StringIO(raw_cwa))
                    if {"SourceID", "CWPTotalDesignFlowNmbr"}.issubset(df_cwa.columns):
                        df_flow = (
                            df_cwa[["SourceID", "CWPTotalDesignFlowNmbr"]]
                            .dropna(subset=["SourceID", "CWPTotalDesignFlowNmbr"])
                            .rename(columns={"SourceID": "_npdes_primary"})
                        )
                        if "_npdes_primary" in df.columns:
                            df = df.merge(df_flow, on="_npdes_primary", how="left")
                            has_flow = df["CWPTotalDesignFlowNmbr"].notna().sum()
                            log.info(f"  Flow data merged: {has_flow} facilities with design flow")
        except Exception as e:
            log.warning(f"CWA flow data unavailable: {e}")

        # ── Step 4: Rename columns ─────────────────────────────────────────────
        df.rename(columns={k: v for k, v in ECHO_COL_MAP.items() if k in df.columns}, inplace=True)
        df.drop(columns=["_npdes_primary"], errors="ignore", inplace=True)

        # ── Step 5: Clean types ────────────────────────────────────────────────
        for col in ("latitude", "longitude", "design_flow_mgd"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "violations_3yr" in df.columns:
            df["violations_3yr"] = (
                pd.to_numeric(df["violations_3yr"], errors="coerce").fillna(0).astype(int)
            )

        if min_flow_mgd and "design_flow_mgd" in df.columns:
            df = df[df["design_flow_mgd"] >= min_flow_mgd]

        df = df.head(max_results).reset_index(drop=True)
        log.info(f"Final dataset: {len(df)} facilities.")

        if output_path:
            df.to_csv(output_path, index=False)
            log.info(f"Saved to {output_path}")

        return df

    # ── Permit limits from Envirofacts ───────────────────────────────────────

    def get_permit_limits(
        self,
        npdes_ids: list[str],
        parameters: list[str] = None,
        output_path: str = None,
    ) -> pd.DataFrame:
        """
        Retrieve effluent permit limits from EPA Envirofacts ICIS-NPDES.

        Parameters
        ----------
        npdes_ids   : List of NPDES permit IDs (e.g. ['TX0001234', 'CA0056789']).
        parameters  : Filter to specific parameters (e.g. ['BOD', 'TSS', 'Ammonia']).
                      None = return all.
        output_path : Optional CSV save path.

        Returns
        -------
        DataFrame with permit limits including parameter, limit type, units, values.

        Key columns
        -----------
        npdes_id, parameter_code, parameter_name, limit_type (Daily Max / Weekly Avg / Monthly Avg),
        limit_value, limit_unit, monitoring_period_desc, receiving_water
        """
        results = []

        for npdes_id in npdes_ids:
            url = (f"{ENVIRO_BASE}/NPDES_LIMITS/EXTERNAL_PERMIT_NMBR/"
                   f"{npdes_id.strip().upper()}/JSON")
            log.info(f"Fetching limits for {npdes_id}")
            try:
                data = self._get(url)
                if isinstance(data, list):
                    for row in data:
                        row["npdes_id"] = npdes_id.upper()
                        results.append(row)
            except Exception as e:
                log.warning(f"Could not fetch limits for {npdes_id}: {e}")

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Normalize column names to snake_case
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Filter to specific parameters
        if parameters and "parameter_name" in df.columns:
            pats = [p.lower() for p in parameters]
            df = df[df["parameter_name"].str.lower().apply(
                lambda x: any(p in str(x) for p in pats)
            )]

        if output_path:
            df.to_csv(output_path, index=False)
        return df

    # ── Discharge Monitoring Reports ─────────────────────────────────────────

    def get_discharge_monitoring(
        self,
        npdes_ids: list[str],
        years: int = 3,
        parameters: list[str] = None,
        output_path: str = None,
    ) -> pd.DataFrame:
        """
        Retrieve Discharge Monitoring Report (DMR) data from Envirofacts.

        DMRs are monthly self-reported measurements that facilities submit.
        Useful for compliance analysis and trend detection.

        Parameters
        ----------
        npdes_ids  : NPDES permit IDs.
        years      : Number of recent years to fetch (default 3).
        parameters : Optional parameter name filter.

        Returns
        -------
        DataFrame with columns: npdes_id, monitoring_period_date, parameter_name,
        quantity_unit, quantity_value, nodi_code, violation_flag.
        """
        results = []
        for npdes_id in npdes_ids:
            url = (f"{ENVIRO_BASE}/NPDES_DMR_MEASUREMENTS/EXTERNAL_PERMIT_NMBR/"
                   f"{npdes_id.strip().upper()}/JSON")
            log.info(f"Fetching DMR data for {npdes_id}")
            try:
                data = self._get(url)
                if isinstance(data, list):
                    for row in data:
                        row["npdes_id"] = npdes_id.upper()
                        results.append(row)
            except Exception as e:
                log.warning(f"DMR fetch failed for {npdes_id}: {e}")

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Date filter
        if "monitoring_period_end_date" in df.columns:
            df["monitoring_period_end_date"] = pd.to_datetime(
                df["monitoring_period_end_date"], errors="coerce"
            )
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
            df = df[df["monitoring_period_end_date"] >= cutoff]

        if parameters and "parameter_name" in df.columns:
            pats = [p.lower() for p in parameters]
            df = df[df["parameter_name"].str.lower().apply(
                lambda x: any(p in str(x) for p in pats)
            )]

        if output_path:
            df.to_csv(output_path, index=False)
        return df

    # ── FRS coordinates ──────────────────────────────────────────────────────

    def enrich_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For facilities missing lat/lon, look them up via EPA FRS.
        Adds/fills 'latitude' and 'longitude' columns.
        """
        if "latitude" not in df.columns:
            df["latitude"] = None
        if "longitude" not in df.columns:
            df["longitude"] = None

        missing = df[df["latitude"].isna()]
        if missing.empty:
            return df

        log.info(f"Enriching coordinates for {len(missing)} facilities via FRS...")
        for idx, row in missing.iterrows():
            registry_id = row.get("registry_id", "")
            if not registry_id:
                continue
            url = f"{FRS_BASE}.get_facilities"
            params = {"registry_id": registry_id, "output": "JSON"}
            try:
                data = self._get(url, params)
                facilities = data.get("Facilities", [])
                if facilities:
                    df.at[idx, "latitude"] = float(facilities[0].get("Latitude83", 0) or 0)
                    df.at[idx, "longitude"] = float(facilities[0].get("Longitude83", 0) or 0)
            except Exception as e:
                log.debug(f"FRS lookup failed for {registry_id}: {e}")

        return df

    # ── Census cross-reference ───────────────────────────────────────────────

    def add_census_demographics(
        self,
        df: pd.DataFrame,
        census_vars: list[str] = None,
        year: int = 2020,
    ) -> pd.DataFrame:
        """
        Join Census data to facilities using state + county FIPS codes.

        Default variables: total population, median household income,
        percent below poverty line, population density.

        Requires a Census API key (free at api.census.gov/data/key_signup.html).

        Parameters
        ----------
        df           : Facilities DataFrame (must have 'state' column).
        census_vars  : ACS variable codes. Defaults to common demographics.
        year         : Census year (2020 = Decennial, 2021-2023 = ACS 5-yr).

        Returns
        -------
        Input DataFrame with additional census columns merged by state/county FIPS.
        """
        if not self.census_api_key:
            log.warning("No Census API key set. Skipping census enrichment. "
                        "Get a free key at https://api.census.gov/data/key_signup.html")
            return df

        if census_vars is None:
            census_vars = [
                "B01003_001E",   # Total population
                "B19013_001E",   # Median household income
                "B17001_002E",   # Population below poverty level
                "B01001_001E",   # Total for age/sex (for density calc)
            ]

        census_records = []
        states = df["state"].dropna().unique()

        for state_abbr in states:
            fips = STATE_FIPS.get(state_abbr.upper())
            if not fips:
                continue

            url = f"{CENSUS_BASE}/{year}/acs/acs5"
            params = {
                "get":    "NAME," + ",".join(census_vars),
                "for":    "county:*",
                "in":     f"state:{fips}",
                "key":    self.census_api_key,
            }
            try:
                data = self._get(url, params)
                if isinstance(data, list) and len(data) > 1:
                    headers = [h.lower() for h in data[0]]
                    for row in data[1:]:
                        rec = dict(zip(headers, row))
                        rec["state_fips"] = fips
                        rec["state_abbr"] = state_abbr
                        census_records.append(rec)
            except Exception as e:
                log.warning(f"Census fetch failed for {state_abbr}: {e}")

        if not census_records:
            return df

        census_df = pd.DataFrame(census_records)
        census_df.rename(columns={
            "b01003_001e": "census_total_pop",
            "b19013_001e": "census_median_income",
            "b17001_002e": "census_poverty_pop",
        }, inplace=True)

        # Merge on state (county-level join requires FIPS in facilities df)
        df = df.merge(
            census_df[["state_abbr", "census_total_pop", "census_median_income", "census_poverty_pop"]
                       if "census_total_pop" in census_df else ["state_abbr"]],
            left_on="state", right_on="state_abbr", how="left"
        )
        return df

    # ── Batch multi-state pull ───────────────────────────────────────────────

    def get_all_states(
        self,
        states: list[str] = None,
        fac_type: str = "POTWs",
        output_dir: str = "data",
        combined_output: str = "data/all_facilities.csv",
    ) -> pd.DataFrame:
        """
        Pull facilities for multiple (or all) states and combine into one DataFrame.

        Parameters
        ----------
        states     : List of 2-letter state codes. None = all 50 states + DC.
        fac_type   : Facility type filter.
        output_dir : Directory to save per-state CSVs.
        combined_output : Path for the merged national CSV.

        Returns
        -------
        Combined DataFrame for all requested states.
        """
        if states is None:
            states = list(STATE_FIPS.keys())

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        all_dfs = []

        for state in states:
            out_path = f"{output_dir}/{state.lower()}_facilities.csv"
            log.info(f"=== Processing {state} ===")
            df_state = self.get_facilities(state=state, fac_type=fac_type,
                                           output_path=out_path)
            if not df_state.empty:
                df_state["state"] = state.upper()
                all_dfs.append(df_state)

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(combined_output, index=False)
        log.info(f"Combined {len(combined)} facilities → {combined_output}")
        return combined
