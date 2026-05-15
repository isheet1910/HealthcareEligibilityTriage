"""
app.py
======
Streamlit dashboard for the Healthcare Eligibility Triage project.
 
Reads the final processed CSV (output of rule_engine.py) and shows
today's appointments grouped into three triage buckets:
  🔴 Re-check needed
  🟢 OK
  🟡 Unknown
 
Run:
    streamlit run app.py
"""
 
from datetime import datetime
from pathlib import Path
 
import pandas as pd
import streamlit as st
 
# ============================================================
# Page config — must be the very first Streamlit call
# ============================================================
 
st.set_page_config(
    page_title="Eligibility Triage Dashboard",
    page_icon="🏥",
    layout="wide",
)
 
# ============================================================
# Constants
# ============================================================
 
DATA_PATH = Path("data/appointments_final.csv")
 
# Columns to show in every triage table, mapped to human labels
DISPLAY_COLUMNS = {
    "appointment_datetime":  "Appt Time",
    "patient_name":          "Patient",
    "provider_name":         "Provider",
    "insurance_on_file":     "Insurance on File",
    "normalized_payer":      "Matched Payer",
    "payer_code":            "Payer Code",
    "confidence":            "Confidence",
    "days_since_last_check": "Days Since Check",
    "reason":                "Reason",
}
 
STATUS_ORDER = ["Re-check needed", "OK", "Unknown"]
 
# Colour mapping for the status badge column
STATUS_COLORS = {
    "Re-check needed": "🔴",
    "OK":              "🟢",
    "Unknown":         "🟡",
}
 
# ============================================================
# Data loading  (cached so filters don't re-read from disk)
#The @st.cache_data decorator is critical. Without it, every user interaction (clicking a filter, adjusting a slider) would re-read the 
# CSV from disk. With it, the data is loaded once and cached in memory for the session. This makes the dashboard much more responsive, especially 
# as the dataset grows. The cache is automatically invalidated 
# if the underlying file changes, ensuring users always see up-to-date information without unnecessary reloads.
# ============================================================
 
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """
    Load the final triage CSV and do light type coercion.
    Only called once per session thanks to @st.cache_data.
    """
    df = pd.read_csv(path)
 
    # Parse appointment datetime for correct sort order
    df["appointment_datetime"] = pd.to_datetime(
        df["appointment_datetime"], errors="coerce"
    )
 
    # Round confidence to 2 decimal points easier to display for display
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].round(2)
 
    return df
 
 
# ============================================================
# Guard: fail gracefully if the pipeline hasn't been run yet check if all the 
#  files have been run simulate normalize rules 
# ============================================================
 
if not DATA_PATH.exists():
    st.error(
        "**Processed appointments file not found.**\n\n"
        "Run the pipeline first:\n"
        "```\n"
        "python simulate.py\n"
        "python src/normalize.py\n"
        "python src/rule_engine.py\n"
        "```\n\n"
        f"Expected file: `{DATA_PATH}`"
    )
    st.stop()
 
# ============================================================
# Load teh data file final file
# ============================================================
 
with st.spinner("Loading today's appointments..."):
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
 
# ============================================================
# Header
# ============================================================
 
st.title("🏥 Eligibility Triage Dashboard")
st.caption(
    f"Today: **{datetime.today().strftime('%A, %B %d, %Y')}**  ·  "
    f"Appointments loaded: **{len(df)}**  ·  "
    f"Last run: `{DATA_PATH}`"
)
 
st.divider()
 
# ============================================================
# Sidebar filters
# ============================================================
 
st.sidebar.header("🔍 Filters")
 
# Provider filter
providers = ["All Providers"] + sorted(df["provider_name"].dropna().unique().tolist())
selected_provider = st.sidebar.selectbox("Provider", providers)
 
# Status filter
selected_status = st.sidebar.selectbox("Status", ["All Statuses"] + STATUS_ORDER)
 
# Confidence filter — only shown when it adds value
min_confidence = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
    help="Filter out rows where the payer match confidence is below this value.",
)
 
st.sidebar.divider()
 
# ============================================================
# Apply filters
# ============================================================
 
filtered = df.copy()
 
if selected_provider != "All Providers":
    filtered = filtered[filtered["provider_name"] == selected_provider]
 
if selected_status != "All Statuses":
    filtered = filtered[filtered["status"] == selected_status]
 
if min_confidence > 0.0:
    filtered = filtered[filtered["confidence"] >= min_confidence]
 
# Sort by appointment time throughout
filtered = filtered.sort_values("appointment_datetime", ascending=True)
 
st.sidebar.metric("Showing", f"{len(filtered)} appointments")
 
# ============================================================
# Top-level metrics row
# ============================================================
 
recheck_n = int((filtered["status"] == "Re-check needed").sum())
ok_n      = int((filtered["status"] == "OK").sum())
unknown_n = int((filtered["status"] == "Unknown").sum())
 
# Deltas relative to total so the coordinator sees proportions at a glance
total = len(filtered) or 1   # avoid div-zero
 
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Appointments",  len(filtered))
col2.metric("🔴 Re-check Needed",  recheck_n,  f"{recheck_n/total:.0%}")
col3.metric("🟢 OK",               ok_n,       f"{ok_n/total:.0%}")
col4.metric("🟡 Unknown",          unknown_n,  f"{unknown_n/total:.0%}")
 
st.divider()
 
# ============================================================
# Triage section renderer
# ============================================================
 
def _render_section(status: str) -> None:
    """
    Render one triage bucket: coloured header, sortable table, row count.
    """
    icon   = STATUS_COLORS[status]
    subset = filtered[filtered["status"] == status].copy()
 
    # Visible columns only — drop anything not in DISPLAY_COLUMNS
    available_cols = [c for c in DISPLAY_COLUMNS if c in subset.columns]
    display = (
        subset[available_cols]
        .rename(columns=DISPLAY_COLUMNS)
        .reset_index(drop=True)
    )
 
    # Coloured subheader
    if status == "Re-check needed":
        st.error(f"{icon} **{status}** — {len(display)} patients")
    elif status == "OK":
        st.success(f"{icon} **{status}** — {len(display)} patients")
    else:
        st.warning(f"{icon} **{status}** — {len(display)} patients")
 
    with st.expander(f"View {status} table", expanded=(status == "Re-check needed")):
        if display.empty:
            st.info("No patients in this category with the current filters.")
        else:
            st.dataframe(
                display,
                width='stretch',
                # use_container_width=True,
                hide_index=True,
                column_config={
                    # Make confidence a progress bar so it reads visually
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.2f",
                    ),
                    # Format appointment time cleanly
                    "Appt Time": st.column_config.DatetimeColumn(
                        "Appt Time",
                        format="HH:mm",
                    ),
                    # Days since check as a number column with help text
                    "Days Since Check": st.column_config.NumberColumn(
                        "Days Since Check",
                        help="Days elapsed since the last eligibility verification.",
                        format="%d",
                    ),
                },
            )
 
# ============================================================
# Render all three buckets
# ============================================================
 
for status_name in STATUS_ORDER:
    _render_section(status_name)
    st.write("")   # breathing room between sections
 
st.divider()
 
# ============================================================
# Download
# ============================================================
 
st.subheader("📥 Export")
 
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
 
st.download_button(
    label="Download filtered results as CSV",
    data=csv_bytes,
    file_name=f"eligibility_triage_{datetime.today().strftime('%Y%m%d')}.csv",
    mime="text/csv",
)
 
# ============================================================
# Pipeline diagnostics footer
# ============================================================
 
st.divider()
st.subheader("🔧 Pipeline Diagnostics")
 
method_counts = (
    filtered["method"]
    .fillna("unknown")
    .value_counts()
    .rename_axis("Match Method")
    .reset_index(name="Count")
)
 
diag_col1, diag_col2 = st.columns([1, 2])
 
with diag_col1:
    st.dataframe(method_counts, hide_index=True, width='stretch')
    # use_container_width=True
 
with diag_col2:
    # Human-readable summary line
    mc = filtered["method"].value_counts()
    alias_n   = int(mc.get("alias_exact", 0))
    exact_n   = int(mc.get("exact", 0))
    fuzzy_n   = int(mc.get("fuzzy", 0))
    llm_n     = int(mc.get("llm", 0))
    ambig_n   = int(mc.get("ambiguous", 0))
    error_n   = int(
        mc.get("llm_error", 0)
        + mc.get("llm_invalid", 0)
        + mc.get("llm_unknown", 0)
        + mc.get("llm_empty_response", 0)
    )
 
    st.markdown(f"""
**How payers were matched:**
 
| Method | Count | What it means |
|---|---|---|
| Alias / Exact | {alias_n + exact_n} | Deterministic — known abbreviation or exact name |
| Fuzzy | {fuzzy_n} | String similarity matched to a canonical name |
| LLM | {llm_n} | Language model resolved an ambiguous string |
| Ambiguous | {ambig_n} | Self-pay, unknown, or genuinely unresolvable |
| LLM unresolved | {error_n} | LLM called but could not match to any payer |
""")
 
st.caption(
    "Built with Claude · Normalization: alias → fuzzy → LLM fallback · "
    "Rule engine: 5 ordered business rules · Data: synthetic (SEED=42)"
)