# import streamlit as st

# st.title("Hello Streamlit App ")

# st.write("Writing to streamlit app is working fine.")

# name = st.text_input("Enter your name: ")

# if name:
#     st.success(f"Hello {name} !")

"""
app.py

Streamlit dashboard for Healthcare Eligibility Triage.

Run:
    streamlit run app.py
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Eligibility Triage Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# CONSTANTS
# ============================================================

DATA_PATH = "data/c2p1/appointments_final.csv"

DISPLAY_COLUMNS = {
    "appointment_datetime": "Appt Time",
    "patient_name": "Patient",
    "provider_name": "Provider",
    "insurance_on_file": "Insurance on File",
    "normalized_payer": "Matched Payer",
    "reason": "Reason"
}

STATUS_ORDER = [
    "Re-check needed",
    "OK",
    "Unknown"
]

# ============================================================
# LOAD DATA
# ============================================================


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load processed eligibility triage CSV.
    """

    df = pd.read_csv(file_path)

    # Parse datetime correctly
    df["appointment_datetime"] = pd.to_datetime(
        df["appointment_datetime"],
        errors="coerce"
    )

    return df


# ============================================================
# HEADER
# ============================================================

today_str = datetime.today().strftime("%B %d, %Y")

st.title("Eligibility Triage Dashboard")
st.caption(f"Today's Date: {today_str}")

# ============================================================
# LOAD FILE
# ============================================================

csv_path = Path(DATA_PATH)

if not csv_path.exists():

    st.error(
        "Processed appointments file not found.\n\n"
        "Please run the normalization and rule engine pipeline first.\n\n"
        f"Expected file:\n{DATA_PATH}"
    )

    st.stop()

with st.spinner("Loading today's appointments..."):

    try:
        df = load_data(DATA_PATH)

    except Exception as e:

        st.error(
            f"Failed to load appointment data.\n\nError: {e}"
        )

        st.stop()

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Filters")

# Provider Filter
provider_options = sorted(
    df["provider_name"]
    .dropna()
    .unique()
    .tolist()
)

selected_provider = st.sidebar.selectbox(
    "Provider",
    options=["All Providers"] + provider_options
)

# Status Filter
selected_status = st.sidebar.selectbox(
    "Status",
    options=["All"] + STATUS_ORDER
)

# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

if selected_provider != "All Providers":

    filtered_df = filtered_df[
        filtered_df["provider_name"] == selected_provider
    ]

if selected_status != "All":

    filtered_df = filtered_df[
        filtered_df["status"] == selected_status
    ]

# Sidebar Count
st.sidebar.markdown("---")
st.sidebar.write(
    f"Filtered Results: **{len(filtered_df)}**"
)

# ============================================================
# TOP METRICS
# ============================================================

recheck_count = (
    filtered_df["status"] == "Re-check needed"
).sum()

ok_count = (
    filtered_df["status"] == "OK"
).sum()

unknown_count = (
    filtered_df["status"] == "Unknown"
).sum()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Re-check Needed",
        value=recheck_count
    )

with col2:
    st.metric(
        label="OK",
        value=ok_count
    )

with col3:
    st.metric(
        label="Unknown",
        value=unknown_count
    )

st.divider()

# ============================================================
# TABLE DISPLAY FUNCTION
# ============================================================


def render_section(
    status_name: str,
    alert_type: str
) -> None:
    """
    Render status section table.
    """

    section_df = filtered_df[
        filtered_df["status"] == status_name
    ].copy()

    section_df = section_df.sort_values(
        by="appointment_datetime",
        ascending=True
    )

    # Rename columns for display
    display_df = section_df[
        list(DISPLAY_COLUMNS.keys())
    ].rename(columns=DISPLAY_COLUMNS)

    # Colored Header
    if alert_type == "error":
        st.error(f"{status_name}")

    elif alert_type == "success":
        st.success(f"{status_name}")

    elif alert_type == "warning":
        st.warning(f"{status_name}")

    # Expandable section
    with st.expander(
        f"View {status_name} Patients ({len(display_df)})",
        expanded=True
    ):

        if display_df.empty:

            st.info(
                "No patients in this category"
            )

        else:

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# MAIN SECTIONS
# ============================================================

render_section(
    "Re-check needed",
    "error"
)

render_section(
    "OK",
    "success"
)

render_section(
    "Unknown",
    "warning"
)

# ============================================================
# DOWNLOAD BUTTON
# ============================================================

st.divider()

csv_export = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Full Report as CSV",
    data=csv_export,
    file_name="eligibility_triage_export.csv",
    mime="text/csv"
)

# ============================================================
# PIPELINE SUMMARY
# ============================================================

method_counts = (
    filtered_df["method"]
    .fillna("unknown")
    .value_counts()
)

alias_count = method_counts.get(
    "alias_exact",
    0
)

fuzzy_count = method_counts.get(
    "fuzzy",
    0
)

llm_count = method_counts.get(
    "llm",
    0
)

unknown_method_count = (
    method_counts.get("ambiguous", 0)
    + method_counts.get("llm_error", 0)
    + method_counts.get("llm_unknown", 0)
    + method_counts.get("llm_invalid", 0)
)

st.markdown("---")

st.caption(
    f"Pipeline: "
    f"{alias_count} matched via alias | "
    f"{fuzzy_count} via fuzzy | "
    f"{llm_count} via LLM | "
    f"{unknown_method_count} unknown"
)