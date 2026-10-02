import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data_generator import generate_ess_dataset
from models import ScreeningEngine, FEATURE_COLS


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Burn-In Screening",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return generate_ess_dataset(
        num_components=1500,
        random_seed=42
    )


@st.cache_resource
def load_engine(_df):
    engine = ScreeningEngine()
    engine.train(_df)
    return engine


df = load_data()
engine = load_engine(df)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Screening Controls")

selected_lot = st.sidebar.selectbox(
    "Manufacturing Lot",
    ["All"] + sorted(df["lot_id"].unique().tolist())
)


# ============================================================
# FILTER DATA BY LOT
# ============================================================

if selected_lot == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[
        df["lot_id"] == selected_lot
    ].copy()


# ============================================================
# HEADER
# ============================================================

st.title("⚡ AI-Driven Component Burn-In & Screening Engine")

st.caption(
    "AI-based early screening using 0h–24h electrical and thermal drift "
    "to predict 168h failure risk."
)

st.markdown("---")


# ============================================================
# COMPONENT SELECTION - MAIN DASHBOARD
# ============================================================

st.subheader("Component Screening")

# Create component display options
component_options = (
    filtered_df["component_id"].astype(str)
    + "  |  Lot: "
    + filtered_df["lot_id"].astype(str)
)

selected_option = st.selectbox(
    "Select Component ID",
    component_options.tolist(),
    index=0,
    help="Select a component to analyze its burn-in behavior."
)

# Extract actual component ID
selected_comp_id = selected_option.split("  |  Lot: ")[0]


# Get selected component row
sample_row = filtered_df[
    filtered_df["component_id"] == selected_comp_id
].iloc[0]


# Display selected component information
info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "Component ID",
        sample_row["component_id"]
    )

with info2:
    st.metric(
        "Manufacturing Lot",
        sample_row["lot_id"]
    )

with info3:
    ground_truth_display = (
        "FAILED"
        if sample_row["actual_failure_168h"] == 1
        else "PASS"
    )

    st.metric(
        "168h Ground Truth",
        ground_truth_display
    )


st.markdown("---")


# ============================================================
# BULK PREDICTIONS
# ============================================================

@st.cache_data
def calculate_bulk_decisions(dataframe, _engine):

    decisions = []

    for _, row in dataframe.iterrows():

        result = _engine.predict_component(
            pd.DataFrame([row[FEATURE_COLS]])
        )

        decisions.append(
            result["decision"]
        )

    return decisions


bulk_decisions = calculate_bulk_decisions(
    filtered_df,
    engine
)


display_df = filtered_df.copy()

display_df["Decision"] = bulk_decisions


# ============================================================
# KPI CALCULATIONS
# ============================================================

pass_count = int(
    (display_df["Decision"] == "PASS").sum()
)

warning_count = int(
    (display_df["Decision"] == "WARNING").sum()
)

reject_count = int(
    (display_df["Decision"] == "REJECT").sum()
)

total_count = len(display_df)

reject_rate = (
    reject_count / total_count
    if total_count
    else 0
)


# ============================================================
# KPI CARDS
# ============================================================

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Components",
    total_count
)

k2.metric(
    "PASS",
    pass_count
)

k3.metric(
    "WARNING",
    warning_count
)

k4.metric(
    "REJECT",
    reject_count
)

k5.metric(
    "Reject Rate",
    f"{reject_rate:.1%}"
)


st.markdown("---")


# ============================================================
# SELECTED COMPONENT AI PREDICTION
# ============================================================

sample_features = pd.DataFrame(
    [sample_row[FEATURE_COLS]]
)

result = engine.predict_component(
    sample_features
)


# ============================================================
# DECISION ICONS
# ============================================================

decision_icons = {
    "PASS": "🟢",
    "WARNING": "🟡",
    "REJECT": "🔴",
}


# ============================================================
# COMPONENT RESULT CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:
    st.metric(
        "Component",
        selected_comp_id
    )


with c2:
    st.metric(
        "Decision",
        f"{decision_icons[result['decision']]} "
        f"{result['decision']}"
    )


with c3:
    st.metric(
        "Predicted 168h Failure Risk",
        f"{result['failure_probability']:.1%}"
    )


with c4:
    st.metric(
        "Anomaly Score",
        f"{result['anomaly_score']:.3f}"
    )


# ============================================================
# RISK MESSAGE
# ============================================================

if result["decision"] == "REJECT":

    st.error(
        "HIGH RISK: Component shows abnormal early behavior "
        "or high predicted 168h failure probability."
    )

elif result["decision"] == "WARNING":

    st.warning(
        "MEDIUM RISK: Component requires additional attention."
    )

else:

    st.success(
        "LOW RISK: Component passes the AI screening criteria."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs([
    "🔍 Component Analysis",
    "📊 Lot Reliability"
])


# ============================================================
# TAB 1 - COMPONENT ANALYSIS
# ============================================================

with tab1:

    left, right = st.columns([1.25, 1])


    # --------------------------------------------------------
    # DRIFT GRAPH
    # --------------------------------------------------------

    with left:

        st.subheader(
            "Parameter Drift: 0h → 24h → 168h"
        )

        time_points = [
            "0h",
            "24h",
            "168h"
        ]

        fig = go.Figure()


        # Voltage
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=[
                    sample_row["v_0h"],
                    sample_row["v_24h"],
                    sample_row["v_168h"],
                ],
                mode="lines+markers",
                name="Voltage (V)",
            )
        )


        # Current
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=[
                    sample_row["i_0h"],
                    sample_row["i_24h"],
                    sample_row["i_168h"],
                ],
                mode="lines+markers",
                name="Current (mA)",
                yaxis="y2",
            )
        )


        # Temperature
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=[
                    sample_row["temp_0h"],
                    sample_row["temp_24h"],
                    sample_row["temp_168h"],
                ],
                mode="lines+markers",
                name="Temperature (°C)",
                yaxis="y3",
            )
        )


        fig.update_layout(

            xaxis_title="ESS Time Point",

            yaxis=dict(
                title="Voltage (V)"
            ),

            yaxis2=dict(
                title="Current (mA)",
                overlaying="y",
                side="right",
            ),

            yaxis3=dict(
                title="Temperature (°C)",
                overlaying="y",
                side="left",
                position=0.05,
            ),

            height=470,

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),

            legend=dict(
                orientation="h"
            ),
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    with right:

        st.subheader(
            "SHAP Feature Impact"
        )

        st.caption(
            "Positive SHAP values increase predicted failure risk; "
            "negative values reduce it."
        )


        shap_df = pd.DataFrame(
            list(
                result["shap_importance"].items()
            ),
            columns=[
                "Feature",
                "Impact"
            ]
        ).sort_values(
            "Impact"
        )


        fig_shap = px.bar(
            shap_df,
            x="Impact",
            y="Feature",
            orientation="h",
            color="Impact",
            color_continuous_scale="RdYlGn_r",
        )


        fig_shap.update_layout(
            height=470,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
        )


        st.plotly_chart(
            fig_shap,
            use_container_width=True
        )


    # --------------------------------------------------------
    # EARLY DRIFT TABLE
    # --------------------------------------------------------

    st.subheader(
        "Early Drift Measurements"
    )


    drift_df = pd.DataFrame({

        "Parameter": [
            "Voltage",
            "Current",
            "Temperature",
        ],

        "0h": [
            sample_row["v_0h"],
            sample_row["i_0h"],
            sample_row["temp_0h"],
        ],

        "24h": [
            sample_row["v_24h"],
            sample_row["i_24h"],
            sample_row["temp_24h"],
        ],

        "0h → 24h Drift": [
            sample_row["delta_v_0_24"],
            sample_row["delta_i_0_24"],
            sample_row["delta_temp_0_24"],
        ],
    })


    st.dataframe(
        drift_df,
        use_container_width=True,
        hide_index=True,
    )


    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    st.subheader(
        "Ground Truth"
    )


    ground_truth = (

        "FAILED at 168h"

        if sample_row["actual_failure_168h"] == 1

        else "DID NOT FAIL at 168h"
    )


    st.info(
        f"Synthetic ground-truth result: "
        f"**{ground_truth}**"
    )


# ============================================================
# TAB 2 - LOT RELIABILITY
# ============================================================

with tab2:

    st.subheader(
        f"Lot Screening Statistics — {selected_lot}"
    )


    # --------------------------------------------------------
    # PIE CHART
    # --------------------------------------------------------

    pie_df = (
        display_df["Decision"]
        .value_counts()
        .reset_index()
    )

    pie_df.columns = [
        "Decision",
        "Count"
    ]


    fig_pie = px.pie(

        pie_df,

        names="Decision",

        values="Count",

        hole=0.55,

        title="Screening Decision Distribution",

        color="Decision",

        color_discrete_map={
            "PASS": "#2ecc71",
            "WARNING": "#f1c40f",
            "REJECT": "#e74c3c",
        },
    )


    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )


    # --------------------------------------------------------
    # LOT SUMMARY
    # --------------------------------------------------------

    lot_summary = (

        df.groupby("lot_id")

        .agg(

            Components=(
                "component_id",
                "count"
            ),

            Actual_Failures=(
                "actual_failure_168h",
                "sum"
            ),
        )

        .reset_index()
    )


    lot_summary["Actual Failure Rate"] = (

        lot_summary["Actual_Failures"]

        / lot_summary["Components"]
    )


    st.subheader(
        "Manufacturing Lot Summary"
    )


    st.dataframe(
        lot_summary,
        use_container_width=True,
        hide_index=True,
    )


    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "Model Performance"
    )


    m1, m2 = st.columns(2)


    m1.metric(
        "Validation Accuracy",
        f"{engine.metrics['accuracy']:.1%}"
    )


    m2.metric(
        "Validation ROC-AUC",
        f"{engine.metrics['roc_auc']:.3f}"
    )


# ============================================================
# DOWNLOAD DATASET
# ============================================================

st.markdown("---")


csv_data = (
    df.to_csv(index=False)
    .encode("utf-8")
)


st.download_button(

    "Download Synthetic ESS Dataset",

    data=csv_data,

    file_name=(
        "ESS_predictive_screening_dataset_1500.csv"
    ),

    mime="text/csv",
)