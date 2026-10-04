import os
import requests
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import textwrap

_original_markdown = st.markdown
def _fixed_markdown(body, *args, **kwargs):
    if isinstance(body, str):
        body = textwrap.dedent(body)
    if kwargs.get("unsafe_allow_html", False):
        kwargs.pop("unsafe_allow_html", None)
        return st.html(body)
    return _original_markdown(body, *args, **kwargs)

st.markdown = _fixed_markdown

from backend.data_generator import generate_ess_dataset
from backend.models import FEATURE_COLS


# ============================================================
# SCREENING AI — INDUSTRIAL OPERATIONS CONSOLE
# Frontend → FastAPI → AI Model → Frontend
# ============================================================

st.set_page_config(
    page_title="Screening AI | Industrial Operations",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIG
# ============================================================


DATA_FILE = (
    ROOT
    / "data"
    / "ESS_predictive_screening_dataset_1500.csv"
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

PAGES = [
    "Command Center",
    "Components",
    "Screening Operations",
    "Risk Center",
    "AI Insights",
    "Manufacturing",
    "Reports",
    "Settings",
]


# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():

    if DATA_FILE.exists():

        df = pd.read_csv(DATA_FILE)

    else:

        df = generate_ess_dataset(
            num_components=1500,
            random_seed=42
        )

    required = set(FEATURE_COLS) | {
        "component_id",
        "lot_id",
        "actual_failure_168h",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
        )

    return df


# ============================================================
# BACKEND API
# ============================================================

def backend_health():

    response = requests.get(
        f"{API_URL}/health",
        timeout=10
    )

    response.raise_for_status()

    return response.json()


def train_backend():

    with open(DATA_FILE, "rb") as file:

        response = requests.post(
            f"{API_URL}/train",
            files={
                "file": (
                    DATA_FILE.name,
                    file,
                    "text/csv"
                )
            },
            timeout=120
        )

    response.raise_for_status()

    return response.json()


def get_batch_predictions(df):

    records = (
        df[FEATURE_COLS]
        .to_dict(orient="records")
    )

    response = requests.post(
        f"{API_URL}/predict-batch",
        json={
            "records": records
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()


def get_component_explanation(row):

    payload = {
        column: float(row[column])
        for column in FEATURE_COLS
    }

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# LOAD DATA + CONNECT BACKEND
# ============================================================

df = load_data()

try:

    health = backend_health()

    if not health["model_trained"]:

        train_backend()

    api_result = get_batch_predictions(df)

except requests.RequestException:

    st.error(
        "Backend API is not running.\n\n"
        "Start it with:\n\n"
        "`python -m uvicorn backend.backend:app "
        "--reload --port 8000`"
    )

    st.stop()


pred = pd.DataFrame(
    api_result["predictions"]
)

model_metrics = api_result.get(
    "metrics",
    {
        "accuracy": 0.0,
        "roc_auc": 0.0
    }
)


# ============================================================
# DISPLAY DATA
# ============================================================

display_df = df.copy()

display_df["Decision"] = (
    pred["Decision"].values
)

display_df["Failure Risk"] = (
    pred["failure_probability"].values
)

display_df["Anomaly Score"] = (
    pred["anomaly_score"].values
)

display_df["Failure Risk %"] = (
    display_df["Failure Risk"] * 100
).round(1)


counts = display_df["Decision"].value_counts()

pass_count = int(
    counts.get("PASS", 0)
)

warning_count = int(
    counts.get("WARNING", 0)
)

reject_count = int(
    counts.get("REJECT", 0)
)

total_count = len(display_df)

reject_rate = (
    reject_count / total_count
    if total_count
    else 0
)


# ============================================================
# SESSION STATE
# ============================================================

if "nav_page" not in st.session_state:

    st.session_state.nav_page = "Command Center"


if "selected_component" not in st.session_state:

    st.session_state.selected_component = (
        str(display_df.iloc[0]["component_id"])
    )


# ============================================================
# INDUSTRIAL THEME
# ============================================================

st.markdown(
    """
<style>

:root{
    --bg:#06101a;
    --panel:#0b1826;
    --panel2:#0e1e2e;
    --line:#20364a;
    --text:#edf4fb;
    --muted:#8196aa;
    --cyan:#4eb6ff;
    --green:#35d58b;
    --amber:#f2b84b;
    --red:#ff5b64;
    --purple:#8c7cff;
}

[data-testid="stAppViewContainer"]{
    background:
        radial-gradient(
            circle at 82% -5%,
            rgba(50,115,165,.18),
            transparent 34%
        ),
        linear-gradient(
            135deg,
            #06101a 0%,
            #07131f 48%,
            #050c14 100%
        );
    color:var(--text);
}

[data-testid="stHeader"]{
    background:transparent;
}

.block-container{
    max-width:1720px;
    padding:1.15rem 1.6rem 2.2rem;
}

[data-testid="stSidebar"]{
    background:#07121e;
    border-right:1px solid #1b3042;
}

[data-testid="stSidebar"] > div:first-child{
    padding:1.1rem .85rem;
}

.brand{
    display:flex;
    align-items:center;
    gap:12px;
    padding:8px 8px 18px;
}

.brand-mark{
    width:40px;
    height:40px;
    border:1px solid #33506a;
    border-radius:10px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-weight:900;
    font-size:19px;
    color:#fff;
    background:linear-gradient(
        145deg,
        #152d43,
        #091522
    );
}

.brand-name{
    font-size:20px;
    font-weight:800;
}

.brand-sub{
    font-size:10px;
    color:#71879b;
    margin-top:2px;
    letter-spacing:1.1px;
    text-transform:uppercase;
}

.nav-label{
    color:#5f778c;
    font-size:10px;
    font-weight:700;
    letter-spacing:1.4px;
    text-transform:uppercase;
    margin:8px 8px 4px;
}

[data-testid="stSidebar"]
div[role="radiogroup"]
label{
    border:1px solid transparent !important;
    border-radius:9px !important;
    padding:8px 10px !important;
    margin:1px 0 !important;
    color:#a8bac9 !important;
    font-size:13px !important;
    font-weight:600 !important;
    background:transparent !important;
}

[data-testid="stSidebar"]
div[role="radiogroup"]
label:hover{
    background:#0d2031 !important;
    color:#eef6fd !important;
}

[data-testid="stSidebar"]
div[role="radiogroup"]
label:has(input:checked){
    background:linear-gradient(
        90deg,
        #132f46,
        #10263a
    ) !important;
    border-color:#284a64 !important;
    color:#fff !important;
    box-shadow:inset 3px 0 0 #4eb6ff !important;
}

[data-testid="stSidebar"]
div[role="radiogroup"]
label > div:first-child{
    display:none !important;
}

.sidebar-status{
    border:1px solid #1d3b32;
    background:#081d18;
    border-radius:10px;
    padding:11px 12px;
    margin-top:12px;
}

.online-dot{
    display:inline-block;
    width:7px;
    height:7px;
    border-radius:50%;
    background:#35d58b;
    box-shadow:0 0 10px rgba(53,213,139,.7);
    margin-right:7px;
}

.sidebar-meta{
    font-size:11px;
    color:#7890a5;
    margin-top:6px;
}

.topbar{
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:20px;
    margin-bottom:14px;
}

.eyebrow{
    color:#6f8ba1;
    font-size:10px;
    letter-spacing:1.7px;
    text-transform:uppercase;
    font-weight:800;
}

.page-title{
    font-size:31px;
    font-weight:800;
    letter-spacing:-.7px;
    margin:4px 0 3px;
}

.page-sub{
    font-size:13px;
    color:#8da4b8;
}

.status-pill{
    display:inline-flex;
    align-items:center;
    gap:7px;
    padding:7px 11px;
    border:1px solid #20513d;
    background:#081d18;
    border-radius:20px;
    color:#55e6a0;
    font-size:11px;
    font-weight:700;
}

.top-controls{
    border:1px solid #20384c;
    background:#0a1724;
    border-radius:9px;
    padding:8px 11px;
    color:#a9bbca;
    font-size:11px;
}

.kpi-card{
    border:1px solid #1d3447;
    border-radius:12px;
    background:linear-gradient(
        145deg,
        rgba(14,31,47,.96),
        rgba(8,19,30,.96)
    );
    padding:15px 16px;
    min-height:108px;
}

.kpi-label{
    font-size:10px;
    letter-spacing:1.2px;
    color:#71899d;
    text-transform:uppercase;
    font-weight:800;
}

.kpi-value{
    font-size:29px;
    font-weight:850;
    line-height:1.15;
    margin-top:8px;
}

.kpi-meta{
    font-size:11px;
    color:#7890a4;
    margin-top:5px;
}

.cyan{color:#4eb6ff}
.green{color:#35d58b}
.amber{color:#f2b84b}
.red{color:#ff5b64}
.purple{color:#9a8cff}

.panel{
    border:1px solid #1d3447;
    border-radius:12px;
    background:rgba(9,23,35,.91);
    padding:15px;
}

.panel-title{
    font-size:14px;
    font-weight:750;
}

.panel-sub{
    font-size:10px;
    color:#71899d;
    margin-bottom:10px;
}

.section-label{
    font-size:10px;
    letter-spacing:1.3px;
    text-transform:uppercase;
    color:#71899d;
    font-weight:800;
    margin:17px 0 8px;
}

.metric-strip{
    display:flex;
    justify-content:space-between;
    gap:12px;
    border-top:1px solid #1a3042;
    padding-top:10px;
    margin-top:8px;
}

.metric-name{
    font-size:10px;
    color:#71899d;
}

.metric-val{
    font-size:15px;
    font-weight:750;
    margin-top:2px;
}

.ops-chip{
    display:inline-flex;
    align-items:center;
    gap:6px;
    border:1px solid #223b50;
    background:#0b1b2a;
    border-radius:7px;
    padding:5px 8px;
    color:#91a8bb;
    font-size:9px;
    font-weight:750;
}

.queue-card{
    border:1px solid #20394d;
    background:#0b1b29;
    border-radius:9px;
    padding:11px 12px;
    margin:7px 0;
}

.queue-top{
    display:flex;
    justify-content:space-between;
    align-items:center;
}

.queue-id{
    font-size:12px;
    font-weight:800;
    color:#edf4fb;
}

.queue-meta{
    font-size:9px;
    color:#71899d;
    margin-top:5px;
}

.queue-risk{
    font-size:16px;
    font-weight:850;
    color:#ff6b73;
}

.diag-grid{
    display:grid;
    grid-template-columns:
        1.1fr 1.1fr 1.1fr;
    gap:10px;
    margin:12px 0 16px;
}

.diag-card{
    border:1px solid #20394d;
    background:#0a1927;
    border-radius:9px;
    padding:12px 13px;
    min-height:92px;
}

.diag-label{
    font-size:9px;
    letter-spacing:1.1px;
    color:#71899d;
    text-transform:uppercase;
    font-weight:800;
}

.diag-value{
    font-size:14px;
    font-weight:800;
    color:#edf4fb;
    margin-top:7px;
}

.diag-copy{
    font-size:10px;
    line-height:1.45;
    color:#8298aa;
    margin-top:5px;
}

.action-hold{
    border-left:3px solid #ff5b64;
}

.action-review{
    border-left:3px solid #f2b84b;
}

.action-release{
    border-left:3px solid #35d58b;
}

.alert-item{
    padding:10px 0;
    border-bottom:1px solid #172b3d;
}

.alert-item:last-child{
    border-bottom:0;
}

.alert-id{
    font-size:12px;
    font-weight:750;
}

.alert-meta{
    font-size:10px;
    color:#7890a4;
    margin-top:3px;
}

.badge{
    display:inline-block;
    padding:3px 7px;
    border-radius:5px;
    font-size:9px;
    font-weight:800;
}

.badge-pass{
    background:#0b3023;
    color:#55e6a0;
    border:1px solid #1a5944;
}

.badge-warning{
    background:#33270d;
    color:#f4c55b;
    border:1px solid #66501f;
}

.badge-reject{
    background:#35171b;
    color:#ff6c73;
    border:1px solid #6b292f;
}

.workspace{
    border:1px solid #29445a;
    border-radius:13px;
    background:linear-gradient(
        145deg,
        #0d2030,
        #091522
    );
    padding:17px;
    margin-bottom:15px;
}

.workspace-title{
    font-size:21px;
    font-weight:800;
}

.workspace-meta{
    color:#7f95a8;
    font-size:11px;
    margin-top:4px;
}

div[data-testid="stButton"] > button{
    border-radius:8px !important;
    border:1px solid #294359 !important;
    background:#0d2031 !important;
    color:#dce8f2 !important;
    font-weight:650 !important;
    min-height:36px !important;
}

div[data-testid="stButton"] > button:hover{
    border-color:#4b7393 !important;
    background:#122a3f !important;
}

button[kind="primary"]{
    background:linear-gradient(
        135deg,
        #1679ba,
        #2454a3
    ) !important;
    border-color:#3187c4 !important;
}

[data-testid="stDataFrame"]{
    border:1px solid #1d3447;
    border-radius:10px;
    overflow:hidden;
}

.stSelectbox label,
.stTextInput label{
    color:#8196aa !important;
    font-size:10px !important;
    text-transform:uppercase;
    letter-spacing:.9px;
    font-weight:750 !important;
}

div[data-baseweb="select"] > div{
    background:#0b1a29 !important;
    border-color:#233d52 !important;
}

hr{
    border-color:#1b3042 !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">S</div>
            <div>
                <div class="brand-name">
                    SCREENING AI
                </div>
                <div class="brand-sub">
                    ESS Reliability Platform
                </div>
            </div>
        </div>

        <div class="nav-label">
            Operations
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = st.radio(
        "NAV",
        PAGES,
        index=PAGES.index(
            st.session_state.nav_page
        ),
        label_visibility="collapsed",
    )

    st.session_state.nav_page = selected_page

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-status">

            <div>
                <span class="online-dot"></span>
                <b style="font-size:11px;">
                    SCREENING ENGINE ONLINE
                </b>
            </div>

            <div class="sidebar-meta">
                XGBoost + Isolation Forest
            </div>

            <div class="sidebar-meta">
                1,500 components loaded
            </div>

        </div>

        <div style="
            padding:12px 4px;
            color:#61798e;
            font-size:10px;
            line-height:1.9;
        ">

            SCREENING WINDOW<br>
            <b style="color:#a8bac9;">
                0h → 24h early signal
            </b>

            <br>

            PREDICTION TARGET<br>

            <b style="color:#a8bac9;">
                168h failure risk
            </b>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HELPERS
# ============================================================

def page_header(
    kicker,
    title,
    subtitle,
):

    st.markdown(
        f"""
        <div class="topbar">

            <div>

                <div class="eyebrow">
                    {kicker}
                </div>

                <div class="page-title">
                    {title}
                </div>

                <div class="page-sub">
                    {subtitle}
                </div>

            </div>

            <div style="
                display:flex;
                gap:8px;
                align-items:center;
                flex-wrap:wrap;
                justify-content:flex-end;
            ">

                <div class="top-controls">
                    PLANT <b>ESS-01</b>
                </div>

                <div class="top-controls">
                    LINE <b>BURN-IN-03</b>
                </div>

                <div class="top-controls">
                    MODE <b>EARLY SCREENING</b>
                </div>

                <div class="status-pill">
                    <span class="online-dot"></span>
                    SYSTEM ONLINE
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi(
    label,
    value,
    meta,
    cls="cyan",
):

    st.markdown(
        f"""
        <div class="kpi-card">

            <div class="kpi-label">
                {label}
            </div>

            <div class="kpi-value {cls}">
                {value}
            </div>

            <div class="kpi-meta">
                {meta}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_title(
    title,
    subtitle="",
):

    st.markdown(
        f"""
        <div class="panel-title">
            {title}
        </div>

        <div class="panel-sub">
            {subtitle}
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_dark(
    fig,
    height=320,
):

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#8ea5b9",
            size=10
        ),
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=1.08,
            x=0
        ),
        xaxis=dict(
            gridcolor="#1b3043",
            zerolinecolor="#1b3043"
        ),
        yaxis=dict(
            gridcolor="#1b3043",
            zerolinecolor="#1b3043"
        ),
    )

    return fig


def decision_badge(
    decision
):

    cls = {
        "PASS": "badge-pass",
        "WARNING": "badge-warning",
        "REJECT": "badge-reject",
    }.get(
        decision,
        "badge-warning"
    )

    return (
        f'<span class="badge {cls}">'
        f'{decision}'
        f'</span>'
    )


def feature_label(
    name
):

    labels = {

        "v_0h":
            "Voltage @ 0h",

        "i_0h":
            "Current @ 0h",

        "temp_0h":
            "Temperature @ 0h",

        "v_24h":
            "Voltage @ 24h",

        "i_24h":
            "Current @ 24h",

        "temp_24h":
            "Temperature @ 24h",

        "delta_v_0_24":
            "Voltage drift (0h → 24h)",

        "delta_i_0_24":
            "Current drift (0h → 24h)",

        "delta_temp_0_24":
            "Temperature drift (0h → 24h)",
    }

    return labels.get(
        name,
        name.replace(
            "_",
            " "
        ).title()
    )


def engineering_action(
    decision
):

    actions = {

        "PASS": (
            "Release to next stage",
            "No high-risk disposition detected in the current screening run.",
            "action-release"
        ),

        "WARNING": (
            "Engineering review",
            "Inspect the strongest early drift signal before release.",
            "action-review"
        ),

        "REJECT": (
            "Hold and investigate",
            "Keep the component out of release flow pending engineering investigation.",
            "action-hold"
        ),
    }

    return actions[decision]


def sensor_chart(
    row,
    height=330
):

    fig = go.Figure()

    specs = [
        (
            "v",
            "Voltage",
            "#4eb6ff"
        ),
        (
            "i",
            "Current",
            "#8c7cff"
        ),
        (
            "temp",
            "Temperature",
            "#f08d92"
        ),
    ]

    for key, name, color in specs:

        fig.add_trace(
            go.Scatter(
                x=[
                    "0h",
                    "24h",
                    "96h",
                    "168h"
                ],
                y=[
                    row[f"{key}_0h"],
                    row[f"{key}_24h"],
                    row[f"{key}_96h"],
                    row[f"{key}_168h"],
                ],
                mode="lines+markers",
                name=name,
                line=dict(
                    color=color,
                    width=2
                ),
                marker=dict(
                    size=6
                ),
            )
        )

    return chart_dark(
        fig,
        height
    )


def open_component(
    component_id
):

    st.session_state.selected_component = (
        str(component_id)
    )

    st.session_state.nav_page = "AI Insights"

    st.rerun()


# ============================================================
# COMMAND CENTER
# ============================================================

if (
    st.session_state.nav_page
    == "Command Center"
):

    page_header(
        "OPERATIONS / COMMAND",
        "Command Center",
        "Plant-level view of early screening, component risk and model health.",
    )

    st.markdown(
        """
        <div style="
            display:flex;
            gap:8px;
            flex-wrap:wrap;
            margin:4px 0 12px;
        ">

            <span class="ops-chip">
                DATA STREAM <b>CONNECTED</b>
            </span>

            <span class="ops-chip">
                WINDOW <b>0h → 24h</b>
            </span>

            <span class="ops-chip">
                TARGET <b>168h RISK</b>
            </span>

            <span class="ops-chip">
                ENGINE <b>XGB + IF</b>
            </span>

        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        kpi(
            "Components",
            f"{total_count:,}",
            "All loaded lots",
            "cyan"
        )

    with c2:
        kpi(
            "Pass",
            f"{pass_count:,}",
            f"{pass_count / total_count:.1%} of population",
            "green"
        )

    with c3:
        kpi(
            "Warning",
            f"{warning_count:,}",
            f"{warning_count / total_count:.1%} of population",
            "amber"
        )

    with c4:
        kpi(
            "Reject",
            f"{reject_count:,}",
            f"{reject_rate:.1%} reject rate",
            "red"
        )

    with c5:
        kpi(
            "Model Accuracy",
            f"{model_metrics['accuracy']:.1%}",
            f"ROC-AUC {model_metrics['roc_auc']:.3f}",
            "purple"
        )

    st.markdown(
        '<div class="section-label">Plant Intelligence</div>',
        unsafe_allow_html=True
    )

    left, mid, right = st.columns(
        [1.15, 1.35, 1.0]
    )

    with left:

        panel_title(
            "Risk Distribution",
            "Current screening disposition"
        )

        risk_fig = px.pie(
            display_df,
            names="Decision",
            hole=.70,
            color="Decision",
            color_discrete_map={
                "PASS": "#35d58b",
                "WARNING": "#f2b84b",
                "REJECT": "#ff5b64"
            },
        )

        risk_fig.update_traces(
            textinfo="none"
        )

        st.plotly_chart(
            chart_dark(
                risk_fig,
                285
            ),
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

        st.markdown(
            f"""
            <div class="metric-strip">

                <div>
                    <div class="metric-name">
                        PASS
                    </div>
                    <div class="metric-val green">
                        {pass_count:,}
                    </div>
                </div>

                <div>
                    <div class="metric-name">
                        WARNING
                    </div>
                    <div class="metric-val amber">
                        {warning_count:,}
                    </div>
                </div>

                <div>
                    <div class="metric-name">
                        REJECT
                    </div>
                    <div class="metric-val red">
                        {reject_count:,}
                    </div>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with mid:

        panel_title(
            "Early Screening Signal",
            "Population mean • electrical and thermal drift"
        )

        mean_cols = [
            "v_0h",
            "v_24h",
            "v_96h",
            "v_168h",
            "i_0h",
            "i_24h",
            "i_96h",
            "i_168h",
            "temp_0h",
            "temp_24h",
            "temp_96h",
            "temp_168h",
        ]

        means = df[
            mean_cols
        ].mean()

        fig = go.Figure()

        for key, name, color in [
            (
                "v",
                "Voltage",
                "#4eb6ff"
            ),
            (
                "i",
                "Current",
                "#8c7cff"
            ),
            (
                "temp",
                "Temperature",
                "#f08d92"
            ),
        ]:

            fig.add_trace(
                go.Scatter(
                    x=[
                        "0h",
                        "24h",
                        "96h",
                        "168h"
                    ],
                    y=[
                        means[f"{key}_0h"],
                        means[f"{key}_24h"],
                        means[f"{key}_96h"],
                        means[f"{key}_168h"],
                    ],
                    name=name,
                    mode="lines+markers",
                    line=dict(
                        color=color,
                        width=2
                    )
                )
            )

        st.plotly_chart(
            chart_dark(
                fig,
                300
            ),
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    with right:

        panel_title(
            "Critical Queue",
            "Highest predicted failure risk"
        )

        critical = (
            display_df
            .sort_values(
                [
                    "Failure Risk",
                    "Anomaly Score"
                ],
                ascending=[
                    False,
                    True
                ]
            )
            .head(5)
        )

        for _, r in critical.iterrows():

            st.markdown(
                f"""
                <div class="queue-card">

                    <div class="queue-top">

                        <div class="queue-id">
                            {r["component_id"]}
                            &nbsp;
                            {decision_badge(r["Decision"])}
                        </div>

                        <div class="queue-risk">
                            {r["Failure Risk %"]:.1f}%
                        </div>

                    </div>

                    <div class="queue-meta">
                        {r["lot_id"]}
                        • anomaly {r["Anomaly Score"]:.3f}
                        • 168h failure risk
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                f"OPEN WORKSPACE · {r['component_id']}",
                key=f"cc_{r['component_id']}",
                use_container_width=True
            ):

                open_component(
                    r["component_id"]
                )

    st.markdown(
        '<div class="section-label">Manufacturing View</div>',
        unsafe_allow_html=True
    )

    lot_summary = (
        display_df
        .groupby("lot_id")
        .agg(
            Components=(
                "component_id",
                "count"
            ),
            Rejects=(
                "Decision",
                lambda s:
                    int(
                        (s == "REJECT").sum()
                    )
            ),
            Warnings=(
                "Decision",
                lambda s:
                    int(
                        (s == "WARNING").sum()
                    )
            ),
            AvgRisk=(
                "Failure Risk",
                "mean"
            ),
        )
        .reset_index()
    )

    lot_summary["Reject Rate"] = (
        lot_summary["Rejects"]
        / lot_summary["Components"]
    )

    a, b = st.columns(
        [1.45, 1.0]
    )

    with a:

        panel_title(
            "Lot Risk Profile",
            "Reject and warning distribution by manufacturing lot"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=lot_summary["lot_id"],
                y=lot_summary["Warnings"],
                name="Warning",
                marker_color="#f2b84b"
            )
        )

        fig.add_trace(
            go.Bar(
                x=lot_summary["lot_id"],
                y=lot_summary["Rejects"],
                name="Reject",
                marker_color="#ff5b64"
            )
        )

        fig.update_layout(
            barmode="stack"
        )

        st.plotly_chart(
            chart_dark(
                fig,
                285
            ),
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    with b:

        panel_title(
            "Model Health",
            "Validation metrics from current training run"
        )

        st.metric(
            "Accuracy",
            f"{model_metrics['accuracy']:.2%}"
        )

        st.metric(
            "ROC-AUC",
            f"{model_metrics['roc_auc']:.3f}"
        )

        st.caption(
            "Current development dataset is synthetic."
        )


# ============================================================
# COMPONENTS
# ============================================================

elif (
    st.session_state.nav_page
    == "Components"
):

    page_header(
        "OPERATIONS / COMPONENTS",
        "Component Explorer",
        "Search, filter and open an individual component reliability workspace.",
    )

    f1, f2, f3 = st.columns(
        [1, 1, 1.5]
    )

    with f1:

        lot_filter = st.selectbox(
            "Manufacturing lot",
            [
                "ALL"
            ]
            + sorted(
                display_df[
                    "lot_id"
                ].unique()
            )
        )

    with f2:

        decision_filter = st.selectbox(
            "Disposition",
            [
                "ALL",
                "PASS",
                "WARNING",
                "REJECT"
            ]
        )

    with f3:

        search = st.text_input(
            "Component search",
            placeholder="e.g. CMP-0042"
        )

    view = display_df.copy()

    if lot_filter != "ALL":

        view = view[
            view["lot_id"]
            == lot_filter
        ]

    if decision_filter != "ALL":

        view = view[
            view["Decision"]
            == decision_filter
        ]

    if search.strip():

        view = view[
            view["component_id"]
            .astype(str)
            .str.contains(
                search.strip(),
                case=False,
                na=False
            )
        ]

    if len(view):

        selected = st.selectbox(
            "Select component",
            view["component_id"].tolist()
        )

        if st.button(
            "OPEN COMPONENT WORKSPACE",
            type="primary"
        ):

            open_component(
                selected
            )

        st.dataframe(
            view[
                [
                    "component_id",
                    "lot_id",
                    "Decision",
                    "Failure Risk %",
                    "Anomaly Score"
                ]
            ].sort_values(
                "Failure Risk %",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "No components match the current filters."
        )


# ============================================================
# SCREENING OPERATIONS
# ============================================================

elif (
    st.session_state.nav_page
    == "Screening Operations"
):

    page_header(
        "OPERATIONS / SCREENING",
        "Screening Operations",
        "Run the early-screening workflow against a manufacturing lot.",
    )

    lot = st.selectbox(
        "Target manufacturing lot",
        [
            "ALL"
        ]
        + sorted(
            display_df[
                "lot_id"
            ].unique()
        )
    )

    target_df = (
        display_df
        if lot == "ALL"
        else display_df[
            display_df["lot_id"]
            == lot
        ]
    )

    st.markdown(
        f"""
        <div class="workspace">

            <div class="workspace-title">
                Early Screening Run
            </div>

            <div class="workspace-meta">
                {len(target_df):,} components queued
                • 0h → 24h signal
                • prediction target 168h failure risk
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)

    with a:
        kpi(
            "Queue",
            f"{len(target_df):,}",
            "Components in scope",
            "cyan"
        )

    with b:
        kpi(
            "Pass",
            f"{int((target_df.Decision == 'PASS').sum()):,}",
            "Within screening limits",
            "green"
        )

    with c:
        kpi(
            "Review",
            f"{int((target_df.Decision == 'WARNING').sum()):,}",
            "Needs engineering review",
            "amber"
        )

    with d:
        kpi(
            "Reject",
            f"{int((target_df.Decision == 'REJECT').sum()):,}",
            "High-risk disposition",
            "red"
        )

    st.markdown(
        '<div class="section-label">Screening Queue</div>',
        unsafe_allow_html=True
    )

    st.success(
        "AI screening results are loaded from the FastAPI backend."
    )

    queue = (
        target_df
        .sort_values(
            "Failure Risk",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        queue[
            [
                "component_id",
                "lot_id",
                "Decision",
                "Failure Risk %",
                "Anomaly Score"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RISK CENTER
# ============================================================

elif (
    st.session_state.nav_page
    == "Risk Center"
):

    page_header(
        "RISK / PRIORITIZATION",
        "Risk Center",
        "Prioritize components and lots requiring engineering attention.",
    )

    high = (
        display_df
        .sort_values(
            [
                "Failure Risk",
                "Anomaly Score"
            ],
            ascending=[
                False,
                True
            ]
        )
    )

    a, b, c, d = st.columns(4)

    with a:
        kpi(
            "High Risk",
            f"{reject_count:,}",
            "REJECT disposition",
            "red"
        )

    with b:
        kpi(
            "Review",
            f"{warning_count:,}",
            "WARNING disposition",
            "amber"
        )

    with c:
        kpi(
            "Peak Risk",
            f"{high.iloc[0]['Failure Risk %']:.1f}%",
            str(
                high.iloc[0]["component_id"]
            ),
            "red"
        )

    with d:
        kpi(
            "Lots",
            f"{display_df['lot_id'].nunique():,}",
            "Manufacturing lots",
            "cyan"
        )

    left, right = st.columns(
        [1.35, 1]
    )

    with left:

        panel_title(
            "Priority Queue",
            "Highest predicted 168h failure risk"
        )

        for _, r in high.head(8).iterrows():

            st.markdown(
                f"""
                <div class="alert-item">

                    <div class="alert-id">

                        {r["component_id"]}

                        &nbsp;

                        {decision_badge(
                            r["Decision"]
                        )}

                    </div>

                    <div class="alert-meta">

                        {r["lot_id"]}
                        • Failure risk
                        {r["Failure Risk %"]:.1f}%
                        • Anomaly
                        {r["Anomaly Score"]:.3f}

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                f"Investigate {r['component_id']}",
                key=f"risk_{r['component_id']}",
                use_container_width=True
            ):

                open_component(
                    r["component_id"]
                )

    with right:

        panel_title(
            "Lot Risk",
            "Average predicted failure probability"
        )

        lot_risk = (
            display_df
            .groupby("lot_id")[
                "Failure Risk"
            ]
            .mean()
            .sort_values(
                ascending=True
            )
            .reset_index()
        )

        fig = px.bar(
            lot_risk,
            x="Failure Risk",
            y="lot_id",
            orientation="h"
        )

        fig.update_traces(
            marker_color="#4eb6ff"
        )

        fig.update_xaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            chart_dark(
                fig,
                360
            ),
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


# ============================================================
# AI INSIGHTS
# ============================================================

elif (
    st.session_state.nav_page
    == "AI Insights"
):

    cid_options = (
        display_df[
            "component_id"
        ].tolist()
    )

    current = (
        st.session_state.selected_component
    )

    if current not in cid_options:

        current = cid_options[0]

    cid = st.selectbox(
        "Component",
        cid_options,
        index=cid_options.index(
            current
        )
    )

    st.session_state.selected_component = cid

    row = (
        display_df[
            display_df["component_id"]
            == cid
        ]
        .iloc[0]
    )

    # IMPORTANT:
    # AI explanation now comes from FastAPI,
    # not directly from ScreeningEngine.

    explanation = get_component_explanation(
        row
    )

    decision = explanation[
        "decision"
    ]

    failure_probability = float(
        explanation[
            "failure_probability"
        ]
    )

    anomaly_score = float(
        explanation[
            "anomaly_score"
        ]
    )

    risk_level = explanation[
        "risk_level"
    ]

    page_header(
        "AI / EXPLANATION",
        f"Component Workspace — {cid}",
        f"{row['lot_id']} • model diagnosis and sensor evidence for the selected component.",
    )

    x1, x2, x3, x4 = st.columns(4)

    with x1:

        kpi(
            "Disposition",
            decision,
            "Screening decision",
            {
                "PASS": "green",
                "WARNING": "amber",
                "REJECT": "red"
            }.get(
                decision,
                "amber"
            )
        )

    with x2:

        risk_cls = (
            "red"
            if failure_probability > .60
            else
            "amber"
            if failure_probability > .25
            else
            "green"
        )

        kpi(
            "168h Failure Risk",
            f"{failure_probability:.1%}",
            "Predicted probability",
            risk_cls
        )

    with x3:

        kpi(
            "Anomaly Score",
            f"{anomaly_score:.3f}",
            "Isolation Forest score",
            "amber"
        )

    with x4:

        kpi(
            "Risk Level",
            risk_level,
            "Engineering priority",
            (
                "red"
                if risk_level == "HIGH RISK"
                else
                "amber"
                if risk_level == "MEDIUM RISK"
                else
                "green"
            )
        )

    shap_items = sorted(
        explanation[
            "shap_importance"
        ].items(),
        key=lambda item:
            abs(item[1]),
        reverse=True
    )

    if shap_items:

        primary_feature = (
            shap_items[0][0]
        )

        primary_impact = float(
            shap_items[0][1]
        )

    else:

        primary_feature = "N/A"
        primary_impact = 0.0

    action_title, action_copy, action_cls = (
        engineering_action(
            decision
        )
    )

    drift_values = {

        "Voltage drift (0h → 24h)":
            abs(
                float(
                    row["delta_v_0_24"]
                )
            ),

        "Current drift (0h → 24h)":
            abs(
                float(
                    row["delta_i_0_24"]
                )
            ),

        "Temperature drift (0h → 24h)":
            abs(
                float(
                    row["delta_temp_0_24"]
                )
            ),
    }

    strongest_signal = max(
        drift_values,
        key=drift_values.get
    )

    st.markdown(
        f"""
        <div class="diag-grid">

            <div class="diag-card">

                <div class="diag-label">
                    Primary AI driver
                </div>

                <div class="diag-value">
                    {feature_label(primary_feature)}
                </div>

                <div class="diag-copy">

                    SHAP contribution:
                    {primary_impact:+.3f}

                    —

                    {"increases"
                    if primary_impact > 0
                    else "reduces"}
                    predicted failure risk.

                </div>

            </div>


            <div class="diag-card">

                <div class="diag-label">
                    Early signal evidence
                </div>

                <div class="diag-value">
                    {strongest_signal}
                </div>

                <div class="diag-copy">

                    Derived from the observed
                    0h → 24h sensor drift.

                </div>

            </div>


            <div class="diag-card {action_cls}">

                <div class="diag-label">
                    Engineering action
                </div>

                <div class="diag-value">
                    {action_title}
                </div>

                <div class="diag-copy">
                    {action_copy}
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1.35, 1]
    )

    with left:

        panel_title(
            "Sensor Behavior",
            "Observed trajectory from 0h through 168h"
        )

        st.plotly_chart(
            sensor_chart(
                row,
                370
            ),
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    with right:

        panel_title(
            "AI Feature Contribution",
            "SHAP impact for the selected component"
        )

        shap_df = pd.DataFrame(
            [
                {
                    "Feature":
                        feature_label(key),
                    "Impact":
                        float(value)
                }
                for key, value
                in explanation[
                    "shap_importance"
                ].items()
            ]
        )

        if not shap_df.empty:

            shap_df = shap_df.sort_values(
                "Impact"
            )

            fig = px.bar(
                shap_df,
                x="Impact",
                y="Feature",
                orientation="h",
                color="Impact",
                color_continuous_scale=[
                    "#ff5b64",
                    "#273c50",
                    "#35d58b"
                ]
            )

            fig.update_coloraxes(
                showscale=False
            )

            st.plotly_chart(
                chart_dark(
                    fig,
                    370
                ),
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )

    st.markdown(
        '<div class="section-label">'
        'Early Signal Evidence'
        '</div>',
        unsafe_allow_html=True
    )

    drift = pd.DataFrame(
        {
            "Parameter": [
                "Voltage",
                "Current",
                "Temperature"
            ],

            "0h": [
                row["v_0h"],
                row["i_0h"],
                row["temp_0h"]
            ],

            "24h": [
                row["v_24h"],
                row["i_24h"],
                row["temp_24h"]
            ],

            "Delta 0h→24h": [
                row["delta_v_0_24"],
                row["delta_i_0_24"],
                row["delta_temp_0_24"]
            ],
        }
    )

    st.dataframe(
        drift,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "AI explanation is generated through the FastAPI "
        "backend using the trained screening model and "
        "SHAP feature contributions."
    )


# ============================================================
# MANUFACTURING
# ============================================================

elif (
    st.session_state.nav_page
    == "Manufacturing"
):

    page_header(
        "MANUFACTURING / LOTS",
        "Manufacturing Intelligence",
        "Compare lot-level screening outcomes and predicted reliability risk.",
    )

    lot_summary = (
        display_df
        .groupby("lot_id")
        .agg(
            Components=(
                "component_id",
                "count"
            ),
            Pass=(
                "Decision",
                lambda s:
                    int(
                        (s == "PASS").sum()
                    )
            ),
            Warning=(
                "Decision",
                lambda s:
                    int(
                        (s == "WARNING").sum()
                    )
            ),
            Reject=(
                "Decision",
                lambda s:
                    int(
                        (s == "REJECT").sum()
                    )
            ),
            AvgRisk=(
                "Failure Risk",
                "mean"
            )
        )
        .reset_index()
    )

    lot_summary["Reject Rate"] = (
        lot_summary["Reject"]
        / lot_summary["Components"]
    )

    st.dataframe(
        lot_summary.sort_values(
            "Reject Rate",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    a, b = st.columns(2)

    with a:

        panel_title(
            "Disposition by Lot",
            "Manufacturing quality distribution"
        )

        fig = go.Figure()

        for name, color in [
            (
                "Pass",
                "#35d58b"
            ),
            (
                "Warning",
                "#f2b84b"
            ),
            (
                "Reject",
                "#ff5b64"
            )
        ]:

            fig.add_trace(
                go.Bar(
                    x=lot_summary["lot_id"],
                    y=lot_summary[name],
                    name=name,
                    marker_color=color
                )
            )

        fig.update_layout(
            barmode="stack"
        )

        st.plotly_chart(
            chart_dark(
                fig,
                330
            ),
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    with b:

        panel_title(
            "Average Failure Risk",
            "Predicted probability by lot"
        )

        fig = px.bar(
            lot_summary.sort_values(
                "AvgRisk"
            ),
            x="AvgRisk",
            y="lot_id",
            orientation="h"
        )

        fig.update_traces(
            marker_color="#8c7cff"
        )

        fig.update_xaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            chart_dark(
                fig,
                330
            ),
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


# ============================================================
# REPORTS
# ============================================================

elif (
    st.session_state.nav_page
    == "Reports"
):

    page_header(
        "REPORTING / OUTPUT",
        "Screening Reports",
        "Export the current model decisions and component-level risk evidence.",
    )

    report = display_df[
        [
            "component_id",
            "lot_id",
            "Decision",
            "Failure Risk",
            "Failure Risk %",
            "Anomaly Score",
            "actual_failure_168h"
        ]
    ].copy()

    a, b, c = st.columns(3)

    with a:

        kpi(
            "Records",
            f"{len(report):,}",
            "Exportable component results",
            "cyan"
        )

    with b:

        kpi(
            "Rejects",
            f"{reject_count:,}",
            "Current disposition",
            "red"
        )

    with c:

        kpi(
            "Warnings",
            f"{warning_count:,}",
            "Current review queue",
            "amber"
        )

    st.markdown(
        '<div class="section-label">'
        'Export'
        '</div>',
        unsafe_allow_html=True
    )

    csv_bytes = (
        report
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "DOWNLOAD SCREENING REPORT",
        data=csv_bytes,
        file_name=(
            "screening_ai_component_report.csv"
        ),
        mime="text/csv",
        type="primary"
    )

    st.dataframe(
        report.head(100),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SETTINGS
# ============================================================

else:

    page_header(
        "SYSTEM / CONFIGURATION",
        "Platform Settings",
        "Current model, screening and dataset configuration.",
    )

    a, b = st.columns(2)

    with a:

        panel_title(
            "Screening Configuration"
        )

        st.markdown(
            """
            <div class="metric-strip">

                <div>
                    <div class="metric-name">
                        EARLY WINDOW
                    </div>
                    <div class="metric-val">
                        0h → 24h
                    </div>
                </div>

                <div>
                    <div class="metric-name">
                        TARGET
                    </div>
                    <div class="metric-val">
                        168h failure
                    </div>
                </div>

            </div>

            <div class="metric-strip">

                <div>
                    <div class="metric-name">
                        ANOMALY
                    </div>
                    <div class="metric-val">
                        Isolation Forest
                    </div>
                </div>

                <div>
                    <div class="metric-name">
                        PREDICTOR
                    </div>
                    <div class="metric-val">
                        XGBoost
                    </div>
                </div>

            </div>

            <div class="metric-strip">

                <div>
                    <div class="metric-name">
                        COMPONENTS
                    </div>
                    <div class="metric-val">
                        1,500
                    </div>
                </div>

                <div>
                    <div class="metric-name">
                        BACKEND
                    </div>
                    <div class="metric-val">
                        FastAPI
                    </div>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with b:

        panel_title(
            "Model Validation"
        )

        st.metric(
            "Accuracy",
            f"{model_metrics['accuracy']:.2%}"
        )

        st.metric(
            "ROC-AUC",
            f"{model_metrics['roc_auc']:.3f}"
        )

        st.caption(
            "Development validation only — "
            "current dataset is synthetic."
        )

    st.markdown(
        '<div class="section-label">'
        'About'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Screening AI is an industrial-style early "
        "screening interface for ESS component "
        "reliability analysis. The frontend communicates "
        "with the FastAPI backend for training, batch "
        "prediction and AI explanations."
    )


