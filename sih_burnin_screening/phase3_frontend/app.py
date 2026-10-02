
import json
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

API_URL = st.sidebar.text_input("Backend API URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="AI Burn-In Screening",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
.main { background: #0b1020; }
.block-container { padding-top: 1.2rem; }
.hero {
    padding: 22px 26px;
    border-radius: 18px;
    background: linear-gradient(135deg,#111a33,#16223f);
    border: 1px solid #26365c;
    margin-bottom: 18px;
}
.hero h1 { margin: 0; color: #eaf2ff; font-size: 34px; }
.hero p { color: #9fb0cf; margin: 8px 0 0; }
.card {
    padding: 18px;
    border-radius: 16px;
    background: #11182b;
    border: 1px solid #26365c;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>⚡ AI-Driven Component Burn-In & Screening</h1>
<p>Phase 3 Frontend connected to the FastAPI + ML backend</p>
</div>
""", unsafe_allow_html=True)

# Backend health
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    if health.ok:
        h = health.json()
        st.sidebar.success(f"Backend online • trained: {h.get('model_trained', 'unknown')}")
    else:
        st.sidebar.warning(f"Backend returned HTTP {health.status_code}")
except Exception:
    st.sidebar.error("Backend offline. Start FastAPI on port 8000.")

tabs = st.tabs(["Dashboard", "Train Model", "API Result"])

with tabs[1]:
    st.subheader("Train the screening model")
    st.caption("Upload the same ESS dataset used in Phase 1.")
    csv_file = st.file_uploader("ESS CSV dataset", type=["csv"], key="train_csv")

    if csv_file is not None:
        try:
            preview = pd.read_csv(csv_file)
            st.write(f"Dataset loaded: **{len(preview):,} rows × {len(preview.columns)} columns**")
            st.dataframe(preview.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if st.button("Train Model", type="primary", disabled=csv_file is None):
        try:
            csv_file.seek(0)
            with st.spinner("Training Isolation Forest + XGBoost..."):
                r = requests.post(
                    f"{API_URL}/train",
                    files={"file": (csv_file.name, csv_file.getvalue(), "text/csv")},
                    timeout=120,
                )
            if r.ok:
                result = r.json()
                st.session_state["last_api_result"] = result
                st.success("Model trained successfully.")
                st.json(result)
            else:
                st.error(f"Training failed: HTTP {r.status_code}")
                st.code(r.text)
        except Exception as e:
            st.error(f"Could not reach backend: {e}")

with tabs[0]:
    st.subheader("Component Screening")

    c1, c2 = st.columns(2)
    with c1:
        component_id = st.text_input("Component ID", "CMP-0001")
        v_0h = st.number_input("Voltage 0h (V)", value=5.02, format="%.4f")
        i_0h = st.number_input("Current 0h (mA)", value=120.5, format="%.4f")
        temp_0h = st.number_input("Temperature 0h (°C)", value=35.2, format="%.4f")
    with c2:
        v_24h = st.number_input("Voltage 24h (V)", value=5.03, format="%.4f")
        i_24h = st.number_input("Current 24h (mA)", value=121.0, format="%.4f")
        temp_24h = st.number_input("Temperature 24h (°C)", value=36.2, format="%.4f")

    payload = {
        "v_0h": v_0h,
        "i_0h": i_0h,
        "temp_0h": temp_0h,
        "v_24h": v_24h,
        "i_24h": i_24h,
        "temp_24h": temp_24h,
        "delta_v_0_24": v_24h - v_0h,
        "delta_i_0_24": i_24h - i_0h,
        "delta_temp_0_24": temp_24h - temp_0h,
    }

    st.markdown("### Early Drift")
    drift_df = pd.DataFrame({
        "Parameter": ["Voltage", "Current", "Temperature"],
        "0h": [v_0h, i_0h, temp_0h],
        "24h": [v_24h, i_24h, temp_24h],
        "Delta": [payload["delta_v_0_24"], payload["delta_i_0_24"], payload["delta_temp_0_24"]],
    })
    st.dataframe(drift_df, use_container_width=True)

    if st.button("Run AI Screening", type="primary"):
        try:
            with st.spinner("Running AI screening..."):
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)

            if r.ok:
                result = r.json()
                st.session_state["last_api_result"] = result
                decision = str(result.get("decision", "UNKNOWN"))
                risk = str(result.get("risk_level", "UNKNOWN"))
                prob = result.get("failure_probability", result.get("predicted_failure_probability", None))
                anomaly = result.get("anomaly_score", None)

                if decision == "PASS":
                    st.success(f"DECISION: {decision} • {risk}")
                elif decision == "WARNING":
                    st.warning(f"DECISION: {decision} • {risk}")
                elif decision == "REJECT":
                    st.error(f"DECISION: {decision} • {risk}")
                else:
                    st.info(f"DECISION: {decision} • {risk}")

                m1, m2, m3 = st.columns(3)
                m1.metric("Component", component_id)
                m2.metric("Failure Probability", f"{float(prob):.1%}" if prob is not None else "N/A")
                m3.metric("Anomaly Score", f"{float(anomaly):.3f}" if anomaly is not None else "N/A")

                st.session_state["last_payload"] = payload
            else:
                st.error(f"Prediction failed: HTTP {r.status_code}")
                st.code(r.text)
        except Exception as e:
            st.error(f"Could not reach backend: {e}")

    result = st.session_state.get("last_api_result")
    if result:
        st.markdown("### AI Explanation")
        shap_data = result.get("shap_importance") or result.get("shap_values")
        if isinstance(shap_data, dict) and shap_data:
            sdf = pd.DataFrame(
                [{"Feature": k, "Impact": float(v)} for k, v in shap_data.items()]
            ).sort_values("Impact")
            fig = go.Figure(go.Bar(
                x=sdf["Impact"], y=sdf["Feature"], orientation="h"
            ))
            fig.update_layout(
                title="SHAP Feature Impact",
                height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10,r=10,t=50,b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SHAP explanation was returned by the backend response.")

with tabs[2]:
    st.subheader("Latest Backend Response")
    if st.session_state.get("last_api_result"):
        st.json(st.session_state["last_api_result"])
    else:
        st.info("Run training or prediction to see the API response here.")
