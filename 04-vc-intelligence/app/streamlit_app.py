@'
# ── app/streamlit_app.py ──────────────────────────────────────────────────────
# Purpose: Two-tab UI — Deal Screener + Market Trends
# Deployed on HuggingFace Spaces (free forever)
# Calls FastAPI on Cloud Run for predictions and search

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = st.secrets.get("API_BASE_URL", "http://localhost:8080")
SAMPLE_PATH = Path("data/sample/startups_sample.csv")

st.set_page_config(
    page_title="VC Intelligence Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size: 2rem; font-weight: 700; color: #1a1a2e;}
    .sub-header {font-size: 1rem; color: #666; margin-bottom: 1.5rem;}
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #4CAF50;
    }
    .sector-tag {
        background: #e3f2fd;
        color: #1565c0;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .risk-tag {
        background: #fce4ec;
        color: #c62828;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────
def call_predict(description: str) -> dict:
    try:
        response = requests.post(
            f"{API_BASE}/predict",
            json={"description": description},
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API returned {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "API unavailable — is the FastAPI server running?"}
    except Exception as e:
        return {"error": str(e)}


def call_search(query: str, top_k: int = 5) -> dict:
    try:
        response = requests.post(
            f"{API_BASE}/search",
            json={"query": query, "top_k": top_k},
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API returned {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def load_sample_data() -> pd.DataFrame:
    if SAMPLE_PATH.exists():
        return pd.read_csv(SAMPLE_PATH)
    return pd.DataFrame()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔍 Startup Funding Intelligence Tool</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Screens a deal in 30 seconds instead of 30 minutes. '
    'Built by a former VC analyst to automate repetitive pattern recognition in deal flow.</p>',
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎯 Deal Screener", "📊 Market Trends"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEAL SCREENER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Paste a startup description")
        description = st.text_area(
            label="Description",
            placeholder=(
                "e.g. AI-powered carbon accounting platform for mid-market companies. "
                "SOC2 certified. Integrates with QuickBooks and SAP. "
                "$1.1M ARR from 85 paying customers."
            ),
            height=180,
            label_visibility="collapsed",
        )

        # Example buttons
        st.caption("Or try an example:")
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        with ex_col1:
            if st.button("🌱 Climate Tech"):
                description = (
                    "Automated Scope 1/2/3 carbon accounting platform for "
                    "mid-market companies. SOC2 certified. Integrates with "
                    "QuickBooks. $1.1M ARR, 85 customers."
                )
        with ex_col2:
            if st.button("🏥 Healthcare AI"):
                description = (
                    "Large language model for automated clinical documentation. "
                    "Integrates with Epic EHR. FDA 510k submitted. "
                    "3 hospital pilots signed."
                )
        with ex_col3:
            if st.button("💳 Fintech"):
                description = (
                    "Cross-border B2B payments infrastructure for emerging markets. "
                    "Settles in local currency without correspondent banks. "
                    "$2.1M TPV monthly, 340 SMB importers."
                )

        screen_btn = st.button("🔍 Screen this deal", type="primary", use_container_width=True)

    with col2:
        if screen_btn and description:
            with st.spinner("Screening deal..."):
                result = call_predict(description)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                pred = result.get("sector_prediction", {})
                llm = result.get("llm_extraction", {})
                latency = result.get("latency_ms", {})

                # Sector prediction
                st.markdown(
                    f'<span class="sector-tag">📂 {pred.get("sector", "Unknown")}</span>',
                    unsafe_allow_html=True,
                )
                st.progress(pred.get("confidence", 0))
                st.caption(f'Confidence: {pred.get("confidence", 0):.1%}')

                # Top 3 predictions
                with st.expander("All sector predictions"):
                    for p in pred.get("top_3", []):
                        st.write(f"**{p['sector']}** — {p['confidence']:.1%}")

                st.divider()

                # LLM extraction results
                if llm.get("success"):
                    m1, m2 = st.columns(2)
                    with m1:
                        score = llm.get("traction_score")
                        if score:
                            st.metric("Traction Score", f"{score}/10")
                    with m2:
                        latency_c = latency.get("classifier_ms", 0)
                        latency_l = latency.get("llm_ms", 0)
                        st.metric("Screened in", f"{latency_c + latency_l:.0f}ms")

                    if llm.get("key_metrics"):
                        st.markdown("**Key Metrics**")
                        for m in llm["key_metrics"]:
                            st.write(f"• {m}")

                    if llm.get("investment_signal"):
                        st.info(f"💡 {llm['investment_signal']}")

                    if llm.get("risk_flags"):
                        st.markdown("**Risk Flags**")
                        for r in llm["risk_flags"]:
                            st.markdown(
                                f'<span class="risk-tag">⚠ {r}</span>',
                                unsafe_allow_html=True,
                            )
                else:
                    st.warning(
                        f"LLM extraction unavailable: {llm.get('error', 'unknown error')}. "
                        "Sector classification still valid."
                    )

        elif screen_btn and not description:
            st.warning("Paste a startup description first.")

    st.divider()

    # ── Semantic search ───────────────────────────────────────────────────────
    st.subheader("🔎 Find comparable startups")
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        query = st.text_input(
            "Search",
            placeholder="e.g. climate tech with enterprise SaaS revenue",
            label_visibility="collapsed",
        )
    with search_col2:
        top_k = st.selectbox("Results", [3, 5, 10], index=1, label_visibility="collapsed")

    if st.button("Search comparables", use_container_width=True):
        if query:
            with st.spinner("Searching..."):
                search_result = call_search(query, top_k)

            if "error" in search_result:
                st.error(search_result["error"])
            else:
                results = search_result.get("results", [])
                if results:
                    st.caption(f"Found {len(results)} comparables in {search_result.get('latency_ms', 0):.0f}ms")
                    for r in results:
                        with st.container():
                            c1, c2, c3 = st.columns([3, 1, 1])
                            with c1:
                                st.markdown(f"**{r['name']}**")
                                st.caption(r["description_snippet"])
                            with c2:
                                st.markdown(f'<span class="sector-tag">{r["sector"]}</span>',
                                           unsafe_allow_html=True)
                            with c3:
                                st.metric("Similarity", f"{r['similarity_score']:.0%}")
                            st.divider()
                else:
                    st.info("No results found. Try a different query.")
        else:
            st.warning("Enter a search query.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    df = load_sample_data()

    if df.empty:
        st.warning("Sample data not found.")
    else:
        df["funding_M"] = df["funding_amount_usd"] / 1e6

        st.subheader("Startup Funding Market Overview")
        st.caption("Based on sample dataset — full dataset in Looker Studio dashboard")

        # ── KPI row ───────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Startups", len(df))
        k2.metric("Sectors Covered", df["sector"].nunique())
        k3.metric("Median Deal Size", f"${df['funding_M'].median():.1f}M")
        k4.metric("Countries", df["hq_country"].nunique())

        st.divider()

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Sector heatmap
            sector_counts = df["sector"].value_counts().reset_index()
            sector_counts.columns = ["sector", "count"]
            fig1 = px.bar(
                sector_counts,
                x="count", y="sector",
                orientation="h",
                title="Deal Count by Sector",
                color="count",
                color_continuous_scale="Blues",
            )
            fig1.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig1, use_container_width=True)

        with chart_col2:
            # Funding by stage
            stage_order = ["Seed", "Series A", "Series B", "Series C"]
            stage_funding = (
                df.groupby("stage")["funding_M"]
                .median()
                .reindex(stage_order)
                .reset_index()
            )
            stage_funding.columns = ["stage", "median_funding_M"]
            fig2 = px.bar(
                stage_funding,
                x="stage", y="median_funding_M",
                title="Median Deal Size by Stage ($M)",
                color="stage",
                color_discrete_sequence=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"],
            )
            fig2.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig2, use_container_width=True)

        # Revenue stage breakdown
        rev_col1, rev_col2 = st.columns(2)
        with rev_col1:
            rev_counts = df["revenue_stage"].value_counts().reset_index()
            rev_counts.columns = ["revenue_stage", "count"]
            fig3 = px.pie(
                rev_counts,
                names="revenue_stage", values="count",
                title="Revenue Stage Distribution",
                hole=0.4,
            )
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)

        with rev_col2:
            # Funding trend by year
            year_funding = (
                df.groupby("funding_year")["funding_M"]
                .sum()
                .reset_index()
            )
            fig4 = px.line(
                year_funding,
                x="funding_year", y="funding_M",
                title="Total Funding by Year ($M)",
                markers=True,
            )
            fig4.update_layout(height=350)
            st.plotly_chart(fig4, use_container_width=True)

        st.divider()
        st.caption(
            "📊 Full interactive dashboard: [Looker Studio](#) · "
            "Data refreshes weekly via GitHub Actions · "
            "Built by Hari Etta — [GitHub](https://github.com/HariEtta)"
        )
'@ | Set-Content -Path "app/streamlit_app.py" -Encoding UTF8

Write-Host "Done" -ForegroundColor Green