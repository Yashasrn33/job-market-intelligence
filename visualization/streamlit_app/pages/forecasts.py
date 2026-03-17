"""
Forecasts page — ML-driven skill demand forecasting.

Wires directly into ml.inference.predictor.SkillDemandPredictor for
live model predictions when a trained model is available, and falls
back to Athena-only trend extrapolation otherwise.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import awswrangler as wr
import boto3
REGION = os.getenv("AWS_REGION", "us-east-1")
BOTO3_SESSION = boto3.Session(region_name=REGION)

st.set_page_config(page_title="Skill Forecasts", page_icon="🔮", layout="wide")

DATABASE = os.getenv("GLUE_DATABASE", "job_market_db")
REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_DIR = os.getenv("MODEL_DIR", "ml/models")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@st.cache_data(ttl=3600)
def run_query(query: str) -> pd.DataFrame:
    try:
        return wr.athena.read_sql_query(
            query,
            database=DATABASE,
            boto3_session=BOTO3_SESSION,
        )
    except Exception as exc:
        st.error(f"Query error: {exc}")
        return pd.DataFrame()


def _model_available() -> bool:
    model_path = PROJECT_ROOT / MODEL_DIR
    return (model_path / "demand_model.pkl").exists()


@st.cache_resource
def get_predictor():
    from ml.inference.predictor import SkillDemandPredictor

    model_path = PROJECT_ROOT / MODEL_DIR
    return SkillDemandPredictor(model_path=str(model_path))


@st.cache_data(ttl=3600)
def get_skill_time_series(skills: list[str]) -> pd.DataFrame:
    skill_list = ", ".join(f"'{s}'" for s in skills)
    return run_query(f"""
        SELECT skill,
               DATE_TRUNC('week', posted_date) AS week,
               COUNT(*)            AS job_count,
               AVG(salary_mid_usd) AS avg_salary
        FROM {DATABASE}.job_skills
        WHERE skill IN ({skill_list})
          AND posted_date >= DATE_ADD('week', -24, CURRENT_DATE)
        GROUP BY skill, DATE_TRUNC('week', posted_date)
        ORDER BY skill, week
    """)


@st.cache_data(ttl=3600)
def get_all_skills() -> list[str]:
    df = run_query(f"SELECT DISTINCT skill FROM {DATABASE}.job_skills ORDER BY skill")
    return df["skill"].tolist() if not df.empty else []


def forecast_chart(ts: pd.DataFrame, skill: str, predicted: float | None = None):
    """Historical line + 4-week forecast with confidence band."""
    df = ts[ts["skill"] == skill].copy()
    if df.empty:
        return None

    df["week"] = pd.to_datetime(df["week"])
    df = df.sort_values("week")

    recent = df.tail(4)["job_count"].values
    trend = (recent[-1] - recent[0]) / max(len(recent), 1) if len(recent) >= 2 else 0
    last_val = df["job_count"].iloc[-1]
    last_date = df["week"].iloc[-1]

    if predicted is not None:
        step = (predicted - last_val) / 4
    else:
        step = trend

    fc_dates = [last_date + timedelta(weeks=i + 1) for i in range(4)]
    fc_vals = [last_val + step * (i + 1) for i in range(4)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["week"], y=df["job_count"],
        mode="lines+markers", name="Historical",
        line=dict(color="#6366f1", width=2), marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=fc_vals,
        mode="lines+markers", name="Forecast",
        line=dict(color="#10b981", width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
    ))
    upper = [v * 1.20 for v in fc_vals]
    lower = [max(0, v * 0.80) for v in fc_vals]
    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1],
        y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(16,185,129,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80 % CI",
    ))
    fig.update_layout(
        title=f"{skill} — Weekly Demand Forecast",
        xaxis_title="Week", yaxis_title="Job Postings",
        hovermode="x unified", height=380,
    )
    return fig


# ── page content ─────────────────────────────────────────────────────────────

st.title("🔮 Skill Demand Forecasts")

has_model = _model_available()

if has_model:
    st.success("Trained model loaded from `ml/models/`")
    source_label = "ML Model (XGBoost)"
else:
    st.info(
        "No trained model found — showing Athena trend extrapolation. "
        "Run `python -m ml.training.train --bucket <bucket> --data-source kaggle` "
        "to enable ML-powered forecasts."
    )
    source_label = "Trend Extrapolation"

st.caption(f"Prediction source: **{source_label}**")
st.divider()

# ── Tab layout ───────────────────────────────────────────────────────────────

tab_overview, tab_drilldown, tab_emerging = st.tabs([
    "📊 Demand Overview", "🔍 Skill Drill-Down", "🚀 Emerging Skills"
])

# ── TAB 1: Demand overview ──────────────────────────────────────────────────

with tab_overview:
    if has_model:
        try:
            predictor = get_predictor()
            forecasts = predictor.forecast_skill_demand(top_n=30)
        except Exception as exc:
            st.error(f"Model prediction failed: {exc}")
            forecasts = pd.DataFrame()
    else:
        forecasts = run_query(f"""
            WITH weekly AS (
                SELECT skill,
                       DATE_TRUNC('week', posted_date) AS week,
                       COUNT(*) AS job_count
                FROM {DATABASE}.job_skills
                WHERE posted_date >= DATE_ADD('week', -12, CURRENT_DATE)
                GROUP BY skill, DATE_TRUNC('week', posted_date)
            ),
            trends AS (
                SELECT skill,
                       SUM(job_count) AS total_jobs,
                       ROUND(AVG(job_count), 1) AS avg_weekly,
                       (MAX(CASE WHEN week >= DATE_ADD('week', -4, CURRENT_DATE) THEN job_count END)
                       - MAX(CASE WHEN week < DATE_ADD('week', -4, CURRENT_DATE)
                                   AND week >= DATE_ADD('week', -8, CURRENT_DATE)
                              THEN job_count END)) * 100.0
                       / NULLIF(MAX(CASE WHEN week < DATE_ADD('week', -4, CURRENT_DATE)
                                          AND week >= DATE_ADD('week', -8, CURRENT_DATE)
                                         THEN job_count END), 0) AS growth_pct
                FROM weekly
                GROUP BY skill HAVING COUNT(*) >= 4
            )
            SELECT skill,
                   total_jobs AS current_demand,
                   ROUND(avg_weekly * (1 + COALESCE(growth_pct, 0)/100), 1) AS predicted_demand,
                   ROUND(COALESCE(growth_pct, 0), 1) AS growth_pct
            FROM trends
            ORDER BY total_jobs DESC LIMIT 30
        """)

    if not forecasts.empty:
        c1, c2, c3 = st.columns(3)
        growing = int((forecasts["growth_pct"] > 0).sum())
        declining = int((forecasts["growth_pct"] < 0).sum())
        c1.metric("Skills Growing", growing, delta=f"{growing}/{len(forecasts)}")
        c2.metric("Skills Declining", declining)
        c3.metric("Avg Growth", f"{forecasts['growth_pct'].mean():.1f} %")

        st.divider()
        left, right = st.columns(2)

        with left:
            st.markdown("### Fastest Growing")
            up = forecasts[forecasts["growth_pct"] > 0].nlargest(10, "growth_pct")
            if not up.empty:
                fig = px.bar(
                    up, x="growth_pct", y="skill", orientation="h",
                    color="growth_pct", color_continuous_scale="Greens",
                    labels={"growth_pct": "Growth %", "skill": ""},
                )
                fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("### Declining Skills")
            down = forecasts[forecasts["growth_pct"] < 0].nsmallest(10, "growth_pct")
            if not down.empty:
                fig = px.bar(
                    down, x="growth_pct", y="skill", orientation="h",
                    color="growth_pct", color_continuous_scale="Reds_r",
                    labels={"growth_pct": "Growth %", "skill": ""},
                )
                fig.update_layout(height=400, yaxis={"categoryorder": "total descending"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No declining skills detected.")

        st.divider()
        st.markdown("### Full Forecast Table")
        disp = forecasts.copy()
        display_cols = [c for c in ["skill", "current_demand", "predicted_demand", "growth_pct",
                                     "current_salary", "demand_change"] if c in disp.columns]
        st.dataframe(disp[display_cols], use_container_width=True, hide_index=True)
    else:
        st.warning("No data available. Run the ingestion pipeline first.")

# ── TAB 2: Skill drill-down ────────────────────────────────────────────────

with tab_drilldown:
    st.markdown("Select specific skills to see their historical trend and 4-week forecast.")

    all_skills = get_all_skills()
    if not all_skills:
        st.warning("No skills data available yet.")
    else:
        default_picks = [s for s in ["python", "react", "aws", "kubernetes"] if s in all_skills]
        selected = st.multiselect(
            "Choose skills to forecast",
            all_skills,
            default=default_picks[:3],
            max_selections=6,
        )

        if selected:
            ts = get_skill_time_series(selected)

            predicted_map: dict[str, float | None] = {}
            if has_model:
                try:
                    predictor = get_predictor()
                    skill_forecast = predictor.forecast_skill_demand(skills=selected)
                    for _, row in skill_forecast.iterrows():
                        predicted_map[row["skill"]] = float(row["predicted_demand"])
                except Exception:
                    pass

            cols = st.columns(min(len(selected), 3))
            for i, skill in enumerate(selected):
                with cols[i % len(cols)]:
                    chart = forecast_chart(ts, skill, predicted_map.get(skill))
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.info(f"No data for {skill}")

            st.divider()
            st.markdown("### Side-by-Side Comparison")
            if not ts.empty:
                ts["week"] = pd.to_datetime(ts["week"])
                fig = px.line(
                    ts, x="week", y="job_count", color="skill",
                    labels={"week": "Week", "job_count": "Job Postings", "skill": "Skill"},
                )
                fig.update_layout(height=400, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Emerging skills ─────────────────────────────────────────────────

with tab_emerging:
    st.markdown("Skills showing unusual growth — potential breakout technologies.")

    if has_model:
        try:
            predictor = get_predictor()
            emerging = predictor.detect_emerging_skills(threshold=0.7, top_n=15)
        except Exception as exc:
            st.error(f"Emergence detection failed: {exc}")
            emerging = pd.DataFrame()
    else:
        emerging = run_query(f"""
            WITH recent AS (
                SELECT skill, COUNT(*) AS current_jobs, AVG(salary_mid_usd) AS avg_salary
                FROM {DATABASE}.job_skills
                WHERE posted_date >= DATE_ADD('week', -4, CURRENT_DATE)
                GROUP BY skill
            ),
            previous AS (
                SELECT skill, COUNT(*) AS prev_jobs
                FROM {DATABASE}.job_skills
                WHERE posted_date >= DATE_ADD('week', -8, CURRENT_DATE)
                  AND posted_date < DATE_ADD('week', -4, CURRENT_DATE)
                GROUP BY skill
            )
            SELECT r.skill,
                   r.current_jobs,
                   ROUND((r.current_jobs - COALESCE(p.prev_jobs, 1)) * 100.0
                         / COALESCE(p.prev_jobs, 1), 1) AS monthly_growth_pct,
                   ROUND(r.avg_salary, 0) AS avg_salary
            FROM recent r LEFT JOIN previous p ON r.skill = p.skill
            WHERE r.current_jobs >= 3
            ORDER BY monthly_growth_pct DESC
            LIMIT 15
        """)
        if not emerging.empty and "emergence_score" not in emerging.columns:
            emerging["emergence_score"] = (
                emerging["monthly_growth_pct"] / emerging["monthly_growth_pct"].max()
            ).round(3)

    if not emerging.empty:
        c1, c2 = st.columns(2)
        c1.metric("Emerging Skills Detected", len(emerging))
        if "monthly_growth_pct" in emerging.columns:
            c2.metric(
                "Avg Monthly Growth",
                f"{emerging['monthly_growth_pct'].mean():.1f} %"
            )

        st.divider()

        size_col = "avg_salary" if "avg_salary" in emerging.columns else None
        fig = px.scatter(
            emerging,
            x="current_jobs" if "current_jobs" in emerging.columns else emerging.columns[1],
            y="monthly_growth_pct" if "monthly_growth_pct" in emerging.columns else "emergence_score",
            size=size_col,
            color="emergence_score",
            hover_name="skill",
            color_continuous_scale="YlOrRd",
            labels={
                "current_jobs": "Current Weekly Demand",
                "monthly_growth_pct": "Monthly Growth %",
                "emergence_score": "Emergence Score",
            },
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### Emerging Skills Detail")
        st.dataframe(emerging, use_container_width=True, hide_index=True)
    else:
        st.info("No emerging skills detected yet. More data may be needed.")

# ── footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
    f"Source: {source_label}"
)
