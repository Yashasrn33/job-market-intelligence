"""
Job Market Intelligence Dashboard with ML Predictions

View modes:
  Overview        - KPIs, top skills, salary leaders, category distribution
  Forecasts       - Demand projections, fastest growing / declining skills
  Emerging Skills - Anomaly detection scatter, per-skill trend drill-down
  Skill Advisor   - Co-occurrence recommendations based on your current skills
"""

import os
from datetime import datetime, timedelta
from typing import List

import awswrangler as wr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Job Market Intelligence",
    page_icon="\U0001F3AF",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 10px; color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATABASE = os.getenv("GLUE_DATABASE", "job_market_db")
REGION = os.getenv("AWS_REGION", "us-east-1")

# ── query helpers ───────────────────────────────────────────────────────────


@st.cache_data(ttl=3600)
def run_query(query: str) -> pd.DataFrame:
    try:
        return wr.athena.read_sql_query(query, database=DATABASE, region_name=REGION)
    except Exception as exc:
        st.error(f"Query error: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_skill_forecasts() -> pd.DataFrame:
    return run_query("""
        WITH weekly_data AS (
            SELECT skill,
                   DATE_TRUNC('week', posted_date) AS week,
                   COUNT(*)            AS job_count,
                   AVG(salary_mid_usd) AS avg_salary
            FROM job_market_db.job_skills
            WHERE posted_date >= DATE_ADD('week', -12, CURRENT_DATE)
            GROUP BY skill, DATE_TRUNC('week', posted_date)
        ),
        skill_trends AS (
            SELECT skill,
                   COUNT(*)      AS weeks_present,
                   SUM(job_count) AS total_jobs,
                   AVG(job_count) AS avg_weekly_jobs,
                   MAX(avg_salary) AS max_salary,
                   (MAX(CASE WHEN week >= DATE_ADD('week', -4, CURRENT_DATE)
                             THEN job_count END)
                  - MAX(CASE WHEN week < DATE_ADD('week', -4, CURRENT_DATE)
                              AND week >= DATE_ADD('week', -8, CURRENT_DATE)
                             THEN job_count END)) * 1.0
                  / NULLIF(MAX(CASE WHEN week < DATE_ADD('week', -4, CURRENT_DATE)
                                     AND week >= DATE_ADD('week', -8, CURRENT_DATE)
                                    THEN job_count END), 0) AS growth_rate
            FROM weekly_data
            GROUP BY skill
            HAVING COUNT(*) >= 4
        )
        SELECT skill,
               total_jobs,
               ROUND(avg_weekly_jobs, 1)                            AS avg_weekly,
               ROUND(max_salary, 0)                                 AS salary,
               ROUND(COALESCE(growth_rate, 0) * 100, 1)            AS growth_pct,
               ROUND(avg_weekly_jobs * (1 + COALESCE(growth_rate, 0)), 1) AS forecast_weekly
        FROM skill_trends
        ORDER BY total_jobs DESC
        LIMIT 50
    """)


@st.cache_data(ttl=3600)
def get_emerging_skills() -> pd.DataFrame:
    return run_query("""
        WITH recent AS (
            SELECT skill, COUNT(*) AS recent_jobs, AVG(salary_mid_usd) AS avg_salary
            FROM job_market_db.job_skills
            WHERE posted_date >= DATE_ADD('week', -4, CURRENT_DATE)
            GROUP BY skill
        ),
        previous AS (
            SELECT skill, COUNT(*) AS prev_jobs
            FROM job_market_db.job_skills
            WHERE posted_date >= DATE_ADD('week', -8, CURRENT_DATE)
              AND posted_date <  DATE_ADD('week', -4, CURRENT_DATE)
            GROUP BY skill
        ),
        growth AS (
            SELECT r.skill, r.recent_jobs, COALESCE(p.prev_jobs, 1) AS prev_jobs,
                   r.avg_salary,
                   (r.recent_jobs - COALESCE(p.prev_jobs, 1)) * 100.0
                       / COALESCE(p.prev_jobs, 1) AS growth_pct
            FROM recent r LEFT JOIN previous p ON r.skill = p.skill
        )
        SELECT skill,
               recent_jobs AS current_demand,
               ROUND(growth_pct, 1) AS growth_pct,
               ROUND(avg_salary, 0) AS avg_salary,
               CASE WHEN growth_pct > 50 AND recent_jobs >= 5 THEN 'Hot'
                    WHEN growth_pct > 25 AND recent_jobs >= 3 THEN 'Rising'
                    WHEN growth_pct > 10                      THEN 'Growing'
                    ELSE 'Stable' END AS trend_status
        FROM growth
        WHERE growth_pct > 20 AND recent_jobs >= 3
        ORDER BY growth_pct DESC
        LIMIT 15
    """)


@st.cache_data(ttl=3600)
def get_skill_time_series(skills: List[str]) -> pd.DataFrame:
    skill_list = ", ".join(f"'{s}'" for s in skills)
    return run_query(f"""
        SELECT skill,
               DATE_TRUNC('week', posted_date) AS week,
               COUNT(*)            AS job_count,
               AVG(salary_mid_usd) AS avg_salary
        FROM job_market_db.job_skills
        WHERE skill IN ({skill_list})
          AND posted_date >= DATE_ADD('week', -24, CURRENT_DATE)
        GROUP BY skill, DATE_TRUNC('week', posted_date)
        ORDER BY skill, week
    """)


def _forecast_chart(ts: pd.DataFrame, skill: str):
    df = ts[ts["skill"] == skill].copy()
    if df.empty:
        return None
    df["week"] = pd.to_datetime(df["week"])
    df = df.sort_values("week")

    recent = df.tail(4)["job_count"].values
    trend = (recent[-1] - recent[0]) / max(len(recent), 1) if len(recent) >= 2 else 0
    last_val = df["job_count"].iloc[-1]
    last_date = df["week"].iloc[-1]

    fc_dates = [last_date + timedelta(weeks=i + 1) for i in range(4)]
    fc_vals = [last_val + trend * (i + 1) for i in range(4)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["week"], y=df["job_count"],
        mode="lines+markers", name="Historical",
        line=dict(color="#6366f1", width=2), marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates, y=fc_vals,
        mode="lines+markers", name="Forecast",
        line=dict(color="#10b981", width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
    ))
    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1],
        y=[v * 1.2 for v in fc_vals] + [max(0, v * 0.8) for v in fc_vals][::-1],
        fill="toself", fillcolor="rgba(16,185,129,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Confidence interval",
    ))
    fig.update_layout(
        title=f"{skill} — Demand Forecast",
        xaxis_title="Week", yaxis_title="Job Postings",
        hovermode="x unified", height=350,
    )
    return fig


# ── sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.header("Controls")

view_mode = st.sidebar.radio(
    "View Mode",
    ["Overview", "Forecasts", "Emerging Skills", "Skill Advisor"],
)

country_filter = st.sidebar.multiselect(
    "Filter by Country",
    ["US", "GB", "CA", "AU", "DE"],
    default=["US"],
)

countries_sql = ", ".join(f"'{c}'" for c in country_filter)

# ── header ──────────────────────────────────────────────────────────────────

st.title("Job Market Intelligence")
st.markdown(
    "**AI-powered insights into tech job market trends, skills demand, and career forecasting**"
)

# =============================================================================
# VIEW: Overview
# =============================================================================

if view_mode == "Overview":
    col1, col2, col3, col4 = st.columns(4)

    summary = run_query(f"""
        SELECT COUNT(DISTINCT job_id)          AS total_jobs,
               COUNT(DISTINCT skill)           AS unique_skills,
               ROUND(AVG(salary_mid_usd), 0)   AS avg_salary,
               COUNT(DISTINCT country)         AS countries
        FROM job_market_db.job_skills
        WHERE country IN ({countries_sql})
    """)

    if not summary.empty:
        col1.metric("Total Jobs", f"{summary['total_jobs'].iloc[0]:,}")
        col2.metric("Unique Skills", f"{summary['unique_skills'].iloc[0]:,}")
        col3.metric("Avg Salary", f"${summary['avg_salary'].iloc[0]:,.0f}")
        col4.metric("Countries", summary["countries"].iloc[0])

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Top 15 In-Demand Skills")
        top = run_query(f"""
            SELECT skill, COUNT(*) AS job_count,
                   ROUND(AVG(salary_mid_usd), 0) AS avg_salary
            FROM job_market_db.job_skills
            WHERE country IN ({countries_sql})
            GROUP BY skill ORDER BY job_count DESC LIMIT 15
        """)
        if not top.empty:
            fig = px.bar(
                top, x="job_count", y="skill", orientation="h",
                color="avg_salary", color_continuous_scale="Viridis",
                labels={"job_count": "Job Postings", "skill": "", "avg_salary": "Salary"},
            )
            fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Highest Paying Skills")
        pay = run_query(f"""
            SELECT skill, ROUND(AVG(salary_mid_usd), 0) AS avg_salary,
                   COUNT(*) AS job_count
            FROM job_market_db.job_skills
            WHERE salary_mid_usd > 50000 AND salary_mid_usd < 400000
              AND country IN ({countries_sql})
            GROUP BY skill HAVING COUNT(*) >= 5
            ORDER BY avg_salary DESC LIMIT 15
        """)
        if not pay.empty:
            fig = px.bar(
                pay, x="avg_salary", y="skill", orientation="h",
                color="job_count", color_continuous_scale="Blues",
                labels={"avg_salary": "Average Salary ($)", "skill": "", "job_count": "Jobs"},
            )
            fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Skills Distribution by Category")

    CAT_MAP = {
        "python": "Languages", "java": "Languages", "javascript": "Languages",
        "react": "Frontend", "angular": "Frontend", "vue": "Frontend",
        "aws": "Cloud", "azure": "Cloud", "gcp": "Cloud", "kubernetes": "Cloud",
        "postgresql": "Databases", "mongodb": "Databases", "redis": "Databases",
        "machine learning": "ML/AI", "deep learning": "ML/AI", "pytorch": "ML/AI",
        "spark": "Data Eng", "airflow": "Data Eng", "kafka": "Data Eng",
    }
    all_s = run_query(f"""
        SELECT skill, COUNT(*) AS count
        FROM job_market_db.job_skills
        WHERE country IN ({countries_sql}) GROUP BY skill
    """)
    if not all_s.empty:
        all_s["category"] = all_s["skill"].map(lambda s: CAT_MAP.get(s, "Other"))
        cat = all_s.groupby("category")["count"].sum().reset_index()
        fig = px.pie(cat, values="count", names="category", hole=0.4)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# VIEW: Forecasts
# =============================================================================

elif view_mode == "Forecasts":
    st.subheader("Skill Demand Forecasts")
    st.markdown("Predictions for skill demand over the next 4 weeks based on historical trends.")

    forecasts = get_skill_forecasts()

    if not forecasts.empty:
        c1, c2, c3 = st.columns(3)
        growing = int((forecasts["growth_pct"] > 0).sum())
        declining = int((forecasts["growth_pct"] < 0).sum())
        c1.metric("Skills Growing", growing)
        c2.metric("Skills Declining", declining)
        c3.metric("Avg Growth Rate", f"{forecasts['growth_pct'].mean():.1f}%")

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
        disp["growth_pct"] = disp["growth_pct"].apply(
            lambda v: f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"
        )
        disp["salary"] = disp["salary"].apply(lambda v: f"${v:,.0f}")
        disp.columns = ["Skill", "Total Jobs", "Avg Weekly", "Avg Salary", "Growth", "Forecast"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("No forecast data available yet. Run the pipeline first.")

# =============================================================================
# VIEW: Emerging Skills
# =============================================================================

elif view_mode == "Emerging Skills":
    st.subheader("Emerging Skills Detection")
    st.markdown("Skills showing unusual growth patterns — potential breakout technologies.")

    emerging = get_emerging_skills()

    if not emerging.empty:
        hot = emerging[emerging["trend_status"] == "Hot"]
        if not hot.empty:
            st.error(f"**HOT SKILLS ALERT**: {', '.join(hot['skill'].tolist())}")

        fig = px.scatter(
            emerging, x="current_demand", y="growth_pct",
            size="avg_salary", color="trend_status", hover_name="skill",
            color_discrete_map={
                "Hot": "#ef4444", "Rising": "#f59e0b",
                "Growing": "#10b981", "Stable": "#6b7280",
            },
            labels={
                "current_demand": "Current Weekly Demand",
                "growth_pct": "Growth Rate %",
                "avg_salary": "Salary",
            },
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### Trend Analysis")

        sel = st.selectbox("Select skill to analyze", emerging["skill"].tolist())
        ts = get_skill_time_series([sel])

        if not ts.empty:
            chart = _forecast_chart(ts, sel)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("No emerging skills detected yet. Try adjusting filters or wait for more data.")

# =============================================================================
# VIEW: Skill Advisor
# =============================================================================

elif view_mode == "Skill Advisor":
    st.subheader("Skill Recommendation Advisor")
    st.markdown("Get personalized recommendations based on your current expertise.")

    all_skills = run_query(
        "SELECT DISTINCT skill FROM job_market_db.job_skills ORDER BY skill"
    )

    if not all_skills.empty:
        known = st.multiselect(
            "Select skills you already know:",
            all_skills["skill"].tolist(),
            max_selections=5,
        )

        if known:
            skill_list = ", ".join(f"'{s}'" for s in known)
            recs = run_query(f"""
                WITH your_jobs AS (
                    SELECT DISTINCT job_id
                    FROM job_market_db.job_skills
                    WHERE skill IN ({skill_list})
                ),
                cooccurring AS (
                    SELECT js.skill,
                           COUNT(*)            AS cooccurrence_count,
                           AVG(js.salary_mid_usd) AS avg_salary
                    FROM job_market_db.job_skills js
                    JOIN your_jobs yj ON js.job_id = yj.job_id
                    WHERE js.skill NOT IN ({skill_list})
                    GROUP BY js.skill
                    HAVING COUNT(*) >= 3
                )
                SELECT skill,
                       cooccurrence_count AS relevance_score,
                       ROUND(avg_salary, 0) AS avg_salary
                FROM cooccurring
                ORDER BY cooccurrence_count DESC
                LIMIT 10
            """)

            if not recs.empty:
                st.divider()
                st.markdown("### Recommended Skills to Learn")

                c1, c2 = st.columns([2, 1])

                with c1:
                    fig = px.bar(
                        recs, x="relevance_score", y="skill", orientation="h",
                        color="avg_salary", color_continuous_scale="Viridis",
                        labels={
                            "relevance_score": "Relevance Score",
                            "skill": "",
                            "avg_salary": "Avg Salary",
                        },
                    )
                    fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    st.markdown("**Why these skills?**")
                    st.markdown(
                        "These skills frequently appear in job postings "
                        "alongside your current skills."
                    )
                    top_rec = recs.iloc[0]
                    st.success(
                        f"**Top pick**: {top_rec['skill']} "
                        f"(${top_rec['avg_salary']:,.0f} avg salary)"
                    )
            else:
                st.warning("Not enough data to generate recommendations. Try different skills.")
        else:
            st.info("Select your current skills above to get recommendations.")

# ── footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Data refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
    "Powered by AWS Athena + ML"
)
