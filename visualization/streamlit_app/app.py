"""
Job Market Intelligence Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import boto3
import awswrangler as wr
from datetime import datetime, timedelta
import os

st.set_page_config(
    page_title="Job Market Intelligence",
    page_icon="📊",
    layout="wide",
)

DATABASE = os.getenv('GLUE_DATABASE', 'job_market_db')
REGION = os.getenv('AWS_REGION', 'us-east-1')


@st.cache_data(ttl=3600)
def run_query(query):
    """Run Athena query and return DataFrame."""
    try:
        return wr.athena.read_sql_query(query, database=DATABASE, region_name=REGION)
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()


# ── Header ──────────────────────────────────────────────────────────────────
st.title("📊 Job Market Intelligence Dashboard")
st.markdown(
    "Real-time insights into tech job market trends, skills demand, and salary analytics"
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
country_filter = st.sidebar.multiselect(
    "Countries",
    ["US", "GB", "CA", "AU", "DE"],
    default=["US"],
)

countries_sql = ",".join(f"'{c}'" for c in country_filter)

# ── KPI Metrics ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

summary_query = f"""
SELECT
    COUNT(DISTINCT job_id) as total_jobs,
    COUNT(DISTINCT skill) as unique_skills,
    ROUND(AVG(salary_mid_usd), 0) as avg_salary,
    COUNT(DISTINCT country) as countries
FROM job_market_db.job_skills
WHERE country IN ({countries_sql})
"""
summary = run_query(summary_query)

if not summary.empty:
    col1.metric("Total Jobs", f"{summary['total_jobs'].iloc[0]:,}")
    col2.metric("Unique Skills", f"{summary['unique_skills'].iloc[0]:,}")
    col3.metric("Avg Salary", f"${summary['avg_salary'].iloc[0]:,.0f}")
    col4.metric("Countries", summary['countries'].iloc[0])

st.divider()

# ── Charts Row 1 ────────────────────────────────────────────────────────────
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("🔥 Top 15 In-Demand Skills")
    top_skills_query = f"""
    SELECT skill, COUNT(*) as job_count, ROUND(AVG(salary_mid_usd), 0) as avg_salary
    FROM job_market_db.job_skills
    WHERE country IN ({countries_sql})
    GROUP BY skill
    ORDER BY job_count DESC
    LIMIT 15
    """
    top_skills = run_query(top_skills_query)
    if not top_skills.empty:
        fig = px.bar(
            top_skills, x='job_count', y='skill',
            orientation='h', color='avg_salary',
            color_continuous_scale='Viridis',
            labels={'job_count': 'Job Count', 'skill': 'Skill', 'avg_salary': 'Avg Salary'},
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("💰 Highest Paying Skills")
    salary_query = f"""
    SELECT skill, ROUND(AVG(salary_mid_usd), 0) as avg_salary, COUNT(*) as job_count
    FROM job_market_db.job_skills
    WHERE salary_mid_usd > 50000 AND salary_mid_usd < 400000
      AND country IN ({countries_sql})
    GROUP BY skill
    HAVING COUNT(*) >= 5
    ORDER BY avg_salary DESC
    LIMIT 15
    """
    salary_skills = run_query(salary_query)
    if not salary_skills.empty:
        fig = px.bar(
            salary_skills, x='avg_salary', y='skill',
            orientation='h', color='job_count',
            color_continuous_scale='Blues',
            labels={'avg_salary': 'Average Salary ($)', 'skill': 'Skill'},
        )
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Skills Distribution ────────────────────────────────────────────────────
st.subheader("📊 Skills Distribution by Category")

CATEGORY_MAP = {
    'python': 'Languages', 'java': 'Languages', 'javascript': 'Languages',
    'react': 'Frontend', 'angular': 'Frontend', 'vue': 'Frontend',
    'aws': 'Cloud', 'azure': 'Cloud', 'gcp': 'Cloud', 'kubernetes': 'Cloud',
    'postgresql': 'Databases', 'mongodb': 'Databases', 'redis': 'Databases',
    'machine learning': 'ML/AI', 'deep learning': 'ML/AI', 'pytorch': 'ML/AI',
    'spark': 'Data Eng', 'airflow': 'Data Eng', 'kafka': 'Data Eng',
}

skills_query = f"""
SELECT skill, COUNT(*) as count
FROM job_market_db.job_skills
WHERE country IN ({countries_sql})
GROUP BY skill
"""
all_skills = run_query(skills_query)

if not all_skills.empty:
    all_skills['category'] = all_skills['skill'].map(
        lambda x: CATEGORY_MAP.get(x, 'Other')
    )
    category_totals = all_skills.groupby('category')['count'].sum().reset_index()
    fig = px.pie(category_totals, values='count', names='category', hole=0.4)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Data refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
    "Powered by AWS Athena"
)
