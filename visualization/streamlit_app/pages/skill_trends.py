"""Skill Trends page -- monthly counts by skill."""
import streamlit as st
import pandas as pd
import plotly.express as px
import awswrangler as wr
import os

st.set_page_config(page_title="Skill Trends", layout="wide")

DATABASE = os.getenv('GLUE_DATABASE', 'job_market_db')
REGION = os.getenv('AWS_REGION', 'us-east-1')


@st.cache_data(ttl=3600)
def run_query(query):
    try:
        return wr.athena.read_sql_query(query, database=DATABASE, region_name=REGION)
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()


st.title("📈 Skill Trends Over Time")

query = """
SELECT skill, year, month, COUNT(*) as job_count
FROM job_market_db.job_skills
GROUP BY skill, year, month
ORDER BY year, month
"""
df = run_query(query)

if not df.empty:
    df['period'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
    )
    top_skills = (
        df.groupby('skill')['job_count'].sum()
        .nlargest(10).index.tolist()
    )
    filtered = df[df['skill'].isin(top_skills)]
    fig = px.line(
        filtered, x='period', y='job_count', color='skill',
        labels={'period': 'Month', 'job_count': 'Job Postings'},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data yet. Run the pipeline first to populate Athena tables.")
