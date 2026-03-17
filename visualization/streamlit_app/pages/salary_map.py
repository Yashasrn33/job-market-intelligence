"""Salary Map page -- average salary by country."""
import streamlit as st
import pandas as pd
import plotly.express as px
import awswrangler as wr
import boto3
import os

st.set_page_config(page_title="Salary Map", layout="wide")

DATABASE = os.getenv('GLUE_DATABASE', 'job_market_db')
REGION = os.getenv('AWS_REGION', 'us-east-1')
BOTO3_SESSION = boto3.Session(region_name=REGION)


@st.cache_data(ttl=3600)
def run_query(query):
    try:
        return wr.athena.read_sql_query(
            query,
            database=DATABASE,
            boto3_session=BOTO3_SESSION,
        )
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()


st.title("🗺️ Salary by Country")

query = """
SELECT country, ROUND(AVG(salary_mid_usd), 0) as avg_salary, COUNT(*) as job_count
FROM job_market_db.job_skills
WHERE salary_mid_usd IS NOT NULL
GROUP BY country
ORDER BY avg_salary DESC
"""
df = run_query(query)

if not df.empty:
    COUNTRY_ISO = {'US': 'USA', 'GB': 'GBR', 'CA': 'CAN', 'AU': 'AUS', 'DE': 'DEU'}
    df['iso_alpha'] = df['country'].map(COUNTRY_ISO)

    fig = px.choropleth(
        df, locations='iso_alpha', color='avg_salary',
        hover_name='country', hover_data=['job_count'],
        color_continuous_scale='Viridis',
        labels={'avg_salary': 'Avg Salary (USD)'},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df[['country', 'avg_salary', 'job_count']], use_container_width=True)
else:
    st.info("No data yet. Run the pipeline first to populate Athena tables.")
