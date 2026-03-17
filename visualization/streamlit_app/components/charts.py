"""Reusable chart helpers for the Streamlit dashboard."""
from typing import Dict, List

import pandas as pd
import plotly.express as px


def skill_bar_chart(rows: List[Dict], top_n: int = 20):
    df = pd.DataFrame(rows)
    if df.empty:
        return px.bar(
            pd.DataFrame({"skill": [], "postings": []}),
            x="skill", y="postings",
        )
    df = df.sort_values("postings", ascending=False).head(top_n)
    return px.bar(df, x="skill", y="postings", title=f"Top {top_n} skills")


def salary_box_chart(rows: List[Dict]):
    df = pd.DataFrame(rows)
    if df.empty:
        return px.box()
    return px.box(
        df, x="skill", y="salary_mid_usd",
        title="Salary distribution by skill",
    )
