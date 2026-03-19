"""
Job Market Intelligence — Streamlit Dashboard

Run with:
    streamlit run dashboard/app.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.data_loader import DashboardDataLoader, SKILL_CATEGORIES

# ── page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Job Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

TREND_COLORS = {
    "Hot": "#ef4444",
    "Rising": "#f97316",
    "Growing": "#22c55e",
    "Stable": "#6b7280",
    "Declining": "#3b82f6",
}

CATEGORY_COLORS = {
    "Languages": "#6366f1",
    "Frontend": "#06b6d4",
    "Backend": "#8b5cf6",
    "Cloud & Infra": "#f59e0b",
    "Databases": "#10b981",
    "ML & AI": "#ef4444",
    "Data Engineering": "#ec4899",
    "DevOps & Tools": "#64748b",
    "Other": "#9ca3af",
}


# ── load data (cached) ─────────────────────────────────────────────────────

@st.cache_resource
def get_loader():
    return DashboardDataLoader()


@st.cache_data(ttl=600)
def load_all_data():
    loader = get_loader()
    return {
        "kpis": loader.load_kpis(),
        "top_skills": loader.load_top_skills(),
        "skill_trends": loader.load_skill_trends(),
        "salary_by_country": loader.load_salary_by_country(),
        "skill_growth": loader.load_skill_growth(),
        "emerging_skills": loader.load_emerging_skills(),
        "cooccurrence": loader.load_cooccurrence(),
        "skills_by_category": loader.load_skills_by_category(),
        "model_metrics": loader.load_model_metrics(),
        "is_demo": loader.is_demo_mode(),
    }


data = load_all_data()

# ── sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Job Market Intelligence")

    if data["is_demo"]:
        st.info("Running with **demo data**. Connect AWS credentials for live data.", icon="🔬")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Skill Demand",
            "Salary Analysis",
            "Growth & Forecasting",
            "Skill Relationships",
            "ML Insights",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    all_categories = sorted(set(SKILL_CATEGORIES.values()))
    selected_categories = st.multiselect(
        "Filter by category",
        all_categories,
        default=[],
        placeholder="All categories",
    )

    country_options = sorted(
        data["top_skills"]["country"].dropna().unique().tolist()
    )
    selected_countries = st.multiselect(
        "Filter by country",
        country_options,
        default=[],
        placeholder="All countries",
    )

    st.divider()
    st.caption("Data refreshes every 10 minutes")
    if st.button("Refresh now"):
        load_all_data.clear()
        st.rerun()


# ── filter helpers ──────────────────────────────────────────────────────────

def _get_category(skill: str) -> str:
    return SKILL_CATEGORIES.get(skill, "Other")


def filter_by_selections(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if selected_categories and "skill" in result.columns:
        cat_col = "category" if "category" in result.columns else None
        if cat_col:
            result = result[result[cat_col].isin(selected_categories)]
        else:
            result = result[result["skill"].apply(_get_category).isin(selected_categories)]
    if selected_countries and "country" in result.columns:
        result = result[result["country"].isin(selected_countries)]
    return result


def fmt_salary(val):
    if pd.isna(val):
        return "N/A"
    return f"${val:,.0f}"


def fmt_number(val):
    if val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if val >= 1_000:
        return f"{val / 1_000:.1f}K"
    return f"{val:,.0f}"


# ── page: overview ──────────────────────────────────────────────────────────

def page_overview():
    st.header("Market Overview")

    kpis = data["kpis"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Job Postings", fmt_number(kpis["total_jobs"]))
    c2.metric("Unique Skills Tracked", fmt_number(kpis["unique_skills"]))
    c3.metric("Avg Salary", fmt_salary(kpis["avg_salary"]))
    c4.metric("Countries", kpis["countries"])

    st.divider()

    col_left, col_right = st.columns(2)

    # Top skills bar
    with col_left:
        st.subheader("Top Skills by Demand")
        top = filter_by_selections(data["top_skills"])
        agg = (
            top.groupby("skill")
            .agg(job_count=("job_count", "sum"), avg_salary=("avg_salary", "mean"))
            .reset_index()
            .nlargest(20, "job_count")
        )
        agg["category"] = agg["skill"].apply(_get_category)
        fig = px.bar(
            agg.sort_values("job_count"),
            x="job_count", y="skill",
            color="category",
            color_discrete_map=CATEGORY_COLORS,
            orientation="h",
            hover_data={"avg_salary": ":$,.0f"},
        )
        fig.update_layout(
            height=500, yaxis_title=None, xaxis_title="Job Count",
            legend_title="Category", margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Skills by category treemap
    with col_right:
        st.subheader("Skills by Category")
        cat_df = data["skills_by_category"].copy()
        if selected_categories:
            cat_df = cat_df[cat_df["category"].isin(selected_categories)]
        cat_agg = cat_df.groupby("category")["job_count"].sum().reset_index()
        cat_agg = cat_agg.sort_values("job_count", ascending=False)
        fig = px.treemap(
            cat_df,
            path=["category", "skill"],
            values="job_count",
            color="category",
            color_discrete_map=CATEGORY_COLORS,
        )
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Skill trends (top 8)
    st.subheader("Weekly Demand Trends (Top 8 Skills)")
    trends = filter_by_selections(data["skill_trends"])
    weekly_agg = trends.groupby(["skill", "week"]).agg(job_count=("job_count", "sum")).reset_index()
    top8 = weekly_agg.groupby("skill")["job_count"].sum().nlargest(8).index
    weekly_top = weekly_agg[weekly_agg["skill"].isin(top8)]

    fig = px.line(
        weekly_top.sort_values("week"),
        x="week", y="job_count", color="skill",
        markers=True,
    )
    fig.update_layout(
        height=400, xaxis_title=None, yaxis_title="Weekly Job Count",
        legend_title="Skill", margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── page: skill demand ──────────────────────────────────────────────────────

def page_skill_demand():
    st.header("Skill Demand Analysis")

    top = filter_by_selections(data["top_skills"])
    agg = (
        top.groupby("skill")
        .agg(job_count=("job_count", "sum"), avg_salary=("avg_salary", "mean"))
        .reset_index()
    )
    agg["category"] = agg["skill"].apply(_get_category)

    n_skills = st.slider("Number of skills to show", 10, 60, 25)
    sort_by = st.radio("Sort by", ["Job Count", "Average Salary"], horizontal=True)

    sort_col = "job_count" if sort_by == "Job Count" else "avg_salary"
    display = agg.nlargest(n_skills, sort_col)

    fig = px.bar(
        display.sort_values(sort_col),
        x=sort_col, y="skill",
        color="avg_salary" if sort_by == "Job Count" else "job_count",
        color_continuous_scale="Viridis",
        orientation="h",
        labels={"avg_salary": "Avg Salary ($)", "job_count": "Job Count"},
    )
    fig.update_layout(
        height=max(400, n_skills * 22),
        yaxis_title=None, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Demand by Country")

    if not selected_countries:
        country_skills = top.copy()
    else:
        country_skills = top[top["country"].isin(selected_countries)]

    pivot = (
        country_skills.groupby(["country", "skill"])["job_count"]
        .sum()
        .reset_index()
    )
    top_skills_list = pivot.groupby("skill")["job_count"].sum().nlargest(15).index
    pivot = pivot[pivot["skill"].isin(top_skills_list)]

    fig = px.bar(
        pivot,
        x="skill", y="job_count", color="country",
        barmode="group",
    )
    fig.update_layout(
        height=450, xaxis_title=None, yaxis_title="Job Count",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Skill Demand Trends")

    trends = filter_by_selections(data["skill_trends"])
    available_skills = sorted(trends["skill"].unique())
    chosen = st.multiselect(
        "Select skills to compare",
        available_skills,
        default=available_skills[:5] if len(available_skills) >= 5 else available_skills,
    )

    if chosen:
        trend_data = trends[trends["skill"].isin(chosen)]
        weekly = trend_data.groupby(["skill", "week"]).agg(
            job_count=("job_count", "sum"),
            avg_salary=("avg_salary", "mean"),
        ).reset_index().sort_values("week")

        tab1, tab2 = st.tabs(["Job Count", "Average Salary"])

        with tab1:
            fig = px.line(weekly, x="week", y="job_count", color="skill", markers=True)
            fig.update_layout(
                height=400, xaxis_title=None, yaxis_title="Weekly Job Count",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = px.line(weekly, x="week", y="avg_salary", color="skill", markers=True)
            fig.update_layout(
                height=400, xaxis_title=None, yaxis_title="Avg Salary ($)",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── page: salary analysis ──────────────────────────────────────────────────

def page_salary_analysis():
    st.header("Salary Analysis")

    salary = data["salary_by_country"].copy()
    if selected_countries:
        salary = salary[salary["country"].isin(selected_countries)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Salary by Country")
        fig = px.bar(
            salary.sort_values("avg_salary", ascending=True),
            x="avg_salary", y="country",
            orientation="h",
            color="avg_salary",
            color_continuous_scale="Greens",
            hover_data={"job_count": ":,.0f", "unique_skills": True},
        )
        fig.update_layout(
            height=400, yaxis_title=None, xaxis_title="Average Salary ($)",
            showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Job Volume by Country")
        fig = px.pie(
            salary,
            values="job_count", names="country",
            hole=0.4,
        )
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Salary vs Demand by Skill")

    top = filter_by_selections(data["top_skills"])
    skill_agg = (
        top.groupby("skill")
        .agg(job_count=("job_count", "sum"), avg_salary=("avg_salary", "mean"))
        .reset_index()
    )
    skill_agg["category"] = skill_agg["skill"].apply(_get_category)
    skill_agg = skill_agg.nlargest(40, "job_count")

    fig = px.scatter(
        skill_agg,
        x="job_count", y="avg_salary",
        size="job_count", color="category",
        color_discrete_map=CATEGORY_COLORS,
        text="skill",
        hover_data={"job_count": ":,.0f", "avg_salary": ":$,.0f"},
    )
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(
        height=550,
        xaxis_title="Total Job Postings",
        yaxis_title="Average Salary ($)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Salary Distribution Across Countries")

    top_full = filter_by_selections(data["top_skills"])
    top_full = top_full[top_full["avg_salary"] > 0]
    top15 = top_full.groupby("skill")["job_count"].sum().nlargest(15).index
    box_data = top_full[top_full["skill"].isin(top15)]

    fig = px.box(
        box_data,
        x="skill", y="avg_salary",
        color="skill",
        points=False,
    )
    fig.update_layout(
        height=450,
        xaxis_title=None, yaxis_title="Salary ($)",
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── page: growth & forecasting ──────────────────────────────────────────────

def page_growth():
    st.header("Growth & Forecasting")

    growth = data["skill_growth"].copy()
    emerging = data["emerging_skills"].copy()

    tab_growth, tab_emerging = st.tabs(["Skill Growth", "Emerging Skills"])

    with tab_growth:
        st.subheader("Skill Growth Rates")
        if selected_categories:
            growth = growth[growth["skill"].apply(_get_category).isin(selected_categories)]

        growth_sorted = growth.sort_values("growth_pct", ascending=True).tail(30)
        fig = px.bar(
            growth_sorted,
            x="growth_pct", y="skill",
            color="trend_status",
            color_discrete_map=TREND_COLORS,
            orientation="h",
            hover_data={
                "total_jobs": ":,.0f",
                "avg_weekly_jobs": ":.1f",
                "forecast_weekly": ":.1f",
            },
        )
        fig.update_layout(
            height=max(400, len(growth_sorted) * 22),
            xaxis_title="Growth %",
            yaxis_title=None,
            legend_title="Trend",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Current vs Forecast Demand")

        forecast_df = growth[growth["forecast_weekly"] > 0].nlargest(20, "total_jobs")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Current (Avg Weekly)",
            y=forecast_df["skill"],
            x=forecast_df["avg_weekly_jobs"],
            orientation="h",
            marker_color="#6366f1",
        ))
        fig.add_trace(go.Bar(
            name="Forecast (Weekly)",
            y=forecast_df["skill"],
            x=forecast_df["forecast_weekly"],
            orientation="h",
            marker_color="#22c55e",
        ))
        fig.update_layout(
            barmode="group", height=500,
            xaxis_title="Weekly Jobs", yaxis_title=None,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_emerging:
        st.subheader("Emerging Skills Detection")

        if selected_categories:
            emerging = emerging[emerging["skill"].apply(_get_category).isin(selected_categories)]

        if emerging.empty:
            st.warning("No emerging skills found for the selected filters.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Emerging Skills", len(emerging[emerging["trend_status"].isin(["Hot", "Rising"])]))
            c2.metric("Highest Growth", f"{emerging['growth_pct'].max():.0f}%")
            c3.metric("Top Emerging", emerging.iloc[0]["skill"] if len(emerging) > 0 else "N/A")

            fig = px.scatter(
                emerging,
                x="current_jobs", y="growth_pct",
                size="current_jobs", color="trend_status",
                color_discrete_map=TREND_COLORS,
                text="skill",
                hover_data={"avg_salary": ":$,.0f", "prev_jobs": True},
            )
            fig.update_traces(textposition="top center", textfont_size=9)
            fig.update_layout(
                height=500,
                xaxis_title="Current Job Count (last 4 weeks)",
                yaxis_title="Growth % (vs prior 4 weeks)",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Emerging Skills Table")
            display_cols = ["skill", "current_jobs", "prev_jobs", "growth_pct", "avg_salary", "trend_status"]
            available = [c for c in display_cols if c in emerging.columns]
            styled = emerging[available].reset_index(drop=True)
            st.dataframe(
                styled,
                use_container_width=True,
                column_config={
                    "skill": st.column_config.TextColumn("Skill"),
                    "current_jobs": st.column_config.NumberColumn("Current Jobs"),
                    "prev_jobs": st.column_config.NumberColumn("Prev Jobs"),
                    "growth_pct": st.column_config.NumberColumn("Growth %", format="%.1f%%"),
                    "avg_salary": st.column_config.NumberColumn("Avg Salary", format="$%,.0f"),
                    "trend_status": st.column_config.TextColumn("Trend"),
                },
                hide_index=True,
            )


# ── page: skill relationships ───────────────────────────────────────────────

def page_relationships():
    st.header("Skill Relationships")

    cooc = data["cooccurrence"].copy()

    tab_heat, tab_network, tab_recs = st.tabs(["Co-occurrence Heatmap", "Top Pairs", "Skill Recommendations"])

    with tab_heat:
        st.subheader("Skill Co-occurrence Matrix")

        top_pairs = cooc.nlargest(200, "cooccurrence_count")
        all_skills_in_pairs = sorted(
            set(top_pairs["skill_a"].unique()) | set(top_pairs["skill_b"].unique())
        )

        if selected_categories:
            all_skills_in_pairs = [
                s for s in all_skills_in_pairs if _get_category(s) in selected_categories
            ]
            top_pairs = top_pairs[
                top_pairs["skill_a"].isin(all_skills_in_pairs)
                & top_pairs["skill_b"].isin(all_skills_in_pairs)
            ]

        n_show = st.slider("Skills to show", 8, min(30, len(all_skills_in_pairs)), min(15, len(all_skills_in_pairs)))

        pair_sums = pd.concat([
            top_pairs.groupby("skill_a")["cooccurrence_count"].sum().rename("total"),
            top_pairs.groupby("skill_b")["cooccurrence_count"].sum().rename("total"),
        ]).groupby(level=0).sum().nlargest(n_show)
        selected_skills = pair_sums.index.tolist()

        matrix = pd.DataFrame(0, index=selected_skills, columns=selected_skills, dtype=float)
        for _, row in top_pairs.iterrows():
            if row["skill_a"] in selected_skills and row["skill_b"] in selected_skills:
                matrix.loc[row["skill_a"], row["skill_b"]] = row["cooccurrence_count"]
                matrix.loc[row["skill_b"], row["skill_a"]] = row["cooccurrence_count"]

        fig = px.imshow(
            matrix,
            labels=dict(color="Co-occurrence Count"),
            color_continuous_scale="Blues",
            aspect="auto",
        )
        fig.update_layout(
            height=max(500, n_show * 30),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_network:
        st.subheader("Top Skill Pairs")

        if selected_categories:
            filtered_cooc = cooc[
                cooc["skill_a"].apply(_get_category).isin(selected_categories)
                | cooc["skill_b"].apply(_get_category).isin(selected_categories)
            ]
        else:
            filtered_cooc = cooc

        top_n = st.slider("Number of pairs", 10, 50, 25, key="pairs_slider")
        top = filtered_cooc.nlargest(top_n, "cooccurrence_count")

        top["pair"] = top["skill_a"] + " + " + top["skill_b"]
        fig = px.bar(
            top.sort_values("cooccurrence_count"),
            x="cooccurrence_count", y="pair",
            orientation="h",
            color="cooccurrence_count",
            color_continuous_scale="Purples",
        )
        fig.update_layout(
            height=max(400, top_n * 22),
            xaxis_title="Co-occurrence Count",
            yaxis_title=None,
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_recs:
        st.subheader("Skill Recommendations")
        st.write("Select skills you already know to discover related skills.")

        metrics = data["model_metrics"]
        cluster_mapping = metrics.get("cluster_model", {}).get("skill_cluster_mapping", {})

        if not cluster_mapping:
            st.warning("No cluster model available. Train the ML models first.")
        else:
            all_skills = sorted(cluster_mapping.keys())
            known = st.multiselect(
                "Your current skills",
                all_skills,
                default=["python", "aws"] if "python" in all_skills else all_skills[:2],
            )

            if known:
                known_clusters = {cluster_mapping[s] for s in known if s in cluster_mapping}
                recs = []
                for skill, cluster in cluster_mapping.items():
                    if cluster in known_clusters and skill not in known:
                        recs.append({"skill": skill, "cluster": cluster})

                if recs:
                    recs_df = pd.DataFrame(recs)
                    recs_df["category"] = recs_df["skill"].apply(_get_category)

                    growth = data["skill_growth"]
                    recs_df = recs_df.merge(
                        growth[["skill", "growth_pct", "avg_weekly_jobs", "trend_status"]],
                        on="skill", how="left",
                    )

                    recs_df = recs_df.sort_values("growth_pct", ascending=False).head(15)

                    st.write(f"Based on your skills, you belong to **{len(known_clusters)} skill cluster(s)**. "
                             f"Here are **{len(recs_df)} recommended skills** to learn:")

                    fig = px.bar(
                        recs_df.sort_values("growth_pct"),
                        x="growth_pct", y="skill",
                        color="category",
                        color_discrete_map=CATEGORY_COLORS,
                        orientation="h",
                        hover_data={"avg_weekly_jobs": ":.1f"},
                    )
                    fig.update_layout(
                        height=max(350, len(recs_df) * 28),
                        xaxis_title="Growth %", yaxis_title=None,
                        legend_title="Category",
                        margin=dict(l=0, r=0, t=10, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(
                        recs_df[["skill", "category", "growth_pct", "avg_weekly_jobs", "trend_status"]].reset_index(drop=True),
                        use_container_width=True,
                        column_config={
                            "skill": "Skill",
                            "category": "Category",
                            "growth_pct": st.column_config.NumberColumn("Growth %", format="%.1f%%"),
                            "avg_weekly_jobs": st.column_config.NumberColumn("Avg Weekly Jobs", format="%.1f"),
                            "trend_status": "Trend",
                        },
                        hide_index=True,
                    )
                else:
                    st.info("No additional recommendations found for the selected skills.")


# ── page: ML insights ───────────────────────────────────────────────────────

def page_ml_insights():
    st.header("ML Model Insights")

    metrics = data["model_metrics"]

    if not metrics:
        st.warning(
            "No model metrics found. Train models with:\n\n"
            "```bash\npython -m ml.training.train --bucket $S3_BUCKET\n```"
        )
        return

    tab_demand, tab_emerge, tab_cluster = st.tabs(
        ["Demand Forecaster", "Emergence Detector", "Skill Clusters"]
    )

    with tab_demand:
        st.subheader("XGBoost Demand Forecaster")

        dm = metrics.get("demand_model", {})
        if dm:
            c1, c2, c3 = st.columns(3)
            c1.metric("CV MAE", f"{dm.get('cv_mae_mean', 0):.2f}", help="Mean Absolute Error (cross-validated)")
            c2.metric("CV RMSE", f"{dm.get('cv_rmse_mean', 0):.2f}", help="Root Mean Squared Error")
            c3.metric("CV R²", f"{dm.get('cv_r2_mean', 0):.3f}", help="Coefficient of determination")

            st.divider()
            st.subheader("Feature Importance (Top 10)")

            top_feats = dm.get("top_features", [])
            if top_feats:
                feat_df = pd.DataFrame(top_feats, columns=["feature", "importance"])
                fig = px.bar(
                    feat_df.sort_values("importance"),
                    x="importance", y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(
                    height=400, yaxis_title=None,
                    xaxis_title="Feature Importance",
                    showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab_emerge:
        st.subheader("Isolation Forest — Emergence Detector")

        em = metrics.get("emergence_model", {})
        if em:
            c1, c2 = st.columns(2)
            c1.metric("Emerging Skills Detected", em.get("n_emerging_detected", 0))
            c2.metric("Anomaly Threshold", f"{em.get('anomaly_threshold', 0):.2e}")

            sample = em.get("emerging_skills_sample", [])
            if sample:
                st.write("**Sample of detected emerging skills:**")
                cols = st.columns(5)
                for i, skill in enumerate(sample):
                    cols[i % 5].code(skill)

    with tab_cluster:
        st.subheader("KMeans Skill Clusters")

        cm = metrics.get("cluster_model", {})
        if cm:
            st.metric("Number of Clusters", cm.get("n_clusters", 0))

            profiles = cm.get("cluster_profiles", {})
            if profiles:
                cluster_data = []
                for cid, info in profiles.items():
                    for skill in info.get("skills", []):
                        cluster_data.append({
                            "cluster": f"Cluster {cid}",
                            "skill": skill,
                            "category": _get_category(skill),
                        })

                cluster_df = pd.DataFrame(cluster_data)
                cluster_counts = cluster_df.groupby("cluster").size().reset_index(name="count")

                fig = px.bar(
                    cluster_counts,
                    x="cluster", y="count",
                    color="cluster",
                    text="count",
                )
                fig.update_layout(
                    height=350,
                    xaxis_title=None, yaxis_title="Skills in Cluster",
                    showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.divider()
                st.subheader("Cluster Composition")

                selected_cluster = st.selectbox(
                    "Select cluster",
                    sorted(profiles.keys()),
                    format_func=lambda x: f"Cluster {x} ({profiles[x]['count']} skills)",
                )

                if selected_cluster:
                    skills_in_cluster = profiles[selected_cluster]["skills"]
                    cluster_skills_df = pd.DataFrame({
                        "skill": skills_in_cluster,
                        "category": [_get_category(s) for s in skills_in_cluster],
                    })

                    fig = px.bar(
                        cluster_skills_df.sort_values("category"),
                        x="skill", y=[1] * len(cluster_skills_df),
                        color="category",
                        color_discrete_map=CATEGORY_COLORS,
                    )
                    fig.update_layout(
                        height=350,
                        xaxis_title=None, yaxis_title=None,
                        yaxis_visible=False,
                        legend_title="Category",
                        margin=dict(l=0, r=0, t=10, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Training Summary")
    summary_cols = st.columns(3)
    summary_cols[0].metric("Training Samples", fmt_number(metrics.get("n_samples", 0)))
    summary_cols[1].metric("Skills Modeled", metrics.get("n_skills", 0))
    summary_cols[2].metric("Features Used", metrics.get("n_features", 0))


# ── routing ─────────────────────────────────────────────────────────────────

PAGES = {
    "Overview": page_overview,
    "Skill Demand": page_skill_demand,
    "Salary Analysis": page_salary_analysis,
    "Growth & Forecasting": page_growth,
    "Skill Relationships": page_relationships,
    "ML Insights": page_ml_insights,
}

PAGES[page]()
