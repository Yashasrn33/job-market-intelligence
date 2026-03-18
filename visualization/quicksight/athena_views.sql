-- ============================================================================
-- Athena Views for QuickSight Dashboards
--
-- These views pre-compute the analytics that power each QuickSight visual.
-- Run once after the initial ETL, or re-run to pick up schema changes.
--
-- Usage:
--   aws athena start-query-execution \
--     --query-string "$(cat visualization/quicksight/athena_views.sql)" \
--     --result-configuration "OutputLocation=s3://${S3_BUCKET}/athena-results/"
--
-- Or deploy via: bash infrastructure/deploy_quicksight.sh
-- ============================================================================

-- ── 1. Top Skills (bar chart + KPIs) ────────────────────────────────────────

CREATE OR REPLACE VIEW job_market_db.vw_top_skills AS
SELECT
    skill,
    country,
    COUNT(*)                        AS job_count,
    COUNT(DISTINCT job_id)          AS unique_jobs,
    ROUND(AVG(salary_mid_usd), 0)  AS avg_salary
FROM job_market_db.job_skills
GROUP BY skill, country;


-- ── 2. Weekly Skill Trends (line chart) ─────────────────────────────────────

CREATE OR REPLACE VIEW job_market_db.vw_skill_trends AS
SELECT
    skill,
    DATE_TRUNC('week', posted_date) AS week,
    country,
    COUNT(*)                        AS job_count,
    AVG(salary_mid_usd)             AS avg_salary
FROM job_market_db.job_skills
WHERE posted_date >= DATE_ADD('week', -24, CURRENT_DATE)
GROUP BY skill, DATE_TRUNC('week', posted_date), country;


-- ── 3. Salary by Country (choropleth / bar) ─────────────────────────────────

CREATE OR REPLACE VIEW job_market_db.vw_salary_by_country AS
SELECT
    country,
    ROUND(AVG(salary_mid_usd), 0) AS avg_salary,
    COUNT(*)                       AS job_count,
    COUNT(DISTINCT skill)          AS unique_skills
FROM job_market_db.job_skills
WHERE salary_mid_usd IS NOT NULL
GROUP BY country;


-- ── 4. Skill Growth (forecast overview) ─────────────────────────────────────

CREATE OR REPLACE VIEW job_market_db.vw_skill_growth AS
WITH weekly AS (
    SELECT
        skill,
        DATE_TRUNC('week', posted_date) AS week,
        COUNT(*)                        AS job_count,
        AVG(salary_mid_usd)             AS avg_salary
    FROM job_market_db.job_skills
    WHERE posted_date >= DATE_ADD('week', -12, CURRENT_DATE)
    GROUP BY skill, DATE_TRUNC('week', posted_date)
),
trends AS (
    SELECT
        skill,
        COUNT(*)       AS weeks_present,
        SUM(job_count) AS total_jobs,
        ROUND(AVG(job_count), 1)  AS avg_weekly_jobs,
        ROUND(MAX(avg_salary), 0) AS max_salary,
        (MAX(CASE WHEN week >= DATE_ADD('week', -4, CURRENT_DATE)
                  THEN job_count END)
       - MAX(CASE WHEN week <  DATE_ADD('week', -4, CURRENT_DATE)
                    AND week >= DATE_ADD('week', -8, CURRENT_DATE)
                  THEN job_count END)) * 100.0
       / NULLIF(MAX(CASE WHEN week <  DATE_ADD('week', -4, CURRENT_DATE)
                           AND week >= DATE_ADD('week', -8, CURRENT_DATE)
                         THEN job_count END), 0) AS growth_pct
    FROM weekly
    GROUP BY skill
    HAVING COUNT(*) >= 4
)
SELECT
    skill,
    total_jobs,
    avg_weekly_jobs,
    max_salary,
    ROUND(COALESCE(growth_pct, 0), 1)                              AS growth_pct,
    ROUND(avg_weekly_jobs * (1 + COALESCE(growth_pct, 0) / 100), 1) AS forecast_weekly,
    CASE
        WHEN growth_pct > 50  THEN 'Hot'
        WHEN growth_pct > 20  THEN 'Rising'
        WHEN growth_pct > 0   THEN 'Growing'
        WHEN growth_pct <= 0  THEN 'Declining'
        ELSE 'Stable'
    END AS trend_status
FROM trends;


-- ── 5. Emerging Skills (scatter / anomaly) ──────────────────────────────────

CREATE OR REPLACE VIEW job_market_db.vw_emerging_skills AS
WITH recent AS (
    SELECT
        skill,
        COUNT(*)            AS current_jobs,
        AVG(salary_mid_usd) AS avg_salary
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
    SELECT
        r.skill,
        r.current_jobs,
        COALESCE(p.prev_jobs, 1)                                     AS prev_jobs,
        ROUND(r.avg_salary, 0)                                       AS avg_salary,
        ROUND((r.current_jobs - COALESCE(p.prev_jobs, 1)) * 100.0
              / COALESCE(p.prev_jobs, 1), 1)                         AS growth_pct
    FROM recent r
    LEFT JOIN previous p ON r.skill = p.skill
)
SELECT
    skill,
    current_jobs,
    prev_jobs,
    avg_salary,
    growth_pct,
    CASE
        WHEN growth_pct > 50 AND current_jobs >= 5 THEN 'Hot'
        WHEN growth_pct > 25 AND current_jobs >= 3 THEN 'Rising'
        WHEN growth_pct > 10                       THEN 'Growing'
        ELSE 'Stable'
    END AS trend_status
FROM growth
WHERE current_jobs >= 3;


-- ── 6. Skill Co-occurrence (recommendations) ────────────────────────────────

CREATE OR REPLACE VIEW job_market_db.vw_skill_cooccurrence AS
SELECT
    a.skill AS skill_a,
    b.skill AS skill_b,
    COUNT(*) AS cooccurrence_count
FROM job_market_db.job_skills a
JOIN job_market_db.job_skills b
    ON a.job_id = b.job_id AND a.skill < b.skill
GROUP BY a.skill, b.skill
HAVING COUNT(*) >= 3;


-- ── 7. KPI Summary (single-row overview) ────────────────────────────────────

CREATE OR REPLACE VIEW job_market_db.vw_dashboard_kpis AS
SELECT
    COUNT(DISTINCT job_id)         AS total_jobs,
    COUNT(DISTINCT skill)          AS unique_skills,
    ROUND(AVG(salary_mid_usd), 0) AS avg_salary,
    COUNT(DISTINCT country)        AS countries
FROM job_market_db.job_skills;


-- ── 8. Skills by Category (pie / treemap) ───────────────────────────────────

CREATE OR REPLACE VIEW job_market_db.vw_skills_by_category AS
SELECT
    skill,
    COUNT(*) AS job_count,
    CASE
        WHEN skill IN ('python','java','javascript','typescript','c++','go','rust','ruby','scala','kotlin','swift','r','sql','bash','c#','php') THEN 'Languages'
        WHEN skill IN ('react','angular','vue','vuejs','nextjs','nuxt') THEN 'Frontend'
        WHEN skill IN ('nodejs','django','flask','fastapi','spring','springboot','express','rails','laravel','dotnet','net') THEN 'Backend'
        WHEN skill IN ('aws','azure','gcp','google_cloud','kubernetes','k8s','docker','terraform','ansible','cloudformation','pulumi','helm','istio','serverless','lambda') THEN 'Cloud & Infra'
        WHEN skill IN ('postgresql','postgres','mysql','mongodb','redis','elasticsearch','cassandra','dynamodb','snowflake','bigquery','redshift','oracle','sql_server','sqlite','neo4j','cockroachdb') THEN 'Databases'
        WHEN skill IN ('machine_learning','deep_learning','tensorflow','pytorch','keras','scikit-learn','nlp','natural_language_processing','computer_vision','mlops','llm','langchain','openai','huggingface','sagemaker','mlflow','kubeflow','ray') THEN 'ML & AI'
        WHEN skill IN ('spark','pyspark','airflow','kafka','flink','dbt','hadoop','hive','presto','trino','databricks') THEN 'Data Engineering'
        WHEN skill IN ('git','linux','unix','ci_cd','jenkins','github_actions','gitlab','circleci') THEN 'DevOps & Tools'
        ELSE 'Other'
    END AS category
FROM job_market_db.job_skills
GROUP BY skill;
