-- Top 20 most in-demand skills
SELECT
    skill,
    COUNT(*) as job_count,
    ROUND(AVG(salary_mid_usd), 0) as avg_salary,
    ROUND(MIN(salary_mid_usd), 0) as min_salary,
    ROUND(MAX(salary_mid_usd), 0) as max_salary
FROM job_market_db.job_skills
WHERE salary_mid_usd IS NOT NULL
GROUP BY skill
ORDER BY job_count DESC
LIMIT 20;
