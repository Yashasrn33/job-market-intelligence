-- Highest paying skills
SELECT
    skill,
    COUNT(*) as job_count,
    ROUND(AVG(salary_mid_usd), 0) as avg_salary,
    ROUND(PERCENTILE_APPROX(salary_mid_usd, 0.5), 0) as median_salary,
    ROUND(PERCENTILE_APPROX(salary_mid_usd, 0.9), 0) as p90_salary
FROM job_market_db.job_skills
WHERE salary_mid_usd > 50000 AND salary_mid_usd < 500000
GROUP BY skill
HAVING COUNT(*) >= 10
ORDER BY avg_salary DESC
LIMIT 30;
