-- Monthly skill trends
SELECT
    skill,
    year,
    month,
    COUNT(*) as job_count,
    ROUND(AVG(salary_mid_usd), 0) as avg_salary
FROM job_market_db.job_skills
GROUP BY skill, year, month
ORDER BY year, month, job_count DESC;
