-- Skills by country
SELECT
    country,
    skill,
    COUNT(*) as job_count,
    ROUND(AVG(salary_mid_usd), 0) as avg_salary
FROM job_market_db.job_skills
GROUP BY country, skill
ORDER BY country, job_count DESC;
