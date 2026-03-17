"""
Skill Extraction Engine - tech skills taxonomy with alias resolution and NLP
"""
import re
from typing import List, Set

TECH_SKILLS = {
    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
    'ruby', 'php', 'scala', 'kotlin', 'swift', 'r',
    # Frontend
    'react', 'angular', 'vue', 'svelte', 'nextjs', 'html', 'css', 'tailwind',
    # Backend
    'nodejs', 'django', 'flask', 'fastapi', 'spring', 'rails', 'express',
    # Databases
    'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'dynamodb',
    'snowflake', 'bigquery', 'redshift',
    # Cloud
    'aws', 'azure', 'gcp', 'kubernetes', 'docker', 'terraform', 'serverless',
    # Data Engineering
    'spark', 'airflow', 'kafka', 'dbt', 'etl', 'data pipeline',
    # ML/AI
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
    'nlp', 'computer vision', 'llm', 'mlops', 'langchain',
    # DevOps
    'ci/cd', 'jenkins', 'github actions', 'ansible', 'prometheus', 'grafana',
}

SKILL_ALIASES = {
    'react.js': 'react', 'reactjs': 'react', 'vue.js': 'vue',
    'node.js': 'nodejs', 'k8s': 'kubernetes', 'postgres': 'postgresql',
    'mongo': 'mongodb', 'ml': 'machine learning', 'dl': 'deep learning',
}


def extract_skills(text: str) -> List[str]:
    """Extract tech skills from a job description using keyword matching."""
    if not text:
        return []

    text_lower = text.lower()
    found: Set[str] = set()

    for alias, canonical in SKILL_ALIASES.items():
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, text_lower):
            found.add(canonical)

    for skill in TECH_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.add(skill)

    return sorted(found)


if __name__ == '__main__':
    sample = "Looking for Python developer with AWS, Kubernetes and machine learning experience"
    print(f"Extracted: {extract_skills(sample)}")
