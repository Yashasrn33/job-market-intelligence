from processing.skill_extractor import extract_skills


def test_extract_skills_basic():
    text = "We use Python, AWS, Docker and Kubernetes daily."
    skills = extract_skills(text)
    assert "python" in skills
    assert "aws" in skills
    assert "docker" in skills
    assert "kubernetes" in skills


def test_extract_skills_aliases():
    text = "Experience with React.js and Node.js required"
    skills = extract_skills(text)
    assert "react" in skills
    assert "nodejs" in skills


def test_extract_skills_ml():
    text = "Deep learning experience with PyTorch or TensorFlow"
    skills = extract_skills(text)
    assert "deep learning" in skills
    assert "pytorch" in skills
    assert "tensorflow" in skills


def test_extract_skills_empty():
    assert extract_skills("") == []
    assert extract_skills(None) == []
