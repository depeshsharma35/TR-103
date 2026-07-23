# pyrefly: ignore [missing-import]
import spacy
import subprocess
import sys
import re
import json
import string
from collections import Counter

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[INFO] spaCy model 'en_core_web_sm' not found — downloading now...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

SKILL_DB = [
    'python', 'machine learning', 'sql', 'scikit-learn', 'pandas', 'matplotlib',
    'javascript', 'react', 'node.js', 'html', 'css', 'nlp', 'deep learning',
    'pytorch', 'tensorflow', 'feature engineering', 'model deployment',
    'excel', 'tableau', 'r', 'agile', 'scrum', 'jira', 'trello', 'predictive modeling',
    'statistical analysis', 'data visualization', 'data cleaning', 'data preprocessing'
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

def extract_skills(text):
    text = text.lower()
    found_skills = []
    for skill in SKILL_DB:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text):
            found_skills.append(skill)
    return list(set(found_skills))

def process_data():
    df = pd.read_csv("resumes.csv")
    df['cleaned_resume'] = df['resume_text'].apply(clean_text)
    df['extracted_skills'] = df['resume_text'].apply(extract_skills)
    df.to_csv("processed_resumes.csv", index=False)
    print("Processed resumes saved to processed_resumes.csv")

    with open("job_description.json", "r") as f:
        jd = json.load(f)

    jd_text = jd['description']
    jd_cleaned = clean_text(jd_text)
    jd_skills = extract_skills(jd_text)

    processed_jd = {
        "role": jd['role'],
        "cleaned_description": jd_cleaned,
        "required_skills": jd_skills
    }

    with open("processed_job_description.json", "w") as f:
        json.dump(processed_jd, f, indent=4)
    print("Processed job description saved to processed_job_description.json")

if __name__ == "__main__":
    process_data()
