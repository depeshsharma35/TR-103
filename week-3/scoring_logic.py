import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

def calculate_scores():
    df = pd.read_csv("processed_resumes.csv")
    with open("processed_job_description.json", "r") as f:
        jd = json.load(f)

    df['extracted_skills'] = df['extracted_skills'].apply(ast.literal_eval)
    required_skills = jd['required_skills']

    texts = [jd['cleaned_description']] + df['cleaned_resume'].tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    df['cosine_similarity'] = cosine_sim

    def get_skill_score(candidate_skills):
        if not required_skills:
            return 0
        matches = set(candidate_skills).intersection(set(required_skills))
        return len(matches) / len(required_skills)

    df['skill_match_score'] = df['extracted_skills'].apply(get_skill_score)

    def get_skill_gap(candidate_skills):
        missing = set(required_skills) - set(candidate_skills)
        return list(missing)

    df['missing_skills'] = df['extracted_skills'].apply(get_skill_gap)

    df['final_score'] = (df['cosine_similarity'] * 0.4) + (df['skill_match_score'] * 0.6)

    df = df.sort_values(by='final_score', ascending=False)

    df.to_csv("ranked_candidates.csv", index=False)
    print("Candidates ranked and saved to ranked_candidates.csv")

    print("\nRanking Summary:")
    print(df[['candidate_name', 'final_score', 'skill_match_score', 'cosine_similarity']])

if __name__ == "__main__":
    calculate_scores()
