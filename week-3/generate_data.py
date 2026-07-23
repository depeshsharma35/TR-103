import pandas as pd
import json

resumes = [
    {
        "candidate_name": "Alice Johnson",
        "resume_text": "Experienced Data Scientist with 5 years in Python, Machine Learning, and SQL. Expert in Scikit-learn, Pandas, and data visualization using Matplotlib. Strong background in statistical analysis and predictive modeling."
    },
    {
        "candidate_name": "Bob Smith",
        "resume_text": "Software Engineer with a focus on web development. Proficient in JavaScript, React, Node.js, and HTML/CSS. Basic knowledge of Python and SQL. Interested in transitioning to Machine Learning roles."
    },
    {
        "candidate_name": "Charlie Brown",
        "resume_text": "Machine Learning Engineer specialized in NLP and Deep Learning. Expert in PyTorch, TensorFlow, and spaCy. Strong skills in Python, feature engineering, and model deployment."
    },
    {
        "candidate_name": "Diana Prince",
        "resume_text": "Data Analyst with 3 years of experience. Proficient in Excel, SQL, and Tableau. Some experience with Python and R for data cleaning and basic statistical analysis."
    },
    {
        "candidate_name": "Edward Norton",
        "resume_text": "Project Manager with experience in Agile and Scrum. Strong communication and leadership skills. Familiar with Jira and Trello. No technical background in programming or ML."
    }
]

job_description = {
    "role": "Senior Machine Learning Engineer",
    "description": "We are looking for a Senior Machine Learning Engineer with strong expertise in Python, Scikit-learn, and NLP. Candidates should have experience in predictive modeling, feature engineering, and deep learning frameworks like PyTorch or TensorFlow. Proficiency in SQL and data preprocessing is required."
}

df_resumes = pd.DataFrame(resumes)
df_resumes.to_csv("resumes.csv", index=False)

with open("job_description.json", "w") as f:
    json.dump(job_description, f, indent=4)

print("Synthetic data generated: resumes.csv and job_description.json")
