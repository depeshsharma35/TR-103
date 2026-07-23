import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import ast

def generate_visuals():
    df = pd.read_csv("ranked_candidates.csv")

    if df['missing_skills'].dtype == object:
        df['missing_skills'] = df['missing_skills'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette('viridis', len(df))
    bars = ax.barh(df['candidate_name'], df['final_score'], color=colors)
    ax.set_title('Candidate Ranking based on Match Score', fontsize=15, fontweight='bold')
    ax.set_xlabel('Final Match Score (0 to 1)', fontsize=12)
    ax.set_ylabel('Candidate Name', fontsize=12)
    ax.set_xlim(0, 1)
    for bar, score in zip(bars, df['final_score']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{score:.3f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('candidate_ranking.png', dpi=150)
    plt.close()
    print("Saved: candidate_ranking.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    df_melted = df.melt(
        id_vars='candidate_name',
        value_vars=['cosine_similarity', 'skill_match_score'],
        var_name='Metric',
        value_name='Score'
    )
    sns.barplot(data=df_melted, x='Score', y='candidate_name', hue='Metric', ax=ax)
    ax.set_title('Comparison of NLP Similarity vs Skill Matching', fontsize=15, fontweight='bold')
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Candidate Name', fontsize=12)
    ax.legend(title='Metric')
    plt.tight_layout()
    plt.savefig('metric_comparison.png', dpi=150)
    plt.close()
    print("Saved: metric_comparison.png")

    all_missing = set()
    for gaps in df['missing_skills']:
        all_missing.update(gaps)
    all_missing = sorted(all_missing)

    heat_data = pd.DataFrame(
        [[1 if skill in gaps else 0 for skill in all_missing] for gaps in df['missing_skills']],
        index=df['candidate_name'],
        columns=all_missing
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(heat_data, annot=True, fmt='d', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, cbar_kws={'label': '1 = Missing skill'})
    ax.set_title('Skill Gap Analysis (1 = Skill Missing)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('skill_gap_heatmap.png', dpi=150)
    plt.close()
    print("Saved: skill_gap_heatmap.png")

    print("\nAll visualisations saved successfully.")

if __name__ == "__main__":
    generate_visuals()
