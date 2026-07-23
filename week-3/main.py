import subprocess
import sys
import os

def run_step(script, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[OK] {script} completed successfully.")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    run_step("generate_data.py",     "Step 1 / 4 — Generating Synthetic Resume & Job Description Data")
    run_step("nlp_pipeline.py",      "Step 2 / 4 — NLP Processing (Cleaning + Skill Extraction)")
    run_step("scoring_logic.py",     "Step 3 / 4 — Scoring & Ranking Candidates")
    run_step("visualize_results.py", "Step 4 / 4 — Generating Visualisations")

    print("\n" + "="*60)
    print("  Pipeline complete!")
    print("  ranked_candidates.csv  — final ranked output")
    print("  candidate_ranking.png  — ranking bar chart")
    print("  metric_comparison.png  — similarity vs skill chart")
    print("  skill_gap_heatmap.png  — skill gap matrix")
    print("="*60 + "\n")
