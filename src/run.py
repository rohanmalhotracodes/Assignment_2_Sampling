import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

from sampling import (
    sample_srs, sample_stratified, sample_systematic, sample_cluster, sample_bootstrap
)
from models import get_models

SAMPLERS = {
    "Sampling1_SRS": sample_srs,
    "Sampling2_Stratified": sample_stratified,
    "Sampling3_Systematic": sample_systematic,
    "Sampling4_Cluster": sample_cluster,
    "Sampling5_Bootstrap": sample_bootstrap,
}

def main():
    ap = argparse.ArgumentParser(description="Sampling Assignment - run experiments")
    ap.add_argument("--data", type=str, default="data/Creditcard_data.csv", help="Path to Creditcard_data.csv")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--sample_size", type=int, default=500, help="Sample size for each sampling technique (from train)")
    ap.add_argument("--outdir", type=str, default="results", help="Output directory")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path.resolve()}")

    df = pd.read_csv(data_path)
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column in dataset.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Balance dataset (50:50) using SMOTE
    smote = SMOTE(random_state=args.seed, k_neighbors=3)
    Xb, yb = smote.fit_resample(X, y)

    # Train/test split on balanced data
    X_train, X_test, y_train, y_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=args.seed, stratify=yb
    )

    models = get_models(seed=args.seed)

    n = min(args.sample_size, len(X_train))
    results = pd.DataFrame(index=models.keys(), columns=SAMPLERS.keys(), dtype=float)

    for sname, sfn in SAMPLERS.items():
        Xs, ys = sfn(pd.DataFrame(X_train), pd.Series(y_train), n, seed=args.seed)
        for mname, model in models.items():
            model.fit(Xs, ys)
            pred = model.predict(X_test)
            results.loc[mname, sname] = accuracy_score(y_test, pred)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results.to_csv(outdir / "accuracy_matrix.csv")
    (outdir / "accuracy_matrix.md").write_text(results.round(4).to_markdown())

    # Print summary
    best_per_model = results.idxmax(axis=1)
    best_vals = results.max(axis=1)

    best_model_per_sampling = results.idxmax(axis=0)
    best_model_vals = results.max(axis=0)

    print("\nAccuracy matrix (test set):\n")
    print(results.round(4).to_string())
    print("\nBest sampling technique for each model:\n")
    print(pd.DataFrame({"Best Sampling": best_per_model, "Best Accuracy": best_vals.round(4)}).to_string())
    print("\nBest model for each sampling technique:\n")
    print(pd.DataFrame({"Best Model": best_model_per_sampling, "Best Accuracy": best_model_vals.round(4)}).to_string())

if __name__ == "__main__":
    main()
