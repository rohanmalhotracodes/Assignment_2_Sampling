import argparse
from pathlib import Path
import pandas as pd

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="../data/Creditcard_data.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_size", type=int, default=500)
    ap.add_argument("--outdir", type=str, default="../results")
    args = ap.parse_args()

    df = pd.read_csv(Path(args.data))
    X = df.drop(columns=["Class"])
    y = df["Class"]

    smote = SMOTE(random_state=args.seed, k_neighbors=3)
    Xb, yb = smote.fit_resample(X, y)

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
    print(results.round(4).to_string())

if __name__ == "__main__":
    main()
