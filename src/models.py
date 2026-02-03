from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def get_models(seed=42):
    return {
        "M1_LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed))
        ]),
        "M2_DecisionTree": DecisionTreeClassifier(random_state=seed),
        "M3_RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
        "M4_GradBoost": GradientBoostingClassifier(random_state=seed),
        "M5_SVC": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", random_state=seed))
        ]),
    }
