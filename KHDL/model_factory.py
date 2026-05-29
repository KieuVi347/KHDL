from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE


def get_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                n_jobs=-1
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            random_state=RANDOM_STATE
        ),

        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=10,
                gamma="scale"
            ))
        ]),

        "Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                solver="adam",
                max_iter=300,
                early_stopping=False,
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            criterion="gini",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    return models