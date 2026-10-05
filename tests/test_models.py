import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_data():
    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names,
    )

    y = pd.Series(data.target)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def test_svm_training():
    X_train, X_test, y_train, y_test = get_data()

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVC(C=1, kernel="rbf")),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_knn_training():
    X_train, X_test, y_train, y_test = get_data()

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_random_forest_training():
    X_train, X_test, y_train, y_test = get_data()

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert len(model.feature_importances_) == X_train.shape[1]