from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)

import matplotlib.pyplot as plt
import joblib
import json
import warnings

warnings.filterwarnings('ignore')


def main():

    # Load Iris Dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names

    print("=" * 60)
    print("DECODELABS PROJECT 2: DATA CLASSIFICATION USING AI")
    print("=" * 60)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Pipeline
    pipe = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier()
    )

    # Hyperparameter Tuning
    param_grid = {
        'kneighborsclassifier__n_neighbors': range(1, 11)
    }

    model = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Cross Validation
    cv_scores = cross_val_score(
        model.best_estimator_,
        X_train,
        y_train,
        cv=5,
        scoring='f1_weighted'
    )

    # Predictions
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')

    # Results Dictionary
    results = {
        "optimal_k": int(
            model.best_params_[
                'kneighborsclassifier__n_neighbors'
            ]
        ),
        "test_accuracy": round(accuracy, 4),
        "test_f1_weighted": round(f1, 4),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4)
    }

    print("\n[OUTPUT] Validation Results")
    print("=" * 60)

    for k, v in results.items():
        print(f"{k}: {v}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=target_names
        )
    )

    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=target_names,
        cmap='Blues',
        values_format='d'
    )

    plt.title("Confusion Matrix - KNN Classification")
    plt.tight_layout()
    plt.savefig(
        "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    # Save Results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nArtifacts saved successfully:")
    print("1. confusion_matrix.png")
    print("2. results.json")

    plt.show()


if __name__ == "__main__":
    main()