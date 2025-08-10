import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from preprocess import load_data, preprocess_data

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data("data/train.csv")
    X, y, preprocessor = preprocess_data(df)

    # Define models and hyperparameters for tuning
    models_params = {
        "logistic": {
            "classifier": LogisticRegression(max_iter=500, random_state=42),
            "params": {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs", "liblinear"]
            }
        },
        "random_forest": {
            "classifier": RandomForestClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5]
            }
        },
        "gradient_boosting": {
            "classifier": GradientBoostingClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1, 0.2],
                "classifier__max_depth": [3, 5]
            }
        }
    }

    best_model = None
    best_score = 0
    best_name = ""
    best_params = None

    # Train and evaluate each model
    for name, mp in models_params.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", mp["classifier"])
        ])

        grid_search = GridSearchCV(
            pipeline,
            mp["params"],
            cv=5,
            scoring="recall",
            n_jobs=-1
        )

        grid_search.fit(X, y)

        print(f"{name} Best CV recall: {grid_search.best_score_:.3f}")
        print(f"{name} Best Params: {grid_search.best_params_}")

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_name = name
            best_params = grid_search.best_params_

    print(f"\nBest Model: {best_name}")
    print(f"Best CV Accuracy: {best_score:.3f}")
    print(f"Best Params: {best_params}")

    # Save the best model
    joblib.dump(best_model, f"models/{best_name}_best_model.pkl")
    print(f"Model saved to models/{best_name}_best_model.pkl")
