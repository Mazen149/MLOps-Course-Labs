"""Refactored experiment runner that delegates to helper modules."""
import mlflow
from logger import get_logger
from training import run_experiment

logger = get_logger(__name__)


def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("churn_prediction")

    # Run 1: Decision Tree with shallow depth
    run_experiment(
        run_name="DT_shallow",
        model_type="decision_tree",
        hyperparams={
            "max_depth": 5,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42,
        },
    )

    # Run 2: Decision Tree with deeper depth
    run_experiment(
        run_name="DT_deep",
        model_type="decision_tree",
        hyperparams={
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
        },
    )

    # Run 3: Random Forest for enhanced performance
    run_experiment(
        run_name="RF_ensemble",
        model_type="random_forest",
        hyperparams={
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1,
        },
    )


if __name__ == "__main__":
    main()
