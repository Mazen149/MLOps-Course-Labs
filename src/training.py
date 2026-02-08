"""Training utilities: model fitting and running experiments.

This module encapsulates model training and experiment run logic so
`train2_dt.py` stays concise.
"""
from typing import Dict
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from preprocessing import preprocess
from logger import get_logger

logger = get_logger(__name__)


def train(X_train, y_train, model):
    model.fit(X_train, y_train)
    sig = mlflow.models.signature.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=sig)
    try:
        mlflow.log_input(mlflow.data.from_pandas(X_train, source="train_data"), context="training")
    except Exception:
        pass
    return model


def run_experiment(run_name: str, model_type: str, hyperparams: Dict):
    with mlflow.start_run(run_name=run_name):
        logger.info("Starting run %s (%s)", run_name, model_type)
        import pandas as pd
        df = pd.read_csv("../../archive/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        mlflow.log_params(hyperparams)
        mlflow.log_param("model_type", model_type)

        # instantiate model based on type
        if model_type == "decision_tree":
            from sklearn.tree import DecisionTreeClassifier

            model = DecisionTreeClassifier(**hyperparams)
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(**hyperparams)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model = train(X_train, y_train, model)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        mlflow.set_tag("model_type", model_type)

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=model.classes_)
        conf_mat_disp.plot()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        logger.info("Completed run %s: acc=%.4f f1=%.4f", run_name, accuracy, f1)
