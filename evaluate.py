"""Evaluate saved phishing URL models within and across datasets."""

from __future__ import annotations

import os
import time
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	balanced_accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
)

from data_loader import load_phreshphish


MODEL_DIRECTORY = "models"
RESULTS_DIRECTORY = "results"
METRIC_NAMES = [
	"accuracy",
	"balanced_accuracy",
	"precision",
	"recall",
	"f1",
	"roc_auc",
	"pr_auc",
]


def _evaluate_model(model: Any, features: pd.DataFrame, labels: pd.Series) -> dict[str, Any]:
	predictions = model.predict(features)
	probabilities = model.predict_proba(features)[:, 1]
	return {
		"accuracy": accuracy_score(labels, predictions),
		"balanced_accuracy": balanced_accuracy_score(labels, predictions),
		"precision": precision_score(labels, predictions, zero_division=0),
		"recall": recall_score(labels, predictions, zero_division=0),
		"f1": f1_score(labels, predictions, zero_division=0),
		"roc_auc": (
			roc_auc_score(labels, probabilities)
			if labels.nunique() == 2
			else float("nan")
		),
		"pr_auc": average_precision_score(labels, probabilities),
		"predictions": predictions,
		"probabilities": probabilities,
	}


def _print_metrics(dataset_name: str, model_name: str, metrics: dict[str, Any]) -> None:
	print(f"\n{model_name} - {dataset_name}")
	for metric_name in METRIC_NAMES:
		print(f"  {metric_name}: {metrics[metric_name]:.4f}")


def _save_confusion_matrix(
	labels: pd.Series,
	predictions: Any,
	output_path: str,
	title: str,
) -> None:
	matrix = confusion_matrix(labels, predictions, labels=[0, 1])
	plt.figure(figsize=(6, 5))
	sns.heatmap(
		matrix,
		annot=True,
		fmt="d",
		cmap="Blues",
		xticklabels=["Legitimate", "Phishing"],
		yticklabels=["Legitimate", "Phishing"],
	)
	plt.title(title)
	plt.xlabel("Predicted label")
	plt.ylabel("True label")
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


def _save_roc_curve(
	labels: pd.Series,
	probabilities: Any,
	output_path: str,
	title: str,
	label: str,
) -> None:
	false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
	plt.figure(figsize=(7, 6))
	plt.plot(
		false_positive_rate,
		true_positive_rate,
		label=f"{label} (AUC = {roc_auc_score(labels, probabilities):.4f})",
	)
	plt.plot([0, 1], [0, 1], "k--", label="Chance")
	plt.xlabel("False positive rate")
	plt.ylabel("True positive rate")
	plt.title(title)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


def _prediction_latency_per_url(model: Any, features: pd.DataFrame) -> float:
	start = time.perf_counter()
	for row_index in range(len(features)):
		model.predict(features.iloc[[row_index]])
	elapsed = time.perf_counter() - start
	return elapsed / len(features) if len(features) else 0.0


def _save_comparison_chart(
	within_metrics: dict[str, dict[str, Any]],
	cross_metrics: dict[str, dict[str, Any]],
) -> None:
	values = []
	labels = []
	for model_name in ("Random Forest", "XGBoost"):
		values.extend(
			[
				[within_metrics[model_name][metric] for metric in METRIC_NAMES],
				[cross_metrics[model_name][metric] for metric in METRIC_NAMES],
			]
		)
		labels.extend([f"{model_name}\nPhiUSIIL", f"{model_name}\nPhreshPhish"])

	positions = range(len(METRIC_NAMES))
	width = 0.2
	plt.figure(figsize=(11, 6))
	for index, (label, metric_values) in enumerate(zip(labels, values)):
		offset = (index - 1.5) * width
		plt.bar([position + offset for position in positions], metric_values, width, label=label)
	plt.xticks(list(positions), METRIC_NAMES)
	plt.ylim(0, 1.05)
	plt.ylabel("Score")
	plt.title("Within-dataset and cross-dataset model comparison")
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(RESULTS_DIRECTORY, "cross_dataset_comparison.png"), dpi=150)
	plt.close()


def _save_combined_roc_plot(
	curves: list[tuple[pd.Series, Any, str]],
) -> None:
	plt.figure(figsize=(8, 7))
	for labels, probabilities, label in curves:
		false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
		auc = roc_auc_score(labels, probabilities)
		plt.plot(false_positive_rate, true_positive_rate, label=f"{label} (AUC = {auc:.4f})")
	plt.plot([0, 1], [0, 1], "k--", label="Chance")
	plt.xlabel("False positive rate")
	plt.ylabel("True positive rate")
	plt.title("ROC comparison across datasets")
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(RESULTS_DIRECTORY, "roc_comparison_all.png"), dpi=150)
	plt.close()


def main() -> None:
	random_forest = joblib.load(os.path.join(MODEL_DIRECTORY, "random_forest.joblib"))
	xgboost = joblib.load(os.path.join(MODEL_DIRECTORY, "xgboost.joblib"))
	test_split = joblib.load(
		os.path.join(MODEL_DIRECTORY, "phiusiil_test_split.joblib")
	)
	X_test = test_split["X_test"]
	y_test = test_split["y_test"]

	os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
	models = {"Random Forest": random_forest, "XGBoost": xgboost}
	within_metrics = {}
	within_curves = {}
	confusion_paths = {
		"Random Forest": "confusion_matrix_rf_within.png",
		"XGBoost": "confusion_matrix_xgb_within.png",
	}
	roc_paths = {"Random Forest": "roc_within_rf.png", "XGBoost": "roc_within_xgb.png"}

	print("=== Within-dataset evaluation: PhiUSIIL ===")
	for model_name, model in models.items():
		metrics = _evaluate_model(model, X_test, y_test)
		within_metrics[model_name] = metrics
		within_curves[model_name] = (y_test, metrics["probabilities"])
		_print_metrics("PhiUSIIL", model_name, metrics)
		_save_confusion_matrix(
			y_test,
			metrics["predictions"],
			os.path.join(RESULTS_DIRECTORY, confusion_paths[model_name]),
			f"{model_name} confusion matrix - PhiUSIIL",
		)
		_save_roc_curve(
			y_test,
			metrics["probabilities"],
			os.path.join(RESULTS_DIRECTORY, roc_paths[model_name]),
			f"{model_name} ROC curve - PhiUSIIL",
			model_name,
		)

	phreshphish = load_phreshphish()
	X_cross = phreshphish.drop(columns=["label"])
	y_cross = phreshphish["label"]
	cross_metrics = {}
	cross_curves = {}

	print("\n=== Cross-dataset evaluation: PhreshPhish ===")
	for model_name, model in models.items():
		metrics = _evaluate_model(model, X_cross, y_cross)
		cross_metrics[model_name] = metrics
		cross_curves[model_name] = (y_cross, metrics["probabilities"])
		_print_metrics("PhreshPhish", model_name, metrics)
		latency = _prediction_latency_per_url(model, X_cross)
		print(f"  average prediction latency per URL: {latency * 1000:.4f} ms")

	_save_comparison_chart(within_metrics, cross_metrics)
	_save_combined_roc_plot(
		[
			(y_test, within_curves["Random Forest"][1], "RF-within"),
			(y_cross, cross_curves["Random Forest"][1], "RF-cross"),
			(y_test, within_curves["XGBoost"][1], "XGB-within"),
			(y_cross, cross_curves["XGBoost"][1], "XGB-cross"),
		]
	)

	print("\n=== Summary ===")
	for model_name in models:
		within_accuracy = within_metrics[model_name]["accuracy"]
		cross_accuracy = cross_metrics[model_name]["accuracy"]
		change = (cross_accuracy - within_accuracy) * 100
		direction = "increased" if change >= 0 else "dropped"
		print(
			f"{model_name}: accuracy {direction} from {within_accuracy:.2f} "
			f"(PhiUSIIL) to {cross_accuracy:.2f} (PhreshPhish), "
			f"a change of {change:+.2f} percentage points."
		)


if __name__ == "__main__":
	main()
