"""Train and persist phishing URL classifiers on the PhiUSIIL dataset."""

from __future__ import annotations

import os
import time

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	balanced_accuracy_score,
	f1_score,
	precision_score,
	recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_loader import load_phiusiil


MODEL_DIRECTORY = "models"


def main() -> None:
	data = load_phiusiil()
	features = data.drop(columns=["label"])
	labels = data["label"]
	X_train, X_test, y_train, y_test = train_test_split(
		features,
		labels,
		test_size=0.2,
		random_state=42,
		stratify=labels,
	)

	class_counts = y_train.value_counts()
	if len(class_counts) != 2 or (class_counts == 0).any():
		raise ValueError("Training data must contain both phishing and legitimate URLs")
	# XGBoost's positive class is phishing (label 1).
	scale_pos_weight = class_counts[0] / class_counts[1]

	os.makedirs(MODEL_DIRECTORY, exist_ok=True)

	random_forest = RandomForestClassifier(
		n_estimators=200,
		random_state=42,
		n_jobs=-1,
		class_weight="balanced",
	)
	random_forest_start = time.perf_counter()
	random_forest.fit(X_train, y_train)
	random_forest_time = time.perf_counter() - random_forest_start
	joblib.dump(random_forest, os.path.join(MODEL_DIRECTORY, "random_forest.joblib"))

	xgboost = XGBClassifier(
		n_estimators=200,
		max_depth=6,
		learning_rate=0.1,
		random_state=42,
		eval_metric="logloss",
		n_jobs=-1,
		scale_pos_weight=scale_pos_weight,
	)
	xgboost_start = time.perf_counter()
	xgboost.fit(X_train, y_train)
	xgboost_time = time.perf_counter() - xgboost_start
	joblib.dump(xgboost, os.path.join(MODEL_DIRECTORY, "xgboost.joblib"))

	joblib.dump(
		{"X_test": X_test, "y_test": y_test},
		os.path.join(MODEL_DIRECTORY, "phiusiil_test_split.joblib"),
	)

	print(f"\nTraining class counts: {class_counts.to_dict()}")
	print(f"XGBoost scale_pos_weight: {scale_pos_weight:.3f}")
	print("\nTraining summary (test split)")
	print(
		f"{'Model':<20}{'Time (s)':>12}{'Bal. acc.':>12}"
		f"{'Precision':>12}{'Recall':>12}{'F1':>12}"
	)
	for name, model, elapsed in (
		("Random Forest", random_forest, random_forest_time),
		("XGBoost", xgboost, xgboost_time),
	):
		predictions = model.predict(X_test)
		print(
			f"{name:<20}{elapsed:>12.2f}"
			f"{balanced_accuracy_score(y_test, predictions):>12.4f}"
			f"{precision_score(y_test, predictions, zero_division=0):>12.4f}"
			f"{recall_score(y_test, predictions, zero_division=0):>12.4f}"
			f"{f1_score(y_test, predictions, zero_division=0):>12.4f}"
		)


if __name__ == "__main__":
	main()
