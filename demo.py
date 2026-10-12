"""Command-line entry point for the phishing URL detection prototype."""

from __future__ import annotations

import os
import sys
import imaplib

import joblib
import pandas as pd
from dotenv import load_dotenv

from email_monitor import check_inbox
from feature_extraction import extract_url_features


MODEL_PATH = os.path.join("models", "random_forest.joblib")


def _load_random_forest():
	try:
		return joblib.load(MODEL_PATH)
	except FileNotFoundError:
		print("No trained model found. Please run 'python train.py' first.")
		return None


def _check_single_url() -> None:
	url = input("Enter a URL to check: ").strip()
	model = _load_random_forest()
	if model is None:
		return

	feature_vector = pd.DataFrame([extract_url_features(url)])
	model_columns = getattr(model, "feature_names_in_", feature_vector.columns)
	feature_vector = feature_vector.reindex(columns=model_columns)
	prediction = int(model.predict(feature_vector)[0])
	probabilities = model.predict_proba(feature_vector)[0]
	predicted_class_index = list(model.classes_).index(prediction)
	confidence = float(probabilities[predicted_class_index])
	result = "PHISHING" if prediction == 1 else "LEGITIMATE"
	print(f"Result: {result} (confidence: {confidence:.1%})")


def _run_evaluation() -> None:
	import evaluate

	evaluate.main()
	print("Evaluation complete. Results and plots were saved to the results/ directory.")


def _start_email_monitoring() -> None:
	print(
		"This will monitor the email inbox configured in your .env file.\n"
		"Only point this at a dedicated TEST account you created yourself."
	)
	confirmation = input("Type YES to confirm and continue: ")
	if confirmation != "YES":
		print("Cancelled.")
		return

	load_dotenv()
	host = os.environ.get("IMAP_HOST")
	user = os.environ.get("IMAP_USER")
	password = os.environ.get("IMAP_PASSWORD")
	missing = [
		name
		for name, value in {
			"IMAP_HOST": host,
			"IMAP_USER": user,
			"IMAP_PASSWORD": password,
		}.items()
		if not value
	]
	if missing:
		print(f"Missing required environment variables: {', '.join(missing)}")
		return

	model = _load_random_forest()
	if model is None:
		return

	try:
		check_inbox(host, user, password, model)
	except KeyboardInterrupt:
		print("Monitoring stopped.")
	except imaplib.IMAP4.error as error:
		print(f"Email monitoring failed: {error}")


def _print_menu() -> None:
	print(
		"\n==================================================\n"
		"Phishing Website Detection Tool\n"
		"==================================================\n"
		"1. Check a single URL\n"
		"2. Run cross-dataset evaluation (Stage 1 + Stage 2)\n"
		"3. Start email monitoring (test inbox only)\n"
		"4. Exit\n"
		"=================================================="
	)


def main() -> None:
	print(
		"Phishing URL Detection Research Prototype\n"
		"This prototype supports generalisation testing across public datasets\n"
		"and includes a background email-scanning module tested only via a\n"
		"dedicated test account."
	)

	while True:
		_print_menu()
		choice = input("Choose an option (1-4): ")
		if choice == "1":
			_check_single_url()
		elif choice == "2":
			_run_evaluation()
		elif choice == "3":
			_start_email_monitoring()
		elif choice == "4":
			print("Goodbye.")
			return
		else:
			print("Invalid option, please choose 1-4.")


if __name__ == "__main__":
	sys.exit(main())
