"""Live end-to-end evaluation of the email phishing detector."""

from __future__ import annotations

import os
import re
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

from data_loader import load_phiusiil_url_labels
from email_monitor import check_inbox


MODEL_PATH = Path("models") / "random_forest.joblib"
ALERTS_PATH = Path("alerts.log")
TEST_CASE_COUNT = 10
RUN_ID_PATTERN = re.compile(r"email-module-eval-[0-9a-f]{12}")


def _required_environment() -> dict[str, str]:
	load_dotenv()
	names = (
		"SMTP_HOST",
		"SMTP_USER",
		"SMTP_PASSWORD",
		"IMAP_HOST",
		"IMAP_USER",
		"IMAP_PASSWORD",
		"TEST_INBOX_ADDRESS",
	)
	values = {name: os.environ.get(name) for name in names}
	missing = [name for name, value in values.items() if not value]
	if missing:
		raise RuntimeError(
			"Missing required environment variables: " + ", ".join(missing)
		)
	return {name: value for name, value in values.items() if value is not None}


def build_test_cases(count: int = TEST_CASE_COUNT) -> list[tuple[str, str, int]]:
	"""Create deterministic cases from real PhiUSIIL URLs and labels."""
	if count < 10:
		raise ValueError("The integration test requires at least 10 test cases")

	data = load_phiusiil_url_labels()
	rows_per_label = (count + 1) // 2
	selected = pd.concat(
		[
			data[data["label"] == label].head(rows_per_label)
			for label in (0, 1)
		],
		ignore_index=True,
	)
	if len(selected) < count:
		raise RuntimeError("PhiUSIIL does not contain enough rows for the test cases")

	run_id = f"email-module-eval-{os.urandom(6).hex()}"
	cases = []
	for index, row in selected.head(count).iterrows():
		url = str(row["url"])
		subject = f"{run_id} case {index + 1}"
		body = (
			f"Plain-text URL for integration case {index + 1}: {url}\n"
			"This message is for the dedicated phishing-detector test inbox."
		)
		cases.append((subject, body, int(row["label"])))
	return cases


def send_test_email(
	host: str,
	user: str,
	password: str,
	recipient: str,
	subject: str,
	body: str,
) -> datetime:
	"""Send one multipart test message and return its send timestamp."""
	url = _url_from_body(body)
	message = EmailMessage()
	message["From"] = user
	message["To"] = recipient
	message["Subject"] = subject
	message.set_content(body)
	message.add_alternative(
		f"<html><body><p>{body.replace(url, f'<a href=\"{url}\">{url}</a>')}</p>"
		"</body></html>",
		subtype="html",
	)
	sent_at = datetime.now(timezone.utc)
	with smtplib.SMTP_SSL(host, 465, timeout=30) as smtp:
		smtp.login(user, password)
		smtp.send_message(message)
	return sent_at


def _url_from_body(body: str) -> str:
	match = re.search(r"https?://\S+", body)
	if match is None:
		raise ValueError("Test email body must contain an HTTP(S) URL")
	return match.group(0).rstrip(".,;:!?)]}>")


def _read_run_alerts(run_id: str) -> dict[str, datetime]:
	alerts: dict[str, datetime] = {}
	if not ALERTS_PATH.exists():
		return alerts
	for line in ALERTS_PATH.read_text(encoding="utf-8").splitlines():
		if run_id not in line:
			continue
		parts = line.split("\t")
		url_part = next((part for part in parts if part.startswith("URL: ")), None)
		if url_part is not None:
			alerts[url_part.removeprefix("URL: ")] = datetime.fromisoformat(parts[0])
	return alerts


def _print_results(
	cases: list[tuple[str, str, int]],
	sent_at: dict[str, datetime],
	processed_at: dict[str, datetime],
	alerts: dict[str, datetime],
) -> None:
	results: list[tuple[str, int, int, bool, float]] = []
	for subject, body, ground_truth in cases:
		url = _url_from_body(body)
		processed = url in processed_at
		predicted = int(url in alerts) if processed else 0
		end_at = alerts.get(url, processed_at.get(url))
		if end_at is None:
			raise RuntimeError(f"No processing event recorded for {url}")
		results.append(
			(url, ground_truth, predicted, processed, (end_at - sent_at[subject]).total_seconds())
		)

	extraction_accuracy = sum(result[3] for result in results) / len(results)
	classification_accuracy = sum(
		result[3] and result[2] == result[1] for result in results
	) / len(results)
	print(f"URL extraction accuracy: {extraction_accuracy:.4f}")
	print(f"Classification accuracy: {classification_accuracy:.4f}")
	print("End-to-end delay per URL:")
	for url, ground_truth, predicted, _, delay in results:
		print(
			f"  {delay:.3f}s | truth={ground_truth} predicted={predicted} | {url}"
		)


def main() -> None:
	settings = _required_environment()
	cases = build_test_cases()
	sent_at: dict[str, datetime] = {}
	processed_at: dict[str, datetime] = {}
	for subject, body, _ in cases:
		sent_at[subject] = send_test_email(
			settings["SMTP_HOST"],
			settings["SMTP_USER"],
			settings["SMTP_PASSWORD"],
			settings["TEST_INBOX_ADDRESS"],
			subject,
			body,
		)

	def record_processed(subject: str, urls: list[str], processed_at_utc: datetime) -> None:
		for url in urls:
			processed_at[url] = processed_at_utc

	check_inbox(
		settings["IMAP_HOST"],
		settings["IMAP_USER"],
		settings["IMAP_PASSWORD"],
		joblib.load(MODEL_PATH),
		poll_interval=2,
		max_cycles=30,
		on_processed=record_processed,
	)
	_run_id = RUN_ID_PATTERN.search(cases[0][0])
	if _run_id is None:
		raise RuntimeError("Could not identify the evaluation run ID")
	_print_results(cases, sent_at, processed_at, _read_run_alerts(_run_id.group(0)))


if __name__ == "__main__":
	main()
