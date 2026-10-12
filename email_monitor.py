"""Monitor a dedicated research test inbox for phishing URLs.

This module must only be pointed at a dedicated test email account created
for this research, never a real third-party inbox, in line with the study's
ethical scope.
"""

from __future__ import annotations

import imaplib
import os
import re
import time
from datetime import datetime, timezone
from email import message_from_bytes
from email.header import decode_header
from email.message import Message
from typing import Any, Callable

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from feature_extraction import extract_url_features


URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
ProcessedCallback = Callable[[str, list[str], datetime], None]


def _decode_header_value(value: str | None) -> str:
	if not value:
		return ""
	decoded_parts = []
	for part, encoding in decode_header(value):
		if isinstance(part, bytes):
			decoded_parts.append(part.decode(encoding or "utf-8", errors="replace"))
		else:
			decoded_parts.append(part)
	return "".join(decoded_parts)


def _decode_message_part(part: Message) -> str:
	payload = part.get_payload(decode=True)
	if payload is None:
		return ""
	charset = part.get_content_charset() or "utf-8"
	return payload.decode(charset, errors="replace")


def _extract_urls(message: Message) -> list[str]:
	urls: list[str] = []
	plain_text_parts: list[str] = []
	html_parts: list[str] = []

	parts = message.walk() if message.is_multipart() else [message]
	for part in parts:
		if part.get_content_disposition() == "attachment":
			continue
		content_type = part.get_content_type()
		content = _decode_message_part(part)
		if content_type == "text/plain":
			plain_text_parts.append(content)
		elif content_type == "text/html":
			html_parts.append(content)

	for text in plain_text_parts:
		urls.extend(URL_PATTERN.findall(text))
	for html in html_parts:
		soup = BeautifulSoup(html, "html.parser")
		urls.extend(anchor.get("href") for anchor in soup.find_all("a") if anchor.get("href"))
		urls.extend(URL_PATTERN.findall(soup.get_text(" ")))

	return list(dict.fromkeys(url.rstrip(".,;:!?)]}>\"'") for url in urls))


def _predict_url(model: Any, url: str) -> tuple[int, float]:
	feature_vector = pd.DataFrame([extract_url_features(url)])
	prediction = int(model.predict(feature_vector)[0])
	probabilities = model.predict_proba(feature_vector)[0]
	phishing_index = list(model.classes_).index(1)
	return prediction, float(probabilities[phishing_index])


def _write_alert(subject: str, sender: str, url: str, confidence: float) -> None:
	timestamp = datetime.now(timezone.utc).isoformat()
	with open("alerts.log", "a", encoding="utf-8") as alert_file:
		alert_file.write(
			f"{timestamp}\tSubject: {subject}\tFrom: {sender}\t"
			f"URL: {url}\tConfidence: {confidence:.4f}\n"
		)


def check_inbox(
	host: str,
	user: str,
	password: str,
	model: Any,
	poll_interval: int = 60,
	max_cycles: int | None = None,
	on_processed: ProcessedCallback | None = None,
) -> None:
	"""Poll an IMAP inbox and log URLs classified as phishing."""
	imap = imaplib.IMAP4_SSL(host)
	mailbox_selected = False
	try:
		imap.login(user, password)
		select_status, _ = imap.select("INBOX")
		mailbox_selected = select_status == "OK"
		if not mailbox_selected:
			raise imaplib.IMAP4.error("Could not select INBOX")
		cycles = 0
		while max_cycles is None or cycles < max_cycles:
			status, message_numbers = imap.search(None, "UNSEEN")
			if status == "OK":
				for number in message_numbers[0].split():
					fetch_status, message_data = imap.fetch(number, "(RFC822)")
					if fetch_status != "OK":
						continue
					raw_message = next(
						(part[1] for part in message_data if isinstance(part, tuple)),
						None,
					)
					if raw_message is None:
						continue
					message = message_from_bytes(raw_message)
					subject = _decode_header_value(message.get("Subject"))
					sender = _decode_header_value(message.get("From"))
					for url in _extract_urls(message):
						prediction, confidence = _predict_url(model, url)
						if prediction == 1:
							_write_alert(subject, sender, url, confidence)
					imap.store(number, "+FLAGS", "\\Seen")
					if on_processed is not None:
						on_processed(subject, _extract_urls(message), datetime.now(timezone.utc))

			cycles += 1
			if max_cycles is not None and cycles >= max_cycles:
				break
			# This is deliberately polling-based: wait before checking UNSEEN again.
			time.sleep(poll_interval)
	finally:
		try:
			if mailbox_selected:
				imap.close()
		finally:
			imap.logout()


def main() -> None:
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
		raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

	import joblib

	model = joblib.load("models/random_forest.joblib")
	check_inbox(host, user, password, model)


if __name__ == "__main__":
	main()
