"""Load phishing URL datasets into a shared feature representation."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from datasets import load_dataset
from ucimlrepo import fetch_ucirepo

from feature_extraction import extract_url_features


FEATURE_COLUMNS = [
	"url_length",
	"special_character_count",
	"is_raw_ip",
	"subdomain_count",
	"is_https",
	"suspicious_keyword_count",
	"domain_length",
	"domain_digit_count",
	"digit_ratio",
]
OUTPUT_COLUMNS = FEATURE_COLUMNS + ["label"]


def _normalise_column_name(column_name: Any) -> str:
	"""Make column names comparable without changing their displayed form."""
	return re.sub(r"[^a-z0-9]", "", str(column_name).lower())


def _find_column(columns: list[Any], candidates: set[str], description: str) -> Any:
	matches = [
		column
		for column in columns
		if _normalise_column_name(column) in candidates
	]
	if len(matches) != 1:
		raise ValueError(
			f"Could not identify exactly one {description} column. "
			f"Candidates found: {matches}; available columns: {columns}"
		)
	return matches[0]


def _normalise_phiusiil_labels(labels: pd.Series) -> pd.Series:
	"""Convert UCI's documented labels: 1 legitimate, 0 phishing."""
	numeric_labels = pd.to_numeric(labels, errors="raise")
	unexpected_labels = set(numeric_labels.dropna().unique()) - {0, 1}
	if unexpected_labels:
		raise ValueError(f"Unexpected PhiUSIIL labels: {unexpected_labels}")
	return (numeric_labels == 0).astype(int)


def _normalise_phreshphish_labels(labels: pd.Series) -> pd.Series:
	"""Convert PhreshPhish's documented benign/phish classes to 0/1."""
	if pd.api.types.is_numeric_dtype(labels):
		numeric_labels = pd.to_numeric(labels, errors="raise")
		if not set(numeric_labels.dropna().unique()).issubset({0, 1}):
			raise ValueError("PhreshPhish numeric labels must contain only 0 and 1")
		return numeric_labels.astype(int)

	normalized_labels = labels.astype(str).str.strip().str.lower()
	phishing_values = {"phish", "phishing", "malicious", "1"}
	legitimate_values = {"benign", "legitimate", "legit", "0"}
	unknown_values = set(normalized_labels.unique()) - phishing_values - legitimate_values
	if unknown_values:
		raise ValueError(f"Unexpected PhreshPhish labels: {unknown_values}")
	return normalized_labels.isin(phishing_values).astype(int)


def _build_feature_dataframe(
	source_data: pd.DataFrame,
	url_column: Any,
	labels: pd.Series,
) -> pd.DataFrame:
	urls = source_data[url_column]
	if urls.isna().any():
		raise ValueError(f"URL column {url_column!r} contains missing values")
	if labels.isna().any():
		raise ValueError("Label column contains missing values")

	feature_rows = [extract_url_features(url) for url in urls]
	feature_frame = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
	feature_frame["label"] = labels.to_numpy()
	return feature_frame.loc[:, OUTPUT_COLUMNS]


def _print_class_balance(dataset_name: str, data: pd.DataFrame) -> None:
	row_count = len(data)
	counts = data["label"].value_counts().reindex([1, 0], fill_value=0)
	print(f"{dataset_name}: {row_count:,} rows")
	for label, class_name in ((1, "phishing"), (0, "legitimate")):
		count = int(counts[label])
		percentage = count / row_count * 100 if row_count else 0.0
		print(f"  {class_name}: {count:,} ({percentage:.2f}%)")


def load_phiusiil() -> pd.DataFrame:
	"""Load UCI PhiUSIIL and map its labels to phishing=1, legitimate=0."""
	dataset = fetch_ucirepo(id=967)
	source_data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
	print("PhiUSIIL raw columns:", list(source_data.columns))

	url_column = _find_column(
		list(source_data.columns), {"url"}, "PhiUSIIL URL"
	)
	label_column = _find_column(
		list(source_data.columns), {"label"}, "PhiUSIIL label"
	)
	labels = _normalise_phiusiil_labels(source_data[label_column])
	result = _build_feature_dataframe(source_data, url_column, labels)
	_print_class_balance("PhiUSIIL", result)
	return result


def load_phiusiil_url_labels() -> pd.DataFrame:
	"""Load PhiUSIIL's original URLs with labels mapped to phishing=1."""
	dataset = fetch_ucirepo(id=967)
	source_data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
	url_column = _find_column(
		list(source_data.columns), {"url"}, "PhiUSIIL URL"
	)
	label_column = _find_column(
		list(source_data.columns), {"label"}, "PhiUSIIL label"
	)
	result = pd.DataFrame(
		{
			"url": source_data[url_column].astype(str),
			"label": _normalise_phiusiil_labels(source_data[label_column]),
		}
	)
	return result


def load_phreshphish() -> pd.DataFrame:
	"""Load PhreshPhish training data and map phishing=1, legitimate=0."""
	dataset = load_dataset("phreshphish/phreshphish", split="train")
	source_data = dataset.to_pandas()
	print("PhreshPhish raw columns:", list(source_data.columns))

	url_column = _find_column(
		list(source_data.columns), {"url", "uri", "link"}, "PhreshPhish URL"
	)
	label_column = _find_column(
		list(source_data.columns), {"label", "class", "target"}, "PhreshPhish label"
	)
	labels = _normalise_phreshphish_labels(source_data[label_column])
	result = _build_feature_dataframe(source_data, url_column, labels)
	_print_class_balance("PhreshPhish", result)
	return result


if __name__ == "__main__":
	phiusiil_data = load_phiusiil()
	phreshphish_data = load_phreshphish()

	print("PhiUSIIL shape:", phiusiil_data.shape)
	print(phiusiil_data.head())
	print("PhreshPhish shape:", phreshphish_data.shape)
	print(phreshphish_data.head())

	assert list(phiusiil_data.columns) == list(phreshphish_data.columns), (
		"Dataset feature columns do not match: "
		f"PhiUSIIL={list(phiusiil_data.columns)}, "
		f"PhreshPhish={list(phreshphish_data.columns)}"
	)
