"""URL feature extraction for phishing detection experiments."""

import re
from urllib.parse import urlparse


SUSPICIOUS_KEYWORDS = [
    "login",
    "verify",
    "secure",
    "update",
    "account",
    "banking",
    "confirm",
    "signin",
]


def extract_url_features(url: str) -> dict:
    """Extract phishing-related features from a URL."""
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ""
    normalized_url = url.lower()

    special_characters = "@-_ %=?&".replace(" ", "")
    special_character_count = sum(url.count(character) for character in special_characters)

    ipv4_pattern = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
    is_raw_ip = int(bool(ipv4_pattern.fullmatch(hostname)))

    subdomain_count = max(hostname.count(".") - 1, 0)
    is_https = int(parsed_url.scheme.lower() == "https")
    url_tokens = re.findall(r"[a-z]+", normalized_url)
    suspicious_keyword_count = sum(
        token in SUSPICIOUS_KEYWORDS for token in url_tokens
    )

    url_length = len(url)
    domain_digit_count = sum(character.isdigit() for character in hostname)

    return {
        "url_length": url_length,
        "special_character_count": special_character_count,
        "is_raw_ip": is_raw_ip,
        "subdomain_count": subdomain_count,
        "is_https": is_https,
        "suspicious_keyword_count": suspicious_keyword_count,
        "domain_length": len(hostname),
        "domain_digit_count": domain_digit_count,
        "digit_ratio": domain_digit_count / url_length if url_length else 0.0,
    }


if __name__ == "__main__":
    example_urls = [
        "https://www.google.com",
        "http://192.168.1.1/secure-login-update",
        "http://mail.accounts.example.co.uk/verify",
    ]

    for example_url in example_urls:
        print(example_url)
        print(extract_url_features(example_url))
