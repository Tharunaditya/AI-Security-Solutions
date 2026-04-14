# Phishing URL Detector

Classifies URLs as **phishing** or **legitimate** using a Random Forest classifier trained on structural and lexical URL features.

## Features Extracted

| Feature | Description |
|---|---|
| `url_length` | Total length of the URL |
| `hostname_length` | Length of the hostname part |
| `path_length` | Length of the path |
| `dot_count` | Number of dots in the URL |
| `hyphen_count` | Number of hyphens |
| `at_count` | Number of `@` symbols |
| `digit_ratio` | Ratio of digits to total characters |
| `subdomain_depth` | Number of subdomain levels |
| `has_ip` | 1 if hostname is an IP address |
| `is_https` | 1 if scheme is HTTPS |
| `suspicious_tld` | 1 if TLD is commonly abused |
| `suspicious_keyword_count` | Count of suspicious keywords (login, verify, etc.) |
| `hostname_entropy` | Shannon entropy of the hostname |
| `hex_encoded_count` | Number of hex-encoded characters |
| `double_slash` | 1 if `//` appears in the path |

## Quick Start

```python
from solutions.phishing_detector import PhishingDetector

# Prepare training data
urls = [
    "https://www.google.com",
    "https://secure-login.paypal.com.malicious.tk/verify?account=123",
    "https://github.com",
    "http://192.168.1.1/login.php?user=admin&pass=1234",
]
labels = [0, 1, 0, 1]  # 0 = legitimate, 1 = phishing

detector = PhishingDetector()
detector.train(urls, labels)

result = detector.predict("http://paypal-secure.verify.tk/login")
print(result)
# {'label': 'phishing', 'confidence': 0.9, 'features': {...}}
```
