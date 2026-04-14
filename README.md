# AI Security Solutions

A collection of machine-learning-based security tools built with Python and scikit-learn.

## Solutions

| Solution | Description |
|---|---|
| [Phishing URL Detector](solutions/phishing_detector/) | Classifies URLs as phishing or legitimate using extracted URL features |
| [Intrusion Detection System](solutions/intrusion_detection/) | Detects malicious network traffic using a Random Forest classifier |
| [Anomaly Detection](solutions/anomaly_detection/) | Identifies anomalous data points in streams using Isolation Forest |
| [Password Strength Analyzer](solutions/password_analyzer/) | Scores password strength using rule-based and ML heuristics |

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
# Clone the repository
git clone https://github.com/Tharunaditya/AI-Security-Solutions.git
cd AI-Security-Solutions

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the tests

```bash
pytest tests/ -v
```

## Solution Overviews

### Phishing URL Detector

Extracts structural features from a URL (length, digit ratio, special-character counts, suspicious TLDs, etc.) and feeds them into a Random Forest classifier trained on labelled examples. Call `PhishingDetector.predict(url)` to get a `"phishing"` / `"legitimate"` label together with a confidence score.

### Intrusion Detection System

Accepts a vector of network-traffic features (packet sizes, protocol flags, connection counts, …) and classifies each connection as `"normal"` or `"attack"`. The underlying Random Forest model is trained via `IDS.train(X, y)` and persisted in memory for repeated inference.

### Anomaly Detection

Wraps scikit-learn's `IsolationForest` to provide a simple `AnomalyDetector.fit(X)` / `AnomalyDetector.predict(X)` interface. Returns `True` for anomalous samples and includes an anomaly score for ranking.

### Password Strength Analyzer

Combines classic rule checks (length, character classes, common passwords) with an ML-derived complexity score to produce a 0-100 strength score and a human-readable label (`Weak`, `Fair`, `Strong`, `Very Strong`).

## Project Structure

```
AI-Security-Solutions/
├── requirements.txt
├── solutions/
│   ├── phishing_detector/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── README.md
│   ├── intrusion_detection/
│   │   ├── __init__.py
│   │   ├── ids.py
│   │   └── README.md
│   ├── anomaly_detection/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── README.md
│   └── password_analyzer/
│       ├── __init__.py
│       ├── analyzer.py
│       └── README.md
└── tests/
    ├── test_phishing_detector.py
    ├── test_intrusion_detection.py
    ├── test_anomaly_detection.py
    └── test_password_analyzer.py
```

## Contributing

Pull requests are welcome. Please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)
