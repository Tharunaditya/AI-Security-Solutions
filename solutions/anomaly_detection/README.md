# Anomaly Detector

Detects anomalous data points in numeric feature vectors using scikit-learn's **Isolation Forest** algorithm.

## How It Works

An Isolation Forest isolates anomalies by randomly partitioning data with axis-aligned splits. Anomalous points are easier to isolate (they require fewer splits) and therefore receive lower decision-function scores. Normal points cluster together and are harder to isolate.

## Quick Start

```python
import numpy as np
from solutions.anomaly_detection import AnomalyDetector

rng = np.random.default_rng(0)

# Training data (predominantly normal)
X_train = rng.normal(loc=0, scale=1, size=(500, 5))

detector = AnomalyDetector(contamination=0.05)
detector.fit(X_train)

# Score a batch
X_test = np.vstack([
    rng.normal(loc=0,  scale=1, size=(10, 5)),  # normal
    rng.normal(loc=10, scale=1, size=(3,  5)),  # anomalies
])

results = detector.predict(X_test)
for r in results:
    print(r)
# {'is_anomaly': False, 'anomaly_score': 0.08}
# {'is_anomaly': True,  'anomaly_score': -0.21}
# ...

# Score a single sample
result = detector.predict_one([10, 10, 10, 10, 10])
print(result)
# {'is_anomaly': True, 'anomaly_score': -0.3}
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `contamination` | `0.05` | Expected proportion of anomalies |
| `n_estimators` | `100` | Number of trees in the forest |
| `random_state` | `42` | Random seed for reproducibility |
