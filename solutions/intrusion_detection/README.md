# Intrusion Detection System (IDS)

Classifies network connections as **normal** or **attack** using a Random Forest classifier trained on 19 standard network-traffic features (inspired by the KDD Cup 1999 dataset schema).

## Feature Vector (19 features)

| Index | Feature | Description |
|---|---|---|
| 0 | `duration` | Connection duration (seconds) |
| 1 | `protocol_type` | Encoded protocol (0=tcp, 1=udp, 2=icmp) |
| 2 | `service` | Encoded service category (0–9) |
| 3 | `flag` | Encoded connection flag (0–10) |
| 4 | `src_bytes` | Bytes from source to destination |
| 5 | `dst_bytes` | Bytes from destination to source |
| 6 | `land` | 1 if same host/port for src and dst |
| 7 | `wrong_fragment` | Number of wrong fragments |
| 8 | `urgent` | Number of urgent packets |
| 9 | `hot` | Number of "hot" indicators |
| 10 | `num_failed_logins` | Number of failed login attempts |
| 11 | `logged_in` | 1 if successfully logged in |
| 12 | `num_compromised` | Number of compromised conditions |
| 13 | `root_shell` | 1 if root shell obtained |
| 14 | `su_attempted` | 1 if su root attempted |
| 15 | `num_root` | Number of root accesses |
| 16 | `num_file_creations` | Number of file-creation operations |
| 17 | `count` | Connections to same host in last 2 s |
| 18 | `srv_count` | Connections to same service in last 2 s |

## Quick Start

```python
import numpy as np
from solutions.intrusion_detection import IDS

# Create synthetic training data
rng = np.random.default_rng(0)
X_normal = rng.normal(loc=0,   scale=1, size=(200, 19))
X_attack = rng.normal(loc=3,   scale=1, size=(100, 19))
X_train  = np.vstack([X_normal, X_attack])
y_train  = np.array([0] * 200 + [1] * 100)

ids = IDS()
ids.train(X_train, y_train)

# Classify a new connection
sample = rng.normal(loc=3, scale=1, size=(19,))
result = ids.predict(sample)
print(result)
# {'label': 'attack', 'confidence': 0.91, 'attack_probability': 0.91}
```
