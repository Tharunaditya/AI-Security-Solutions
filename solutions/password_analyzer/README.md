# Password Strength Analyzer

Scores a password on a **0–100 scale** and classifies it as *Weak*, *Fair*, *Strong*, or *Very Strong* using rule-based checks and Shannon entropy.

## Strength Labels

| Label | Score Range |
|---|---|
| Weak | 0–39 |
| Fair | 40–59 |
| Strong | 60–79 |
| Very Strong | 80–100 |

## Checks Performed

| Check | Description |
|---|---|
| `length` | At least 8 characters |
| `uppercase` | Contains uppercase letters |
| `lowercase` | Contains lowercase letters |
| `digits` | Contains digits |
| `special_chars` | Contains special characters |
| `not_common` | Not in the top-50 common passwords list |
| `no_keyboard_walk` | Does not contain keyboard walk patterns |
| `no_repeated_chars` | No character repeated 3+ times in a row |

## Quick Start

```python
from solutions.password_analyzer import PasswordAnalyzer

analyzer = PasswordAnalyzer()

result = analyzer.analyze("P@ssw0rd!")
print(result)
# {
#     'score': 68,
#     'label': 'Strong',
#     'feedback': ['Consider making it longer (12+ characters).'],
#     'checks': {
#         'length': True, 'uppercase': True, 'lowercase': True,
#         'digits': True, 'special_chars': True, 'not_common': True,
#         'no_keyboard_walk': True, 'no_repeated_chars': True
#     }
# }

result = analyzer.analyze("password")
print(result['label'])  # 'Weak'
print(result['feedback'])
# ['Consider making it longer (12+ characters).',
#  'Include at least one special character (e.g. !@#$%).',
#  'Avoid commonly used passwords.']
```
