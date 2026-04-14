"""Tests for the Password Strength Analyzer."""

import pytest
from solutions.password_analyzer import PasswordAnalyzer


@pytest.fixture
def analyzer():
    return PasswordAnalyzer()


# ---------------------------------------------------------------------------
# Result structure tests
# ---------------------------------------------------------------------------

class TestPasswordAnalyzerStructure:
    def test_returns_dict(self, analyzer):
        result = analyzer.analyze("TestPass1!")
        assert isinstance(result, dict)

    def test_expected_keys(self, analyzer):
        result = analyzer.analyze("TestPass1!")
        assert set(result.keys()) == {"score", "label", "feedback", "checks"}

    def test_score_is_int(self, analyzer):
        result = analyzer.analyze("TestPass1!")
        assert isinstance(result["score"], int)

    def test_score_in_range(self, analyzer):
        for pwd in ["a", "password", "TestPass1!", "Tr0ub4dor&3_long_passphrase!"]:
            result = analyzer.analyze(pwd)
            assert 0 <= result["score"] <= 100, f"Score out of range for '{pwd}'"

    def test_label_is_valid(self, analyzer):
        valid_labels = {"Weak", "Fair", "Strong", "Very Strong"}
        for pwd in ["a", "password", "TestPass1!", "Tr0ub4dor&3_long_passphrase!"]:
            assert analyzer.analyze(pwd)["label"] in valid_labels

    def test_feedback_is_list(self, analyzer):
        result = analyzer.analyze("TestPass1!")
        assert isinstance(result["feedback"], list)

    def test_checks_is_dict(self, analyzer):
        result = analyzer.analyze("TestPass1!")
        assert isinstance(result["checks"], dict)

    def test_checks_expected_keys(self, analyzer):
        result = analyzer.analyze("TestPass1!")
        expected = {
            "length", "uppercase", "lowercase", "digits",
            "special_chars", "not_common", "no_keyboard_walk",
            "no_repeated_chars",
        }
        assert set(result["checks"].keys()) == expected


# ---------------------------------------------------------------------------
# Strength ranking tests (stronger password → higher score)
# ---------------------------------------------------------------------------

class TestPasswordStrengthRanking:
    def test_weak_password(self, analyzer):
        result = analyzer.analyze("password")
        assert result["label"] == "Weak"
        assert result["score"] < 40

    def test_common_password_is_weak(self, analyzer):
        result = analyzer.analyze("123456")
        assert result["label"] == "Weak"

    def test_strong_password_scores_higher_than_weak(self, analyzer):
        weak_score = analyzer.analyze("password")["score"]
        strong_score = analyzer.analyze("P@ssw0rd!Xyz42")["score"]
        assert strong_score > weak_score

    def test_very_long_complex_password_is_very_strong(self, analyzer):
        result = analyzer.analyze("Tr0ub4dor&3_V3ryL0ng_P@ssphrase!")
        assert result["label"] == "Very Strong"
        assert result["score"] >= 80

    def test_short_password_scores_low(self, analyzer):
        result = analyzer.analyze("Ab1!")
        assert result["score"] < 40


# ---------------------------------------------------------------------------
# Check flag tests
# ---------------------------------------------------------------------------

class TestPasswordChecks:
    def test_length_check_true_when_long_enough(self, analyzer):
        assert analyzer.analyze("LongPassw0rd!")["checks"]["length"] is True

    def test_length_check_false_when_short(self, analyzer):
        assert analyzer.analyze("Ab1!")["checks"]["length"] is False

    def test_uppercase_check(self, analyzer):
        assert analyzer.analyze("ALLCAPS1!")["checks"]["uppercase"] is True
        assert analyzer.analyze("nouppercase1!")["checks"]["uppercase"] is False

    def test_lowercase_check(self, analyzer):
        assert analyzer.analyze("alllower1!")["checks"]["lowercase"] is True
        assert analyzer.analyze("NOLOWER1!")["checks"]["lowercase"] is False

    def test_digits_check(self, analyzer):
        assert analyzer.analyze("NoDigits!")["checks"]["digits"] is False
        assert analyzer.analyze("HasDigit1!")["checks"]["digits"] is True

    def test_special_chars_check(self, analyzer):
        assert analyzer.analyze("NoSpecial1A")["checks"]["special_chars"] is False
        assert analyzer.analyze("HasSpecial1!")["checks"]["special_chars"] is True

    def test_not_common_check_for_common_password(self, analyzer):
        assert analyzer.analyze("password")["checks"]["not_common"] is False

    def test_not_common_check_for_uncommon_password(self, analyzer):
        assert analyzer.analyze("Xk7#mNpQ!")["checks"]["not_common"] is True

    def test_no_keyboard_walk_fails_for_qwerty(self, analyzer):
        assert analyzer.analyze("qwerty12!")["checks"]["no_keyboard_walk"] is False

    def test_no_repeated_chars_fails(self, analyzer):
        assert analyzer.analyze("aaabbbCCC1!")["checks"]["no_repeated_chars"] is False

    def test_no_repeated_chars_passes(self, analyzer):
        assert analyzer.analyze("AbCdEfGh1!")["checks"]["no_repeated_chars"] is True


# ---------------------------------------------------------------------------
# Feedback tests
# ---------------------------------------------------------------------------

class TestPasswordFeedback:
    def test_feedback_empty_for_strong_password(self, analyzer):
        result = analyzer.analyze("Tr0ub4dor&3_V3ryL0ng_P@ssphrase!")
        assert result["feedback"] == []

    def test_feedback_suggests_special_chars(self, analyzer):
        result = analyzer.analyze("NoSpecialChar1A")
        tips = result["feedback"]
        assert any("special" in tip.lower() for tip in tips)

    def test_feedback_suggests_length(self, analyzer):
        result = analyzer.analyze("Ab1!")
        tips = result["feedback"]
        assert any("character" in tip.lower() for tip in tips)

    def test_feedback_warns_about_common_password(self, analyzer):
        result = analyzer.analyze("password")
        tips = result["feedback"]
        assert any("common" in tip.lower() for tip in tips)
