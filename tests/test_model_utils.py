import pandas as pd

from model_utils import train_and_evaluate


def test_decision_tree_returns_metrics():
    df = pd.DataFrame({
        "hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "practice": ["no", "no", "no", "no", "yes", "yes", "yes", "yes", "yes", "yes"],
        "passed": ["no", "no", "no", "no", "yes", "yes", "yes", "yes", "yes", "yes"],
    })
    result = train_and_evaluate(df, "passed", ["hours", "practice"], "Decision Tree", 0.2)
    assert 0 <= result["train_accuracy"] <= 1
    assert 0 <= result["test_accuracy"] <= 1
    assert 0 <= result["baseline_accuracy"] <= 1
    assert result["confusion_matrix"].shape[0] == result["confusion_matrix"].shape[1]

def test_invalid_single_class_target_raises():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "target": ["yes"] * 6})
    try:
        train_and_evaluate(df, "target", ["x"], "Decision Tree")
    except ValueError as exc:
        assert "at least two classes" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a single-class target")
