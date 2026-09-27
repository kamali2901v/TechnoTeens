"""
Grading engine — kept separate from the AI model so official thresholds
can be updated without retraining. Thresholds are currently PLACEHOLDERS
pending sourcing from verified AGMARK/NAFED onion grading standards.
"""

import json


def load_rules(path="grading_rules.json"):
    with open(path, "r") as f:
        return json.load(f)


def confidence_level(confidence, rules):
    """Returns 'High' / 'Medium' / 'Low' per the configured thresholds."""
    high = rules["confidence_thresholds"]["high"]
    low = rules["confidence_thresholds"]["low"]
    if confidence >= high:
        return "High"
    if confidence >= low:
        return "Medium"
    return "Low"


def apply_grading(prediction, confidence, rules):
    level = confidence_level(confidence, rules)
    grades = rules["grades"]

    if level == "Low":
        grade = grades["uncertain"]
        review_needed = True
    elif prediction == "healthy":
        if level == "High":
            grade = grades["healthy_high"]
            review_needed = False
        else:
            grade = grades["healthy_review"]
            review_needed = True
    else:  # damaged or rotten
        if level == "High":
            grade = grades["defective_high"]
            review_needed = False
        else:
            grade = grades["defective_review"]
            review_needed = True

    return {
        "grade": grade,
        "confidence_level": level,
        "manual_review_required": review_needed,
    }