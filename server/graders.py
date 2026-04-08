"""
Explicit grader functions for LegaLoom-Env.

Each grader takes the submitted answer and ground truth,
and returns a normalised score STRICTLY in (0.0, 1.0) exclusive.
The hackathon validator requires: score > 0.0 AND score < 1.0.
These functions are deterministic — same inputs always give same output.
"""

from typing import Dict, Any

AMOUNT_TOLERANCE_INR = 1.0

# Strict bounds — never touch 0.0 or 1.0
_SCORE_MIN = 0.05   # minimum meaningful score (not zero)
_SCORE_MAX = 0.95   # maximum score (not one)


def _clamp(score: float) -> float:
    """Clamp to strictly open interval (0.0, 1.0)."""
    return round(min(max(float(score), _SCORE_MIN), _SCORE_MAX), 4)


def grade_submission(
    params: Dict[str, Any],
    ground_truth: Dict[str, Any],
    task_id: str = "task_easy",
) -> Dict[str, Any]:
    """
    Primary grader. Weights adapt per task_id.
    Always returns score strictly in (_SCORE_MIN, _SCORE_MAX).
    """
    submitted_amount  = float(params.get("tds_amount_inr", -1))
    submitted_section = str(params.get("section", "")).strip().upper()
    submitted_rate    = float(params.get("rate_percent", -1))
    no_tds_flag       = str(params.get("no_tds", "")).lower() == "true"

    feedback  = []
    breakdown = {}
    score     = 0.0

    # Case 1: No TDS applicable
    if not ground_truth["tds_applicable"]:
        correct = (submitted_amount == 0.0 or no_tds_flag)
        # 0.85 for correct, 0.10 for wrong — both strictly inside (0, 1)
        raw_score = 0.85 if correct else 0.10
        feedback.append(
            "Correctly identified: no TDS applicable (below threshold)."
            if correct else
            f"No TDS required (below threshold). Submitted INR {submitted_amount:,.2f} — should be 0."
        )
        breakdown["no_tds_correct"] = correct
        return {
            "score": _clamp(raw_score),
            "correct": correct,
            "feedback": feedback,
            "breakdown": breakdown,
        }

    goods = ground_truth.get("goods_amount", 0.0)
    is_inop_pan = not ground_truth["pan_valid"]

    # Task-specific weights that sum to 1.0
    if task_id == "task_easy":
        W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = 0.0, 0.30, 0.30, 0.0, 0.40
    elif task_id == "task_medium":
        if goods > 0:
            W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = 0.0, 0.25, 0.15, 0.20, 0.40
        else:
            W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = 0.0, 0.25, 0.15, 0.0, 0.60
    elif task_id == "task_expert":
        W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = 0.0, 0.40, 0.25, 0.0, 0.35
    else:  # task_hard
        if is_inop_pan:
            if goods > 0:
                W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = 0.35, 0.10, 0.15, 0.10, 0.30
            else:
                W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = 0.35, 0.10, 0.15, 0.0, 0.40
        else:
            if goods > 0:
                W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = 0.0, 0.20, 0.15, 0.20, 0.45
            else:
                W_PAN, W_SECT, W_RATE, W_GOODS, W_AMOUNT = 0.0, 0.20, 0.15, 0.0, 0.65

    # Normalise weights to sum=1.0
    total_w = W_PAN + W_SECT + W_RATE + W_GOODS + W_AMOUNT
    if total_w > 0:
        W_PAN /= total_w
        W_SECT /= total_w
        W_RATE /= total_w
        W_GOODS /= total_w
        W_AMOUNT /= total_w

    # Case 2: Inoperative PAN
    if is_inop_pan:
        pan_ok = abs(submitted_rate - 20.0) < 0.01
        breakdown["pan_inoperative_detected"] = pan_ok
        if pan_ok:
            score += W_PAN
            feedback.append("Inoperative PAN detected — 20% rate applied.")
        else:
            feedback.append(
                f"PAN is INOPERATIVE. Must apply 20% (Section 206AA). "
                f"Submitted {submitted_rate}%."
            )

    # Case 3: Section correct
    expected_section = ground_truth["section"]
    split = expected_section in ("SPLIT", "SPLIT_194J_194I")
    section_ok = (
        submitted_section == expected_section or
        (split and submitted_section in ("194J", "194I", "194C"))
    )
    breakdown["section_correct"] = section_ok
    if section_ok:
        score += W_SECT
        feedback.append(f"Section {submitted_section} is correct.")
    else:
        feedback.append(
            f"Section wrong: submitted {submitted_section}, "
            f"expected {expected_section}."
        )

    # Case 4: Rate correct
    rate_ok = abs(submitted_rate - ground_truth["tds_rate_percent"]) < 0.01
    breakdown["rate_correct"] = rate_ok
    if rate_ok:
        score += W_RATE
        feedback.append(f"Rate {submitted_rate}% is correct.")
    else:
        feedback.append(
            f"Rate wrong: submitted {submitted_rate}%, "
            f"expected {ground_truth['tds_rate_percent']}%."
        )

    # Case 5: Goods exclusion
    if goods > 0:
        goods_ok = submitted_amount <= ground_truth["taxable_amount"] + AMOUNT_TOLERANCE_INR
        breakdown["goods_excluded"] = goods_ok
        if goods_ok:
            score += W_GOODS
            feedback.append(
                f"Goods (INR {goods:,.0f}) correctly excluded from TDS base."
            )
        else:
            feedback.append(
                f"Goods (INR {goods:,.0f}) must be excluded — TDS on services only."
            )

    # Case 6: Final amount
    amount_ok = abs(submitted_amount - ground_truth["tds_amount_inr"]) <= AMOUNT_TOLERANCE_INR
    breakdown["amount_correct"] = amount_ok
    if amount_ok:
        score += W_AMOUNT
        feedback.append(
            f"TDS amount INR {submitted_amount:,.2f} is CORRECT "
            f"(expected INR {ground_truth['tds_amount_inr']:,.2f})."
        )
    else:
        feedback.append(
            f"TDS amount wrong: submitted INR {submitted_amount:,.2f}, "
            f"correct is INR {ground_truth['tds_amount_inr']:,.2f}."
        )

    correct = amount_ok and (section_ok or is_inop_pan)

    # Scale raw score (0.0–1.0) into strictly open interval (_SCORE_MIN, _SCORE_MAX)
    # Perfect score maps to _SCORE_MAX; zero score maps to _SCORE_MIN.
    scaled = _SCORE_MIN + score * (_SCORE_MAX - _SCORE_MIN)
    return {
        "score": _clamp(scaled),
        "correct": correct,
        "feedback": feedback,
        "breakdown": breakdown,
    }


def grade_easy(params: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grader for task_easy — weights: section=0.30, rate=0.30, amount=0.40."""
    return grade_submission(params, ground_truth, task_id="task_easy")["score"]


def grade_medium(params: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grader for task_medium — adds goods-exclusion weight."""
    return grade_submission(params, ground_truth, task_id="task_medium")["score"]


def grade_hard(params: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grader for task_hard — inop-PAN detection heavily weighted (0.35)."""
    return grade_submission(params, ground_truth, task_id="task_hard")["score"]


def grade_expert(params: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """Grader for task_expert — section heavily weighted (0.40): 194T/194Q identification."""
    return grade_submission(params, ground_truth, task_id="task_expert")["score"]


# Map task_id -> grader function
GRADERS = {
    "task_easy":   grade_easy,
    "task_medium": grade_medium,
    "task_hard":   grade_hard,
    "task_expert": grade_expert,
}
