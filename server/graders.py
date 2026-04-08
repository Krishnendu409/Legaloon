"""
Explicit grader functions for LegaLoom-Env.

Each grader takes the submitted answer and ground truth,
and returns a normalised score STRICTLY in (0.0, 1.0) exclusive.
The hackathon validator requires: score > 0.0 AND score < 1.0.
These functions are deterministic — same inputs always give same output.
"""

from typing import Dict, Any

AMOUNT_TOLERANCE_INR = 1.0
GST_BUNDLED_INDICATORS = (
    "inclusive of all taxes",
    "gst included in invoice value",
    "gst bundled",
    "gst not shown separately",
)
PENALTY_SECTION_RATE_MISMATCH = 0.10
PENALTY_INOP_PAN_MISSED = 0.10

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
        correct = (submitted_amount == 0.0 and no_tds_flag)
        # Keep non-zero floor per existing score-range contract, while strongly penalizing wrong no-TDS submissions.
        raw_score = 0.85 if correct else 0.05
        feedback.append(
            "Correctly identified: no TDS applicable (below threshold)."
            if correct else
            f"No TDS required (below threshold). Submitted INR {submitted_amount:,.2f} — should be 0."
        )
        breakdown["no_tds_correct"] = correct
        breakdown["gst_base_correct"] = False
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
    split_allowed_sections = {
        "SPLIT": {"194J", "194C"},
        "SPLIT_194J_194I": {"194J", "194I"},
    }
    section_ok = (
        submitted_section == expected_section or
        (split and submitted_section in split_allowed_sections.get(expected_section, set()))
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
        goods_ok = False
        if submitted_rate > 0:
            implied_taxable_base = (submitted_amount * 100.0) / submitted_rate
            expected_taxable_base = float(ground_truth.get("taxable_amount", 0.0))
            base_tolerance = max(AMOUNT_TOLERANCE_INR, expected_taxable_base * 0.02)
            goods_ok = abs(implied_taxable_base - expected_taxable_base) <= base_tolerance
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

    # Case 5b: GST-bundled base correctness (if applicable)
    note_text = str(ground_truth.get("note", "")).lower()
    gst_bundled_case = any(
        token in note_text for token in GST_BUNDLED_INDICATORS
    )
    gst_base_ok = False
    if gst_bundled_case and submitted_rate > 0:
        implied_taxable_base = (submitted_amount * 100.0) / submitted_rate
        expected_taxable_base = float(ground_truth.get("taxable_amount", 0.0))
        base_tolerance = max(AMOUNT_TOLERANCE_INR, expected_taxable_base * 0.02)
        gst_base_ok = abs(implied_taxable_base - expected_taxable_base) <= base_tolerance
        if gst_base_ok:
            feedback.append("GST-bundled base handled correctly (TDS on full amount).")
        else:
            feedback.append("GST-bundled case: taxable base appears incorrect.")
    breakdown["gst_base_correct"] = gst_base_ok

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

    no_tds_invalid_for_applicable = ground_truth.get("tds_applicable", True) and no_tds_flag
    if no_tds_invalid_for_applicable:
        score = max(0.0, score - 0.25)
        feedback.append(
            "Invalid no_tds flag for a TDS-applicable invoice. no_tds must be false."
        )
    breakdown["no_tds_invalid_for_applicable"] = no_tds_invalid_for_applicable

    if not section_ok and not rate_ok:
        score = max(0.0, score - PENALTY_SECTION_RATE_MISMATCH)
    if is_inop_pan and not breakdown.get("pan_inoperative_detected", False):
        score = max(0.0, score - PENALTY_INOP_PAN_MISSED)

    correct = (
        amount_ok
        and section_ok
        and rate_ok
        and (not is_inop_pan or breakdown.get("pan_inoperative_detected", False))
        and not no_tds_invalid_for_applicable
    )

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
