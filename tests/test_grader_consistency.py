import json

from server.graders import grade_submission


def test_grade_submission_is_deterministic_and_bounded():
    db = json.load(open("/home/runner/work/Legaloon/Legaloon/server/invoice_db.json", encoding="utf-8"))
    gt = db[0]["ground_truth"]
    params = {
        "tds_amount_inr": float(gt["tds_amount_inr"]),
        "section": str(gt["section"]),
        "rate_percent": float(gt["tds_rate_percent"]),
    }
    first = grade_submission(params, gt, task_id="task_easy")
    second = grade_submission(params, gt, task_id="task_easy")
    assert first == second
    assert 0.0 <= first["score"] <= 1.0
