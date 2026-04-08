"""
tasks.py — LegaLoom-Env Task Definitions (Upgraded)

Four tasks of increasing difficulty, each drawing from the
260-invoice database. Every reset() picks a random invoice
from the appropriate difficulty pool.

Task 1 — Easy   : Single service, valid PAN, above threshold
Task 2 — Medium : Mixed invoice (goods + service) or threshold boundary
Task 3 — Hard   : Inoperative PAN or GST-bundled base
Task 4 — Expert : Multiple simultaneous traps
"""

import json
import os
import random
from typing import Optional

DEFAULT_TASK_SEED = 42

# ---------------------------------------------------------------------------
# Load invoice database
# ---------------------------------------------------------------------------
_DB_PATH = os.path.join(os.path.dirname(__file__), "invoice_db.json")

def _load_db() -> list:
    with open(_DB_PATH, encoding="utf-8") as f:
        return json.load(f)

_INVOICE_DB: Optional[list] = None

def get_db() -> list:
    global _INVOICE_DB
    if _INVOICE_DB is None:
        _INVOICE_DB = _load_db()
    return _INVOICE_DB


# ---------------------------------------------------------------------------
# Difficulty pools — which categories go into which task
# ---------------------------------------------------------------------------

DIFFICULTY_POOLS = {
    "task_easy": {
        "difficulty": "easy",
        "description": "Single invoice, clear service description, valid operative PAN, amount above threshold",
        "max_steps": 6,
        "categories": [
            "194J_professional",
            "194C_contractor",
            "194I_rent",
            "194H_commission",
        ],
        "hint_enabled": True,
    },
    "task_medium": {
        "difficulty": "medium",
        "description": (
            "Mixed invoice with goods and service line items, or threshold boundary case. "
            "Agent must split correctly and track cumulative YTD payments."
        ),
        "max_steps": 8,
        "categories": [
            "mixed_invoice",
            "threshold_boundary",
            "194J_technical",
            "194I_machinery",
        ],
        "hint_enabled": True,
    },
    "task_hard": {
        "difficulty": "hard",
        "description": (
            "Inoperative PAN (20% override) or GST-bundled invoice (TDS on full amount). "
            "Agent must check PAN status before computing any rate."
        ),
        "max_steps": 8,
        "categories": [
            "inoperative_pan",
            "gst_bundled_tds_base",
            "below_threshold_new_limits",
        ],
        "hint_enabled": False,   # No hints on hard — agent must reason independently
    },
    "task_expert": {
        "difficulty": "expert",
        "description": (
            "Multiple simultaneous traps: split invoices + inoperative PAN, "
            "or new 194T section + threshold crossing, or high-value 194Q goods. "
            "Genuinely challenges frontier models."
        ),
        "max_steps": 10,
        "categories": [
            "194T_partner",
            "194T_partner_extra",
            "194Q_goods",
        ],
        "hint_enabled": False,
    },
}

TASK_ORDER = ["task_easy", "task_medium", "task_hard", "task_expert"]


def _build_scenario_noise(task_id: str, chosen: dict, seed: Optional[int]) -> dict:
    """
    Deterministic scenario-noise generator for realism/novelty.
    Adds non-authoritative memos that may be noisy/conflicting and
    should be validated by proper rule reasoning.
    """
    combined_seed = f"{task_id}:{chosen.get('invoice_id')}:{seed}"
    rng = random.Random(combined_seed)
    noisy_memos = [
        "Internal AP memo: Prior reviewer suggested 194J, pending final verification.",
        "Vendor email note: Claims no deduction needed due to prior annual deductions.",
        "Finance handoff note: Mentions legacy threshold interpretation from last FY.",
        "Payment ops note: Tax treatment marked provisional until compliance check.",
    ]
    conflicting_memos = [
        "Legacy note: Apply old benchmark; may conflict with current FY rule.",
        "Ops comment: Section marker appears copied from a different contract type.",
    ]
    category = chosen.get("category", "")
    gt_section = str(chosen.get("ground_truth", {}).get("section", "")).upper()
    ambiguous = category in {
        "mixed_invoice",
        "threshold_boundary",
        "gst_bundled_tds_base",
        "inoperative_pan",
    } or gt_section in {"SPLIT", "SPLIT_194J_194I"}
    conflicting = task_id in {"task_hard", "task_expert"} and ambiguous and rng.random() < 0.6
    memo = rng.choice(conflicting_memos if conflicting else noisy_memos)
    return {
        "ambiguous_signals": bool(ambiguous),
        "conflicting_signal": bool(conflicting),
        "requires_multi_step": bool(task_id in {"task_medium", "task_hard", "task_expert"}),
        "memo": memo,
    }


# ---------------------------------------------------------------------------
# Task sampling
# ---------------------------------------------------------------------------

def sample_task(task_id: str, seed: Optional[int] = None) -> dict:
    """
    Sample one random invoice from the appropriate difficulty pool.

    Args:
        task_id : "task_easy" | "task_medium" | "task_hard" | "task_expert"
        seed    : optional random seed for reproducibility

    Returns:
        A complete task dict combining pool config + sampled invoice
    """
    if task_id not in DIFFICULTY_POOLS:
        raise KeyError(f"Unknown task_id: {task_id!r}. Valid: {TASK_ORDER}")

    pool_config = DIFFICULTY_POOLS[task_id]
    db = get_db()

    # Filter invoices matching this pool's categories
    candidates = [
        inv for inv in db
        if inv["category"] in pool_config["categories"]
    ]

    if not candidates:
        raise RuntimeError(f"No invoices found for task_id={task_id!r}")

    rng = random.Random(seed if seed is not None else DEFAULT_TASK_SEED)
    chosen = rng.choice(candidates)
    scenario_noise = _build_scenario_noise(task_id, chosen, seed)
    noisy_invoice_text = (
        f"{chosen['invoice_text']}\n\n"
        f"[Compliance Memo - Non-authoritative]\n{scenario_noise['memo']}\n"
        "Treat this memo as advisory only; verify against invoice facts and statutory rules."
    )

    return {
        "task_id":      task_id,
        "difficulty":   pool_config["difficulty"],
        "description":  pool_config["description"],
        "max_steps":    pool_config["max_steps"],
        "hint_enabled": pool_config["hint_enabled"],

        # From the sampled invoice
        "invoice_id":      chosen["invoice_id"],
        "invoice_text":    noisy_invoice_text,
        "vendor_pan":      chosen["vendor_pan"],
        "cumulative_ytd":  chosen["cumulative_ytd"],
        "category":        chosen["category"],
        "task_hint":       chosen["task_hint"],
        "ground_truth":    chosen["ground_truth"],
        "scenario_noise":  scenario_noise,
        "required_evidence_actions": _required_evidence_actions(chosen["category"], task_id),

        # Reward breakpoints vary by difficulty
        "reward_breakpoints": _build_breakpoints(pool_config["difficulty"], chosen),
    }


def _build_breakpoints(difficulty: str, invoice: dict) -> dict:
    """
    Build reward breakpoints for a sampled invoice.
    Weights vary by difficulty — harder tasks reward PAN check more.
    """
    gt = invoice["ground_truth"]
    is_inop_pan  = not gt["pan_valid"]
    is_mixed     = gt.get("goods_amount", 0) > 0
    is_bundled   = invoice["category"] == "gst_bundled_tds_base"
    is_threshold = invoice["category"] in ("threshold_boundary", "below_threshold_new_limits", "below_threshold")
    is_split     = gt["section"] in ("SPLIT", "SPLIT_194J_194I")

    bp = {}

    # PAN check reward — always present
    if difficulty in ("hard", "expert") and is_inop_pan:
        bp["pan_checked"]             = 0.20   # worth more — it's the key insight
        bp["pan_inoperative_flagged"] = 0.20   # extra credit for explicitly flagging
    else:
        bp["pan_checked"] = 0.10

    # Section identification
    if not is_inop_pan and not is_split:
        bp["section_correct"] = 0.25 if difficulty in ("easy", "medium") else 0.15

    # Goods exclusion (mixed invoices)
    if is_mixed or is_split:
        bp["goods_excluded"] = 0.20

    # Threshold check
    if is_threshold:
        bp["query_ytd_checked"] = 0.075
        bp["threshold_checked"] = 0.075

    # GST base
    if is_bundled:
        bp["gst_base_correct"] = 0.15

    # Final amount — always the highest-value breakpoint
    bp["amount_exact"] = 0.40 if not is_inop_pan else 0.30

    # Normalise to sum to 1.0
    total = sum(bp.values())
    if total > 0:
        bp = {k: round(v / total, 4) for k, v in bp.items()}

    return bp


def _required_evidence_actions(category: str, task_id: str) -> list[str]:
    """
    Return the ordered evidence actions expected for a scenario.

    Args:
        category: Invoice category used for the sampled task.
        task_id: Current task identifier (easy/medium/hard/expert).

    Returns:
        Ordered de-duplicated action_type list used by submit-time reasoning checks.
    """
    base = ["check_pan"]
    if category in {"mixed_invoice", "gst_bundled_tds_base"}:
        base.append("lookup_section")
    if category in {"threshold_boundary", "below_threshold_new_limits", "below_threshold"}:
        base.extend(["query_ytd", "check_threshold"])
    if task_id in {"task_hard", "task_expert"}:
        base.append("query_law")
    # preserve order while deduplicating
    out = []
    for item in base:
        if item not in out:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def all_task_ids() -> list:
    return TASK_ORDER.copy()


def get_task(task_id: str, seed: Optional[int] = None) -> dict:
    """Alias for sample_task — matches old interface."""
    return sample_task(task_id, seed=seed)


def pool_size(task_id: str) -> int:
    """Return how many invoices are available for a given task."""
    config = DIFFICULTY_POOLS.get(task_id, {})
    db = get_db()
    return sum(1 for inv in db if inv["category"] in config.get("categories", []))
