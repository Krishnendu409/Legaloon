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

    if seed is not None:
        random.seed(seed)

    chosen = random.choice(candidates)

    return {
        "task_id":      task_id,
        "difficulty":   pool_config["difficulty"],
        "description":  pool_config["description"],
        "max_steps":    pool_config["max_steps"],
        "hint_enabled": pool_config["hint_enabled"],

        # From the sampled invoice
        "invoice_id":      chosen["invoice_id"],
        "invoice_text":    chosen["invoice_text"],
        "vendor_pan":      chosen["vendor_pan"],
        "cumulative_ytd":  chosen["cumulative_ytd"],
        "category":        chosen["category"],
        "task_hint":       chosen["task_hint"],
        "ground_truth":    chosen["ground_truth"],

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
        bp["threshold_checked"] = 0.15

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
