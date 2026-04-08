---
title: LegaLoom-Env
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# LegaLoom-Env — TDS Compliance Environment for AI Agents

An OpenEnv-compatible reinforcement learning environment where AI agents learn
to perform **Tax Deducted at Source (TDS) compliance** — one of the most
critical and error-prone financial back-office tasks in India.

---

## Why TDS Compliance?

Every Indian company processes hundreds of vendor invoices monthly. Each invoice requires the accounts team to:

1. Identify the correct TDS section (194C, 194J, 194I, 194H, 194T, 194Q…)
2. Verify the vendor's PAN status — inoperative PAN triggers a 20% fallback rate under Section 206AA
3. Check if cumulative annual payments cross the deduction threshold for that section
4. Compute the exact INR amount to deduct, excluding goods and GST where applicable

A single systemic error — for example applying 2% instead of 10% across all consulting invoices — results in tax shortfall, 1.5% monthly interest, and heavy penalties during audit. This environment models that exact real-world workflow with 260 realistic vendor invoices and verified FY 2025-26 tax rules.

---

## Action Space

The agent takes one action per step by submitting a JSON object:

| Action | Parameters | What it does |
|---|---|---|
| `read_invoice` | — | Retrieves the full formatted invoice text |
| `check_pan` | `pan: str` | Returns PAN operative status and vendor type (company/individual/firm) |
| `check_threshold` | `section: str, amount: float` | Checks if the annual deduction threshold is crossed |
| `query_ytd` | `pan: str` | Returns cumulative year-to-date payments to this vendor |
| `lookup_section` | `description: str` | Classifies a service description into the applicable TDS section and rate |
| `query_law` | `section: str` | Returns the full FY 2025-26 rules for a section |
| `submit_answer` | `tds_amount_inr: float, section: str, rate_percent: float` | Submits the final answer — ends the episode |

Invalid action type incurs a penalty of −0.05.

---

## Observation Space

Each step returns a `TDSObservation` with these fields:

| Field | Type | Description |
|---|---|---|
| `invoice_text` | str | Full invoice text (populated after `read_invoice`) |
| `action_result` | str | Environment's response to the last action |
| `available_actions` | list[str] | Valid action types for the next step |
| `steps_used` | int | Steps taken so far this episode |
| `max_steps` | int | Maximum steps allowed for this task |
| `hint` | str | Coaching hint (empty on hard/expert — agent must reason independently) |
| `reward` | float | Reward for this step (always float, never None) |
| `done` | bool | True when the episode is complete |

---

## Tasks

There are four tasks of increasing difficulty. Each `reset()` draws a random invoice from the task pool — the agent never sees the same scenario twice in exactly the same way.

---

### Task 1 — Easy (`task_easy`)

**Goal:** Identify the TDS section from a clear service description, verify the vendor PAN, and compute the exact deduction.

**Pool:** 102 invoices — single-service invoices across four sections:

| Category | Invoices | Section | Rate | Example |
|---|---|---|---|---|
| Professional services | 40 | 194J | 10% | Legal advisory, audit fees, architectural services |
| Contractor services | 30 | 194C | 1–2% | Catering, security guards, event management, printing |
| Rent | 17 | 194I | 10% | Office space, warehouse, virtual office, meeting rooms |
| Commission | 15 | 194H | 2% | Sales commission, brokerage, referral fees |

**Constraints:** Valid operative PAN, single line item, no GST confusion. Max 6 steps. Hints enabled.

**Grading:** Full score for correct section + rate + exact INR amount (±₹1). Partial credit for correct section identification even if the final amount is wrong. If the invoice is below threshold, a correct `no_tds=true` submission scores 1.0.

**Recommended strategy:**
1. `read_invoice` — extract vendor PAN and service description
2. `check_pan` — confirm PAN is operative and get vendor type
3. `lookup_section` — get applicable section and rate
4. `submit_answer` — amount = taxable × rate / 100

---

### Task 2 — Medium (`task_medium`)

**Goal:** Handle mixed invoices (goods + services) or threshold boundary cases requiring cumulative YTD payment history.

**Pool:** 88 invoices across four categories:

| Category | Invoices | Challenge |
|---|---|---|
| Mixed invoice | 25 | Goods line items (no TDS) + service line items (TDS applies) — must split and exclude goods |
| Threshold boundary | 20 | Single invoice below threshold but YTD + current invoice crosses it |
| 194J Technical | 30 | IT/cloud/BPO services by company vendor — rate is 2% not 10% |
| 194I Machinery hire | 13 | Equipment hire — rate is 2% not 10% like building rent |

**Constraints:** Valid PAN, up to 2 line items, YTD may be non-zero. Max 8 steps. Hints enabled.

**Grading:** Goods exclusion carries its own reward breakpoint (+0.20). Threshold check carries a breakpoint (+0.15) on boundary invoices. Final amount must be within ±₹1 of ground truth.

**Recommended strategy:**
1. `read_invoice` — identify all line items (goods vs services)
2. `check_pan` — confirm vendor type (affects 194C rate: 1% individual, 2% company)
3. `lookup_section` — classify the service portion
4. `query_ytd` — get cumulative payments (for threshold boundary invoices)
5. `check_threshold` — confirm whether TDS applies at all
6. `submit_answer` — amount = service-only taxable × rate / 100

---

### Task 3 — Hard (`task_hard`)

**Goal:** Detect special conditions that override normal section rates — inoperative PAN, GST bundled into the taxable base, or below-threshold cases under new FY 2025-26 limits.

**Pool:** 43 invoices:

| Category | Invoices | Challenge |
|---|---|---|
| Inoperative PAN | 25 | PAN not linked to Aadhaar → 20% under Section 206AA, regardless of service section |
| GST-bundled base | 8 | GST not itemised on invoice → TDS on full invoice amount including GST |
| Below-threshold (new FY 2025-26 limits) | 10 | Thresholds raised: 194J ₹50k, 194I ₹6L, 194H ₹20k — agent must apply new limits |

**Constraints:** No hints. Max 8 steps. Agent must independently reason through edge cases.

**Grading:** Inoperative PAN detection earns an extra breakpoint (+0.20 on top of the base pan_checked reward). For GST-bundled invoices, computing TDS on the correct base earns a separate breakpoint (+0.15). Final amount ±₹1 tolerance.

**Key rule:** If PAN is INOPERATIVE, always deduct at 20% regardless of which section would otherwise apply. This overrides everything.

---

### Task 4 — Expert (`task_expert`)

**Goal:** Apply new FY 2025-26 sections that most models have not seen in training data.

**Pool:** 13 invoices:

| Category | Invoices | Section | Rate | Challenge |
|---|---|---|---|---|
| 194T — Partner payments | 8 | 194T | 10% | New section from April 2025 — partnerships paying their own partners: 10% on payments above ₹20,000/year |
| 194Q — Bulk goods purchase | 5 | 194Q | 0.1% | Buyer deducts 0.1% on goods purchases exceeding ₹50L/year from a single vendor |

**Why expert:** Section 194T was introduced in April 2025 — after most model training cutoffs. Section 194Q (0.1%) is easily confused with 194C (1–2%). The agent must use `query_law` or `lookup_section` correctly to distinguish these sections. No hints. Max 10 steps.

**Grading:** Same framework as other tasks. Partial credit awarded for correct PAN check and section identification even if the final amount is wrong.

---

## Reward Function

Rewards are shaped across the full trajectory — not just binary end-of-episode.

| Breakpoint | Reward share | When awarded |
|---|---|---|
| `pan_checked` | 0.10–0.20 | Any `check_pan` call with the vendor PAN |
| `pan_inoperative_flagged` | +0.20 | Correctly identifying an inoperative PAN (Task 3) |
| `section_correct` | 0.15–0.25 | `lookup_section` returns the correct section |
| `threshold_checked` | 0.15 | `check_threshold` on a threshold-boundary invoice |
| `goods_excluded` | 0.20 | Correct split computation on mixed invoices |
| `gst_base_correct` | 0.15 | Correct taxable base on GST-bundled invoices |
| `amount_exact` | 0.30–0.50 | Final TDS amount within ±₹1 of ground truth |
| Unknown action | −0.05 | Any unrecognised `action_type` |

Each breakpoint is awarded **at most once per episode**. Total episode score is clamped to [0.0, 1.0].

---

## Grader Functions

Explicit, deterministic grader functions live in `server/graders.py`:

```python
grade_easy(params, ground_truth)    # → float in [0.0, 1.0]
grade_medium(params, ground_truth)  # → float in [0.0, 1.0]
grade_hard(params, ground_truth)    # → float in [0.0, 1.0]
grade_expert(params, ground_truth)  # → float in [0.0, 1.0]
```

All graders are **deterministic** — same inputs always produce the same score. Scores are rounded to 4 decimal places and clamped to [0.0, 1.0].

**Grading logic for `submit_answer`:**

```
Case 1 — No TDS applicable (below threshold):
  correct no_tds=true submission  → 1.0
  non-zero amount submitted        → 0.0

Case 2 — Inoperative PAN (Section 206AA):
  rate=20% correctly applied       → +0.40
  correct section identified       → +0.20
  correct rate value               → +0.10
  exact amount (±₹1)              → +0.30 (capped at 1.0 total)

Case 3 — Normal TDS:
  correct section                  → +0.20
  correct rate                     → +0.10
  goods excluded (mixed invoices)  → +0.10
  exact amount (±₹1)              → +0.40 (or +0.30 for mixed invoices)
```

---

## TDS Rules (FY 2025-26)

| Section | Nature | Default Rate | Annual Threshold |
|---|---|---|---|
| 194C | Contractor — company | 2% | ₹30k single / ₹1,00,000 annual |
| 194C | Contractor — individual/firm | 1% | ₹30k single / ₹1,00,000 annual |
| 194J | Professional services | 10% | ₹50,000 |
| 194J | Technical services (company vendor) | 2% | ₹50,000 |
| 194I | Building / office rent | 10% | ₹6,00,000 |
| 194I | Machinery / equipment hire | 2% | ₹6,00,000 |
| 194H | Commission / brokerage | 2% | ₹20,000 |
| 194T | Partner payments (NEW April 2025) | 10% | ₹20,000 |
| 194Q | Buyer's goods purchase above ₹50L | 0.1% | ₹50,00,000 |
| 206AA | Inoperative / missing PAN | 20% | Overrides all sections |

---

## Baseline Scores

Evaluated using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router:

| Task | Score | Steps | Notes |
|---|---|---|---|
| task_easy | 1.000 | 5 | Correct section, rate, exact amount |
| task_medium | 1.000 | 5 | Goods excluded, threshold checked correctly |
| task_hard | 1.000 | 5 | Inoperative PAN detected, 20% correctly applied |
| task_expert | 1.000 | 5 | 194T partner payment correctly identified |
| **Average** | **1.000** | | Qwen/Qwen2.5-72B-Instruct via HF Inference |

Secondary evaluation using `llama-3.3-70b-versatile` via Groq:

| Task | Score | Notes |
|---|---|---|
| task_easy | 0.917 | Occasional GST base confusion on complex invoices |
| task_medium | 1.000 | |
| task_hard | 1.000 | |
| task_expert | 1.000 | Both 194T and 194Q correctly handled |
| **Average** | **0.979** | |

---

## Setup & Usage

### Install

```bash
pip install openenv-core openai
```

### Run locally

```bash
git clone https://github.com/aaravgupta0202/legaloom_env.git
cd legaloom_env
pip install openenv-core fastapi uvicorn pydantic openai
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Test endpoints

```bash
# Start a new episode
curl -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" -d "{}"

# Take a step
curl -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"action": {"action_type": "read_invoice", "parameters": {}}}'

# Health check
curl http://localhost:7860/health
```

### Run with Docker

```bash
docker build -t legaloom-env .
docker run -p 7860:7860 legaloom-env
```

### Run baseline inference

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

---

## Project Structure

```
legaloom_env/
├── inference.py                     ← Baseline inference script (all 4 tasks)
├── models.py                        ← TDSAction, TDSObservation, TDSState
├── client.py                        ← LegaloomEnv WebSocket client
├── openenv.yaml                     ← OpenEnv manifest with tasks + action/obs schemas
├── Dockerfile                       ← Container definition (port 7860)
└── server/
    ├── app.py                       ← FastAPI server
    ├── legaloom_env_environment.py  ← Environment logic (reset, step, state)
    ├── graders.py                   ← Explicit deterministic grader functions
    ├── tasks.py                     ← Task pool definitions and invoice sampling
    ├── tds_rules.py                 ← FY 2025-26 TDS rules and section classifier
    ├── pan_registry.py              ← Vendor PAN database (32 vendors, 4 inoperative)
    └── invoice_db.json              ← 260 verified invoices with ground truth
```

---

## Live Endpoint

**HF Space:** https://huggingface.co/spaces/aarav0202/legaloom-env

**API base:** https://aarav0202-legaloom-env.hf.space
