"""
inference.py — LegaLoom-Env Baseline Inference Script

Mandatory variables:
    API_BASE_URL     : LLM endpoint
    MODEL_NAME       : model identifier
    HF_TOKEN         : Hugging Face API key
    LOCAL_IMAGE_NAME : Docker image name (optional)

Stdout format (strictly enforced by hackathon spec):
    [START] task=<task_id> env=legaloom_env model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Reward contract: every reward value is strictly in (0.0, 1.0) exclusive.
"""

import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Mandatory environment variables
# ---------------------------------------------------------------------------
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")
API_KEY          = HF_TOKEN or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK   = "legaloom_env"
MAX_STEPS   = 10
TEMPERATURE = 0
MAX_TOKENS  = 300

# Reward bounds — strictly open (0.0, 1.0)
_R_MIN = 0.05
_R_MAX = 0.95


def _clamp(v: float) -> float:
    return round(min(max(float(v), _R_MIN), _R_MAX), 4)


# ---------------------------------------------------------------------------
# Mandatory stdout loggers — exact spec format
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val    = error if error else "null"
    done_val     = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    # Reward must be strictly in (0.0, 1.0) — clamp here as final safety net
    logged_reward = _clamp(reward)
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={logged_reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # Clamp all individual rewards — evaluator checks each one
    safe_rewards = [_clamp(r) for r in rewards]
    rewards_str  = ",".join(f"{r:.2f}" for r in safe_rewards)
    success_val  = "true" if success else "false"
    # Clamp the overall score too
    safe_score   = _clamp(score)
    print(
        f"[END] success={success_val} steps={steps} rewards={rewards_str}",
        flush=True,
    )
    print(f"[SCORE] {safe_score:.3f}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Indian tax compliance agent specialising in TDS
(Tax Deducted at Source) under the Income Tax Act 1961, FY 2025-26.

Your job: read an invoice and compute the exact TDS deduction in INR.

OUTPUT FORMAT: Each turn output ONLY a valid JSON object. No explanation,
no markdown, no extra text. Just the JSON.

AVAILABLE ACTIONS:

1. Read the invoice (always first step):
   {"action_type": "read_invoice", "parameters": {}}

2. Check vendor PAN status (always do this before computing):
   {"action_type": "check_pan", "parameters": {"pan": "<10-char PAN>"}}

3. Check cumulative YTD payments to vendor (for threshold scenarios):
   {"action_type": "query_ytd", "parameters": {"pan": "<PAN>"}}

4. Look up applicable TDS section for a service description:
   {"action_type": "lookup_section", "parameters": {"description": "<service>"}}

5. Check if TDS threshold is crossed:
   {"action_type": "check_threshold", "parameters": {"section": "194J", "amount": 85000}}

6. Look up exact law text for a section:
   {"action_type": "query_law", "parameters": {"section": "194J"}}

7. Submit final answer (ends the episode):
   {"action_type": "submit_answer", "parameters": {
     "tds_amount_inr": <AMOUNT>,
     "section": "<194J>",
     "rate_percent": <RATE>
   }}

   For NO TDS cases (below threshold):
   {"action_type": "submit_answer", "parameters": {
     "tds_amount_inr": 0.0, "no_tds": "true",
     "section": "<section>", "rate_percent": 0.0
   }}

TDS RULES FY 2025-26 (memorise these):

SECTION IDENTIFICATION:
  194J Professional (10%): legal, CA/audit, medical, architect, CS
                            Individual/LLP vendor = professional
  194J Technical (2%):     IT support, cloud, software dev, BPO, data
                            Company (Pvt Ltd) vendor = always technical
  194C Contractor (2%):    security, catering, housekeeping, events,
                            manpower, printing, labour supply
  194I Rent (10%):         office/warehouse/land/building rent
  194I Rent (2%):          machinery/equipment/vehicle hire (no driver)
  194H Commission (2%):    sales commission, brokerage, referral fees
  194T Partner (10%):      NEW — partner salary/commission from firm
  194Q Goods (0.1%):       goods purchase >50L/year (buyer turnover >10Cr)

CRITICAL RULES:
  1. ALWAYS check PAN status first
  2. INOPERATIVE PAN: 20% flat, overrides ALL section rates (Section 206AA)
  3. MISSING PAN: also 20%
  4. Thresholds FY 2025-26: 194J=50,000 | 194C=30,000/1,00,000 |
     194I=6,00,000/year | 194H=20,000
  5. GST shown separately: TDS on pre-GST amount
     GST bundled in total: TDS on FULL invoice amount
  6. Goods line items (hardware, products, materials): NO TDS
     TDS only on service/rent/commission portion
  7. Mixed invoice: split line items, apply TDS only to service portion
  8. Below threshold: submit with tds_amount_inr=0.0 and no_tds="true"

RECOMMENDED STRATEGY:
  Step 1: read_invoice
  Step 2: check_pan (with vendor PAN from invoice)
  Step 3: lookup_section (with service description)
  Step 4: query_ytd IF invoice mentions cumulative payments OR threshold is borderline
  Step 5: check_threshold IF amount is near section limit
  Step 6: submit_answer

Output ONLY the JSON. Nothing else.
""").strip()


# ---------------------------------------------------------------------------
# Build user prompt
# ---------------------------------------------------------------------------

def build_user_prompt(step: int, obs: dict, history: List[str]) -> str:
    history_block   = "\n".join(history[-8:]) if history else "None"
    invoice_block   = obs.get("invoice_text", "")
    invoice_section = f"\nINVOICE:\n{invoice_block}\n" if invoice_block else ""
    hint            = obs.get("hint", "")
    hint_line       = f"\nHint: {hint}" if hint else ""

    return textwrap.dedent(f"""
Step {step} of {obs.get('max_steps', 10)}.
{invoice_section}
Last result: {obs.get('action_result', '')}
Available actions: {obs.get('available_actions', [])}
{hint_line}

Previous steps:
{history_block}

Output your next action as a JSON object only.
""").strip()


# ---------------------------------------------------------------------------
# Call LLM and parse JSON action
# ---------------------------------------------------------------------------

def get_agent_action(client: OpenAI, step: int, obs: dict,
                     history: List[str]) -> dict:
    user_prompt = build_user_prompt(step, obs, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if model wraps JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error at step {step}: {e}", flush=True)
        return {"action_type": "submit_answer",
                "parameters": {"tds_amount_inr": 0.0, "section": "194J",
                                "rate_percent": 0.0, "no_tds": "true"}}
    except Exception as e:
        print(f"[DEBUG] LLM call failed at step {step}: {e}", flush=True)
        return {"action_type": "submit_answer",
                "parameters": {"tds_amount_inr": 0.0, "section": "194J",
                                "rate_percent": 0.0, "no_tds": "true"}}


# ---------------------------------------------------------------------------
# Run one full episode
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, env, task_id: str) -> dict:
    from models import TDSAction

    rewards: List[float] = []
    history: List[str]   = []
    steps_taken = 0
    success     = False
    score       = _R_MIN

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(task_id=task_id)

        if hasattr(result, "observation"):
            obs  = result.observation.__dict__ if hasattr(result.observation, "__dict__") else {}
            done = result.done
        else:
            obs  = result.__dict__ if hasattr(result, "__dict__") else {}
            done = getattr(result, "done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_dict = get_agent_action(client, step, obs, history)
            action_type = action_dict.get("action_type", "read_invoice")
            parameters  = action_dict.get("parameters", {})

            action_str = (
                f"{action_type}("
                + ",".join(f"{k}={v}" for k, v in parameters.items())
                + ")"
            )

            error = None
            try:
                result = env.step(TDSAction(
                    action_type=action_type,
                    parameters=parameters,
                ))
                if hasattr(result, "observation"):
                    obs    = result.observation.__dict__ if hasattr(result.observation, "__dict__") else {}
                    reward = float(result.reward) if result.reward is not None else _R_MIN
                    done   = result.done
                else:
                    obs    = result.__dict__ if hasattr(result, "__dict__") else {}
                    reward = float(getattr(result, "reward", _R_MIN) or _R_MIN)
                    done   = getattr(result, "done", False)

            except Exception as e:
                reward = _R_MIN
                done   = False
                error  = str(e)[:120]

            # Every per-step reward must be strictly in (0.0, 1.0)
            reward = _clamp(reward)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str,
                     reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_str} -> reward={reward:+.3f} | "
                f"{obs.get('action_result', '')[:120]}"
            )

            if done:
                break

        # Final score: use the last reward from done=True step (the grader score)
        # That reward is already clamped by the environment.
        # Fallback: average of all step rewards.
        if rewards:
            # The final reward (done=True) is the authoritative grader score
            last_reward = rewards[-1]
            score = _clamp(last_reward)
        else:
            score = _R_MIN

        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Episode error for {task_id}: {e}", flush=True)
        score = _R_MIN

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score":   score,
        "success": success,
        "steps":   steps_taken,
    }


# ---------------------------------------------------------------------------
# Main — runs all 4 tasks sequentially
# ---------------------------------------------------------------------------

def main() -> None:
    client   = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_ids = ["task_easy", "task_medium", "task_hard", "task_expert"]
    results  = []

    if LOCAL_IMAGE_NAME:
        print(f"[INFO] Docker mode: {LOCAL_IMAGE_NAME}", file=sys.stderr)
        from client import LegaloomEnv
        env_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")
        for task_id in task_ids:
            with LegaloomEnv(base_url=env_url).sync() as env:
                result = run_episode(client, env, task_id)
                results.append(result)
    else:
        # In-process mode (default) — no Docker required for baseline run
        from server.legaloom_env_environment import LegaloomEnvironment

        for task_id in task_ids:
            env = LegaloomEnvironment()

            class LocalEnvWrapper:
                def __init__(self, e): self._env = e
                def reset(self, task_id="task_easy", **kw):
                    return self._env.reset(task_id=task_id)
                def step(self, action):
                    return self._env.step(action)

            result = run_episode(client, LocalEnvWrapper(env), task_id)
            results.append(result)

    print("\n[SUMMARY]", file=sys.stderr)
    for r in results:
        print(
            f"  {r['task_id']}: score={r['score']:.3f} "
            f"success={r['success']} steps={r['steps']}",
            file=sys.stderr,
        )
    avg = sum(r["score"] for r in results) / len(results)
    if (avg>=0.5):
        final_output=1
    else:
        final_output=0
    print(f"  Average score: {final_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
