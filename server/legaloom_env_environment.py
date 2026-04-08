"""
LegaLoom-Env — TDS Compliance Environment

Simulates Indian Tax Deducted at Source (TDS) compliance back-office tasks.
Agent reads vendor invoices and computes the correct TDS deduction.

Reward contract: every reward value is STRICTLY in (0.0, 1.0) exclusive.
The hackathon validator rejects exactly 0.0 and exactly 1.0.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TDSAction, TDSObservation, TDSState
except ImportError:
    from models import TDSAction, TDSObservation, TDSState

try:
    from .tds_rules import (get_rate, threshold_crossed, compute_tds,
                             TDS_SECTIONS, section_summary, classify_service)
    from .pan_registry import is_pan_valid, is_company, pan_status_message
    from .tasks import get_task, all_task_ids, sample_task
    from .graders import grade_submission, GRADERS
except ImportError:
    from server.tds_rules import (get_rate, threshold_crossed, compute_tds,
                                   TDS_SECTIONS, section_summary, classify_service)
    from server.pan_registry import is_pan_valid, is_company, pan_status_message
    from server.tasks import get_task, all_task_ids, sample_task
    from server.graders import grade_submission, GRADERS

AMOUNT_TOLERANCE_INR = 1.0

# Reward bounds — strictly open interval required by hackathon
_R_MIN = 0.05
_R_MAX = 0.95


def _r(v: float) -> float:
    """Clamp a reward to strictly open (0.0, 1.0)."""
    return round(min(max(float(v), _R_MIN), _R_MAX), 4)


class LegaloomEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state           = TDSState()
        self._task            = None
        self._invoice_read    = False
        self._ytd_queried     = False
        self._law_queried     = False
        self._reward_earned   = {}
        self._episode_reward  = _R_MIN
        self._current_task_id = "task_easy"
        self._pan_checked_pan = None
        self._section_found   = None

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task_easy", seed: int = None, **kwargs) -> TDSObservation:
        if task_id not in all_task_ids():
            task_id = "task_easy"

        self._task            = sample_task(task_id, seed=seed)
        self._current_task_id = task_id
        self._invoice_read    = False
        self._ytd_queried     = False
        self._law_queried     = False
        self._pan_checked_pan = None
        self._section_found   = None
        self._reward_earned   = {k: False for k in self._task["reward_breakpoints"]}
        self._episode_reward  = _R_MIN   # start just above 0.0

        self._state = TDSState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task_id,
            difficulty=self._task["difficulty"],
            pan_checked=False,
            section_identified=False,
            answer_submitted=False,
        )

        return TDSObservation(
            done=False,
            reward=_R_MIN,
            invoice_text="",
            action_result=(
                f"New episode started. Task: {task_id} "
                f"({self._task['difficulty']} difficulty). "
                f"Invoice: {self._task['invoice_id']}. "
                f"You have {self._task['max_steps']} steps. "
                "Start with action_type='read_invoice'."
            ),
            available_actions=["read_invoice"],
            steps_used=0,
            max_steps=self._task["max_steps"],
            hint=self._build_hint(),
        )

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: TDSAction, **kwargs) -> TDSObservation:
        try:
            if self._task is None:
                self.reset(task_id=self._current_task_id)

            self._state.step_count += 1
            steps_used = self._state.step_count
            max_steps  = self._task["max_steps"]

            if steps_used > max_steps:
                return self._force_close(steps_used, max_steps)

            action_type = (action.action_type or "").strip().lower()
            params      = action.parameters or {}

            handlers = {
                "read_invoice":    self._handle_read_invoice,
                "check_pan":       self._handle_check_pan,
                "check_threshold": self._handle_check_threshold,
                "query_ytd":       self._handle_query_ytd,
                "lookup_section":  self._handle_lookup_section,
                "query_law":       self._handle_query_law,
                "submit_answer":   self._handle_submit_answer,
            }

            handler = handlers.get(action_type)
            if handler:
                if action_type == "submit_answer":
                    return handler(params, steps_used)
                return handler(params, steps_used, max_steps)

            # Unknown action — penalise (but still above floor)
            return TDSObservation(
                done=False, reward=_R_MIN,
                invoice_text=self._invoice_text(),
                action_result=(
                    f"Unknown action_type: '{action_type}'. "
                    "Valid: read_invoice, check_pan, check_threshold, "
                    "query_ytd, lookup_section, query_law, submit_answer."
                ),
                available_actions=self._available_actions(),
                steps_used=steps_used, max_steps=max_steps,
                hint=self._build_hint(),
            )
        except Exception as exc:
            return TDSObservation(
                done=False, reward=_R_MIN,
                invoice_text=self._invoice_text() if self._task else "",
                action_result=f"Environment error: {exc}. Please retry your action.",
                available_actions=self._available_actions() if self._task else ["read_invoice"],
                steps_used=getattr(self._state, "step_count", 0),
                max_steps=self._task["max_steps"] if self._task else 8,
                hint="",
            )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_read_invoice(self, params, steps_used, max_steps):
        self._invoice_read = True
        gt = self._task["ground_truth"]
        guidance = "Note the vendor PAN, service description, and amount(s)."
        if gt.get("goods_amount", 0) > 0:
            guidance += " This invoice has multiple line items — check each carefully."
        if not gt["pan_valid"]:
            guidance += " IMPORTANT: Always verify PAN status before computing TDS."
        return TDSObservation(
            done=False, reward=_R_MIN,
            invoice_text=self._task["invoice_text"],
            action_result=f"Invoice retrieved. {guidance}",
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_check_pan(self, params, steps_used, max_steps):
        pan = str(params.get("pan", "")).strip().upper()
        if not pan:
            return self._error_obs(
                'check_pan requires parameter pan. Example: {"pan": "ABCDE1234F"}',
                steps_used, max_steps
            )

        reward = self._award("pan_checked")
        self._state.pan_checked = True

        try:
            from server.pan_registry import PAN_DB
        except ImportError:
            from pan_registry import PAN_DB
        record = PAN_DB.get(pan)
        if record:
            result = pan_status_message(pan)
        else:
            gt = self._task["ground_truth"]
            vendor_pan = self._task["vendor_pan"]
            if pan == vendor_pan:
                if gt["pan_valid"]:
                    result = f"PAN {pan} is operative and valid. TDS at section rate applies."
                else:
                    result = (
                        f"PAN {pan} is INOPERATIVE (not linked to Aadhaar). "
                        "TDS rate: 20% regardless of section — Section 206AA applies."
                    )
            else:
                result = f"PAN {pan} not found in registry. Verify the PAN from the invoice."

        gt = self._task["ground_truth"]
        if not gt["pan_valid"] and "INOPERATIVE" in result.upper():
            reward = _r(reward + self._award("pan_inoperative_flagged"))

        return TDSObservation(
            done=False, reward=reward,
            invoice_text=self._invoice_text(),
            action_result=result,
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_check_threshold(self, params, steps_used, max_steps):
        section = str(params.get("section", "")).strip().upper()
        amount  = float(params.get("amount", 0))
        ytd     = self._task.get("cumulative_ytd", 0.0)

        if not section:
            return self._error_obs(
                'check_threshold requires section and amount. Example: {"section": "194J", "amount": 85000}',
                steps_used, max_steps
            )
        if section not in TDS_SECTIONS:
            return self._error_obs(
                f"Unknown section '{section}'. Valid: {list(TDS_SECTIONS.keys())}",
                steps_used, max_steps
            )

        crossed = threshold_crossed(section, amount, ytd)
        s = TDS_SECTIONS[section]
        result = (
            f"Section {section} threshold check — Invoice: INR {amount:,.0f} | "
            f"YTD paid: INR {ytd:,.0f} | Running total: INR {ytd+amount:,.0f}. "
            f"Single limit: INR {s['threshold_single']:,} | "
            f"Annual limit: INR {s['threshold_annual']:,}. "
            f"Result: {'TDS IS applicable — threshold crossed.' if crossed else 'TDS NOT applicable — below threshold.'}"
        )

        reward = self._award("threshold_checked")
        return TDSObservation(
            done=False, reward=reward,
            invoice_text=self._invoice_text(),
            action_result=result,
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_query_ytd(self, params, steps_used, max_steps):
        pan = str(params.get("pan", "")).strip().upper()
        ytd = self._task.get("cumulative_ytd", 0.0)
        self._ytd_queried = True

        if not pan:
            result = (
                f"YTD payments to vendor for current financial year: "
                f"INR {ytd:,.0f}. "
                "This is the cumulative amount paid BEFORE this invoice."
            )
        else:
            vendor_pan = self._task["vendor_pan"]
            if pan == vendor_pan:
                result = (
                    f"YTD payments to vendor PAN {pan} in FY 2024-25: "
                    f"INR {ytd:,.0f}. "
                    "Add the current invoice amount to determine if annual threshold is crossed."
                )
            else:
                result = f"No payment history found for PAN {pan} in current financial year."

        reward = self._award("threshold_checked")
        return TDSObservation(
            done=False, reward=reward,
            invoice_text=self._invoice_text(),
            action_result=result,
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_lookup_section(self, params, steps_used, max_steps):
        description = str(params.get("description", "")).strip()
        if not description:
            return self._error_obs(
                'lookup_section requires description. Example: {"description": "legal consultation"}',
                steps_used, max_steps
            )

        gt = self._task["ground_truth"]
        vendor_pan = self._task["vendor_pan"]
        is_company_vendor = is_company(vendor_pan)

        result_dict = classify_service(description, vendor_is_company=is_company_vendor)
        matched_section = result_dict["section"]
        matched_rate    = result_dict["rate"]

        result = (
            f"Service classification for '{description}': "
            f"Section {matched_section} — {TDS_SECTIONS.get(matched_section, {}).get('description', 'Unknown')}. "
            f"Applicable rate: {matched_rate}% "
            f"(vendor is {'company' if is_company_vendor else 'individual/LLP'}). "
            f"Confidence: {result_dict.get('confidence', 'medium')}."
        )

        reward = _R_MIN
        expected_section = gt["section"]
        tds_applicable = gt.get("tds_applicable", True)
        section_match = (
            matched_section == expected_section and
            expected_section not in ("SPLIT", "SPLIT_194J_194I")
        )
        if section_match or not tds_applicable:
            reward = self._award("section_correct")
            self._state.section_identified = True

        return TDSObservation(
            done=False, reward=reward,
            invoice_text=self._invoice_text(),
            action_result=result,
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_query_law(self, params, steps_used, max_steps):
        section = str(params.get("section", "")).strip().upper()
        self._law_queried = True

        if not section or section not in TDS_SECTIONS:
            result = "TDS Sections overview (FY 2025-26):\n"
            for code, data in TDS_SECTIONS.items():
                result += (
                    f"  {code}: {data['description']} — "
                    f"Rate: {data['rate_default']}% — "
                    f"Threshold: INR {data.get('threshold_annual',0):,}/year\n"
                )
            result += '\nUse query_law with section parameter for details. E.g.: {"section": "194J"}'
        else:
            s = TDS_SECTIONS[section]
            result = (
                f"Section {section} — {s['description']} (FY 2025-26)\n"
                f"Nature: {s['nature']}\n"
                f"Default rate: {s['rate_default']}%\n"
            )
            if "rate_company" in s:
                result += f"Company rate: {s['rate_company']}%\n"
            if "rate_technical" in s:
                result += f"Technical services rate: {s['rate_technical']}%\n"
            if "rate_machinery" in s:
                result += f"Machinery/equipment rate: {s['rate_machinery']}%\n"
            result += (
                f"No-PAN / inoperative PAN rate: {s['rate_no_pan']}% (Section 206AA)\n"
                f"Single payment threshold: INR {s['threshold_single']:,}\n"
                f"Annual threshold: INR {s['threshold_annual']:,}\n"
            )

        return TDSObservation(
            done=False, reward=_R_MIN,
            invoice_text=self._invoice_text(),
            action_result=result,
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _handle_submit_answer(self, params, steps_used) -> TDSObservation:
        self._state.answer_submitted = True
        gt = self._task["ground_truth"]

        grader_result = grade_submission(params, gt, task_id=self._current_task_id)
        fb = grader_result["feedback"]

        if grader_result["breakdown"].get("no_tds_correct") or \
                grader_result["breakdown"].get("amount_correct"):
            self._award("amount_exact")
        if grader_result["breakdown"].get("pan_inoperative_detected"):
            self._award("pan_inoperative_flagged")
        if grader_result["breakdown"].get("goods_excluded"):
            self._award("goods_excluded")
        if grader_result["breakdown"].get("gst_base_correct"):
            self._award("gst_base_correct")

        # Grader score is already clamped strictly in (0.0, 1.0)
        final_reward = grader_result["score"]
        self._episode_reward = final_reward
        return self._end_episode(final_reward, fb, steps_used)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _end_episode(self, reward, feedback_parts, steps_used) -> TDSObservation:
        total_score = _r(reward)   # strictly in (0.0, 1.0)
        result = (
            f"Episode complete. {' | '.join(feedback_parts)} "
            f"Final score: {total_score:.3f}."
        )
        return TDSObservation(
            done=True,
            reward=total_score,
            invoice_text=self._task["invoice_text"],
            action_result=result,
            available_actions=[],
            steps_used=steps_used,
            max_steps=self._task["max_steps"],
            hint="",
        )

    def _award(self, key: str) -> float:
        """Award reward for a breakpoint exactly once per episode. Always > 0."""
        if self._reward_earned.get(key, True):
            return _R_MIN
        reward = float(self._task["reward_breakpoints"].get(key, _R_MIN))
        reward = _r(reward)
        self._reward_earned[key] = True
        self._episode_reward = _r(self._episode_reward + reward)
        return reward

    def _invoice_text(self) -> str:
        return self._task["invoice_text"] if self._invoice_read else ""

    def _available_actions(self) -> list:
        if not self._invoice_read:
            return ["read_invoice"]
        return ["check_pan", "check_threshold", "query_ytd",
                "lookup_section", "query_law", "submit_answer"]

    def _build_hint(self) -> str:
        if self._task and not self._task.get("hint_enabled", True):
            return ""
        if not self._invoice_read:
            return "Start by calling read_invoice to see the invoice."
        if not self._state.pan_checked:
            return "Call check_pan with the vendor PAN from the invoice."
        if not self._state.section_identified:
            return "Call lookup_section with the service description."
        if self._ytd_queried is False and self._task.get("cumulative_ytd", 0) > 0:
            return "Call query_ytd to check cumulative payments before computing TDS."
        return "Call submit_answer with tds_amount_inr, section, and rate_percent."

    def _error_obs(self, message: str, steps_used: int, max_steps: int) -> TDSObservation:
        return TDSObservation(
            done=False, reward=_R_MIN,
            invoice_text=self._invoice_text(),
            action_result=message,
            available_actions=self._available_actions(),
            steps_used=steps_used, max_steps=max_steps,
            hint=self._build_hint(),
        )

    def _force_close(self, steps_used: int, max_steps: int) -> TDSObservation:
        return TDSObservation(
            done=True, reward=_R_MIN,
            invoice_text=self._invoice_text(),
            action_result=(
                f"Episode terminated: exceeded {max_steps} steps without submitting. "
                "Score: 0.000."
            ),
            available_actions=[],
            steps_used=steps_used, max_steps=max_steps,
            hint="",
        )

    @property
    def state(self) -> TDSState:
        return self._state

    def get_state(self) -> TDSState:
        """Callable method alias for OpenEnv HTTP routing (state() endpoint)."""
        return self._state
