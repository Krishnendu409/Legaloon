from models import TDSAction
from server.legaloom_env_environment import LegaloomEnvironment


def _read_invoice_text(env: LegaloomEnvironment, task_id: str, seed: int) -> str:
    env.reset(task_id=task_id, seed=seed)
    obs = env.step(TDSAction(action_type="read_invoice", parameters={}))
    return obs.invoice_text


def test_reset_and_sampling_deterministic_with_seed():
    e1 = LegaloomEnvironment()
    e2 = LegaloomEnvironment()
    text1 = _read_invoice_text(e1, "task_medium", 20260408)
    text2 = _read_invoice_text(e2, "task_medium", 20260408)
    assert text1 == text2
