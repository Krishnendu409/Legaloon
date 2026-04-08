import yaml
from pathlib import Path

from server.legaloom_env_environment import LegaloomEnvironment


def test_openenv_schema_contains_reward_info():
    repo_root = Path(__file__).resolve().parents[1]
    with open(repo_root / "openenv.yaml", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    obs_props = manifest["observation_schema"]["properties"]
    assert "reward_info" in obs_props
    assert obs_props["reward_info"]["type"] == "object"


def test_state_api_is_callable():
    env = LegaloomEnvironment()
    assert callable(getattr(env, "state", None))
