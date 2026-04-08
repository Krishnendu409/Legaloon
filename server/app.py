"""
FastAPI server for LegaLoom-Env.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Run: pip install openenv-core") from e

from models import TDSAction, TDSObservation
from server.legaloom_env_environment import LegaloomEnvironment


app = create_app(
    LegaloomEnvironment,
    TDSAction,
    TDSObservation,
    env_name="legaloom_env",
    max_concurrent_envs=10,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
