import io
import re
from contextlib import redirect_stdout

from inference import log_end, log_start, log_step


def test_strict_log_lines_format():
    buf = io.StringIO()
    with redirect_stdout(buf):
        log_start(task="task_easy", env="legaloom_env", model="demo", seed=42)
        log_step(step=1, action="read_invoice()", reward=0.12, done=False, error=None)
        log_end(success=True, steps=1, score=0.12, rewards=[0.12])
    lines = [ln.strip() for ln in buf.getvalue().splitlines() if ln.strip()]
    assert re.match(r"^\[START\] task=.+ env=.+ model=.+ seed=\d+$", lines[0])
    assert re.match(r"^\[STEP\] step=\d+ action=.+ reward=-?\d+\.\d{2} done=(true|false) error=.*$", lines[1])
    assert re.match(r"^\[END\] success=(true|false) steps=\d+ rewards=.*$", lines[2])
