import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_examples_match_the_readme() -> None:
    for script, expected in (
        ("examples/calculator_agent.py", "response: 125 multiplied by 48 is 6000."),
        ("examples/multi_step_agent.py", "response: The discounted total is 508.98."),
        ("examples/stateful_agent.py", "answer_response: The remembered unit price is 49.90."),
    ):
        completed = subprocess.run(
            [sys.executable, script],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        assert expected in completed.stdout
