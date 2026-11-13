import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = HERE / "inp.txt"


def read_config(path):
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Malformed configuration line: {raw_line}")
        key, value = line.split(":", 1)
        values[key.strip().lower()] = value.strip()
    return values


def enabled(value):
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main():
    config = read_config(CONFIG)
    model = config.get("model", "").lower()
    if model not in {"heston", "bates"}:
        raise ValueError("model must be Heston or Bates")
    if enabled(config.get("run p errors", "1")):
        source = HERE / f"P_errors_{model}.cpp"
        if not source.is_file():
            raise FileNotFoundError(f"Missing P-error handler for {model}: {source}")
        binary = Path(tempfile.gettempdir()) / f"joint_calibration_p_errors_{model}"
        print(f"Running {model.title()} P-space error analysis", flush=True)
        subprocess.run(
            ["g++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", str(source), "-o", str(binary)],
            cwd=ROOT,
            check=True,
        )
        subprocess.run([str(binary)], cwd=ROOT, check=True)

    if enabled(config.get("run q errors", "1")):
        q_handler = HERE / f"Q_errors_{model}.py"
        if not q_handler.is_file():
            raise FileNotFoundError(f"Missing Q-error handler for {model}: {q_handler}")
        print(f"Running {model.title()} Q-space error analysis", flush=True)
        subprocess.run(
            ["python3", str(q_handler)],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
