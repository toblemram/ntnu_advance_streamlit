import json
from pathlib import Path

DEFAULT_INPUTS = {
    "TBM_diameter": {"LB": 7.1, "Mean": 7.1, "UB": 7.1},
    "Cutter_diameter": {"LB": 483, "Mean": 483, "UB": 483},
    "RPM": {"LB": 6, "Mean": 6, "UB": 7},
    "Cutters": {"LB": 45, "Mean": 45, "UB": 45},
    "Thrust_MB": {"LB": 260, "Mean": 300, "UB": 360},
    "Power_Ptbm": {"LB": 3500, "Mean": 3500, "UB": 3500},
    "rmc": {"LB": 0.59, "Mean": 0.59, "UB": 0.59},
    "Stroke_length_ls": {"LB": 1.8, "Mean": 1.8, "UB": 1.8},
    "ts": {"LB": 25, "Mean": 25, "UB": 25},
    "tc": {"LB": 60, "Mean": 60, "UB": 60},
    "TTBM": {"LB": 0.12, "Mean": 0.12, "UB": 0.12},
    "Tback": {"LB": 0.0, "Mean": 0.0, "UB": 0.0},
    "Tm": {"LB": 0.12, "Mean": 0.12, "UB": 0.12},
    "Effective_hours_Te": {"LB": 12, "Mean": 12, "UB": 12}
}

JSON_PATH = Path("data/machine_inputs.json")


def load_inputs():
    """Returnerer input fra JSON, eller default hvis filen ikke finnes."""
    if not JSON_PATH.exists():
        return DEFAULT_INPUTS
    try:
        with open(JSON_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_INPUTS


def save_inputs(data: dict):
    """Lagrer input til JSON."""
    JSON_PATH.parent.mkdir(exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)
