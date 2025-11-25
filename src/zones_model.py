import json
from pathlib import Path

JSON_PATH = Path("data/zones.json")

def load_zones():
    """Load saved zones from JSON, or return empty structure."""
    if JSON_PATH.exists():
        try:
            with open(JSON_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"zones": []}


def save_zones(data: dict):
    """Save zones to JSON."""
    JSON_PATH.parent.mkdir(exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)


def default_zone_template():
    """Default empty zone."""
    return {
        "zone_name": "",
        "rock_domain": "",
        "chainage_from": "",
        "chainage_to": "",
        "sd": "",
        "station": "",
        "DRI": {"Mean": 0.0, "LB": 0.0, "UB": 0.0},
        "CLI": {"Mean": 0.0, "LB": 0.0, "UB": 0.0},
        "Q": {"Mean": 0.0, "LB": 0.0, "UB": 0.0},
        "Porosity": {"Mean": 0.0, "LB": 0.0, "UB": 0.0},
        "tunnel_direction": 0,
        "set1": {"strike": 0, "dip": 0, "Fr_mean": "", "Fr_LB": "", "Fr_UB": ""},
        "set2": {"strike": 0, "dip": 0, "Fr_mean": "", "Fr_LB": "", "Fr_UB": ""},
        "set3": {"strike": 0, "dip": 0, "Fr_mean": "", "Fr_LB": "", "Fr_UB": ""},
        "length_m": 0
    }
