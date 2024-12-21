import requests
from pathlib import Path
import yaml

PACKAGE_YML = Path(__file__).parents[2] / "libs" / "packages.yml"

def _get_downloads(p: dict) -> int:
    url = f"https://pypistats.org/api/packages/{p['name']}/recent?period=month"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()["data"]["last_month"]

with open(PACKAGE_YML) as f:
    data = yaml.safe_load(f)
    for p in data["packages"]:
        p["downloads"] = _get_downloads(p)

with open(PACKAGE_YML, "w") as f:
    yaml.dump(data, f)