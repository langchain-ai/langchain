import requests
from pathlib import Path
import yaml
from datetime import datetime, timezone, timedelta

PACKAGE_YML = Path(__file__).parents[2] / "libs" / "packages.yml"

def _get_downloads(p: dict) -> int:
    url = f"https://pypistats.org/api/packages/{p['name']}/recent?period=month"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()["data"]["last_month"]

current_datetime = datetime.now(timezone.utc)
yesterday = current_datetime - timedelta(days=1)

with open(PACKAGE_YML) as f:
    data = yaml.safe_load(f)
for p in data["packages"]:
    downloads_updated_at = datetime.fromisoformat(p["downloads_updated_at"])

    if downloads_updated_at > yesterday:
        print(f"done: {p['name']}: {p['downloads']}")
        continue

    p["downloads"] = _get_downloads(p)
    p['downloads_updated_at'] = current_datetime.isoformat()
    with open(PACKAGE_YML, "w") as f:
        yaml.dump(data, f)
    print(f"{p['name']}: {p['downloads']}")


with open(PACKAGE_YML, "w") as f:
    yaml.dump(data, f)
