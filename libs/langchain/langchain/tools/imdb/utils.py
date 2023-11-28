from typing import Dict, List


def movies_to_dicts(movies: List) -> List[Dict]:
    return [
        {"title": m.get("long imdb title"), "id": m.getID()}
        for m in movies
        if m.getID()
    ]


def people_to_dicts(people: List) -> List[Dict]:
    return [
        {"name": p.get("long imdb name"), "id": p.getID()} for p in people if p.getID()
    ]


def companies_to_strs(companies: List) -> List[str]:
    return [company.get("name") for company in companies if "name" in company]
