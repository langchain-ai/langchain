def movies_to_dicts(movies):
    return [
        {'title': m.get('long imdb title'), 'id': m.getID()}
        for m in movies
        if m.getID()
    ]

def people_to_dicts(people):
    return [
        {'name': p.get('long imdb name'), 'id': p.getID()}
        for p in people
        if p.getID()
    ]

def companies_to_strs(companies):
    return [company.get('name') for company in companies]