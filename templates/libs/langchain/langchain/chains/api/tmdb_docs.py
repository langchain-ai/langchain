# flake8: noqa
TMDB_DOCS = """API documentation:
Endpoint: https://api.themoviedb.org/3
GET /search/movie

This API is for searching movies.

Query parameters table:
language | string | Pass a ISO 639-1 value to display translated data for the fields that support it. minLength: 2, pattern: ([a-z]{2})-([A-Z]{2}), default: en-US | optional
query | string | Pass a text query to search. This value should be URI encoded. minLength: 1 | required
page | integer | Specify which page to query. minimum: 1, maximum: 1000, default: 1 | optional
include_adult | boolean | Choose whether to include adult (pornography) content in the results. default | optional
region | string | Specify a ISO 3166-1 code to filter release dates. Must be uppercase. pattern: ^[A-Z]{2}$ | optional
year | integer  | optional
primary_release_year | integer | optional

Response schema (JSON object):
page | integer | optional
total_results | integer | optional
total_pages | integer | optional
results | array[object] (Movie List Result Object)

Each object in the "results" key has the following schema:
poster_path | string or null | optional
adult | boolean | optional
overview | string | optional
release_date | string | optional
genre_ids | array[integer] | optional
id | integer | optional
original_title | string | optional
original_language | string | optional
title | string | optional
backdrop_path | string or null | optional
popularity | number | optional
vote_count | integer | optional
video | boolean | optional
vote_average | number | optional"""
