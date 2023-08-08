# flake8: noqa
PODCAST_DOCS = """API documentation:
Endpoint: https://listen-api.listennotes.com/api/v2
GET /search

This API is for searching podcasts or episodes.

Query parameters table:
q | string | Search term, e.g., person, place, topic... You can use double quotes to do verbatim match, e.g., "game of thrones". Otherwise, it's fuzzy search. | required
type | string | What type of contents do you want to search for? Available values: episode, podcast, curated. default: episode | optional
page_size | integer | The maximum number of search results per page. A valid value should be an integer between 1 and 10 (inclusive). default: 3 | optional
language | string | Limit search results to a specific language, e.g., English, Chinese ... If not specified, it'll be any language. It works only when type is episode or podcast. | optional
region | string | Limit search results to a specific region (e.g., us, gb, in...). If not specified, it'll be any region. It works only when type is episode or podcast. | optional
len_min | integer | Minimum audio length in minutes. Applicable only when type parameter is episode or podcast. If type parameter is episode, it's for audio length of an episode. If type parameter is podcast, it's for average audio length of all episodes in a podcast. | optional
len_max | integer | Maximum audio length in minutes. Applicable only when type parameter is episode or podcast. If type parameter is episode, it's for audio length of an episode. If type parameter is podcast, it's for average audio length of all episodes in a podcast. | optional

Response schema (JSON object):
next_offset | integer | optional
total | integer | optional
results | array[object] (Episode / Podcast List Result Object)

Each object in the "results" key has the following schema:
listennotes_url | string | optional
id | integer | optional
title_highlighted | string | optional

Use page_size: 3
"""
