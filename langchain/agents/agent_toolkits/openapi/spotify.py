"""Throwaway code for experimenting with Spotify's 100k+ token length spec.
"""
import os, yaml, subprocess
import spotipy.util as util


SPOTIFY_API_SPEC_URL = "https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/spotify.com/1.0.0/openapi.yaml"


def fetch_spotify_api_spec():    
    if not os.path.exists('./openapi.yaml'):
        subprocess.run(["wget", SPOTIFY_API_SPEC_URL])
    with open("./openapi.yaml", "r") as f:
        spec = yaml.load(f, Loader=yaml.Loader)
    return spec

def construct_spotify_auth_headers(raw_spec: dict):
    scopes = list(raw_spec['components']['securitySchemes']['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
    access_token = util.prompt_for_user_token(scope=','.join(scopes))
    return {
        'Authorization': f'Bearer {access_token}'
    }
