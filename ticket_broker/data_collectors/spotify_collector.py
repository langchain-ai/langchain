"""
Spotify Data Collector for Artist Analysis
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from ..config.settings import settings


logger = logging.getLogger(__name__)


class SpotifyCollector:
    """Collects artist data from Spotify API"""
    
    def __init__(self):
        self.client_id = settings.spotify_client_id
        self.client_secret = settings.spotify_client_secret
        self.sp = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Spotify client with credentials"""
        if not self.client_id or not self.client_secret:
            logger.warning("Spotify credentials not provided. Spotify data collection disabled.")
            return
        
        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            logger.info("Spotify client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Spotify client: {e}")
    
    def search_artist(self, artist_name: str) -> Optional[Dict[str, Any]]:
        """Search for artist and return basic information"""
        if not self.sp:
            return None
        
        try:
            results = self.sp.search(q=artist_name, type='artist', limit=1)
            if results['artists']['items']:
                artist = results['artists']['items'][0]
                return {
                    'spotify_id': artist['id'],
                    'name': artist['name'],
                    'popularity': artist['popularity'],
                    'follower_count': artist['followers']['total'],
                    'genres': artist['genres'],
                    'external_urls': artist['external_urls']
                }
        except Exception as e:
            logger.error(f"Error searching for artist {artist_name}: {e}")
        
        return None
    
    def get_artist_details(self, artist_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed artist information by Spotify ID"""
        if not self.sp:
            return None
        
        try:
            artist = self.sp.artist(artist_id)
            albums = self.sp.artist_albums(artist_id, album_type='album', limit=50)
            top_tracks = self.sp.artist_top_tracks(artist_id)
            
            # Calculate career metrics
            latest_album = None
            if albums['items']:
                latest_album = max(albums['items'], key=lambda x: x['release_date'])
            
            # Analyze track popularity trend
            track_popularities = [track['popularity'] for track in top_tracks['tracks']]
            avg_track_popularity = sum(track_popularities) / len(track_popularities) if track_popularities else 0
            
            return {
                'spotify_id': artist['id'],
                'name': artist['name'],
                'popularity': artist['popularity'],
                'follower_count': artist['followers']['total'],
                'genres': artist['genres'],
                'latest_album': {
                    'name': latest_album['name'] if latest_album else None,
                    'release_date': latest_album['release_date'] if latest_album else None,
                    'type': latest_album['album_type'] if latest_album else None
                } if latest_album else None,
                'total_albums': len(albums['items']),
                'avg_track_popularity': avg_track_popularity,
                'top_tracks_count': len(top_tracks['tracks']),
                'monthly_listeners': None,  # Not available in API
                'last_updated': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting artist details for {artist_id}: {e}")
        
        return None
    
    def get_artist_albums(self, artist_id: str, album_type: str = 'album') -> List[Dict[str, Any]]:
        """Get artist's albums with release dates"""
        if not self.sp:
            return []
        
        try:
            albums = self.sp.artist_albums(artist_id, album_type=album_type, limit=50)
            album_list = []
            
            for album in albums['items']:
                album_list.append({
                    'id': album['id'],
                    'name': album['name'],
                    'release_date': album['release_date'],
                    'album_type': album['album_type'],
                    'total_tracks': album['total_tracks'],
                    'external_urls': album['external_urls']
                })
            
            return sorted(album_list, key=lambda x: x['release_date'], reverse=True)
        except Exception as e:
            logger.error(f"Error getting albums for artist {artist_id}: {e}")
        
        return []
    
    def analyze_popularity_trend(self, artist_id: str) -> Dict[str, Any]:
        """Analyze artist's popularity trend based on releases and tracks"""
        if not self.sp:
            return {}
        
        try:
            # Get albums from last 2 years
            albums = self.sp.artist_albums(artist_id, album_type='album,single', limit=50)
            recent_albums = []
            
            for album in albums['items']:
                release_date = datetime.strptime(album['release_date'], '%Y-%m-%d' if len(album['release_date']) == 10 else '%Y')
                if release_date > datetime.now() - timedelta(days=730):  # Last 2 years
                    recent_albums.append(album)
            
            # Get top tracks
            top_tracks = self.sp.artist_top_tracks(artist_id)
            
            # Calculate trend indicators
            recent_release_count = len(recent_albums)
            avg_track_popularity = sum(track['popularity'] for track in top_tracks['tracks']) / len(top_tracks['tracks']) if top_tracks['tracks'] else 0
            
            # Determine trend
            trend = "stable"
            if recent_release_count >= 2 and avg_track_popularity > 70:
                trend = "rising"
            elif avg_track_popularity < 40 or recent_release_count == 0:
                trend = "declining"
            
            return {
                'trend': trend,
                'recent_releases_count': recent_release_count,
                'avg_track_popularity': avg_track_popularity,
                'latest_release_date': recent_albums[0]['release_date'] if recent_albums else None,
                'analysis_date': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing popularity trend for {artist_id}: {e}")
        
        return {}
    
    def check_new_releases(self, artist_id: str, months_back: int = 6) -> bool:
        """Check if artist has released new content in the specified timeframe"""
        if not self.sp:
            return False
        
        try:
            albums = self.sp.artist_albums(artist_id, album_type='album,single', limit=20)
            cutoff_date = datetime.now() - timedelta(days=months_back * 30)
            
            for album in albums['items']:
                release_date = datetime.strptime(album['release_date'], '%Y-%m-%d' if len(album['release_date']) == 10 else '%Y')
                if release_date > cutoff_date:
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking new releases for {artist_id}: {e}")
        
        return False
    
    def get_artist_genre_classification(self, artist_id: str) -> Dict[str, Any]:
        """Classify artist by genre and market appeal"""
        if not self.sp:
            return {}
        
        try:
            artist = self.sp.artist(artist_id)
            genres = artist['genres']
            
            # Genre classification for market analysis
            mainstream_genres = ['pop', 'rock', 'hip hop', 'country', 'r&b']
            niche_genres = ['jazz', 'classical', 'folk', 'indie', 'electronic']
            
            genre_classification = "other"
            market_appeal = "niche"
            
            for genre in genres:
                for mainstream in mainstream_genres:
                    if mainstream in genre.lower():
                        genre_classification = mainstream
                        market_appeal = "mainstream"
                        break
                
                if market_appeal == "mainstream":
                    break
                
                for niche in niche_genres:
                    if niche in genre.lower():
                        genre_classification = niche
                        market_appeal = "niche"
                        break
            
            return {
                'primary_genre': genre_classification,
                'all_genres': genres,
                'market_appeal': market_appeal,
                'popularity_score': artist['popularity']
            }
        except Exception as e:
            logger.error(f"Error classifying artist genre for {artist_id}: {e}")
        
        return {}
    
    def batch_artist_lookup(self, artist_names: List[str]) -> Dict[str, Any]:
        """Lookup multiple artists efficiently"""
        if not self.sp:
            return {}
        
        results = {}
        
        for artist_name in artist_names:
            try:
                artist_data = self.search_artist(artist_name)
                if artist_data:
                    detailed_data = self.get_artist_details(artist_data['spotify_id'])
                    if detailed_data:
                        results[artist_name] = detailed_data
                else:
                    logger.warning(f"Artist not found on Spotify: {artist_name}")
            except Exception as e:
                logger.error(f"Error in batch lookup for {artist_name}: {e}")
        
        return results