"""
ShowsOnSale.com Data Collector for Tour and Event Information
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import urljoin, urlparse, parse_qs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from ..config.settings import settings


logger = logging.getLogger(__name__)


class ShowsOnSaleCollector:
    """Collects tour and event data from ShowsOnSale.com"""
    
    def __init__(self):
        self.base_url = "https://www.showsonsale.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': settings.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.selenium_driver = None
    
    def _init_selenium(self):
        """Initialize Selenium WebDriver for dynamic content"""
        if self.selenium_driver is None:
            try:
                chrome_options = Options()
                chrome_options.add_argument('--headless')
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument(f'--user-agent={settings.user_agent}')
                
                self.selenium_driver = webdriver.Chrome(options=chrome_options)
                logger.info("Selenium WebDriver initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Selenium WebDriver: {e}")
    
    def _close_selenium(self):
        """Close Selenium WebDriver"""
        if self.selenium_driver:
            self.selenium_driver.quit()
            self.selenium_driver = None
    
    def search_artist_tours(self, artist_name: str) -> List[Dict[str, Any]]:
        """Search for artist tours and upcoming shows"""
        try:
            search_url = f"{self.base_url}/search"
            params = {'q': artist_name}
            
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            shows = []
            
            # Parse search results for show listings
            show_elements = soup.find_all('div', class_=['show-item', 'event-item', 'concert-listing'])
            
            for show_element in show_elements:
                show_data = self._parse_show_element(show_element, artist_name)
                if show_data:
                    shows.append(show_data)
            
            return shows
            
        except Exception as e:
            logger.error(f"Error searching for {artist_name} tours: {e}")
            return []
    
    def _parse_show_element(self, element, artist_name: str) -> Optional[Dict[str, Any]]:
        """Parse individual show element from search results"""
        try:
            show_data = {
                'artist_name': artist_name,
                'event_title': None,
                'venue_name': None,
                'venue_city': None,
                'venue_state': None,
                'event_date': None,
                'on_sale_date': None,
                'presale_date': None,
                'ticket_url': None,
                'price_range': None,
                'venue_capacity': None,
                'tour_name': None,
                'special_notes': None,
                'scraped_at': datetime.utcnow().isoformat()
            }
            
            # Extract event title
            title_elem = element.find(['h2', 'h3', 'a'], class_=['title', 'event-title', 'show-title'])
            if title_elem:
                show_data['event_title'] = title_elem.get_text(strip=True)
            
            # Extract venue information
            venue_elem = element.find(['div', 'span'], class_=['venue', 'location', 'venue-name'])
            if venue_elem:
                venue_text = venue_elem.get_text(strip=True)
                # Parse venue name and location
                venue_parts = venue_text.split(',')
                if len(venue_parts) >= 2:
                    show_data['venue_name'] = venue_parts[0].strip()
                    location_parts = venue_parts[1].strip().split(' ')
                    if len(location_parts) >= 2:
                        show_data['venue_state'] = location_parts[-1].strip()
                        show_data['venue_city'] = ' '.join(location_parts[:-1]).strip()
            
            # Extract dates
            date_elem = element.find(['div', 'span'], class_=['date', 'event-date', 'show-date'])
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                show_data['event_date'] = self._parse_date(date_text)
            
            # Extract on-sale information
            onsale_elem = element.find(['div', 'span'], class_=['onsale', 'on-sale', 'sale-date'])
            if onsale_elem:
                onsale_text = onsale_elem.get_text(strip=True)
                show_data['on_sale_date'] = self._parse_date(onsale_text)
            
            # Extract ticket URL
            link_elem = element.find('a', href=True)
            if link_elem:
                show_data['ticket_url'] = urljoin(self.base_url, link_elem['href'])
            
            # Extract price information
            price_elem = element.find(['div', 'span'], class_=['price', 'ticket-price', 'cost'])
            if price_elem:
                price_text = price_elem.get_text(strip=True)
                show_data['price_range'] = self._parse_price_range(price_text)
            
            return show_data
            
        except Exception as e:
            logger.error(f"Error parsing show element: {e}")
            return None
    
    def get_detailed_show_info(self, show_url: str) -> Dict[str, Any]:
        """Get detailed information for a specific show"""
        try:
            response = self.session.get(show_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            detailed_info = {
                'url': show_url,
                'venue_details': {},
                'ticket_info': {},
                'presale_info': {},
                'tour_info': {},
                'special_restrictions': [],
                'scraped_at': datetime.utcnow().isoformat()
            }
            
            # Extract venue capacity and details
            venue_section = soup.find('div', class_=['venue-info', 'venue-details'])
            if venue_section:
                capacity_elem = venue_section.find(text=re.compile(r'capacity|seats', re.I))
                if capacity_elem:
                    capacity_match = re.search(r'(\d+,?\d*)', capacity_elem)
                    if capacity_match:
                        detailed_info['venue_details']['capacity'] = int(capacity_match.group(1).replace(',', ''))
                
                # Extract venue type
                venue_type_elem = venue_section.find(text=re.compile(r'arena|stadium|theater|amphitheater|hall', re.I))
                if venue_type_elem:
                    detailed_info['venue_details']['type'] = venue_type_elem.strip().lower()
            
            # Extract presale information
            presale_section = soup.find('div', class_=['presale', 'pre-sale'])
            if presale_section:
                presale_dates = presale_section.find_all(text=re.compile(r'\d{1,2}/\d{1,2}/\d{4}'))
                detailed_info['presale_info']['dates'] = [self._parse_date(date) for date in presale_dates]
                
                presale_codes = presale_section.find_all('code')
                detailed_info['presale_info']['codes'] = [code.get_text(strip=True) for code in presale_codes]
            
            # Extract ticket information
            ticket_section = soup.find('div', class_=['ticket-info', 'pricing'])
            if ticket_section:
                price_elements = ticket_section.find_all(text=re.compile(r'\$\d+'))
                prices = []
                for price_text in price_elements:
                    price_match = re.search(r'\$(\d+(?:\.\d{2})?)', price_text)
                    if price_match:
                        prices.append(float(price_match.group(1)))
                
                if prices:
                    detailed_info['ticket_info']['price_min'] = min(prices)
                    detailed_info['ticket_info']['price_max'] = max(prices)
            
            # Extract tour information
            tour_section = soup.find('div', class_=['tour', 'tour-info'])
            if tour_section:
                tour_name_elem = tour_section.find(['h1', 'h2', 'h3'])
                if tour_name_elem:
                    detailed_info['tour_info']['name'] = tour_name_elem.get_text(strip=True)
            
            # Extract special notes or restrictions
            restrictions_section = soup.find('div', class_=['restrictions', 'notes', 'important'])
            if restrictions_section:
                restriction_items = restrictions_section.find_all(['li', 'p'])
                detailed_info['special_restrictions'] = [item.get_text(strip=True) for item in restriction_items]
            
            return detailed_info
            
        except Exception as e:
            logger.error(f"Error getting detailed show info from {show_url}: {e}")
            return {}
    
    def get_upcoming_onsales(self, days_ahead: int = 30) -> List[Dict[str, Any]]:
        """Get upcoming on-sale dates for all shows"""
        try:
            # Use Selenium for dynamic content
            self._init_selenium()
            
            onsale_url = f"{self.base_url}/upcoming-onsales"
            self.selenium_driver.get(onsale_url)
            
            # Wait for content to load
            WebDriverWait(self.selenium_driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "onsale-item"))
            )
            
            soup = BeautifulSoup(self.selenium_driver.page_source, 'html.parser')
            onsales = []
            
            onsale_elements = soup.find_all('div', class_=['onsale-item', 'upcoming-sale'])
            
            for element in onsale_elements:
                onsale_data = self._parse_onsale_element(element)
                if onsale_data:
                    # Filter by date range
                    onsale_date = onsale_data.get('on_sale_date')
                    if onsale_date:
                        try:
                            onsale_datetime = datetime.fromisoformat(onsale_date.replace('Z', '+00:00'))
                            if onsale_datetime <= datetime.utcnow() + timedelta(days=days_ahead):
                                onsales.append(onsale_data)
                        except:
                            onsales.append(onsale_data)  # Include if date parsing fails
            
            return onsales
            
        except Exception as e:
            logger.error(f"Error getting upcoming on-sales: {e}")
            return []
        finally:
            self._close_selenium()
    
    def _parse_onsale_element(self, element) -> Optional[Dict[str, Any]]:
        """Parse individual on-sale element"""
        try:
            onsale_data = {
                'artist_name': None,
                'event_title': None,
                'venue_name': None,
                'venue_city': None,
                'venue_state': None,
                'event_date': None,
                'on_sale_date': None,
                'on_sale_time': None,
                'ticket_source': None,
                'is_presale': False,
                'presale_code_required': False,
                'scraped_at': datetime.utcnow().isoformat()
            }
            
            # Extract artist/event title
            title_elem = element.find(['h2', 'h3', 'a'], class_=['title', 'artist', 'event-name'])
            if title_elem:
                title_text = title_elem.get_text(strip=True)
                onsale_data['artist_name'] = title_text.split(' - ')[0] if ' - ' in title_text else title_text
                onsale_data['event_title'] = title_text
            
            # Extract on-sale date and time
            onsale_elem = element.find(['div', 'span'], class_=['onsale-date', 'sale-time'])
            if onsale_elem:
                onsale_text = onsale_elem.get_text(strip=True)
                onsale_data['on_sale_date'] = self._parse_date(onsale_text)
                
                # Extract time if present
                time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM))', onsale_text, re.I)
                if time_match:
                    onsale_data['on_sale_time'] = time_match.group(1)
            
            # Extract venue information
            venue_elem = element.find(['div', 'span'], class_=['venue', 'location'])
            if venue_elem:
                venue_text = venue_elem.get_text(strip=True)
                venue_parts = venue_text.split(',')
                if len(venue_parts) >= 2:
                    onsale_data['venue_name'] = venue_parts[0].strip()
                    location = venue_parts[1].strip()
                    location_parts = location.split(' ')
                    if len(location_parts) >= 2:
                        onsale_data['venue_state'] = location_parts[-1].strip()
                        onsale_data['venue_city'] = ' '.join(location_parts[:-1]).strip()
            
            # Check for presale indicators
            presale_indicators = element.find_all(text=re.compile(r'presale|pre-sale|fan club|vip', re.I))
            if presale_indicators:
                onsale_data['is_presale'] = True
                
                code_required = element.find_all(text=re.compile(r'code|password', re.I))
                onsale_data['presale_code_required'] = bool(code_required)
            
            # Extract ticket source
            source_elem = element.find('a', href=True)
            if source_elem:
                href = source_elem['href']
                if 'ticketmaster' in href.lower():
                    onsale_data['ticket_source'] = 'ticketmaster'
                elif 'stubhub' in href.lower():
                    onsale_data['ticket_source'] = 'stubhub'
                elif 'seatgeek' in href.lower():
                    onsale_data['ticket_source'] = 'seatgeek'
                else:
                    parsed_url = urlparse(href)
                    onsale_data['ticket_source'] = parsed_url.netloc
            
            return onsale_data
            
        except Exception as e:
            logger.error(f"Error parsing on-sale element: {e}")
            return None
    
    def get_venue_information(self, venue_name: str, city: str = None) -> Dict[str, Any]:
        """Get detailed venue information"""
        try:
            search_query = f"{venue_name}"
            if city:
                search_query += f" {city}"
            
            search_url = f"{self.base_url}/venues/search"
            params = {'q': search_query}
            
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            venue_info = {
                'name': venue_name,
                'city': city,
                'capacity': None,
                'venue_type': None,
                'address': None,
                'parking_info': None,
                'public_transport': None,
                'nearby_venues': [],
                'typical_genres': [],
                'scraped_at': datetime.utcnow().isoformat()
            }
            
            # Parse venue details from search results
            venue_element = soup.find('div', class_=['venue-result', 'venue-info'])
            if venue_element:
                # Extract capacity
                capacity_text = venue_element.find(text=re.compile(r'capacity|seats', re.I))
                if capacity_text:
                    capacity_match = re.search(r'(\d+,?\d*)', capacity_text)
                    if capacity_match:
                        venue_info['capacity'] = int(capacity_match.group(1).replace(',', ''))
                
                # Extract venue type
                type_elem = venue_element.find(text=re.compile(r'arena|stadium|theater|amphitheater|hall|club', re.I))
                if type_elem:
                    venue_info['venue_type'] = type_elem.strip().lower()
                
                # Extract address
                address_elem = venue_element.find('div', class_=['address', 'location'])
                if address_elem:
                    venue_info['address'] = address_elem.get_text(strip=True)
            
            return venue_info
            
        except Exception as e:
            logger.error(f"Error getting venue information for {venue_name}: {e}")
            return {}
    
    def get_market_trends(self, city: str = None, genre: str = None) -> Dict[str, Any]:
        """Get market trends for specific city or genre"""
        try:
            trends_data = {
                'city': city,
                'genre': genre,
                'hot_artists': [],
                'upcoming_major_events': [],
                'market_saturation': 'unknown',
                'average_ticket_prices': {},
                'popular_venues': [],
                'seasonal_patterns': {},
                'scraped_at': datetime.utcnow().isoformat()
            }
            
            # Build search parameters
            params = {}
            if city:
                params['city'] = city
            if genre:
                params['genre'] = genre
            
            trends_url = f"{self.base_url}/trends"
            response = self.session.get(trends_url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse trending artists
            trending_section = soup.find('div', class_=['trending', 'hot-artists'])
            if trending_section:
                artist_elements = trending_section.find_all('a', class_=['artist-link', 'trending-artist'])
                trends_data['hot_artists'] = [elem.get_text(strip=True) for elem in artist_elements[:10]]
            
            # Parse upcoming major events
            major_events_section = soup.find('div', class_=['major-events', 'featured-shows'])
            if major_events_section:
                event_elements = major_events_section.find_all('div', class_=['event-item', 'major-event'])
                for event_elem in event_elements[:5]:
                    event_title = event_elem.find(['h3', 'h4'])
                    if event_title:
                        trends_data['upcoming_major_events'].append(event_title.get_text(strip=True))
            
            return trends_data
            
        except Exception as e:
            logger.error(f"Error getting market trends: {e}")
            return {}
    
    def monitor_sellout_alerts(self, artist_names: List[str]) -> List[Dict[str, Any]]:
        """Monitor for shows that have sold out quickly"""
        try:
            alerts = []
            
            for artist_name in artist_names:
                shows = self.search_artist_tours(artist_name)
                
                for show in shows:
                    # Check if show details indicate recent sellout
                    if show.get('ticket_url'):
                        detailed_info = self.get_detailed_show_info(show['ticket_url'])
                        
                        # Look for sellout indicators in the page content
                        if self._check_sellout_indicators(detailed_info):
                            alert_data = {
                                'artist_name': artist_name,
                                'event_title': show.get('event_title'),
                                'venue_name': show.get('venue_name'),
                                'venue_city': show.get('venue_city'),
                                'event_date': show.get('event_date'),
                                'sellout_detected_at': datetime.utcnow().isoformat(),
                                'show_url': show.get('ticket_url')
                            }
                            alerts.append(alert_data)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring sellout alerts: {e}")
            return []
    
    def _check_sellout_indicators(self, detailed_info: Dict[str, Any]) -> bool:
        """Check if show appears to be sold out"""
        # Look for sellout keywords in special restrictions or notes
        restrictions = detailed_info.get('special_restrictions', [])
        for restriction in restrictions:
            if any(keyword in restriction.lower() for keyword in ['sold out', 'no tickets', 'unavailable', 'waitlist']):
                return True
        
        return False
    
    def _parse_date(self, date_text: str) -> Optional[str]:
        """Parse date string and return ISO format"""
        try:
            # Handle various date formats
            date_patterns = [
                r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
                r'(\d{4}-\d{2}-\d{2})',      # YYYY-MM-DD
                r'(\w+ \d{1,2}, \d{4})',     # Month DD, YYYY
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, date_text)
                if match:
                    date_str = match.group(1)
                    # Try to parse the date
                    try:
                        if '/' in date_str:
                            parsed_date = datetime.strptime(date_str, '%m/%d/%Y')
                        elif '-' in date_str:
                            parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
                        else:
                            parsed_date = datetime.strptime(date_str, '%B %d, %Y')
                        
                        return parsed_date.isoformat()
                    except ValueError:
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing date '{date_text}': {e}")
            return None
    
    def _parse_price_range(self, price_text: str) -> Optional[Dict[str, float]]:
        """Parse price range from text"""
        try:
            price_matches = re.findall(r'\$(\d+(?:\.\d{2})?)', price_text)
            if price_matches:
                prices = [float(price) for price in price_matches]
                return {
                    'min': min(prices),
                    'max': max(prices)
                }
            return None
        except Exception as e:
            logger.error(f"Error parsing price range '{price_text}': {e}")
            return None
    
    def __del__(self):
        """Cleanup Selenium driver on object destruction"""
        self._close_selenium()