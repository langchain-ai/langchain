"""Persian text normalization utilities."""
from typing import Dict


class PersianTextNormalizer:
    """Normalizer for Persian text."""
    
    # Character replacement mappings
    CHAR_REPLACEMENTS: Dict[str, str] = {
        # Normalize Persian Ye
        'ي': 'ی',
        'ئ': 'ی',
        
        # Normalize Persian Kaf
        'ك': 'ک',
        
        # Normalize Persian numbers
        '٠': '۰',
        '١': '۱',
        '٢': '۲',
        '٣': '۳',
        '٤': '۴',
        '٥': '۵',
        '٦': '۶',
        '٧': '۷',
        '٨': '۸',
        '٩': '۹',
        
        # Normalize spaces
        '\u200B': '',  # Zero-width space
        '\u200D': '',  # Zero-width joiner
        '\u2003': ' ', # Em space
        '\u2002': ' ', # En space
        '\u00A0': ' ', # Non-breaking space
        
        # Normalize various forms of Hamza
        'أ': 'ا',
        'إ': 'ا',
        'ٱ': 'ا',
        'ة': 'ه',
        'ؤ': 'و',
        'ء': '',
    }
    
    # Diacritics to remove
    DIACRITICS = ''.join([
        '\u064B',  # Fathatan
        '\u064C',  # Dammatan
        '\u064D',  # Kasratan
        '\u064E',  # Fatha
        '\u064F',  # Damma
        '\u0650',  # Kasra
        '\u0651',  # Shadda
        '\u0652',  # Sukun
        '\u0653',  # Maddah Above
        '\u0654',  # Hamza Above
        '\u0655',  # Hamza Below
    ])
    
    def __init__(self, remove_diacritics: bool = True):
        """Initialize the normalizer.
        
        Args:
            remove_diacritics: Whether to remove diacritical marks
        """
        self.remove_diacritics = remove_diacritics
        
    def normalize(self, text: str) -> str:
        """Normalize Persian text by applying character replacements and cleaning.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        # Replace characters according to mapping
        for old, new in self.CHAR_REPLACEMENTS.items():
            text = text.replace(old, new)
            
        # Remove diacritics if enabled
        if self.remove_diacritics:
            text = self.remove_diacritical_marks(text)
            
        # Normalize whitespace
        text = ' '.join(text.split())
            
        return text
    
    def remove_diacritical_marks(self, text: str) -> str:
        """Remove diacritical marks from text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with diacritical marks removed
        """
        return text.translate(str.maketrans('', '', self.DIACRITICS))
    
    @staticmethod
    def is_diacritical_mark(char: str) -> bool:
        """Check if a character is a diacritical mark.
        
        Args:
            char: Character to check
            
        Returns:
            True if character is a diacritical mark
        """
        return '\u064B' <= char <= '\u0655' 