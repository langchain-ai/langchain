"""Persian number conversion utilities."""
from typing import Dict


class PersianNumberConverter:
    """Converter for Persian numbers and digits."""
    
    # Mapping between Persian and English digits
    PERSIAN_TO_ENGLISH: Dict[str, str] = {
        '۰': '0',
        '۱': '1',
        '۲': '2',
        '۳': '3',
        '۴': '4',
        '۵': '5',
        '۶': '6',
        '۷': '7',
        '۸': '8',
        '۹': '9',
    }
    
    ENGLISH_TO_PERSIAN: Dict[str, str] = {v: k for k, v in PERSIAN_TO_ENGLISH.items()}
    
    # Arabic digits (also used in Persian)
    ARABIC_TO_ENGLISH: Dict[str, str] = {
        '٠': '0',
        '١': '1',
        '٢': '2',
        '٣': '3',
        '٤': '4',
        '٥': '5',
        '٦': '6',
        '٧': '7',
        '٨': '8',
        '٩': '9',
    }
    
    def to_persian(self, text: str) -> str:
        """Convert English digits in text to Persian digits.
        
        Args:
            text: Text containing numbers to convert
            
        Returns:
            Text with Persian digits
        """
        for english, persian in self.ENGLISH_TO_PERSIAN.items():
            text = text.replace(english, persian)
        return text
    
    def to_english(self, text: str) -> str:
        """Convert Persian/Arabic digits in text to English digits.
        
        Args:
            text: Text containing numbers to convert
            
        Returns:
            Text with English digits
        """
        # First convert Arabic digits
        for arabic, english in self.ARABIC_TO_ENGLISH.items():
            text = text.replace(arabic, english)
            
        # Then convert Persian digits
        for persian, english in self.PERSIAN_TO_ENGLISH.items():
            text = text.replace(persian, english)
            
        return text
    
    @staticmethod
    def is_persian_digit(char: str) -> bool:
        """Check if a character is a Persian digit.
        
        Args:
            char: Character to check
            
        Returns:
            True if character is a Persian digit
        """
        return '\u06F0' <= char <= '\u06F9'
    
    @staticmethod
    def is_arabic_digit(char: str) -> bool:
        """Check if a character is an Arabic digit.
        
        Args:
            char: Character to check
            
        Returns:
            True if character is an Arabic digit
        """
        return '\u0660' <= char <= '\u0669' 