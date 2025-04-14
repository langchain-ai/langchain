"""
Persian text tokenization utilities.
"""
import re
from typing import List, Optional, Pattern


class PersianTokenizer:
    """A tokenizer specifically designed for Persian text."""

    # Persian word boundary pattern
    WORD_PATTERN: Pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+|[A-Za-z0-9]+|\s+')
    
    # ZWNJ character (نیم‌فاصله)
    ZWNJ: str = '\u200C'
    
    # Common Persian punctuation marks
    PERSIAN_PUNCTUATION: str = '،؛؟»«'
    
    def __init__(self, preserve_whitespace: bool = False, handle_zwnj: bool = True):
        """
        Initialize the Persian tokenizer.
        
        Args:
            preserve_whitespace: Whether to keep whitespace tokens
            handle_zwnj: Whether to handle Zero-Width Non-Joiner character specially
        """
        self.preserve_whitespace = preserve_whitespace
        self.handle_zwnj = handle_zwnj

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Persian text into words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []

        # Split text into tokens using word boundary pattern
        tokens = self.WORD_PATTERN.findall(text)
        
        # Filter out empty strings and handle whitespace
        if not self.preserve_whitespace:
            tokens = [t for t in tokens if t.strip()]
        
        # Handle ZWNJ if enabled
        if self.handle_zwnj:
            tokens = self._handle_zwnj_tokens(tokens)
            
        return tokens

    def _handle_zwnj_tokens(self, tokens: List[str]) -> List[str]:
        """
        Handle tokens containing ZWNJ character.
        
        Args:
            tokens: List of tokens to process
            
        Returns:
            Processed list of tokens
        """
        result = []
        for token in tokens:
            if self.ZWNJ in token:
                # Keep ZWNJ-connected parts together
                parts = token.split(self.ZWNJ)
                # Rejoin parts with ZWNJ if they're Persian words
                if all(re.match(r'[\u0600-\u06FF]+', part) for part in parts if part):
                    result.append(token)
                else:
                    # Split on ZWNJ if parts aren't all Persian
                    result.extend(parts)
            else:
                result.append(token)
        return result

    def join_tokens(self, tokens: List[str]) -> str:
        """
        Join tokens back into text, handling ZWNJ and spaces properly.
        
        Args:
            tokens: List of tokens to join
            
        Returns:
            Joined text
        """
        if not tokens:
            return ""
            
        result = []
        for i, token in enumerate(tokens):
            if i > 0:
                # Add space between tokens unless they're connected by ZWNJ
                if not (token.startswith(self.ZWNJ) or tokens[i-1].endswith(self.ZWNJ)):
                    result.append(" ")
            result.append(token)
            
        return "".join(result)

    @staticmethod
    def is_persian_word(text: str) -> bool:
        """
        Check if text contains Persian characters.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains Persian characters
        """
        return bool(re.search(r'[\u0600-\u06FF]', text)) 