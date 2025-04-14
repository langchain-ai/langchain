"""
Tests for Persian text processing components.
"""
import unittest
from .tokenizer import PersianTokenizer
from .normalizer import PersianTextNormalizer
from .numbers import PersianNumberConverter


class TestPersianTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = PersianTokenizer()
        
    def test_basic_tokenization(self):
        text = "سلام دنیا"
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(tokens, ["سلام", "دنیا"])
        
    def test_mixed_text(self):
        text = "این یک text ترکیبی است"
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(tokens, ["این", "یک", "text", "ترکیبی", "است"])
        
    def test_zwnj_handling(self):
        text = "می‌خواهم"  # Text with ZWNJ
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(tokens, ["می‌خواهم"])
        
    def test_join_tokens(self):
        tokens = ["این", "یک", "متن", "است"]
        text = self.tokenizer.join_tokens(tokens)
        self.assertEqual(text, "این یک متن است")


class TestPersianTextNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = PersianTextNormalizer()
        
    def test_character_normalization(self):
        text = "كتاب"  # With Arabic kaf
        normalized = self.normalizer.normalize(text)
        self.assertEqual(normalized, "کتاب")  # With Persian kaf
        
    def test_diacritic_removal(self):
        text = "کِتاب"  # With kasra
        normalized = self.normalizer.normalize(text)
        self.assertEqual(normalized, "کتاب")
        
    def test_space_normalization(self):
        text = "این  متن    دارای    فاصله    است"
        normalized = self.normalizer.normalize(text)
        self.assertEqual(normalized, "این متن دارای فاصله است")


class TestPersianNumberConverter(unittest.TestCase):
    def setUp(self):
        self.converter = PersianNumberConverter()
        
    def test_persian_to_english(self):
        text = "۱۲۳۴۵"
        converted = self.converter.to_english(text)
        self.assertEqual(converted, "12345")
        
    def test_english_to_persian(self):
        text = "12345"
        converted = self.converter.to_persian(text)
        self.assertEqual(converted, "۱۲۳۴۵")
        
    def test_mixed_numbers(self):
        text = "The number ۱۲۳ is equal to 123"
        converted = self.converter.to_persian(text)
        self.assertEqual(converted, "The number ۱۲۳ is equal to ۱۲۳")
        
    def test_arabic_numbers(self):
        text = "١٢٣٤٥"
        converted = self.converter.to_english(text)
        self.assertEqual(converted, "12345")


if __name__ == '__main__':
    unittest.main() 