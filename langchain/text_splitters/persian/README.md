# Persian Language Processing

This module provides comprehensive Persian language processing utilities for LangChain, including text tokenization, normalization, and number system conversion.

## Features

- **PersianTokenizer**: Tokenizes Persian text with support for:
  - Persian word boundaries
  - ZWNJ (نیم‌فاصله) character handling
  - Mixed Persian-English text
  - Proper token joining

- **PersianTextNormalizer**: Normalizes Persian text with:
  - Character normalization (ی/ي، ک/ك)
  - Diacritics (اِعراب) removal
  - Space normalization
  - Hamza form normalization

- **PersianNumberConverter**: Converts between:
  - Persian digits (۰-۹)
  - Arabic digits (٠-٩)
  - English digits (0-9)

## Installation

The Persian language processing utilities are included in the main LangChain package.

## Usage

```python
from langchain.text_splitters.persian import (
    PersianTokenizer,
    PersianTextNormalizer,
    PersianNumberConverter
)

# Initialize components
tokenizer = PersianTokenizer(handle_zwnj=True)
normalizer = PersianTextNormalizer(remove_diacritics=True)
number_converter = PersianNumberConverter()

# Process Persian text
text = "این متن شامل عدد ۱۲۳ و کلمه‌ی ترکیبی است"

# Normalize text
normalized = normalizer.normalize(text)

# Tokenize text
tokens = tokenizer.tokenize(normalized)

# Convert numbers
english_numbers = number_converter.to_english(text)
persian_numbers = number_converter.to_persian("Text with number 123")
```

## Examples

### Text Tokenization
```python
from langchain.text_splitters.persian import PersianTokenizer

tokenizer = PersianTokenizer()
text = "سلام دنیا"
tokens = tokenizer.tokenize(text)  # ["سلام", "دنیا"]
```

### Text Normalization
```python
from langchain.text_splitters.persian import PersianTextNormalizer

normalizer = PersianTextNormalizer()
text = "كتاب"  # With Arabic kaf
normalized = normalizer.normalize(text)  # "کتاب" with Persian kaf
```

### Number Conversion
```python
from langchain.text_splitters.persian import PersianNumberConverter

converter = PersianNumberConverter()
text = "۱۲۳۴۵"
english = converter.to_english(text)  # "12345"
persian = converter.to_persian("12345")  # "۱۲۳۴۵"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 