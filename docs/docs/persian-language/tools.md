# Persian Natural Processing Tools
## Table Of Contents
- [Part-of-Speech Tagger](#part-of-speech-tagger)
- [Language Detection](#language-detection)
- [Tokenization & Segmentation](#tokenization--segmentation)
- [Normalizer And Text Cleaner](#normalizer-and-text-cleaner)
- [Translator](#translator)
- [Transliterator](#transliterator)
- [Morphological Analysis](#morphological-analysis)
- [Stemmer](#stemmer)
- [Sentiment Analysis](#sentiment-analysis)
- [Spell Checking](#spell-checking)
- [Dependency Parser](#dependency-parser)
- [Shallow Parser](#shallow-parser)
- [Information Extraction](#information-extraction)
- [Text To Speech Preprocessing](#text-to-speech-preprocessing)
- [Text To Speech](#text-to-speech)
- [MISC](#misc)
- [Keyphrase Extractor](#keyphrase-extractor)
- [Speech Recognition](#speech-recognition)
- [Persian Phonemizer](#persian-phonemizer)

## Part-of-Speech Tagger
- [farsiNLPTools](https://github.com/wfeely/farsiNLPTools) - Open-source dependency parser, part-of-speech tagger, and text normalizer for Farsi (Persian).
- [Hazm](https://github.com/roshan-research/hazm) - Persian NLP Toolkit.
- [Persian Language Model for HunPoS](http://stp.lingfil.uu.se/~mojgan/tagper.html) - HunPoS (Halacsy et al, 2007) is an open source reimplementation of the statistical part-of-speech tagger Trigrams'n Tags, also called TnT (Brants, 2000) allowing the user to tune the tagger by using different feature settings.
- [Maryam Tavafi POS Tagger ](https://sites.google.com/site/maryamtavafi/persian-pos-tagger) - This software includes implementation of a Persian part of speech tagger based on Structured Support Vector Machines.
- [Perstem](https://github.com/jonsafari/perstem) - Perstem is a Persian (Farsi) stemmer, morphological analyzer, transliterator, and partial part-of-speech tagger. Inflexional morphemes are separated or removed from their stems. Perstem can also tokenize and transliterate between various character set encodings and romanizations.
- [Persianp Toolbox](http://www.persianp.ir/toolbox.html) - Multi-purpose persian NLP toolbox.
- [UM-wtlab pos tagger](http://wtlab.um.ac.ir/index.php?option=com_content&view=article&id=326&Itemid=224&lang=en) - This software is a C# implementation of the Viberbi and Brill part-of-speech taggers.
- [RDRPOSTagger](https://github.com/datquocnguyen/RDRPOSTagger) - Provides a pre-trained part-of-speech (POS) tagging model for Persian. This POS tagging toolkit is implemented in both Python and Java.
- [jPTDP](https://github.com/datquocnguyen/jPTDP) - Provides a pre-trained model for joint POS tagging and dependency parsing for Persian.
- [Parsivar](https://github.com/ICTRC/Parsivar) - A Language Processing Toolkit for Persian

## Language Detection
- [Google language detect (python port)](https://github.com/Mimino666/langdetect) - Light Weight language detector, its performance for persian is excellent.

## Tokenization & Segmentation
- [Hazm](https://github.com/roshan-research/hazm) - Persian NLP Toolkit.
- [polyglot](https://github.com/aboSamoor/polyglot) -  Natural language pipeline that supports massive multilingual applications (like lokenization (165 languages), language detection (196 languages), named entity recognition (40 languages), part of speech tagging (16 languages), sentiment analysis (136 languages), word embeddings (137 languages), morphological analysis (135 languages), transliteration (69 Languages)).
- [tok-tok](https://github.com/jonsafari/tok-tok) - Tok-tok is a fast, simple, multilingual tokenizer(single .pl file).
- [segmental](https://github.com/jonsafari/segmental) - You can train your model based on plain-text corpus for text segmentation by powerful deep learning platform.
- [Persian Sentence Segmenter and Tokenizer: SeTPer](http://stp.lingfil.uu.se/~mojgan/setper.html) - Regex based sentence segmenter.
- [Farsi-Verb-Tokenizer](https://github.com/mehdi-manshadi/Farsi-Verb-Tokenizer) - Tokenizes Farsi Verbs.
- [Parsivar](https://github.com/ICTRC/Parsivar) - A Language Processing Toolkit for Persian
- [ParsiAnalyzer](https://github.com/NarimanN2/ParsiAnalyzer) -  Persian Analyzer For Elasticsearch.
- [ParsiNorm](https://github.com/haraai/ParsiNorm) - Persain Text Pre-Proceesing Tool
- [Persian Tools](https://github.com/persian-tools/py-persian-tools) - An anthology of a variety of tools for the Persian language in Python

## Normalizer And Text Cleaner
- [Hazm](https://github.com/roshan-research/hazm) - Persian NLP Toolkit.
- [Persian Pre-processor: PrePer](http://stp.lingfil.uu.se/~mojgan/preper.html) - Another signle .pl tools that normals your persian text.
- [virastar](https://github.com/aziz/virastar) - Cleaning up Persian text!.replace double dash to ndash and triple dash to mdash, replace English numbers with their Persian equivalent, correct :;,.?! spacing (one space after and no space before), replace English percent sign to its Persian equivalent and many other normalization. Virastar is written by ruby and has [python port](https://github.com/JKhakpour/virastar.py).
- [Virastyar](http://www.virastyar.ir/development) - A collection of C# libraries for Persian text processing (Spell Checking, Purification, Punctuation Correction, Persian Character Standardization, Pinglish Conversion & ...)
- [Parsivar](https://github.com/ICTRC/Parsivar) - A Language Processing Toolkit for Persian (Has Half-Space Normalizer and Pinglish Conversion)
- [ParsiAnalyzer](https://github.com/NarimanN2/ParsiAnalyzer) -  Persian Analyzer For Elasticsearch.
- [ParsiNorm](https://github.com/haraai/ParsiNorm) - Persain Text Pre-Proceesing Tool
- [Persian Tools](https://github.com/persian-tools/py-persian-tools) - An anthology of a variety of tools for the Persian language in Python
 
## Translator
- [SPL](https://github.com/stanford-oval/SPL) - Semantic Parser Localizer toolkit can be used to translate text between any language pairs for which an NMT model exists. We currently support [Marian](https://github.com/marian-nmt/marian) models and Google Translate. In general, for translations to or from Persian, Google Translate has higher quality.

## Transliterator
- [Perstem](https://github.com/jonsafari/perstem) - Perstem is a Persian (Farsi) stemmer, morphological analyzer, transliterator, and partial part-of-speech tagger. Inflexional morphemes are separated or removed from their stems. Perstem can also tokenize and transliterate between various character set encodings and romanizations.

## Morphological Analysis
- [polyglot](https://github.com/aboSamoor/polyglot) - Natural language pipeline that supports massive multilingual applications (like lokenization (165 languages), language detection (196 languages), named entity recognition (40 languages), part of speech tagging (16 languages), sentiment analysis (136 languages), word embeddings (137 languages), morphological analysis (135 languages), transliteration (69 Languages)).

## Stemmer
- [Hazm](https://github.com/roshan-research/hazm) - Persian NLP Toolkit.
- [PersianStemmer](https://github.com/MrHTZ/PersianStemmer-Java/) - ([Java](https://github.com/MrHTZ/PersianStemmer-Java/), [Delphi](https://github.com/MrHTZ/PersianStemmer/),[C#](https://github.com/MrHTZ/PersianStemmer-CSharp/) and [Python](https://github.com/MrHTZ/PersianStemmer-Python/)) - PersianStemmer is a longest-match stemming algorithm that is based on pattern matching. It uses a knowledge base which consist of a collection of rules named "patterns". Furthermore, the exceptions and problems in the Persian morphology have been studied, and a solution is presented for each of them. So our stemmer evaluated. Its result was much better than the previous stemmers.
- [Perstem](https://github.com/jonsafari/perstem) - Perstem is a Persian (Farsi) stemmer, morphological analyzer, transliterator, and partial part-of-speech tagger. Inflexional morphemes are separated or removed from their stems. Perstem can also tokenize and transliterate between various character set encodings and romanizations.
- [polyglot](https://github.com/aboSamoor/polyglot) -  Natural language pipeline that supports massive multilingual applications (like lokenization (165 languages), language detection (196 languages), named entity recognition (40 languages), part of speech tagging (16 languages), sentiment analysis (136 languages), word embeddings (137 languages), morphological analysis (135 languages), transliteration (69 Languages)).
- [Parsivar](https://github.com/ICTRC/Parsivar) - A Language Processing Toolkit for Persian
- [ParsiAnalyzer](https://github.com/NarimanN2/ParsiAnalyzer) -  Persian Analyzer For Elasticsearch.

## Sentiment Analysis
- [polyglot (polarity)](https://github.com/aboSamoor/polyglot) -  Natural language pipeline that supports massive multilingual applications (like lokenization (165 languages), language detection (196 languages), named entity recognition (40 languages), part of speech tagging (16 languages), sentiment analysis (136 languages), word embeddings (137 languages), morphological analysis (135 languages), transliteration (69 Languages)).

## Spell Checking
- [async_faspell](https://github.com/eteamin/async_faspell) - Persian spellchecker. An algorithm that suggests words for misspelled words.

## Dependency Parser
- [Hazm](https://github.com/roshan-research/hazm) - Persian NLP Toolkit.
  
## Shallow Parser
- [Hazm](https://github.com/roshan-research/hazm) - Persian NLP Toolkit.
- [Parsivar](https://github.com/ICTRC/Parsivar) - A Language Processing Toolkit for Persian

## Information Extraction
- [Baaz](https://github.com/sobhe/information-extraction) - Open information extraction from Persian web.

## Text To Speech Preprocessing
- [ParsiNorm](https://github.com/haraai/ParsiNorm) - Persain Text Pre-Proceesing Tool
- [Persian Tools](https://github.com/persian-tools/py-persian-tools) - An anthology of a variety of tools for the Persian language in Python

## Text To Speech
- [AlisterTA TTS](https://github.com/AlisterTA/Persian-text-to-speech) - A convolutional sequence to sequence model for Persian text to speech based on Tachibana et al with a few modifications.

## Persian Phonemizer
- [persian_phonemizer](https://github.com/de-mh/persian_phonemizer) - A tool for translating Persian text to IPA (International Phonetic Alphabet).

## MISC
- [petit](https://github.com/JKhakpour/petit) - Convert alphabet-written numbers to digit-form

## Keyphrase Extractor
- [Perke](https://github.com/AlirezaTheH/perke) - Perke is a Python keyphrase extraction package for Persian language. It provides an end-to-end keyphrase extraction pipeline in which each component can be easily modified or extended to develop new models.

## Speech Recognition
- [Vosk](https://github.com/alphacep/vosk-api) - Vosk is an offline open source speech recognition toolkit. It enables speech recognition for 20+ languages and dialects. Supports Persian.
- [m3hrdadfi/wav2vec](https://huggingface.co/m3hrdadfi/wav2vec2-large-xlsr-persian-v3) - Persian speech recognition model based on XLS-R.

## Metrics
- [Rouge](https://pypi.org/project/rouge/) - Full Python ROUGE Score Implementation (not a wrapper)
