# langchain-sarvamcloud

[![PyPI version](https://img.shields.io/pypi/v/langchain-sarvamcloud.svg)](https://pypi.org/project/langchain-sarvamcloud/)
[![Python versions](https://img.shields.io/pypi/pyversions/langchain-sarvamcloud.svg)](https://pypi.org/project/langchain-sarvamcloud/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LangChain integration for [Sarvam AI](https://www.sarvam.ai) — India's sovereign AI platform with comprehensive support for 22+ Indian languages.

**Author:** Srinivasulu Kethanaboina

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Authentication](#authentication)
- [Chat Model — ChatSarvam](#chat-model--chatsarvam)
- [Speech-to-Text — SarvamSTT](#speech-to-text--sarvamstt)
- [Batch Speech-to-Text — SarvamBatchSTT](#batch-speech-to-text--sarvambatchstt)
- [Text-to-Speech — SarvamTTS](#text-to-speech--sarvamtts)
- [Translation — SarvamTranslator](#translation--sarvamtranslator)
- [Transliteration — SarvamTransliterator](#transliteration--sarvamtransliterator)
- [Language Detection — SarvamLanguageDetector](#language-detection--sarvamlanguagedetector)
- [Document Intelligence — SarvamDocumentIntelligence](#document-intelligence--sarvamdocumentintelligence)
- [Supported Languages Reference](#supported-languages-reference)
- [Error Handling](#error-handling)
- [Development](#development)

---

## Features

| Class | Service | Languages |
|-------|---------|-----------|
| `ChatSarvam` | LLM chat completions with tool calling and streaming | 22+ Indian languages |
| `SarvamSTT` | Speech-to-text (REST, max 30 sec) | 23 languages |
| `SarvamBatchSTT` | Batch speech-to-text (async, max 1 hr/file) | 23 languages |
| `SarvamTTS` | Text-to-speech, 30+ voices | 11 languages |
| `SarvamTranslator` | Text translation with style modes | 22 languages |
| `SarvamTransliterator` | Script conversion (e.g. Devanagari ↔ Roman) | 11 languages |
| `SarvamLanguageDetector` | Language and script identification | 11 languages |
| `SarvamDocumentIntelligence` | PDF/image OCR and digitization | 23 languages |

---

## Installation

```bash
pip install langchain-sarvamcloud
```

This automatically installs all required dependencies: `sarvamai`, `httpx`, `pydantic`, and `langchain-core`.

---

## Authentication

All classes read the API key from the `SARVAM_API_KEY` environment variable:

```bash
export SARVAM_API_KEY="your-api-key"
```

Or pass it directly when creating any class:

```python
from langchain_sarvamcloud import ChatSarvam

model = ChatSarvam(api_subscription_key="your-api-key")
```

Get your API key from [dashboard.sarvam.ai](https://dashboard.sarvam.ai).

---

## Chat Model — `ChatSarvam`

A LangChain-compatible chat model backed by Sarvam's language models. Supports tool calling, streaming, async calls, and structured output.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"sarvam-105b"` | Model name (alias: `model_name`) |
| `temperature` | `float` | `0.2` | Sampling temperature (Sarvam default is 0.2, not 0.7) |
| `top_p` | `float` | `1.0` | Nucleus sampling probability |
| `max_tokens` | `int \| None` | `None` | Maximum tokens to generate |
| `reasoning_effort` | `"low" \| "medium" \| "high" \| None` | `None` | Thinking depth (only for sarvam-30b, sarvam-105b) |
| `streaming` | `bool` | `False` | Enable streaming by default |
| `max_retries` | `int` | `2` | Number of retries on failure |
| `api_subscription_key` | `SecretStr \| None` | `None` | API key (reads from `SARVAM_API_KEY`) |
| `base_url` | `str` | `"https://api.sarvam.ai/v1"` | API base URL |
| `model_kwargs` | `dict` | `{}` | Extra keyword arguments passed to the API |

### Available Models

| Model | Context Window | Tool Calling | Notes |
|-------|---------------|-------------|-------|
| `sarvam-105b` | 128,000 tokens | Yes | Flagship model (default) |
| `sarvam-105b-32k` | 32,000 tokens | Yes | Short-context variant |
| `sarvam-30b` | 32,000 tokens | Yes | Balanced speed/quality |
| `sarvam-30b-16k` | 16,000 tokens | Yes | Short-context variant |
| `sarvam-m` | 8,192 tokens | No | Legacy 24B model |

All models are **free to use** — no per-token cost.

### Basic Usage

```python
from langchain_sarvamcloud import ChatSarvam

model = ChatSarvam(model="sarvam-105b", temperature=0.2)
response = model.invoke("हिंदी में मेरा परिचय दो।")
print(response.content)
```

### Streaming

```python
from langchain_sarvamcloud import ChatSarvam

model = ChatSarvam(model="sarvam-105b")

for chunk in model.stream("Tell me a short story in Hindi"):
    print(chunk.content, end="", flush=True)
```

### Async Usage

```python
import asyncio
from langchain_sarvamcloud import ChatSarvam

async def main():
    model = ChatSarvam(model="sarvam-105b")
    response = await model.ainvoke("Explain photosynthesis in Tamil")
    print(response.content)

    # Async streaming
    async for chunk in model.astream("Write a poem in Telugu"):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

### Tool Calling

```python
from langchain_core.tools import tool
from langchain_sarvamcloud import ChatSarvam

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 28°C."

model = ChatSarvam(model="sarvam-105b")
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What is the weather in Mumbai?")
print(response.tool_calls)
# [{'name': 'get_weather', 'args': {'city': 'Mumbai'}, 'id': '...'}]
```

### Structured Output

```python
from pydantic import BaseModel
from langchain_sarvamcloud import ChatSarvam

class PersonInfo(BaseModel):
    name: str
    age: int
    city: str

model = ChatSarvam(model="sarvam-105b")
structured_model = model.with_structured_output(PersonInfo)

result = structured_model.invoke("Ravi is 28 years old and lives in Chennai")
print(result.name)   # Ravi
print(result.age)    # 28
print(result.city)   # Chennai
```

### Reasoning Effort (Unique to Sarvam)

Control how much the model "thinks" before responding. Available for `sarvam-30b` and `sarvam-105b`:

```python
from langchain_sarvamcloud import ChatSarvam

# For complex reasoning tasks
model = ChatSarvam(model="sarvam-105b", reasoning_effort="high")
response = model.invoke("Solve this math problem step by step: ...")

# For fast, simple responses
model = ChatSarvam(model="sarvam-105b", reasoning_effort="low")
response = model.invoke("What is the capital of India?")
```

---

## Speech-to-Text — `SarvamSTT`

Transcribes audio files into text. Supports 23 Indian languages, multiple transcription modes.

**Limit:** Max **30 seconds** per audio file. For longer audio, use `SarvamBatchSTT`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `"saaras:v3" \| "saarika:v2.5"` | `"saaras:v3"` | STT model (`saaras:v3` recommended) |
| `mode` | `str` | `"transcribe"` | Transcription mode (see below) |
| `language_code` | `str \| None` | `None` | BCP-47 language code. Auto-detected if `None` |
| `api_subscription_key` | `SecretStr \| None` | `None` | API key |

### Transcription Modes

| Mode | Description |
|------|-------------|
| `transcribe` | Return text in the source language (default) |
| `translate` | Translate speech directly to English text |
| `verbatim` | Exact verbatim transcription including filler words |
| `translit` | Transcribe and romanize (convert to Latin script) |
| `codemix` | Code-mixed output (mix of source language and English) |

### Supported Audio Formats

WAV, MP3, AAC, OGG, OPUS, FLAC, MP4, AMR, WebM, PCM

### Usage

```python
from langchain_sarvamcloud import SarvamSTT

stt = SarvamSTT(model="saaras:v3", mode="transcribe")

with open("audio.wav", "rb") as f:
    result = stt.transcribe(f, language_code="hi-IN")

print(result["transcript"])           # Transcribed text
print(result["language_code"])        # Detected/used language code
print(result["language_probability"]) # Confidence (0.0–1.0)
```

### Translate Speech to English

```python
stt = SarvamSTT(model="saaras:v3", mode="translate")

with open("hindi_audio.wav", "rb") as f:
    result = stt.transcribe(f)  # language_code auto-detected

print(result["transcript"])  # English translation of Hindi audio
```

### Romanize (Transliterate) Output

```python
stt = SarvamSTT(model="saaras:v3", mode="translit")

with open("hindi_audio.wav", "rb") as f:
    result = stt.transcribe(f, language_code="hi-IN")

print(result["transcript"])  # Hindi speech in Roman script, e.g. "namaste"
```

### `transcribe()` Method Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | `BinaryIO` | Binary audio file object |
| `language_code` | `str \| None` | Override instance language code |
| `mode` | `str \| None` | Override instance mode |
| `input_audio_codec` | `str \| None` | PCM codec for raw PCM files (e.g. `pcm_s16le`) |

### Return Value Keys

| Key | Type | Description |
|-----|------|-------------|
| `transcript` | `str` | Transcribed/translated text |
| `language_code` | `str` | BCP-47 code of detected/used language |
| `language_probability` | `float` | Confidence score (0.0–1.0) |
| `timestamps` | `list \| None` | Word-level timestamps (if available) |
| `request_id` | `str` | Unique request identifier |

---

## Batch Speech-to-Text — `SarvamBatchSTT`

Processes multiple long audio files asynchronously. Use this when:
- Audio is longer than 30 seconds (up to 1 hour per file)
- You need to process multiple files at once (up to 20 per job)
- You need speaker diarization

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_subscription_key` | `SecretStr \| None` | `None` | API key |

### Full Workflow

```python
import httpx
from langchain_sarvamcloud import SarvamBatchSTT

batch = SarvamBatchSTT()

# Step 1: Create a job
job_id = batch.create_job(
    model="saaras:v3",
    mode="transcribe",
    language_code="hi-IN",      # None = auto-detect
    with_timestamps=True,        # Include word-level timestamps
    with_diarization=True,       # Identify different speakers
    num_speakers=2,              # Expected number of speakers (1–8)
)
print(f"Job created: {job_id}")

# Step 2: Get signed upload URLs
filenames = ["interview_part1.mp3", "interview_part2.mp3"]
upload_urls = batch.get_upload_urls(job_id, filenames)

# Step 3: Upload files to signed URLs
for filename, url_info in upload_urls.items():
    with open(filename, "rb") as f:
        httpx.put(url_info["url"], content=f.read())
    print(f"Uploaded: {filename}")

# Step 4: Start processing
batch.start_job(job_id)
print("Processing started...")

# Step 5: Wait for completion (polls every 5 seconds by default)
status = batch.wait_for_completion(job_id, poll_interval=10.0)

if status["job_state"] == "Completed":
    print(f"Completed: {status['successful_files_count']} files processed")
    for detail in status["job_details"]:
        print(f"  {detail['file']}: {detail['output']}")
else:
    print(f"Failed: {status}")
```

### `create_job()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `"saaras:v3" \| "saarika:v2.5"` | `"saaras:v3"` | STT model |
| `mode` | `str` | `"transcribe"` | Transcription mode |
| `language_code` | `str \| None` | `None` | BCP-47 code. Auto-detected if `None` |
| `with_timestamps` | `bool` | `False` | Include chunk-level timestamps in output |
| `with_diarization` | `bool` | `False` | Enable speaker diarization |
| `num_speakers` | `int \| None` | `None` | Expected number of speakers (1–8) |
| `input_audio_codec` | `str \| None` | `None` | PCM codec for raw audio |
| `callback` | `str \| None` | `None` | Webhook URL for completion notification |

### Job States

```
Accepted → Pending → Running → Completed
                             → Failed
```

### `get_status()` Return Keys

| Key | Description |
|-----|-------------|
| `job_state` | Current state |
| `job_id` | Job identifier |
| `total_files` | Number of files submitted |
| `successful_files_count` | Files processed successfully |
| `failed_files_count` | Files that failed |
| `job_details` | List of per-file results with `file`, `output`, `error` |

---

## Text-to-Speech — `SarvamTTS`

Converts text to natural-sounding speech in 11 Indian languages with 30+ voice options.

**Limit:** Max **2500 characters** per request.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `"bulbul:v3"` | `"bulbul:v3"` | TTS model |
| `speaker` | `str` | `"shubh"` | Voice speaker name (see list below) |
| `pace` | `float` | `1.0` | Speech rate. Range: `0.5` (slow) to `2.0` (fast) |
| `speech_sample_rate` | `int` | `24000` | Audio sample rate in Hz |
| `api_subscription_key` | `SecretStr \| None` | `None` | API key |

### Available Speakers (30+)

```
anushka   abhilash   manisha   vidya     arya
karun     hitesh     aditya    ritu      priya
neha      rahul      pooja     rohan     simran
kavya     amit       dev       ishita    shreya
ratan     varun      manan     sumit     roopa
kabir     aayan      shubh     ashutosh  advait
amelia    sophia     anand     tanya     tarun
sunny     mani       gokul     vijay     shruti
suhani    mohit      kavitha   rehan     soham
rupali
```

### Supported Sample Rates

`8000`, `16000`, `22050`, `24000`, `44100`, `48000` Hz

### Save Audio to File

```python
from langchain_sarvamcloud import SarvamTTS

tts = SarvamTTS(speaker="shubh", pace=1.0)

# Returns raw WAV bytes — simplest method
audio_bytes = tts.synthesize_to_bytes(
    "नमस्ते! मैं सर्वम AI हूँ।",
    target_language_code="hi-IN",
)

with open("output.wav", "wb") as f:
    f.write(audio_bytes)
print("Saved output.wav")
```

### Access Raw API Response

```python
import base64
from langchain_sarvamcloud import SarvamTTS

tts = SarvamTTS(speaker="anushka", pace=0.9, speech_sample_rate=44100)

result = tts.synthesize(
    "வணக்கம்! நான் சர்வம் AI.",
    target_language_code="ta-IN",
)

print(result["request_id"])   # Request ID for tracking
audio_bytes = base64.b64decode(result["audios"][0])
```

### Override Speaker Per Call

```python
tts = SarvamTTS()  # default speaker: shubh

# Override speaker for this specific call
audio = tts.synthesize_to_bytes(
    "Hello, how are you?",
    target_language_code="en-IN",
    speaker="amelia",
    pace=1.2,
)
```

### Supported Languages

`hi-IN`, `en-IN`, `ta-IN`, `te-IN`, `kn-IN`, `ml-IN`, `mr-IN`, `gu-IN`, `pa-IN`, `bn-IN`, `od-IN`

---

## Translation — `SarvamTranslator`

Translates text between 22 Indian languages with rich style controls including code-mixed output and spoken-form variants.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `"sarvam-translate:v1" \| "mayura:v1"` | `"sarvam-translate:v1"` | Translation model |
| `mode` | `str` | `"formal"` | Translation style |
| `speaker_gender` | `"Male" \| "Female" \| None` | `None` | Gender for gender-aware translations |
| `output_script` | `str \| None` | `None` | Output script format |
| `numerals_format` | `"international" \| "native"` | `"international"` | Numeral style |
| `api_subscription_key` | `SecretStr \| None` | `None` | API key |

### Models

| Model | Max Characters | Notes |
|-------|---------------|-------|
| `sarvam-translate:v1` | 2,000 | Default, higher quality |
| `mayura:v1` | 1,000 | Supports `source_language_code="auto"` for auto-detection |

### Translation Modes

| Mode | Description | Example Output |
|------|-------------|----------------|
| `formal` | Formal register (default) | Standard written language |
| `modern-colloquial` | Modern conversational | Everyday spoken style |
| `classic-colloquial` | Classical conversational | Traditional spoken style |
| `code-mixed` | Mix of source and English | Hinglish, Tanglish, etc. |

### Basic Translation

```python
from langchain_sarvamcloud import SarvamTranslator

translator = SarvamTranslator(model="sarvam-translate:v1")

result = translator.translate(
    "Hello, how are you?",
    source_language_code="en-IN",
    target_language_code="hi-IN",
    mode="formal",
)
print(result["translated_text"])         # नमस्ते, आप कैसे हैं?
print(result["source_language_code"])    # en-IN
print(result["request_id"])             # Request ID
```

### Code-Mixed Output (Hinglish)

```python
translator = SarvamTranslator(model="sarvam-translate:v1")

result = translator.translate(
    "I am going to the market to buy vegetables.",
    source_language_code="en-IN",
    target_language_code="hi-IN",
    mode="code-mixed",
)
print(result["translated_text"])  # e.g. "Main market जा रहा हूँ vegetables buy करने के लिए।"
```

### Auto-Detect Source Language

```python
translator = SarvamTranslator(model="mayura:v1")

result = translator.translate(
    "நான் சென்னையில் வசிக்கிறேன்",
    source_language_code="auto",   # Only with mayura:v1
    target_language_code="en-IN",
)
print(result["translated_text"])  # I live in Chennai
```

### `translate()` Method Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to translate |
| `source_language_code` | `str` | BCP-47 source language. Use `"auto"` with `mayura:v1` |
| `target_language_code` | `str` | BCP-47 target language |
| `mode` | `str \| None` | Overrides instance `mode` |
| `speaker_gender` | `"Male" \| "Female" \| None` | Gender-aware translation |
| `output_script` | `str \| None` | `roman`, `fully-native`, or `spoken-form-in-native` |
| `numerals_format` | `str \| None` | `international` (0–9) or `native` (e.g. ०–९ for Hindi) |

### Output Script Options

```python
# Roman script output (even for Indic languages)
result = translator.translate(
    "Hello",
    source_language_code="en-IN",
    target_language_code="hi-IN",
    output_script="roman",
)
print(result["translated_text"])  # "Namaste"

# Native numerals
result = translator.translate(
    "I have 42 mangoes.",
    source_language_code="en-IN",
    target_language_code="hi-IN",
    numerals_format="native",
)
print(result["translated_text"])  # "मेरे पास ४२ आम हैं।"
```

---

## Transliteration — `SarvamTransliterator`

Converts text **between scripts** without changing the language. For example, write Hindi in Roman letters, or write English words in Devanagari script.

> **Note:** This is NOT translation. Transliteration preserves pronunciation, not meaning.
> Use `SarvamTranslator` for cross-language conversion.

> **Limitation:** Indic-to-Indic transliteration (e.g. Hindi → Bengali script) is **not supported**.
> One of the two languages must be `en-IN`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numerals_format` | `"international" \| "native"` | `"international"` | Numeral style in output |
| `spoken_form` | `bool` | `False` | Convert to natural spoken form |
| `spoken_form_numerals_language` | `"english" \| "native"` | `"english"` | How to pronounce numbers in spoken form |
| `api_subscription_key` | `SecretStr \| None` | `None` | API key |

### Supported Languages

`bn-IN`, `en-IN`, `gu-IN`, `hi-IN`, `kn-IN`, `ml-IN`, `mr-IN`, `od-IN`, `pa-IN`, `ta-IN`, `te-IN`

### Devanagari → Roman (Hindi to English script)

```python
from langchain_sarvamcloud import SarvamTransliterator

tl = SarvamTransliterator()

result = tl.transliterate(
    "नमस्ते, आप कैसे हैं?",
    source_language_code="hi-IN",
    target_language_code="en-IN",
)
print(result["transliterated_text"])  # namaste, aap kaise hain?
```

### Roman → Telugu Script

```python
result = tl.transliterate(
    "namaskaram",
    source_language_code="en-IN",
    target_language_code="te-IN",
)
print(result["transliterated_text"])  # నమస్కారం
```

### Spoken Form Conversion

Convert written text to how it would be spoken naturally (useful for TTS preprocessing):

```python
tl = SarvamTransliterator(spoken_form=True)

result = tl.transliterate(
    "Dr. Sharma has 3 appointments on 15/08/2024",
    source_language_code="en-IN",
    target_language_code="hi-IN",
    spoken_form=True,
    spoken_form_numerals_language="english",
)
# Converts abbreviations and numbers to spoken words
```

### `transliterate()` Method Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Text to transliterate. Max 1000 characters |
| `source_language_code` | `str` | BCP-47 source language |
| `target_language_code` | `str` | BCP-47 target language |
| `spoken_form` | `bool \| None` | Convert to spoken form. Overrides instance default |
| `numerals_format` | `str \| None` | Overrides instance default |
| `spoken_form_numerals_language` | `str \| None` | Overrides instance default |

**Raises:** `ValueError` if both source and target are non-English Indic languages.

---

## Language Detection — `SarvamLanguageDetector`

Identifies the language and writing script of input text, with a confidence score.

**Limit:** Max **1000 characters** per request.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_subscription_key` | `SecretStr \| None` | `None` | API key |

### Supported Languages

`en-IN`, `hi-IN`, `bn-IN`, `gu-IN`, `kn-IN`, `ml-IN`, `mr-IN`, `od-IN`, `pa-IN`, `ta-IN`, `te-IN`

### Supported Scripts

| Script | Languages |
|--------|-----------|
| Latin | English |
| Devanagari | Hindi, Marathi |
| Bengali | Bengali |
| Gujarati | Gujarati |
| Kannada | Kannada |
| Malayalam | Malayalam |
| Odia | Odia |
| Gurmukhi | Punjabi |
| Tamil | Tamil |
| Telugu | Telugu |

### Usage

```python
from langchain_sarvamcloud import SarvamLanguageDetector

detector = SarvamLanguageDetector()

result = detector.detect("నమస్కారం, మీరు ఎలా ఉన్నారు?")

print(result["language_code"])  # te-IN
print(result["script_code"])    # Telu
print(result["confidence"])     # e.g. 0.97
print(result["request_id"])     # Request ID (always present)
```

### Return Value Keys

| Key | Type | Nullable | Description |
|-----|------|---------|-------------|
| `language_code` | `str` | Yes | BCP-47 detected language code |
| `script_code` | `str` | Yes | ISO 15924 script code |
| `confidence` | `float` | Yes | Confidence score (0.0–1.0) |
| `request_id` | `str` | No | Unique request identifier |

> **Note:** `language_code`, `script_code`, and `confidence` may be `None` if detection fails.

---

## Document Intelligence — `SarvamDocumentIntelligence`

Digitizes PDF documents, images, and ZIP archives using Sarvam's Vision model (3B VLM). Extracts text with layout preservation, table extraction, and reading order detection.

**Limits:**
- Max **500 pages** per job
- Max **200 MB** per file
- Input formats: PDF, PNG, JPEG, ZIP (flat structure only)

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_paths` | `list[str]` | required | Local paths to files to digitize |
| `language` | `str` | `"hi-IN"` | BCP-47 language code for OCR |
| `output_format` | `"md" \| "html" \| "json"` | `"md"` | Format of extracted content |
| `callback` | `str \| None` | `None` | Webhook URL for job completion notification |
| `poll_interval` | `float` | `5.0` | Seconds between status poll requests |
| `api_subscription_key` | `SecretStr \| None` | `None` | API key |

### Output Formats

| Format | Description | Best For |
|--------|-------------|----------|
| `md` | Markdown with headings and tables | LLM processing, RAG pipelines |
| `html` | HTML with full layout preservation | Web display, structure-sensitive tasks |
| `json` | Structured JSON with bounding boxes | Programmatic processing, data extraction |

### Basic Usage

```python
from langchain_sarvamcloud import SarvamDocumentIntelligence

loader = SarvamDocumentIntelligence(
    file_paths=["document.pdf"],
    language="hi-IN",
    output_format="md",
)

docs = loader.load()  # Returns list of LangChain Document objects

for doc in docs:
    print(f"Source: {doc.metadata['source']}")
    print(f"Language: {doc.metadata['language']}")
    print(doc.page_content[:500])
```

### Multiple Files

```python
loader = SarvamDocumentIntelligence(
    file_paths=["form1.pdf", "form2.pdf", "scan.png"],
    language="ta-IN",
    output_format="json",
)

docs = loader.load()
print(f"Got {len(docs)} documents")
```

### Use with LangChain RAG Pipeline

```python
from langchain_sarvamcloud import SarvamDocumentIntelligence
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# Load and digitize
loader = SarvamDocumentIntelligence(
    file_paths=["hindi_report.pdf"],
    language="hi-IN",
    output_format="md",
)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks for indexing")
```

### Lazy Loading (Memory-Efficient)

```python
loader = SarvamDocumentIntelligence(
    file_paths=["large_document.pdf"],
    language="en-IN",
)

# Process one document at a time instead of loading all at once
for doc in loader.lazy_load():
    process(doc)
```

### Document Metadata

Each returned `Document` object has these metadata keys:

| Key | Description |
|-----|-------------|
| `source` | Filename of the output file |
| `format` | Output format used (`md`, `html`, `json`) |
| `language` | BCP-47 language code used for OCR |

### How the Workflow Works Internally

The loader automatically runs this 5-step async job:

```
1. Create job   →  POST /doc-digitization/job/v1
2. Get URLs     →  POST /doc-digitization/job/v1/{id}/upload-urls
3. Upload files →  PUT to signed S3 URLs
4. Start job    →  POST /doc-digitization/job/v1/{id}/start
5. Poll status  →  GET /doc-digitization/job/v1/{id}/status (until Completed/Failed)
6. Download     →  GET signed download URLs → parse as Documents
```

---

## Supported Languages Reference

Full language support matrix across all services:

| Language | BCP-47 | Chat | STT | TTS | Translation | Transliteration | LID | Document |
|----------|--------|------|-----|-----|-------------|-----------------|-----|----------|
| Hindi | `hi-IN` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| English | `en-IN` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tamil | `ta-IN` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Telugu | `te-IN` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Kannada | `kn-IN` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Malayalam | `ml-IN` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Marathi | `mr-IN` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Gujarati | `gu-IN` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bengali | `bn-IN` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Punjabi | `pa-IN` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Odia | `od-IN` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Assamese | `as-IN` | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Sanskrit | `sa-IN` | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Urdu | `ur-IN` | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Bodo | `bodo-IN` | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Kashmiri | `ks-IN` | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Konkani | `kok-IN` | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Maithili | `mai-IN` | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Manipuri | `mni-IN` | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Nepali | `ne-IN` | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Santali | `sat-IN` | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Sindhi | `sd-IN` | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| Dogri | `doi-IN` | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |

---

## Error Handling

### Missing API Key

```python
import os
os.environ.pop("SARVAM_API_KEY", None)

from langchain_sarvamcloud import ChatSarvam

# Key is None but won't error until actual API call is made
model = ChatSarvam()
response = model.invoke("Hello")  # Will fail with authentication error from API
```

Always set `SARVAM_API_KEY` before use:
```bash
export SARVAM_API_KEY="your-key"
```

### Document Intelligence Job Failure

```python
from langchain_sarvamcloud import SarvamDocumentIntelligence

loader = SarvamDocumentIntelligence(file_paths=["document.pdf"])

try:
    docs = loader.load()
except RuntimeError as e:
    print(f"Digitization failed: {e}")
    # e.g. "Sarvam Document Intelligence job abc-123 failed: {'job_state': 'Failed', ...}"
```

### Indic-to-Indic Transliteration

```python
from langchain_sarvamcloud import SarvamTransliterator

tl = SarvamTransliterator()

try:
    result = tl.transliterate(
        "नमस्ते",
        source_language_code="hi-IN",
        target_language_code="ta-IN",  # Both are non-English Indic — will raise!
    )
except ValueError as e:
    print(e)
    # "Indic-to-Indic transliteration is not supported by Sarvam AI.
    #  Use SarvamTranslator for cross-Indic language conversion instead."

# Fix: use SarvamTranslator instead
from langchain_sarvamcloud import SarvamTranslator
translator = SarvamTranslator()
result = translator.translate("नमस्ते", source_language_code="hi-IN", target_language_code="ta-IN")
```

### Missing SDK

```python
# If sarvamai is not installed:
from langchain_sarvamcloud import ChatSarvam

model = ChatSarvam(api_subscription_key="key")
# ImportError: Could not import sarvamai python package.
# Please install it with `pip install sarvamai`.
```

Fix: `pip install langchain-sarvamcloud` installs all dependencies automatically.

---

## Development

### Setup

```bash
git clone https://github.com/langchain-ai/langchain.git
cd langchain/libs/partners/sarvamcloud

# Install all dependencies
uv sync --all-groups
```

### Running Tests

```bash
# Unit tests (no network calls)
make test
# or
uv run --group test pytest tests/unit_tests/ -v

# Integration tests (requires real API key)
export SARVAM_API_KEY="your-api-key"
uv run --group test pytest tests/integration_tests/ -v
```

### Linting and Formatting

```bash
make lint      # ruff check + mypy type checking
make format    # ruff format (auto-fix style issues)
```

### Project Structure

```
langchain_sarvamcloud/
├── __init__.py           # Public exports
├── chat_models.py        # ChatSarvam
├── speech.py             # SarvamSTT, SarvamBatchSTT, SarvamTTS
├── text.py               # SarvamTranslator, SarvamTransliterator, SarvamLanguageDetector
├── document_loaders.py   # SarvamDocumentIntelligence
├── version.py            # Package version
└── data/
    ├── _profiles.py              # Model profiles (context windows, capabilities)
    └── profile_augmentations.toml
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

**Author:** Srinivasulu Kethanaboina
