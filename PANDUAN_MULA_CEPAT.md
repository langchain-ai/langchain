# Panduan Mula Cepat (Inisiasi LangChain)

Selamat datang di LangChain! Dokumen ini adalah panduan singkat berbahasa Indonesia untuk memulai dan menginisiasi (*initialize*) project pertama Anda menggunakan kerangka kerja LangChain.

## 1. Persiapan Awal

Sebelum memulai, pastikan Anda telah menginstal Python (minimal versi 3.9) di komputer Anda. Disarankan untuk menggunakan *Virtual Environment* (seperti `venv` atau `conda`).

Instal library utama LangChain menggunakan perintah berikut di terminal/command prompt:

```bash
pip install langchain
```

Jika Anda ingin menggunakan integrasi model tertentu, misalnya OpenAI, Anda perlu menginstal *package* spesifik mereka:

```bash
pip install -U langchain-openai
```

## 2. Inisiasi Model (*LLM/Chat Models*)

Langkah pertama dalam menggunakan LangChain adalah menginisiasi sebuah model AI. Kita akan mengambil contoh inisiasi menggunakan model dari OpenAI.

Pastikan Anda telah memiliki API Key dari OpenAI dan atur parameternya:

```python
import os
from langchain_openai import ChatOpenAI

# Set Environment Variable untuk API Key
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxx" # Ganti dengan API key asli Anda

# Inisiasi Model Chat (menggunakan GPT-4o sebagai contoh default)
llm = ChatOpenAI(model="gpt-4o")

# Coba jalankan model
respons = llm.invoke("Halo, perkenalkan diri Anda dalam satu kalimat!")
print(respons.content)
```

## 3. Inisiasi "Prompt Template"

Seringkali kita tidak ingin mengirimkan teks mentah kepada model, melainkan sebuah format template yang bisa diisi sesuai kebutuhan pengguna. Untuk itulah *Prompt Template* digunakan.

```python
from langchain_core.prompts import PromptTemplate

# Inisiasi Template
template = PromptTemplate.from_template(
    "Tolong buatkan deskripsi singkat terkait topik ini: {topik}"
)

# Coba isi parameter topik
prompt_hasil = template.format(topik="Kecerdasan Buatan")
print(prompt_hasil)
```

## 4. Inisiasi "Chain" Sederhana

Dalam LangChain, sebuah *Chain* (rantai) adalah kumpulan urutan tugas. Pola standar yang kini sering digunakan adalah *LangChain Expression Language* (LCEL) dengan operator `|`.

Mari kita gabungkan `PromptTemplate` dan `LLM` yang telah kita inisiasi:

```python
# Membuat chain dengan LCEL
chain = template | llm

# Eksekusi chain tersebut
hasil = chain.invoke({"topik": "Kecerdasan Buatan"})
print(hasil.content)
```

## 5. Menambahkan Output Parser

Jika kita ingin hasil akhir dari model bukan sebuah objek obrolan (chat message object) tapi berupa teks *string* murni, kita bisa menginisiasi dan menambahkan `StrOutputParser` di akhir *chain*:

```python
from langchain_core.output_parsers import StrOutputParser

# Inisiasi Parser
parser = StrOutputParser()

# Perbarui chain kita
chain = template | llm | parser

# Jalankan kembali
hasil_teks = chain.invoke({"topik": "Pemrograman Python"})
print(hasil_teks) # Kini langsung mengembalikan string
```

---
**Catatan untuk Kontributor:** 
Dokumentasi ini ditulis sebagai langkah awal (inisiasi) agar pengembang berbahasa Indonesia dapat dengan cepat memanfaatkan framework ini. Anda dapat menambahkan kasus penggunaan lain (seperti inisiasi *Vector Store* dan *Retrievers*) di panduan yang lebih lanjut.
