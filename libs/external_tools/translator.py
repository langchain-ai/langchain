"""Тул для автоматического перевода промптов 
    библиотеки LangChain с помощью OpenAI API."""
import ast
import os
import time

import openai
import tiktoken

IGNORED_DIRS = {
    "venv",
    ".venv",
    "build",
    ".git",
    "__pycache__",
    "tests",
    "venv_clear",
    "docs",
    "external_tools",
}
enc = tiktoken.get_encoding("cl100k_base")

ALREADY_PROCESSED_STORE = "libs/external_tools/translator_processed.txt"


def translate_to_russian(text):
    # Restart after pause in case of openai.error.RateLimitError
    try:
        messages = [
            {
                "role": "system",
                "content": """Я отправлю тебе программу на python или текстовый файл. В этой программе или тексте могут быть запросы (промпты) к большой языковой модели. Нужно перевести их на русский.
    В промпте могут содержаться плейсхолдеры в фигурных скобках, например {question} или {answer}. Это нормально, их нужно сохранить без перевода.
    Всю остальную программу нужно оставить как есть. Если в программе нет промптов, то нужно просто переписать её полностью. Больше ничего не надо выводить - только код с перевеёнными промптами.
    Не переводи комментарии в коде, не переводи docstring! Не переводи строки, которые не похожи на запросы или части запросов, например названия полей, имена ключей в словарях и тому подобное. Если ты не уверен, что это промпт, то лучше вообще не переводи.
    Если в файле нет ни одного промпта, верни "NO" без каких-либо пояснений. Общайся на `ты`, а не на `вы`, например `сделай`, а не `сделайте`. В промптах обращение к сети обязательно должно быть на "ты".
    Ты должен вернуть полный код программы, которую тебе прислали без сокращений или дополнительных пояснений или своих комментариев. Сразу пиши код.
    Не пиши в начале фразу "Код программы" и тому подобное. Начинай сразу с кода, первым словом в твоем ответе должна сразу быть программа""",  # noqa: E501
            },
            {"role": "user", "content": text},
        ]

        # Use tiktoken to check text size
        if len(enc.encode(text)) > 3500:
            with open(ALREADY_PROCESSED_STORE, "a", encoding="utf-8") as f:
                f.write("File is too big:\n")
            return text

        completion = openai.ChatCompletion.create(
            model="gpt-4", messages=messages, temperature=0.0, max_tokens=4500
        )

        translated_text = completion["choices"][0]["message"]["content"]
        if translated_text.startswith("NO"):
            return text
        return translated_text
    except Exception as ex:
        print("Exception occured: ", ex)
        print("Rate limit error. Restarting in (30s)...")
        time.sleep(30)
        return translate_to_russian(text)


def is_russian(s):
    return any(["а" <= ch <= "я" for ch in s.lower()])


ERROR_PHRASES = {
    "error",
    "exception",
    "failed",
    "cannot",
    "unable",
    "not found",
    "invalid",
    "unexpected",
    "could not",
    "please report",
    "stop sequences found ",
    "select ",
}


def is_not_error_message(s):
    return all(phrase not in s.lower() for phrase in ERROR_PHRASES)


def set_parent(node):
    for child in ast.iter_child_nodes(node):
        child.parent = node
        set_parent(child)


def process_file(file_path):
    try:
        # Check file is not processed yet. Create it if not exists
        if not os.path.exists(ALREADY_PROCESSED_STORE):
            with open(ALREADY_PROCESSED_STORE, "w", encoding="utf-8") as f:
                f.write("")
        with open(ALREADY_PROCESSED_STORE, "r", encoding="utf-8") as f:
            processed = f.read().splitlines()

        if file_path in processed:
            return False

        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        if (
            ("prompt" in source)
            or file_path.endswith(".txt")
            or "template = " in source.lower()
        ):
            print(f"Found file: {file_path}")
            print(f"Source: {source}\n\n")
            print("Do you see prompts here? (y/n)")

            answer = input()
            if answer.lower() != "n":
                translated = translate_to_russian(source)
                if not translated.endswith("\n"):
                    translated += "\n"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(translated)
                # Save file to processed list
                with open(ALREADY_PROCESSED_STORE, "a", encoding="utf-8") as f:
                    f.write(file_path + "\n")
                pass
            else:
                with open(ALREADY_PROCESSED_STORE, "a", encoding="utf-8") as f:
                    f.write("No prompts:" + "\n" + file_path + "\n")
            return True

    except UnicodeDecodeError:
        pass
    return False


def main(directory):
    total = 0
    for root, dirs, files in os.walk(directory):
        # Игнорируем ненужные директории
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for file in files:
            if file.endswith(".py") or file.endswith(".txt"):
                if process_file(os.path.join(root, file)):
                    total += 1
    print(f"Total files: {total}")


if __name__ == "__main__":
    main(".")
