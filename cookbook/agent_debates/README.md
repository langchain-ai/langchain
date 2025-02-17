# Дебаты агентов

Пример мультиагентой системы, в которой два AI агента спорят на заданную тему.

![alt text](image.png)

Чтобы запустить пример:

1. Склонируйте репозиторий.
2. Создайте чистое окружение Python.
3. Установите зависимости:

   ```sh
   pip install -r requirements.txt
   ```

4. Создайте файл .env с настройками доступа к GigaChat API:

   ```sh
   GIGACHAT_CREDENTIALS=ключ_авторизации
   GIGACHAT_BASE_URL=...
   ```

   Образец заполнения файла .env в папке [coobook/sample_data](https://github.com/ai-forever/gigachain/tree/master/cookbook/sample_data).

5. Запустите приложение

   ```sh
   streamlit run debates.py
   ```

6. Откройте в браузере http://localhost:8501/

## Отладка с помощью LangGraph Studio

1. Установить LangGraph Studio:

   ```sh
   pip install -U "langgraph-cli[inmem]"
   ```

2. Запустите LangGraph Studio:

   ```sh
   langgraph dev
   ```

3. Перейдите в запустившийся браузер:

   ![alt text](image-2.png)

4. Для старта обсуждения задайте поле "Main Topic".