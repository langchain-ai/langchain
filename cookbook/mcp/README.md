# Работа GigaChat-агента с MCP-сервером

Пример демонстрирует работу агента с MCP-сервером ([`math_server.py`](math_server.py)), который дает доступ к функциям сложения, умножения и поиска данных человека по имени.
Доступны два варианта агента:

* [`agent.py`](agent.py) — реализует MCP-клиент для локального взаимодействия и не требует отдельного запуска MCP-сервера;
* [`agent_http.py`](agent_http.py) — реализует MCP-клиент для взаимодействия по HTTP. Для работы агента нужно запустить MCP-серввер в режиме SSE.

Агент разработан с помощью фреймворков [LangChain](https://python.langchain.com/docs/introduction/) и [LangGraph](https://langchain-ai.github.io/langgraph/), а также библиотеки [langchain-mcp-adapters](https://pypi.org/project/langchain-mcp-adapters/).
Библиотека предоставляет обертку для функций, описанных в формате MCP, и позволяет использовать их совместно с LangChain и LangGraph.

В роли LLM агент использует модель GigaChat.
Обмен сообщениями с моделью выполняется через [GigaChat API](https://developers.sber.ru/docs/ru/gigachat/api/overview).

## Model Context Protocol (MCP)

Model Context Protocol (или MCP) — открытый протокол, который унифицирует обмен контекстом между приложением и LLM. Другими словами, использование MCP упрощает подключение больших языковых моделей к различным функциям (*инструментам*) и источникам данных.

Протокол предоставляет:

* растущий список готовых интеграций, доступных для подключения LLM;
* простоту переключения между различными моделями и их поставщиками;
* набор лучших практик для обеспечения безопасности данных внутри вашей инфраструктуры.

MCP разработан компанией Anthropic.
Подробнее о протоколе — в [официальной документации](https://modelcontextprotocol.io/introduction).

## Подготовка к работе

Установите зависимости:

```sh
pip install langchain-gigachat langchain_mcp_adapters langgraph rich
```

В папке примера создайте файл с переменными окружения `.env` и добавьте в него переменную `GIGACHAT_CREDENTIALS`:

```sh
GIGACHAT_CREDENTIALS=<ключ_авторизации>
```

О том как получить ключ авторизации — в [официальной документации GigaChat](https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-using-api).

> [!TIP]
> Вы также можете указать другие переменные окружения, которые поддерживает [Python-библиотека GigaChat](https://github.com/ai-forever/gigachat#настройка-переменных-окружения).

## Быстрый старт

Для запуска локального клиента выполните команду:

```sh
python agent.py
```

Для запуска HTTP-клиента:

1. Запустите MCP-сервер в режиме SSE с помощью команды:

   ```sh
   python math_server.py sse
   ```

2. Запустите клиент с помощью команды:

   ```sh
   python agent_http.py
   ```

Пример вывода агента:

```sh
[HumanMessage] Сколько будет (3 + 5) x 12? 
[AIMessage]  [{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': '99f7f6c7-baac-4e61-9577-03903e83f3a7', 'type': 'tool_call'}]
[ToolMessage] 8.0 
[AIMessage]  [{'name': 'multiply', 'args': {'a': 8, 'b': 12}, 'id': 'c923315e-0888-47c3-a380-2f91d95c3177', 'type': 'tool_call'}]
[ToolMessage] 96.0 
[AIMessage] Результат выражения $(3+5)\times12$ равен $96$. []
[HumanMessage] Найди сколько лет Джону Доу? 
[AIMessage]  [{'name': 'find_preson', 'args': {'name': {'query': 'Джон Доу'}}, 'id': 'fa2ecddc-c446-477b-adc7-7d4f09281953', 'type': 'tool_call'}]
[ToolMessage] {"name": "John Doe", "age": 30} 
[AIMessage] Джону Доу 30 лет. []
```
