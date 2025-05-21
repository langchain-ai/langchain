<br />
<div align="center">

  <a href="https://github.com/ai-forever/gigachain">
    <img src="static/img/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h1 align="center">🦜️🔗 GigaChain (GigaChat + LangChain)</h1>

  <p align="center">
    Набор решений для разработки LLM-приложений и мультиагентых систем на русском языке с поддержкой GigaChat, LangChain, LangGraph, LangChain4j.
    <br />
    <a href="https://github.com/ai-forever/gigachain/issues">Создать issue</a>
    ·
    <a href="https://developers.sber.ru/docs/ru/gigachat/sdk/overview">Документация GigaChain</a>
  </p>
</div>


![Product Name Screen Shot](/static/img/logo-with-backgroung.png)

---

# О GigaChain

GigaChain – это набор решений для создания приложений с использованием больших языковых моделей (*LLM*). GigaChain охватывает все этапы разработки от прототипирования и исследования, до запуска в эксплуатацию и поддержки.

Для работы вам понадобится [ключ авторизации для доступа к GigaChat API](https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-using-api?tool=python#poluchenie-avtorizatsionnyh-dannyh).

Наши решения: [Фреймворки](#фреймворки) | [SDK для работы с моделями GigaChat](#sdk-для-работы-с-моделями-gigachat) | [Утилиты и MCP-сервера](#утилиты-и-mcp-сервера) 


## Фреймворки

В состав GigaChain входят библиотеки для интеграции с популярными фреймворками LangChain, LangGraph и LangChain4j.

Библиотеки доступны на [Python](#python-), [JavaScript/TypeScript](#javascripttypescript-) и [Java](#java-).

Они позволяют использовать модели GigaChat со всеми возможностями и инфраструктурой, которую предоставляют фреймворки для разработки комплексных LLM-приложений, AI-агентов и мультиагентных систем.

### Python [![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-gigachat?style=flat-round)](https://pypistats.org/packages/langchain-gigachat)[![GitHub star chart](https://img.shields.io/github/stars/ai-forever/langchain-gigachat?style=flat-round)](https://www.star-history.com/#ai-forever/langchain-gigachat)

[`langchain-gigachat`](https://github.com/ai-forever/langchain-gigachat) – интеграционная библиотека для работы с LangChain и LangGraph.

[Быстрый старт](https://github.com/ai-forever/langchain-gigachat) | [Сборник примеров](/cookbook/README.md)

[Документация LangChain](https://python.langchain.com/docs/introduction/) | [Документация LangGraph](https://langchain-ai.github.io/langgraph/) | [Чат-бот по документации](https://chat.langchain.com)

### JavaScript/TypeScript ![npm](https://img.shields.io/npm/dm/langchain-gigachat)[![GitHub star chart](https://img.shields.io/github/stars/ai-forever/langchainjs?style=flat-round)](https://www.star-history.com/#ai-forever/langchainjs)

[`langchain-gigachat`](https://github.com/ai-forever/langchainjs) интеграционная библиотека для работы с LangChainJS и LangGraphJS.

[Быстрый старт](https://github.com/ai-forever/langchain-gigachat) | [Сборник примеров](/cookbook/js/README.md)

[Документация LangChainJS](https://js.langchain.com/docs/introduction/) | [Документация LangGraphJS](https://langchain-ai.github.io/langgraphjs/) | [Чат-бот по JS-документации](https://chatjs.langchain.com/)

### Java [![GitHub star chart](https://img.shields.io/github/stars/ai-forever/langchain4j-gigachat?style=flat-round)](https://www.star-history.com/#ai-forever/langchain4j-gigachat)

[`langchain4j-gigachat`](https://github.com/ai-forever/langchain4j-gigachat) – библиотека для работы с фреймворком LangChain4j.

[Быстрый старт](https://github.com/ai-forever/langchain4j-gigachat?tab=readme-ov-file#%D1%83%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0) | [Сборник примеров](https://github.com/ai-forever/langchain4j-gigachat/tree/main/langchain4j-gigachat-examples#%D0%BF%D1%80%D0%B8%D0%BC%D0%B5%D1%80%D1%8B-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B-%D1%81-langchain4j-gigachat)

[Документация LangChain4j](https://docs.langchain4j.dev/)


## SDK для работы с моделями GigaChat

Библиотеки-обертки для работы с [REST API GigaChat](https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/gigachat-api).
Они управляют авторизацией запросов, упрощают отправку сообщений и дают доступ к другим методам API. 

SDK доступны на языках:

### Python [![GitHub Downloads (all assets, all releases)](https://img.shields.io/pypi/dm/gigachat?style=flat-square?style=flat-round)](https://pypistats.org/packages/gigachat)[![GitHub Repo stars](https://img.shields.io/github/stars/ai-forever/gigachat?style=flat-round)](https://star-history.com/#ai-forever/gigachat)

[`gigachat`](https://github.com/ai-forever/gigachat/) | [Сборник примеров](https://github.com/ai-forever/gigachat/tree/main/examples#примеры-работы-с-gigachat) 

### JavaScript/TypeScript ![GitHub Downloads (all assets, all releases)](https://img.shields.io/npm/dm/gigachat?style=flat-square?style=flat-round)[![GitHub Repo stars](https://img.shields.io/github/stars/ai-forever/gigachat-js?style=flat-round)](https://star-history.com/#ai-forever/gigachat-js)

[`gigachat`](https://github.com/ai-forever/gigachat-js) | [Сборник примеров](https://github.com/ai-forever/gigachat-java/blob/main/gigachat-java-example/README.md#примеры-работы-с-библиотекой-gigachat)

### Java [![GitHub Repo stars](https://img.shields.io/github/stars/ai-forever/gigachat-java?style=flat-round)](https://star-history.com/#ai-forever/gigachat-java)

[`gigachat-java`](https://github.com/ai-forever/gigachat-java) | [Сборник примеров](https://github.com/ai-forever/gigachat-js/tree/master/examples#примеры-работы-с-gigachat)

## Утилиты и MCP-сервера

### GPT2GIGA [![GitHub Downloads (all assets, all releases)](https://img.shields.io/pypi/dm/gpt2giga?style=flat-square?style=flat-round)](https://pypistats.org/packages/gpt2giga)[![GitHub Repo stars](https://img.shields.io/github/stars/ai-forever/gpt2giga?style=flat-round)](https://star-history.com/#ai-forever/gpt2giga)

[`gpt2giga`](https://github.com/ai-forever/gpt2giga) — прокси-сервер, перенаправляющий отправленные в OpenAI API запросы в GigaChat API. 

Список протестированных приложений, работающих с GPT2GIGA и GigaChat:

* [Aider](https://aider.chat/) — AI-ассистент для написания приложений. [Запуск и настройка Aider](https://github.com/ai-forever/gpt2giga/tree/main/integrations/aider).
* [n8n](https://n8n.io/) — платформа для создания no-code-агентов.
* [Cline](https://github.com/cline/cline?tab=readme-ov-file#cline--1-on-openrouter) | [Roo Code](https://github.com/RooVetGit/Roo-Code/blob/main/locales/ru/README.md#roo-code-%D1%80%D0%B0%D0%BD%D0%B5%D0%B5-roo-cline) — AI-ассистенты для разработки, которые можно интегрировать в редактор кода.

### MCP-сервера

Model Context Protocol — открытый протокол, который унифицирует обмен контекстом между приложением и LLM. Использование MCP упрощает подключение больших языковых моделей к различным функциям (*инструментам*) и источникам данных.

Подробнее о протоколе — в [официальной документации](https://modelcontextprotocol.io/introduction).

Список MCP-серверов, предоставляющих инструменты для работы с GigaChat и другими сервисами Сбера:

* [Think MCP](https://github.com/ai-forever/think-mcp) — инструмент для реализации размышлений («think») при работе AI-агентов
* [MCP Giga Checker](https://github.com/ai-forever/mcp_giga_checker) — инструмент для проверки переданного текста на наличие содержимого, сгенерированного с помощью нейросетевых моделей.
* [MCP Voice Salute](https://github.com/ai-forever/mcp_voice_salute) — инструменты для работы с [API сервиса синтеза и распознавания речи SaluteSpeech](https://developers.sber.ru/docs/ru/salutespeech/overview);
* [MCP Kandinsky](https://github.com/ai-forever/mcp_kandinsky) — инструмент для генерации изображений с помощью нейросети Kandinsky 3.1.