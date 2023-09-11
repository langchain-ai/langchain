# 🦜️🔗 GigaChain (GigaChat + LangChain)

⚡ Разработка LangChain-style приложений на русском языке с поддержкой GigaChat ⚡

<!--
[![Release Notes](https://img.shields.io/github/release/langchain-ai/langchain)](https://github.com/ai-forever/gigachain/releases)
-->
[![CI](https://github.com/ai-forever/gigachain/actions/workflows/langchain_ci.yml/badge.svg)](https://github.com/ai-forever/gigachain/actions/workflows/langchain_ci.yml)
<!--
[![Experimental CI](https://github.com/ai-forever/gigachain/actions/workflows/langchain_experimental_ci.yml/badge.svg)](https://github.com/ai-forever/gigachain/actions/workflows/langchain_experimental_ci.yml)-->
[![Downloads](https://static.pepy.tech/badge/gigachain/month)](https://pepy.tech/project/gigachain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!--[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)
[![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.gg/6adMQxSpJS)
[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/langchain-ai/langchain)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/langchain-ai/langchain)
[![GitHub star chart](https://img.shields.io/github/stars/langchain-ai/langchain?style=social)](https://star-history.com/#langchain-ai/langchain)
[![Dependency Status](https://img.shields.io/librariesio/github/langchain-ai/langchain)](https://libraries.io/github/langchain-ai/langchain)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langchain)](https://github.com/ai-forever/gigachain/issues)
-->

<!-- Ищете версию на JS/TS? Ознакомьтесь с [LangChain.js](https://github.com/hwchase17/langchainjs).-->

💡Данная библиотека является адаптированной версией библиотеки [LangChain](https://github.com/langchain-ai/langchain) для русского языка с поддержкой GigaChat.

<!--
**Production Support:** As you move your LangChains into production, we'd love to offer more hands-on support.
Fill out [this form](https://airtable.com/appwQzlErAS2qiP0L/shrGtGaVBVAz7NcV2) to share more about what you're building, and our team will get in touch.
-->

## 🚨🚨🚨ВАЖНО!!!🚨🚨🚨

Данная библиотека - очень ранняя альфа-версия. Она находится в процессе перевода и адаптации к GigaChat. Большая часть компонентов пока не проверена на совместимость с GigaChat, поэтому могут возникать ошибки. Пожалуйста, будьте осторожны при использовании этой библиотеки в своих проектах. Будем рады видеть ваши pull request'ы и issues.

Библиотека GigaChain обратно совместима с исходной библиотекой LangChain, вы можете свободно использовать её не только для GigaChat, но и для других LLM в различных комбинациях.

## 🚨Кардинальные изменения для отдельных цепочек (SQLDatabase) с 28.07.23

В попытке сделать `gigachain` более компактным и безопасным, мы переносим отдельные цепочки в langchain_experimental.
Миграция уже началась, но мы сохраняем обратную совместимость до 28.07.
С этой даты мы удалим функциональность из `gigachain`.
Узнайте больше о мотивации и ходе изменений [здесь](https://github.com/hwchase17/langchain/discussions/8043).
О том, как мигрировать ваш код, читайте [здесь](MIGRATE.md).

## Быстрая установка

`pip install gigachain`
<!--
`pip install langchain`
or
`pip install langsmith && conda install langchain -c conda-forge`
-->

## Hello world
```python
"""Пример работы с чатом через gigachain """
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat

chat = GigaChat(user=<user_name>, password=<password>)

messages = [
    SystemMessage(
        content="Ты эмпатичный бот-психолог, который помогает пользователю решить его проблемы."
    )
]

while(True):
    user_input = input("User: ")
    messages.append(HumanMessage(content=user_input))
    res = chat(messages)
    messages.append(res)
    print("Bot: ", res.content)
```

Боле подробный пример работы с GigaChat см. в notebook [Работа с GigaChat](docs/extras/integrations/chat/gigachat.ipynb)

## 🤔 Что это?

Большие языковые модели (LLMs) стали прорывной технологией, позволяя разработчикам создавать приложения, которые раньше были недоступны. Однако использование этих LLMs в изоляции часто недостаточно для создания действительно мощного приложения - настоящая сила проявляется, когда вы можете сочетать их с другими источниками вычислений или знаний.

Эта библиотека направлена на помощь в разработке таких приложений. Примеры таких приложений включают:

**❓ Ответы на вопросы**

- [Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- Пример: [Ответы на вопросы по статьям из wikipedia](https://github.com/ai-forever/gigachain/blob/master/docs/extras/integrations/retrievers/wikipedia.ipynb)

**❓ Ответы на вопросы по конкретным документам**

...
<!--- [Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- Полный пример: [Question Answering over Notion Database](https://github.com/hwchase17/notion-qa)
-->
**💬 Чат-боты**

...
<!-- - [Documentation](https://python.langchain.com/docs/use_cases/chatbots/)
- Полный пример: [Chat-LangChain](https://github.com/hwchase17/chat-langchain)
-->
**🤖 Агенты**

- [Documentation](https://python.langchain.com/docs/modules/agents/)
<!--- Полный пример: [GPT+WolframAlpha](https://huggingface.co/spaces/JavaFXpert/Chat-GPT-LangChain)-->
- Пример: [Игра в стиле DnD с GPT-3.5 и GigaChat](docs/extras/use_cases/agent_simulations/multi_llm_thre_player_dnd.ipynb)

## 📖 Documentation

Please see [here](https://python.langchain.com) for full documentation on:

- Getting started (installation, setting up the environment, simple examples)
- How-To examples (demos, integrations, helper functions)
- Reference (full API docs)
- Resources (high-level explanation of core concepts)






Для получения более подробной информации о данных концепциях, пожалуйста, обратитесь к нашей полной документации.




## 🚀 Что может GigaChain

Есть шесть ключевых направлений, в которых GigaChain может оказать помощь. Ниже они перечислены от самых простых к более сложным:

**📃 LLM и Запросы (Prompts):**

Включает в себя управление запросами, оптимизацию запросов, универсальный интерфейс для всех LLM и стандартные инструменты для работы с LLM.

**🔗 Цепочки (Chains):**

Цепочки выходят за рамки одного вызова LLM и включают в себя последовательность вызовов (будь то к LLM или другому инструменту). GigaChain предоставляет стандартный интерфейс для цепочек, множество интеграций с другими инструментами и цепочки "от начала до конца" для популярных приложений.

**📚 Аугментация данных (Data Augmented Generation):**

Генерация с дополнением данными включает в себя специфические типы цепочек, которые сначала взаимодействуют с внешним источником данных для получения данных, которые затем используются в генерации. Примеры включают в себя суммирование больших текстов и ответы на вопросы по конкретным источникам данных.

**🤖 Агенты (Agents):**

Агенты включают в себя LLM, принимающие решения о том, какие действия предпринимать, выполняя это действие, наблюдая за результатом и повторяя процесс до завершения. GigaChain предоставляет стандартный интерфейс для агентов, выбор агентов и примеры агентов "от начала до конца".

**🧠 Память (Memory):**

Память означает сохранение состояния между вызовами цепочки или агента. GigaChain предоставляет стандартный интерфейс для памяти, коллекцию реализаций памяти и примеры цепочек/агентов, использующих память.

**🧐 Самооценка (Evaluation):**

[BETA] Генеративные модели традиционно сложно оценивать с помощью стандартных метрик. Один из новых способов оценки - использование самих языковых моделей для этой цели. GigaChain предоставляет некоторые запросы и цепочки для помощи в этом.

Для получения более подробной информации о данных концепциях, пожалуйста, обратитесь к нашей [полной документации](https://python.langchain.com).

## 📚 Примеры использования, адаптированные для GigaChat

- [Ответы на вопросы по статьям из wikipedia](docs/extras/integrations/retrievers/wikipedia.ipynb)
- [Суммаризация map-reduce](docs/extras/use_cases/summarization.ipynb) (см. раздел map/reduce)
- [Игра Blade Runner: GPT-4 и GigaChat выясняют, кто из них бот](docs/extras/use_cases/more/fun/blade_runner.ipynb)
- [Работа с хабом промптов, цепочками и парсером JSON](docs/extras/modules/model_io/output_parsers/json.ipynb)
- [Игра в стиле DnD с GPT-3.5 и GigaChat](docs/extras/use_cases/agent_simulations/multi_llm_thre_player_dnd.ipynb)
- [Парсинг списков, содержащихся в ответе](docs/extras/modules/model_io/output_parsers/list.ipynb)
- [Асинхронная работа с LLM](docs/extras/modules/model_io/models/llms/async_llm.ipynb)
- [Использование Elastic для поиска ответов по документам](docs/extras/integrations/retrievers/elastic_qna.ipynb)

## Примеры использования, связанные с другими LLM
- [Агент-менеджер по продажам с автоматическим поиском по каталогу и формированием заказа](docs/extras/modules/agents/how_to/add_memory_openai_functions.ipynb)
- [Поиск ответов в интернете с автоматическими промежуточными вопросами (self-ask)](docs/extras/modules/agents/agent_types/self_ask_with_search.ipynb)

## 💁 Помощь

Как проект с открытым исходным кодом в быстро развивающейся области, мы чрезвычайно открыты для вклада, будь то в виде новой функции, улучшенной инфраструктуры или улучшенной документации.

Подробную информацию о том, как внести свой вклад, можно найти [здесь](.github/CONTRIBUTING.md).
