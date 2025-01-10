[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<br />
<div align="center">

  <a href="https://github.com/ai-forever/gigachain">
    <img src="docs/static/img/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h1 align="center">🦜️🔗 GigaChain (GigaChat + LangChain)</h1>

  <p align="center">
    Набор решений для разработки LLM-приложений на русском языке с поддержкой GigaChat
    <br />
    <a href="https://github.com/ai-forever/gigachain/issues">Создать issue</a>
    ·
    <a href="https://developers.sber.ru/docs/ru/gigachat/sdk/overview">Документация GigaChain</a>
  </p>
</div>


![Product Name Screen Shot](/docs/static/img/logo-with-backgroung.png)

---

> [!NOTE]
> С 29.10.2024 GigaChain предоставляет всю функциональность в рамках партнерского пакета [langchain-gigachat](https://github.com/ai-forever/langchain-gigachat/tree/master/libs/gigachat).
>
> Это значительно упрощает разработку, обеспечивая доступ ко всем интеграциям, которые [поддерживает LangChain](https://python.langchain.com/docs/integrations/providers/), а также поддержку новых версий фреймворка в момент выпуска.
>
> Предыдущую версию GigaChain (v0.2.x) вы можете найти в ветке [v_2.x_legacy](https://github.com/ai-forever/gigachain/tree/v_2.x_legacy).

# О GigaChain

GigaChain – это набор решений для создания приложений с использованием больших языковых моделей (*LLM*), который охватывает все этапы разработки от прототипирования и исследования, до запуска в эксплуатацию и поддержки.

В состав GigaChain входят две библиотеки для работы с [моделями GigaChat](https://developers.sber.ru/docs/ru/gigachat/models):

* langchain-gigachat — партнерский пакет LangChain, популярного open source фреймворка для разработки комплексных LLM-приложений. Библиотека позволяет использовать все возможности фреймворка и моделей GigaChat, в том числе создание агентов с помощью [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/).
  * [Официальная документация LangChain](https://python.langchain.com/docs/introduction/).
  * [Страница партнерского пакета](https://python.langchain.com/docs/integrations/llms/gigachat/) в официальной документации LangChain.
* gigachat — обертка для [REST API GigaChat](https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/gigachat-api). Библиотека управляет авторизацией запросов, упрощает отправку сообщений в модели GigaChat и предоставляет другие методы для работы с API.

  Подробная информация о библиотеке — [в репозитории](https://github.com/ai-forever/gigachat).

В этом репозитории вы найдете краткие инструкции по началу работы с библиотеками, а так же ссылки на различные примеры их использования.

## Требования

Для работы с библиотеками langchain-gigachat и gigachat вам понадобятся:

* Python версии 3.9 и выше;
* Ключ авторизации для работы с API. О том, как получить ключ авторизации — в [документации GigaChat API](https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-using-api#poluchenie-avtorizatsionnyh-dannyh);
* [Сертификаты НУЦ Минцифры](https://developers.sber.ru/docs/ru/gigachat/certificates).

  Если нужно, вы можете отключить проверку сертификатов. Подробнее — в примерах ниже.

## Быстрый старт

### langchain-gigachat

Для установки библиотеки используйте менеджер пакетов pip:

```sh
pip install langchain-gigachat
```

Запустите простой пример:

```py
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat

giga = GigaChat(
    # Для авторизации запросов используйте ключ, полученный в проекте GigaChat API
    credentials="ваш_ключ_авторизации",
    verify_ssl_certs=False,
)

messages = [
    SystemMessage(
        content="Ты эмпатичный бот-психолог, который помогает пользователю решить его проблемы."
    )
]

while(True):
    user_input = input("Пользователь: ")
    if user_input == "пока":
      break
    messages.append(HumanMessage(content=user_input))
    res = llm.invoke(messages)
    messages.append(res)
    print("GigaChat: ", res.content)
```

Объект GigaChat принимает параметры:

* `credentials` — ключ авторизации для обмена сообщениями с GigaChat API. [Подробнее о получении ключа авторизации](https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-using-api#poluchenie-avtorizatsionnyh-dannyh).
* `scope` — необязательный параметр, в котором можно указать версию API. По умолчанию запросы передаются в версию для физических лиц. Возможные значения:
  
  * `GIGACHAT_API_PERS` — версия API для физических лиц;
  * `GIGACHAT_API_B2B` — доступ для ИП и юридических лиц [по предоплате](https://developers.sber.ru/docs/ru/gigachat/api/tariffs#platnye-pakety2);
  * `GIGACHAT_API_CORP` — доступ для ИП и юридических лиц [по схеме pay-as-you-go](https://developers.sber.ru/docs/ru/gigachat/api/tariffs#oplata-pay-as-you-go).

* `model` — необязательный параметр, в котором можно задать [модель GigaChat](https://developers.sber.ru/docs/ru/gigachat/models). По умолчанию запросы передаются в модель GigaChat Lite (`model="GigaChat"`).
* `verify_ssl_certs` — необязательный параметр, с помощью которого можно отключить проверку [сертификатов НУЦ Минцифры](/https://developers.sber.ru/docs/ru/gigachat/certificates).
* `streaming` — необязательный параметр, который включает и отключает [потоковую генерацию токенов](https://developers.sber.ru/docs/ru/gigachat/api/response-token-streaming). По умолчанию `False`. Потоковая генерация позволяет повысить отзывчивость интерфейса программы при работе с длинными текстами.

> [!TIP]
> Спросите [чат-бот LangChain](https://chat.langchain.com/), о том, как использовать GigaChat с инструментами фреймворка.
> 
> Исходный код чат-бота — в [репозитории chat-langchain](https://github.com/langchain-ai/chat-langchain).

### gigachat

Для установки библиотеки используйте менеджер пакетов pip:

```sh
pip install gigachat
```

Вызовите подходящий метод для запроса в API:

```py
from gigachat import GigaChat

# Для авторизации запросов используйте ключ, полученный в проекте GigaChat API
with GigaChat(credentials="ваш_ключ_авторизации", verify_ssl_certs=False) as giga:
    response = giga.chat("Какие факторы влияют на стоимость страховки на дом?")
    print(response.choices[0].message.content)
```

Объект GigaChat принимает параметры:

* `credentials` — ключ авторизации для обмена сообщениями с GigaChat API. [Подробнее о получении ключа авторизации](https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-using-api#poluchenie-avtorizatsionnyh-dannyh).
* `scope` — необязательный параметр, в котором можно указать версию API. Возможные значения:
  
  * `GIGACHAT_API_PERS` — версия API для физических лиц;
  * `GIGACHAT_API_B2B` — доступ для ИП и юридических лиц [по предоплате](https://developers.sber.ru/docs/ru/gigachat/api/tariffs#platnye-pakety2);
  * `GIGACHAT_API_CORP` — доступ для ИП и юридических лиц [по схеме pay-as-you-go](https://developers.sber.ru/docs/ru/gigachat/api/tariffs#oplata-pay-as-you-go).

  По умолчанию запросы передаются в версию для физических лиц.

* `model` — необязательный параметр, в котором можно явно задать [модель GigaChat](https://developers.sber.ru/docs/ru/gigachat/models). По умолчанию запросы передаются в модель GigaChat Lite (значение поля `GigaChat`).
* `verify_ssl_certs` — необязательный параметр, с помощью которого можно отключить проверку [сертификатов НУЦ Минцифры](/https://developers.sber.ru/docs/ru/gigachat/certificates).

В отличие от библиотеки langchain-gigachat, для запуска потоковой передачи используйте метод `stream()`:

```py
from gigachat import GigaChat

for chunk in GigaChat(credentials="ваш_ключ_авторизации",verify_ssl_certs=False, scope="GIGACHAT_API_PERS", model="GigaChat-Max").stream("Напиши рассказ про двух котят."):
    print(chunk.choices[0].delta.content, end="", flush=True)
```
> [!TIP]
> Больше примеров работы с библиотекой — [в репозитории](https://github.com/ai-forever/gigachat/tree/main/examples).

## Примеры

При запуске примеров могут возникать проблемы связанные с особенностями локального окружения Python.
Чтобы их избежать используйте чистое виртуальное окружение.

Список интерактивных примеров в формате блокнотов Jupyter:

* Retrieval-Augmented Generation (RAG):
  * [Ответы на вопросы по заданной книге](/docs/docs_ru/cookbook/gigachat_qa.ipynb)
  * [RAG с текстовым поиском на основе Yandex Search API](/cookbook/yandex_search/retriever.ipynb)
* Агенты:
  * [Агент для работы с функциями](/docs/docs_ru/cookbook/gigachat_functions_agent.ipynb)
  * [Агент «Продавец телефонов»](/docs/docs_ru/cookbook/agents/gigachat_phone_agent.ipynb)
  * [Агент с текстовым поиском на основе Yandex Search API](/cookbook/yandex_search/tool.ipynb)
  * [Агент риэлтор, с функциями GigaChat](/cookbook/realestate/realestate.ipynb)
  * [Дебаты агентов с разными ролями](/cookbook/agent_debates/README.md)
  * [Агент для получения рекомендаций Spotify](/docs/docs_ru/cookbook/playlists.ipynb)
* [Извлечение структурированной информации](/docs/docs_ru/cookbook/extraction.ipynb)
* Работа с изображениями:
  * [Распознавание изображения](/cookbook/gigachat_vision/gigachat_vision_simple.ipynb)
  * [Генерация структурированных данных на основе изображений](/cookbook/gigachat_vision/gigachat_vision.ipynb)
  * [Получение изображений и видео после генерации](/cookbook/images_and_videos/gigachat_with_images.ipynb)

## Смотрите также

* Документация GigaChat API:
  * [Быстрый страт для физических лиц](https://developers.sber.ru/docs/ru/gigachat/individuals-quickstart)
  * [Быстрый страт для ИП и юридических лиц](https://developers.sber.ru/docs/ru/gigachat/legal-quickstart)
* [Документация LangChain](https://python.langchain.com/docs/introduction/)
* [Документация LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
