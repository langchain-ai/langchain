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

> [!WARNING]
> С 29.10.2024 GigaChain изменяет способ взаимодействия с LangChain.
> Проект перестает быть ответвлением LangChain и будет предоставлять всю функциональность в рамках партнерского пакета [langchain-gigachat](https://github.com/ai-forever/langchain-gigachat/tree/master/libs/gigachat).
>
> Это значительно упростит разработку и позволит развивать оригинальные решения GigaChain, а также даст доступ ко всем интеграциям, которые [поддерживает LangChain](https://python.langchain.com/docs/integrations/providers/) и обеспечит поддержку новых версий фреймворка в момент выпуска.
>
> Предыдущую версию GigaChain (v0.2.x) вы можете найти в ветке [v_2.x_legacy](https://github.com/ai-forever/gigachain/tree/v_2.x_legacy).

## О GigaChain

GigaChain – это набор решений для создания приложений с использованием больших языковых моделей (*LLM*), который охватывает все этапы разработки от прототипирования и исследования, до запуска в эксплуатацию и поддержки.

Один из компонентов GigaChain — партнерский пакет [langchain-gigachat](https://github.com/ai-forever/langchain-gigachat/tree/master/libs/gigachat), который позволяет использовать [модели GigaChat](https://developers.sber.ru/docs/ru/gigachat/models) при работе с LangChain.
Также, благодаря этому пакету, вы сможете использовать GigaChat для разработки агентов с помощью [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/).

## Быстрый старт

Чтобы начать работу, установите партнерский пакет:

```sh
pip install langchain-gigachat
```

Запустите простой пример:

```py
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat

# Авторизация в GigaChat
llm = GigaChat(
    credentials="ключ_авторизации",
    scope="GIGACHAT_API_PERS",
    model="GigaChat",
    # Отключает проверку наличия сертификатов НУЦ Минцифры
    verify_ssl_certs=False,
    streaming=False,
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

* `credentials` — ключ_авторизации для обмена сообщениями с GigaChat API. О том, как их получить — в [официальной документации](https://developers.sber.ru/docs/ru/gigachat/quickstart/ind-using-api#poluchenie-avtorizatsionnyh-dannyh).
* `scope` — необязательный параметр, в котором можно указать версию API. Возможные значения:
  
  * `GIGACHAT_API_PERS` — версия API для физических лиц;
  * `GIGACHAT_API_B2B` — доступ для ИП и юридических лиц [по предоплате](https://developers.sber.ru/docs/ru/gigachat/api/tariffs#platnye-pakety2);
  * `GIGACHAT_API_CORP` — доступ для ИП и юридических лиц [по схеме pay-as-you-go](https://developers.sber.ru/docs/ru/gigachat/api/tariffs#oplata-pay-as-you-go).

  По умолчанию запросы передаются в версию для физических лиц.

* `model` — необязательный параметр, в котором можно явно задать [модель GigaChat](https://developers.sber.ru/docs/ru/gigachat/models). По умолчанию запросы передаются в модель `GigaChat`.
* `verify_ssl_certs` — необязательный параметр, с помощью которого можно отключить проверку [сертификатов НУЦ Минцифры](/https://developers.sber.ru/docs/ru/gigachat/certificates).
* `streaming` — необязательный параметр, который включает и отключает [потоковую генерацию токенов](https://developers.sber.ru/docs/ru/gigachat/api/response-token-streaming). Потоковая генерация позволяет повысить отзывчивость интерфейса программы при работе с длинными текстами. По умолчанию `False`.

## Смотрите также

* [Официальная документация LangChain](https://python.langchain.com/docs/introduction/)
* [Официальная документация LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
