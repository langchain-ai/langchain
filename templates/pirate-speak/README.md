# pirate-speak

Этот шаблон преобразует пользовательский ввод в `пиратский жаргон`.

## Настройка среды

Установите переменную среды `OPENAI_API_KEY` для доступа к моделям OpenAI.

## Использование

Чтобы использовать этот пакет, у вас должен быть установлен CLI для GigaChain:

```shell
pip install -U gigachain-cli
```

Чтобы создать новый проект GigaChain и установить этот пакет как единственный, вы можете сделать следующее:

```shell
gigachain app new my-app --package pirate-speak
```

Если вы хотите добавить это в существующий проект, просто выполните:

```shell
gigachain app add pirate-speak
```

И добавьте следующий код в файл `server.py`:
```python
from pirate_speak.chain import chain as pirate_speak_chain

add_routes(app, pirate_speak_chain, path="/pirate-speak")
```

(Необязательно) Давайте теперь настроим LangSmith.
LangSmith поможет нам отслеживать, мониторить и отлаживать приложения GigaChain.
LangSmith сейчас находится в частном бета-тестировании, вы можете зарегистрироваться [здесь](https://smith.langchain.com/).
Если у вас нет доступа, вы можете пропустить этот раздел.

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<ваш-api-ключ>
export LANGCHAIN_PROJECT=<ваш-проект>  # если не указано, по умолчанию используется "default"
```

Если вы находитесь в этой директории, то вы можете напрямую запустить экземпляр GigaServe с помощью:

```shell
gigachain serve
```

Это запустит приложение FastAPI, и сервер будет работать локально по адресу 
[http://localhost:8000](http://localhost:8000)

Мы можем увидеть все шаблоны по адресу [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
Мы можем получить доступ к площадке для игр по адресу [http://127.0.0.1:8000/pirate-speak/playground](http://127.0.0.1:8000/pirate-speak/playground)  

Мы можем получить доступ к шаблону из кода с помощью:

```python
from gigaserve.client import RemoteRunnable

runnable = RemoteRunnable("http://localhost:8000/pirate-speak")
```
