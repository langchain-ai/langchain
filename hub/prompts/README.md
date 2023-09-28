# Работа с промптами

В этом разделе вы найдете шаблоны промптов и примеры работы с ними с помощью библиотеки GigaChain.

Коллекция содержит следующие примеры:

- [Привет, мир!](hello_world/README.md)

  Самый простой пример, который показывает как обращаться к GigaChat с помощью SDK. В ответ на запрос SDK нейросетевая модель возвращает реплику «Привет, мир!».

- [Вопрос-Ответ](qna/README.md)

  Пример содержит шаблоны промптов для работы с GigaChat в формате Вопрос-Ответ.

- [Суммаризатор](summarize/map_reduce/README.md)

  Пример содержит шаблоны промптов для суммаризации текстов по алгоритму MapReduce.

- [Генерирование синонимов](synonyms/README.md)

  Пример содержит шаблоны промптов для создания заданного количества синонимов к указанному слову.

## Авторизация запросов

Авторизация запросов к GigaChat выполняется с помощью токена авторизации. Токен авторизации можно получить после [подключения проекта GigaChat API в личном кабинете](https://developers.sber.ru/docs/ru/gigachat/api/integration).

> [!NOTE]
> Сейчас проект GigaChat API доступен только юридическим лицам и индивидуальным предпринимателям после подписания договора.

## Шаблоны промптов

Каждый из примеров коллекции содержит один или несколько yaml-файлов с шаблонами промптов.

### Пример шаблона

```yaml
input_variables: [dataset_size_min, dataset_size_max, subject, examples]
output_parser: null
template: 'Сгенерируй от {dataset_size_min} до {dataset_size_max} синонимов для слова "{subject}". Примеры фраз: {examples}. Результат верни в формате JSON-списка без каких либо пояснений, например, ["синоним1", "синоним2", "синоним3", "синоним4"]. Не повторяй фразы из примера и не дублируй фразы.'
template_format: f-string
_type: prompt
```

Шаблон промптов может содержать следующие поля:

- `input_variables` — список переменных, заданных в тексте шаблона промпта. Значения переменных задаются при вызове метода, использующего промпт.

  Пример:

  ```yaml
  input_variables: [dataset_size_min, dataset_size_max, subject]
  ```

- `output_parser` — парсер выходных данных, полученных от нейросетевой модели. Используется для дополнительной обработки и структуризации ответов. Значение по умолчанию — `null`.

  > [!NOTE]
  > Подробнее о парсерах выходных данных читайте в [документации LangChain](https://python.langchain.com/docs/modules/model_io/output_parsers/).

- `template` — текст шаблона. Может содержать переменные, заданные с помощью фигурных скобок. Переменные, использованные в тексте, должны быть заданы в списке `input_variables`.
  
  Пример:

  ```yaml
  template: 'Сгенерируй от {dataset_size_min} до {dataset_size_max} синонимов для слова "{subject}".'
  ```

- `template_format` — формат данных шаблона. Значение по умолчанию: `f-string`.
- `_type` — тип шаблона. Для шаблонов промптов используйте значение `promt`.

### Использование шаблона

Пример использования шаблона:

```python
from langchain.prompts import load_prompt
from langchain.chat_models import GigaChat
from langchain.chains import LLMChain

giga = GigaChat(oauth_token="...")
synonyms_with_examples = load_prompt('lc://prompts/synonyms/synonyms_generation_with_examples.yaml')
text = prompt.format(dataset_size_min=5,
                        dataset_size_max=10,
                        subject="кошка",
                        examples='["кот", "котёнок"]')
```