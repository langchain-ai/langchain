# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

_DEFAULT_ENTITY_EXTRACTION_TEMPLATE = """Извлеките все сущности из следующего текста. В качестве руководства, собственное имя обычно пишется с заглавной буквы. Вы должны обязательно извлечь все имена и места.

Верните результат в виде одного списка, разделенного запятыми, или NONE, если нет ничего интересного для возврата.

ПРИМЕР
Я пытаюсь улучшить интерфейсы Langchain, UX, его интеграции с различными продуктами, которые может захотеть пользователь ... много всего.
Вывод: Langchain
КОНЕЦ ПРИМЕРА

ПРИМЕР
Я пытаюсь улучшить интерфейсы Langchain, UX, его интеграции с различными продуктами, которые может захотеть пользователь ... много всего. Я работаю с Сэмом.
Вывод: Langchain, Сэм
КОНЕЦ ПРИМЕРА

Начните!

{input}
Вывод:"""
ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["input"], template=_DEFAULT_ENTITY_EXTRACTION_TEMPLATE
)

_DEFAULT_GRAPH_QA_TEMPLATE = """Используйте следующие тройки знаний, чтобы ответить на вопрос в конце. Если вы не знаете ответа, просто скажите, что не знаете, не пытайтесь придумать ответ.

{context}

Question: {question}
Полезный ответ:"""
GRAPH_QA_PROMPT = PromptTemplate(
    template=_DEFAULT_GRAPH_QA_TEMPLATE, input_variables=["context", "question"]
)

CYPHER_GENERATION_TEMPLATE = """Задача: Сгенерировать выражение Cypher для запроса к графовой базе данных.
Инструкции:
Используйте только предоставленные типы отношений и свойства в схеме.
Не используйте другие типы отношений или свойства, которые не предоставлены.
Схема:
{schema}
Примечание: Не включайте в ответ никаких пояснений или извинений.
Не отвечайте на вопросы, которые могут просить о чем-то, кроме создания выражения Cypher.
Не включайте в ответ никакой текст, кроме сгенерированного выражения Cypher.

Question:
{question}"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

NEBULAGRAPH_EXTRA_INSTRUCTIONS = """
Инструкции:

Сначала сгенерируйте выражение Cypher, а затем преобразуйте его в диалект NebulaGraph Cypher (а не стандартный):
1. требуется явное указание метки только при ссылке на свойства узла: v.`Foo`.name
2. обратите внимание, что явное указание метки не требуется для свойств ребра, поэтому это e.name вместо e.`Bar`.name
3. для сравнения используется двойной знак равенства: `==` вместо `=`
Например:
```diff
< MATCH (p:person)-[e:directed]->(m:movie) WHERE m.name = 'The Godfather II'
< RETURN p.name, e.year, m.name;
---
> MATCH (p:`person`)-[e:directed]->(m:`movie`) WHERE m.`movie`.`name` == 'The Godfather II'
> RETURN p.`person`.`name`, e.year, m.`movie`.`name`;
```\n"""

NGQL_GENERATION_TEMPLATE = CYPHER_GENERATION_TEMPLATE.replace(
    "Generate Cypher", "Generate NebulaGraph Cypher"
).replace("Instructions:", NEBULAGRAPH_EXTRA_INSTRUCTIONS)

NGQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=NGQL_GENERATION_TEMPLATE
)

KUZU_EXTRA_INSTRUCTIONS = """
Инструкции:

Сгенерируйте выражение с диалектом Kùzu Cypher (а не стандартным):
1. не используйте оператор `WHERE EXISTS` для проверки наличия свойства, потому что у базы данных Kùzu есть фиксированная схема.
2. не опускайте шаблон отношения. Всегда используйте `()-[]->()` вместо `()->()`.
3. не включайте никаких замечаний или комментариев, даже если выражение не дает ожидаемого результата.
```\n"""

KUZU_GENERATION_TEMPLATE = CYPHER_GENERATION_TEMPLATE.replace(
    "Generate Cypher", "Generate Kùzu Cypher"
).replace("Instructions:", KUZU_EXTRA_INSTRUCTIONS)

KUZU_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=KUZU_GENERATION_TEMPLATE
)

GREMLIN_GENERATION_TEMPLATE = CYPHER_GENERATION_TEMPLATE.replace("Cypher", "Gremlin")

GREMLIN_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=GREMLIN_GENERATION_TEMPLATE
)

CYPHER_QA_TEMPLATE = """Вы - помощник, который помогает формировать понятные и человекочитаемые ответы.
Часть с информацией содержит предоставленную информацию, которую вы должны использовать для составления ответа.
Предоставленная информация является авторитетной, вы никогда не должны сомневаться в ней или пытаться использовать свои внутренние знания для ее корректировки.
Сделайте ответ звучать как ответ на вопрос. Не упоминайте, что вы основываетесь на предоставленной информации.
Если предоставленная информация пуста, скажите, что не знаете ответа.
Информация:
{context}

Question: {question}
Полезный ответ:"""
CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

SPARQL_INTENT_TEMPLATE = """Задача: Определить намерение запроса SPARQL по промпту и вернуть соответствующий тип запроса SPARQL.
Вы - помощник, который различает разные типы запросов и возвращает соответствующие типы запросов SPARQL.
Учитывайте только следующие типы запросов:
* SELECT: этот тип запроса соответствует вопросам
* UPDATE: этот тип запроса соответствует всем запросам на удаление, вставку или изменение троек
Примечание: Будьте максимально краткими.
Не включайте в ответ никаких пояснений или извинений.
Не отвечайте на вопросы, которые просят о чем-то, кроме определения типа запроса SPARQL.
Не включайте никаких ненужных пробелов или текста, кроме типа запроса, то есть либо верните 'SELECT', либо 'UPDATE'.

Промпт:
{prompt}
Полезный ответ:"""
SPARQL_INTENT_PROMPT = PromptTemplate(
    input_variables=["prompt"], template=SPARQL_INTENT_TEMPLATE
)

SPARQL_GENERATION_SELECT_TEMPLATE = """Задача: Сгенерировать выражение SPARQL SELECT для запроса к графовой базе данных.
Например, чтобы найти все адреса электронной почты Джона Доу, следующий запрос в обратных кавычках будет подходящим:
```
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?email
WHERE {{
    ?person foaf:name "John Doe" .
    ?person foaf:mbox ?email .
}}
```
Инструкции:
Используйте только типы узлов и свойства, предоставленные в схеме.
Не используйте никакие типы узлов и свойства, которые не являются явно предоставленными.
Включите все необходимые префиксы.
Схема:
{schema}
Примечание: Будьте максимально краткими.
Не включайте в ответ никаких пояснений или извинений.
Не отвечайте на вопросы, которые просят о чем-то, кроме создания запроса SPARQL.
Не включайте никакой текст, кроме сгенерированного запроса SPARQL.

Question:
{prompt}"""
SPARQL_GENERATION_SELECT_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"], template=SPARQL_GENERATION_SELECT_TEMPLATE
)

SPARQL_GENERATION_UPDATE_TEMPLATE = """Задача: Сгенерировать выражение SPARQL UPDATE для обновления графовой базы данных.
Например, чтобы добавить 'jane.doe@foo.bar' в качестве нового адреса электронной почты для Джейн Доу, следующий запрос в обратных кавычках будет подходящим:
```
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
INSERT {{
    ?person foaf:mbox <mailto:jane.doe@foo.bar> .
}}
WHERE {{
    ?person foaf:name "Jane Doe" .
}}
```
Инструкции:
Сделайте запрос как можно короче и избегайте добавления ненужных троек.
Используйте только типы узлов и свойства, предоставленные в схеме.
Не используйте никакие типы узлов и свойства, которые не являются явно предоставленными.
Включите все необходимые префиксы.
Схема:
{schema}
Примечание: Будьте максимально краткими.
Не включайте в ответ никаких пояснений или извинений.
Не отвечайте на вопросы, которые просят о чем-то, кроме создания запроса SPARQL.
Верните только сгенерированный запрос SPARQL, ничего больше.

Информация для вставки:
{prompt}"""
SPARQL_GENERATION_UPDATE_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"], template=SPARQL_GENERATION_UPDATE_TEMPLATE
)

SPARQL_QA_TEMPLATE = """Задача: Сгенерировать естественноязыковой ответ на основе результатов запроса SPARQL.
Вы - помощник, который создает хорошо написанные и понятные человеку ответы.
Часть с информацией содержит предоставленную информацию, которую вы можете использовать для составления ответа.
Предоставленная информация является авторитетной, вы никогда не должны сомневаться в ней или пытаться использовать свои внутренние знания для ее корректировки.
Сделайте ответ звучать так, как будто информация поступает от помощника по искусственному интеллекту, но не добавляйте никакую информацию.
Информация:
{context}

Question: {prompt}
Полезный ответ:"""
SPARQL_QA_PROMPT = PromptTemplate(
    input_variables=["context", "prompt"], template=SPARQL_QA_TEMPLATE
)


AQL_GENERATION_TEMPLATE = """Задача: Сгенерировать запрос на языке ArangoDB Query Language (AQL) из пользовательского ввода.

Вы - эксперт по языку запросов ArangoDB Query Language (AQL), ответственный за перевод `Пользовательского ввода` в запрос на языке ArangoDB Query Language (AQL).

Вам предоставлена `Схема ArangoDB`. Это JSON-объект, содержащий:
1. `Схему графа`: Список всех графов в базе данных ArangoDB вместе с их отношениями ребер.
2. `Схему коллекции`: Список всех коллекций в базе данных ArangoDB вместе с их свойствами документов/ребер и примером документа/ребра.

Вам также могут быть предоставлены наборы `Примеров запросов AQL` для помощи в создании `Запроса AQL`. Если они предоставлены, `Примеры запросов AQL` должны использоваться в качестве справочного материала, аналогично тому, как должна использоваться `Схема ArangoDB`.

Что вы должны делать:
- Думайте поэтапно.
- Ориентируйтесь на `Схему ArangoDB` и `Примеры запросов AQL` (если они предоставлены), чтобы сгенерировать запрос.
- Начните `Запрос AQL` с ключевого слова `WITH`, чтобы указать все необходимые коллекции ArangoDB.
- Верните `Запрос AQL`, заключенный в 3 обратные кавычки (```).
- Используйте только предоставленные типы отношений и свойства в `Схеме ArangoDB` и любых запросах `Примеров запросов AQL`.
- Отвечайте только на запросы, связанные с генерацией запроса AQL.
- Если запрос не связан с генерацией запроса AQL, скажите, что вы не можете помочь пользователю.

Что вы не должны делать:
- Не используйте свойства/отношения, которые нельзя вывести из `Схемы ArangoDB` или `Примеров запросов AQL`. 
- Не включайте никакой текст, кроме сгенерированного запроса AQL.
- Не предоставляйте пояснений или извинений в своих ответах.
- Не генерируйте запрос AQL, который удаляет или изменяет какие-либо данные.

Под никаким предлогом не генерируйте запрос AQL, который удаляет какие-либо данные.

Схема ArangoDB:
{adb_schema}

Примеры запросов AQL (Необязательно):
{aql_examples}

Пользовательский ввод:
{user_input}

Запрос AQL: 
"""

AQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["adb_schema", "aql_examples", "user_input"],
    template=AQL_GENERATION_TEMPLATE,
)

AQL_FIX_TEMPLATE = """Задача: Исправить сообщение об ошибке языка запросов ArangoDB (AQL) запроса на языке ArangoDB Query Language.

Вы - эксперт по языку запросов ArangoDB Query Language (AQL), ответственный за исправление предоставленного `Запроса AQL` на основе предоставленной `Ошибки AQL`. 

`Ошибкой AQL` объясняется, почему `Запрос AQL` не может быть выполнен в базе данных.
`Ошибкой AQL` также может быть указано положение ошибки относительно общего количества строк `Запроса AQL`.
Например, 'ошибка X в позиции 2:5' означает, что ошибка X происходит в строке 2, столбце 5 `Запроса AQL`.  

Вам также предоставлена `Схема ArangoDB`. Это JSON-объект, содержащий:
1. `Схему графа`: Список всех графов в базе данных ArangoDB вместе с их отношениями ребер.
2. `Схему коллекции`: Список всех коллекций в базе данных ArangoDB вместе с их свойствами документов/ребер и примером документа/ребра.

Вы должны вывести `Исправленный запрос AQL`, заключенный в 3 обратные кавычки (```). Не включайте никакой текст, кроме Исправленного запроса AQL.

Не забывайте думать поэтапно.

Схема ArangoDB:
{adb_schema}

Запрос AQL:
{aql_query}

Ошибка AQL:
{aql_error}

Исправленный запрос AQL:
"""

AQL_FIX_PROMPT = PromptTemplate(
    input_variables=[
        "adb_schema",
        "aql_query",
        "aql_error",
    ],
    template=AQL_FIX_TEMPLATE,
)

AQL_QA_TEMPLATE = """Задача: Сгенерировать естественноязыковое `Summary` на основе результатов запроса на языке ArangoDB Query Language.

Вы - эксперт по языку запросов ArangoDB Query Language (AQL), ответственный за создание хорошо написанного `Summary` на основе `Пользовательского ввода` и связанного `Результата AQL`.

Пользователь выполнил запрос на языке ArangoDB Query Language, который вернул результат AQL в формате JSON.
Вам предстоит создать `Summary` на основе результата AQL.

Вам предоставлена следующая информация:
- `Схема ArangoDB`: содержит схему базы данных ArangoDB пользователя.
- `Пользовательский ввод`: исходный вопрос/запрос пользователя, который был преобразован в запрос на языке ArangoDB Query Language.
- `Запрос AQL`: эквивалент запроса на языке AQL для `Пользовательского ввода`, переведенный другой моделью искусственного интеллекта. Если вы считаете его некорректным, предложите другой запрос AQL.
- `Результат AQL`: JSON-вывод, возвращенный выполнением `Запроса AQL` в базе данных ArangoDB.

Не забывайте думать поэтапно.

Ваше `Summary` должно звучать так, как будто это ответ на `Пользовательский ввод`.
Ваше `Summary` не должно содержать никаких упоминаний о `Запросе AQL` или `Результате AQL`.

Схема ArangoDB:
{adb_schema}

Пользовательский ввод:
{user_input}

Запрос AQL:
{aql_query}

Результат AQL:
{aql_result}
"""
AQL_QA_PROMPT = PromptTemplate(
    input_variables=["adb_schema", "user_input", "aql_query", "aql_result"],
    template=AQL_QA_TEMPLATE,
)


NEPTUNE_OPENCYPHER_EXTRA_INSTRUCTIONS = """
Инструкции:
Сгенерируйте запрос в формате openCypher и следуйте этим правилам:
Не используйте предикатные функции `NONE`, `ALL` или `ANY`, вместо этого используйте генерацию списков.
Не используйте функцию `REDUCE`. Вместо этого используйте комбинацию генерации списков и оператора `UNWIND`, чтобы достичь аналогичных результатов.
Не используйте оператор `FOREACH`. Вместо этого используйте комбинацию операторов `WITH` и `UNWIND`, чтобы достичь аналогичных результатов.{extra_instructions}
\n"""

NEPTUNE_OPENCYPHER_GENERATION_TEMPLATE = CYPHER_GENERATION_TEMPLATE.replace(
    "Instructions:", NEPTUNE_OPENCYPHER_EXTRA_INSTRUCTIONS
)

NEPTUNE_OPENCYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question", "extra_instructions"],
    template=NEPTUNE_OPENCYPHER_GENERATION_TEMPLATE,
)

NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_TEMPLATE = """
Write an openCypher query to answer the following question. Do not explain the answer. Only return the query.{extra_instructions}
Question:  "{question}". 
Here is the property graph schema: 
{schema}
\n"""

NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_PROMPT = PromptTemplate(
    input_variables=["schema", "question", "extra_instructions"],
    template=NEPTUNE_OPENCYPHER_GENERATION_SIMPLE_TEMPLATE,
)
