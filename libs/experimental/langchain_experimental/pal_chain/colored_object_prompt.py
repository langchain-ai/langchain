# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

template = (
    """
# Генерация кода Python3 для решения задач
# Q: На тумбочке находятся красный карандаш, фиолетовая кружка, бордовый брелок, розовый мишка, черная тарелка и синий мяч для снятия стресса. Какого цвета мяч для снятия стресса?
# Поместим объекты в словарь для быстрого поиска
objects = dict()
objects['pencil'] = 'red'
objects['mug'] = 'purple'
objects['keychain'] = 'burgundy'
objects['teddy bear'] = 'fuchsia'
objects['plate'] = 'black'
objects['stress ball'] = 'blue'

# Найдем цвет мяча для снятия стресса
stress_ball_color = objects['stress ball']
answer = stress_ball_color


# Q: На столе вы видите ряд объектов: фиолетовую скрепку, розовый мяч для снятия стресса, коричневый брелок, зеленый зарядный кабель, малиновый спиннер и бордовую ручку. Какого цвета объект сразу справа от мяча для снятия стресса?
# Поместим объекты в список для сохранения порядка
objects = []
objects += [('paperclip', 'purple')] * 1
objects += [('stress ball', 'pink')] * 1
objects += [('keychain', 'brown')] * 1
objects += [('scrunchiephone charger', 'green')] * 1
objects += [('fidget spinner', 'mauve')] * 1
objects += [('pen', 'burgundy')] * 1

# Найдем индекс мяча для снятия стресса
stress_ball_idx = None
for i, object in enumerate(objects):
    if object[0] == 'stress ball':
        stress_ball_idx = i
        break

# Найдем объект сразу справа
direct_right = objects[i+1]

# Проверим цвет объекта справа
direct_right_color = direct_right[1]
answer = direct_right_color


# Q: На тумбочке вы видите следующие предметы, расположенные в ряд: бирюзовую тарелку, бордовый брелок, желтый зарядный кабель, оранжевую кружку, розовую тетрадь и серую чашку. Сколько неоранжевых предметов вы видите слева от бирюзового предмета?
# Поместим объекты в список для сохранения порядка
objects = []
objects += [('plate', 'teal')] * 1
objects += [('keychain', 'burgundy')] * 1
objects += [('scrunchiephone charger', 'yellow')] * 1
objects += [('mug', 'orange')] * 1
objects += [('notebook', 'pink')] * 1
objects += [('cup', 'grey')] * 1

# Найдем индекс бирюзового предмета
teal_idx = None
for i, object in enumerate(objects):
    if object[1] == 'teal':
        teal_idx = i
        break

# Найдем неоранжевые предметы слева от бирюзового предмета
non_orange = [object for object in objects[:i] if object[1] != 'orange']

# Подсчитаем количество неоранжевых предметов
num_non_orange = len(non_orange)
answer = num_non_orange


# Q: {question}
""".strip()
    + "\n"
)

COLORED_OBJECT_PROMPT = PromptTemplate(input_variables=["question"], template=template)
