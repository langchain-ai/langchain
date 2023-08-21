# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

template = (
    '''
Q: У Оливии было $23. Она купила пять бейглов по $3 каждый. Сколько денег у неё осталось?

# solution in Python:


def solution():
    """У Оливии было $23. Она купила пять бейглов по $3 каждый. Сколько денег у неё осталось?"""
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result





Q: У Майкла было 58 мячей для гольфа. Во вторник он потерял 23 мяча. В среду он потерял еще 2. Сколько мячей для гольфа у него осталось в конце среды?

# solution in Python:


def solution():
    """У Майкла было 58 мячей для гольфа. Во вторник он потерял 23 мяча. В среду он потерял еще 2. Сколько мячей для гольфа у него осталось в конце среды?"""
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result





Q: В серверной было девять компьютеров. С понедельника по четверг каждый день устанавливали по пять новых компьютеров. Сколько компьютеров теперь в серверной?

# solution in Python:


def solution():
    """В серверной было девять компьютеров. С понедельника по четверг каждый день устанавливали по пять новых компьютеров. Сколько компьютеров теперь в серверной?"""
    computers_initial = 9
    computers_per_day = 5
    num_days = 4  # 4 days between monday and thursday
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result





Q: У Шона было пять игрушек. На Рождество он получил по две игрушки от мамы и папы. Сколько игрушек у него теперь?

# solution in Python:


def solution():
    """У Шона было пять игрушек. На Рождество он получил по две игрушки от мамы и папы. Сколько игрушек у него теперь?"""
    toys_initial = 5
    mom_toys = 2
    dad_toys = 2
    total_received = mom_toys + dad_toys
    total_toys = toys_initial + total_received
    result = total_toys
    return result





Q: У Джейсона было 20 леденцов. Он отдал некоторые леденцы Денни. Теперь у Джейсона 12 леденцов. Сколько леденцов Джейсон отдал Денни?

# solution in Python:


def solution():
    """У Джейсона было 20 леденцов. Он отдал некоторые леденцы Денни. Теперь у Джейсона 12 леденцов. Сколько леденцов Джейсон отдал Денни?"""
    jason_lollipops_initial = 20
    jason_lollipops_after = 12
    denny_lollipops = jason_lollipops_initial - jason_lollipops_after
    result = denny_lollipops
    return result





Q: У Лии было 32 шоколадки, а у её сестры - 42. Если они съели 35, сколько шоколадок у них осталось в общей сложности?

# solution in Python:


def solution():
    """У Лии было 32 шоколадки, а у её сестры - 42. Если они съели 35, сколько шоколадок у них осталось в общей сложности?"""
    leah_chocolates = 32
    sister_chocolates = 42
    total_chocolates = leah_chocolates + sister_chocolates
    chocolates_eaten = 35
    chocolates_left = total_chocolates - chocolates_eaten
    result = chocolates_left
    return result





Q: Если на парковке 3 автомобиля и приезжают еще 2, сколько автомобилей теперь на парковке?

# solution in Python:


def solution():
    """Если на парковке 3 автомобиля и приезжают еще 2, сколько автомобилей теперь на парковке?"""
    cars_initial = 3
    cars_arrived = 2
    total_cars = cars_initial + cars_arrived
    result = total_cars
    return result





Q: В роще 15 деревьев. Сегодня садовники посадят еще деревья. Когда они закончат, в роще будет 21 дерево. Сколько деревьев садовники посадили сегодня?

# solution in Python:


def solution():
    """В роще 15 деревьев. Сегодня садовники посадят еще деревья. Когда они закончат, в роще будет 21 дерево. Сколько деревьев садовники посадили сегодня?"""
    trees_initial = 15
    trees_after = 21
    trees_added = trees_after - trees_initial
    result = trees_added
    return result





Q: {question}

# solution in Python:
'''.strip()
    + "\n\n\n"
)
MATH_PROMPT = PromptTemplate(input_variables=["question"], template=template)
