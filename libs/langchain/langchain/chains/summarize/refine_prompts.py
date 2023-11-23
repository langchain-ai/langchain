from langchain_core.prompts import PromptTemplate

REFINE_PROMPT_TMPL = (
    "Твоя задача - создать окончательное резюме\n"
    "Мы предоставили существующее резюме до определенного момента: {existing_answer}\n"
    "У нас есть возможность улучшить существующее резюме"
    "(только если это необходимо) с некоторым дополнительным контекстом ниже.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Учитывая новый контекст, улучши оригинальное резюме\n"
    "Если контекст не полезен, верни оригинальное резюме."
)
REFINE_PROMPT = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=REFINE_PROMPT_TMPL,
)


prompt_template = """Напиши краткое резюме следующего:


"{text}"


КРАТКОЕ РЕЗЮМЕ:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
