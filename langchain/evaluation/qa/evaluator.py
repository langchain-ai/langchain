from pydantic import BaseModel
from typing import Optional, List
from langchain.chains.base import Chain



class QADataPoint(BaseModel):
    question: str
    answer: str
    error: Optional[str]
    prediction: Optional[str]
    ai_grade: Optional[str]


def predict_qa(chain: Chain, datapoints: List[QADataPoint], question_key: str="question", prediction_key: Optional[str] = None, silent_errors: bool = False) -> None:
    for data in datapoints:
        try:
            prediction_dict = chain({question_key: data.question}, return_only_outputs=True)
        except Exception as e:
            if silent_errors:
                data.error = str(e)
                continue
            else:
                raise e
        if prediction_key is not None:
            data.prediction = prediction_dict[prediction_key]
        elif len(prediction_dict) == 1:
            data.prediction = list(prediction_dict.values())[0]
        else:
            raise ValueError(
                "No prediction key was specified, and got multiple outputs so not "
                f"sure which one to use: {prediction_dict}. Please either "
                f"specify a `prediction_key` or change the chain to return "
                f"a single output"
            )

def eval_qa(
        eval_chain: Chain,
        datapoints: List[QADataPoint],
        question_key: str = "query",
        answer_key: str = "answer",
        prediction_key: str = "result",
        grade_key: str = "text"
    ) -> None:
        """Evaluate question answering examples and predictions."""
        data_to_grade = [d for d in datapoints if d.prediction is not None]
        inputs = [
            {
                question_key: data.question,
                answer_key: data.answer,
                prediction_key: data.prediction,
            }
            for data in data_to_grade
        ]

        grades = eval_chain.apply(inputs)
        for i, grade in enumerate(grades):
            data_to_grade[i].ai_grade = grade[grade_key]
    