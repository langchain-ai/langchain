import copy
import uuid
from typing import Dict, List

import numpy as np
from langchain.tools.base import BaseTool

from langchain_experimental.autonomous_agents.hugginggpt.task_planner import Plan


class Task:
    def __init__(self, task: str, id: int, dep: List[int], args: Dict, tool: BaseTool):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool
        self.status = "pending"
        self.message = ""
        self.result = ""

    def __str__(self) -> str:
        return f"{self.task}({self.args})"

    def save_product(self) -> None:
        import cv2

        if self.task == "video_generator":
            # ndarray to video
            product = np.array(self.product)
            nframe, height, width, _ = product.shape
            video_filename = uuid.uuid4().hex[:6] + ".mp4"
            fps = 30  # Frames per second
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            video_out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            for frame in self.product:
                video_out.write(frame)
            video_out.release()
            self.result = video_filename
        elif self.task == "image_generator":
            # PIL.Image to image
            filename = uuid.uuid4().hex[:6] + ".png"
            self.product.save(filename)  # type: ignore
            self.result = filename

    def completed(self) -> bool:
        return self.status == "completed"

    def failed(self) -> bool:
        return self.status == "failed"

    def pending(self) -> bool:
        return self.status == "pending"

    def run(self) -> str:
        from diffusers.utils import load_image

        try:
            new_args = copy.deepcopy(self.args)
            for k, v in new_args.items():
                if k == "image":
                    new_args["image"] = load_image(v)
            if self.task in ["video_generator", "image_generator", "text_reader"]:
                self.product = self.tool(**new_args)
            else:
                self.result = self.tool(**new_args)
        except Exception as e:
            self.status = "failed"
            self.message = str(e)
        self.status = "completed"
        self.save_product()

        return self.result


class TaskExecutor:
    """Load tools to execute tasks."""

    def __init__(self, plan: Plan):
        self.plan = plan
        self.tasks = []
        self.id_task_map = {}
        self.status = "pending"
        for step in self.plan.steps:
            task = Task(step.task, step.id, step.dep, step.args, step.tool)
            self.tasks.append(task)
            self.id_task_map[step.id] = task

    def completed(self) -> bool:
        return all(task.completed() for task in self.tasks)

    def failed(self) -> bool:
        return any(task.failed() for task in self.tasks)

    def pending(self) -> bool:
        return any(task.pending() for task in self.tasks)

    def check_dependency(self, task: Task) -> bool:
        for dep_id in task.dep:
            if dep_id == -1:
                continue
            dep_task = self.id_task_map[dep_id]
            if dep_task.failed() or dep_task.pending():
                return False
        return True

    def update_args(self, task: Task) -> None:
        for dep_id in task.dep:
            if dep_id == -1:
                continue
            dep_task = self.id_task_map[dep_id]
            for k, v in task.args.items():
                if f"<resource-{dep_id}>" in v:
                    task.args[k].replace(f"<resource-{dep_id}>", dep_task.result)

    def run(self) -> str:
        for task in self.tasks:
            print(f"running {task}")
            if task.pending() and self.check_dependency(task):
                self.update_args(task)
                task.run()
        if self.completed():
            self.status = "completed"
        elif self.failed():
            self.status = "failed"
        else:
            self.status = "pending"
        return self.status

    def __str__(self) -> str:
        result = ""
        for task in self.tasks:
            result += f"{task}\n"
            result += f"status: {task.status}\n"
            if task.failed():
                result += f"message: {task.message}\n"
            if task.completed():
                result += f"result: {task.result}\n"
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def describe(self) -> str:
        return self.__str__()
