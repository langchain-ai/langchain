# execute task by calling huggingface tools
import copy
import uuid
import numpy as np
import cv2
from typing import List, Dict
from langchain.experimental.plan_and_execute.schema import Plan
from diffusers.utils import load_image

class Task:
    def __init__(self, task: str, id: int, dep: List[int], args: Dict, tool):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool= tool
        self.result = None
        self.status = "pending"
        self.message = None

    def __str__(self):
        return f"{self.task}({self.args})"

    def save_result(self):
        if self.task == "video_generator":
            # ndarray to video
            result = np.array(self.result)
            nframe, height, width, _ = result.shape
            video_filename = uuid.uuid4().hex[:6] + ".mp4"
            fps = 30  # Frames per second
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 video
            video_out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            for frame in self.result:
                video_out.write(frame)
            video_out.release()
            self.result = video_filename
        elif self.task == "image_generator":
            # PIL.Image to image
            filename = uuid.uuid4().hex[:6] + ".png"
            self.result.save(filename)
            self.result = filename
            

    def completed(self):
        return self.status == "completed"
    
    def failed(self):
        return self.status == "failed"
    
    def pending(self):
        return self.status == "pending"

    def run(self):
        try:
            new_args = copy.deepcopy(self.args)
            for k,v in new_args.items():
                if k == "image":
                    new_args["image"] = load_image(v)
            self.result = self.tool(**new_args)
        except Exception as e:
            self.status = "failed"
            self.message = str(e)
        self.status = "completed"
        self.save_result()

        return self.result

class TaskExecutor:
    """Load tools to execute tasks."""

    def __init__(
        self,   
        plan: Plan
    ):
        self.plan = plan
        self.tasks = []
        self.id_task_map = {}
        self.status = "pending"
        for step in self.plan.steps:
            task = Task(step.task, step.id, step.dep, step.args, step.tool)
            self.tasks.append(task)
            self.id_task_map[step.id] = task
    
    def completed(self):
        return all(task.completed() for task in self.tasks)
    
    def failed(self):
        return any(task.failed() for task in self.tasks)
    
    def pending(self):
        return any(task.pending() for task in self.tasks)

    def check_dependency(self, task):
        for dep_id in task.dep:
            if dep_id == -1:
                continue
            dep_task = self.id_task_map[dep_id]
            if dep_task.failed() or dep_task.pending():
                return False
        return True
    
    def update_args(self, task):
        for dep_id in task.dep:
            if dep_id == -1:
                continue
            dep_task = self.id_task_map[dep_id]
            for k, v in task.args.items():
                if f"<resource-{dep_id}>" in v:
                    task.args[k].replace(f"<resource-{dep_id}>", dep_task.result)

    def run(self):
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

    def __str__(self):
        result = ""
        for task in self.tasks:
            result += f"{task}\n"
            result += f"status: {task.status}\n"
            if task.failed():
                result += f"message: {task.message}\n"
            if task.completed():
                result += f"result: {task.result}\n"
        return result
    
    def __repr__(self):
        return self.__str__()
    
    def describe(self):
        return self.__str__()