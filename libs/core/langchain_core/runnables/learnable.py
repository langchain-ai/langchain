# from langchain_core.runnables.base import RunnableBinding


# class RunnableLearnable(RunnableBinding):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.parameters = []

#     def backward(self):
#         for param in self.parameters:
#             param.backward()

#     def update(self, optimizer):
#         for param in self.parameters:
#             optimizer.update(param)
