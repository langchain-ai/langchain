# File changes (if any)
class ChatOpenAI:
def __init__(self, model, **kwargs):
# Existing initialization code...
self.model = self._map_model_name(model)

def _map_model_name(self, model_name):
# Add mapping for Tongyi Qianwen model
model_mapping = {
"gpt-4o": "tongyi-qianwen-model-id",  # Replace with actual model ID
# Other model mappings...
}
return model_mapping.get(model_name, model_name)