# File changes (hypothetical path and function)
def emit_on_tool_end_event(tool_output, tool_input, run_id, tool_name):
event_data = {
'event': 'on_tool_end',
'data': {
'output': tool_output[0],  # Assuming tool_output is a tuple (content, artifact)
'artifact': tool_output[1],  # Include the artifact
'input': tool_input,
},
'run_id': run_id,
'name': tool_name,
'tags': [],
'metadata': {},
'parent_ids': [],  # Populate as necessary
}
# Emit the event with the updated data
emit_event(event_data)