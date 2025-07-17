import sys
sys.path.insert(0, 'libs/core')
from langchain_core.utils.json_schema import dereference_refs

schema = {
    'type': 'object',
    'properties': {
        'payload': {
            'anyOf': [
                {
                    'type': 'object',
                    'properties': {
                        'kind': {'type': 'string', 'const': 'ONE'}
                    }
                },
                {
                    'type': 'object',
                    'properties': {
                        'kind': {'type': 'string', 'const': 'TWO'},
                        'startDate': {
                            'type': 'string',
                            'pattern': r'^\d{4}-\d{2}-\d{2}$'
                        },
                        'endDate': {
                            '$ref': '#/properties/payload/anyOf/1/properties/startDate'
                        }
                    }
                }
            ]
        }
    }
}

print('Testing original issue...')
try:
    result = dereference_refs(schema, skip_keys=())
    print('SUCCESS: Fix works!')
    end_date = result['properties']['payload']['anyOf'][1]['properties']['endDate']
    start_date = result['properties']['payload']['anyOf'][1]['properties']['startDate']
    print('endDate schema:', end_date)
    print('startDate schema:', start_date)
    print('Match:', end_date == start_date)
except Exception as e:
    print(f'FAILURE: {e}')
    import traceback
    traceback.print_exc()
