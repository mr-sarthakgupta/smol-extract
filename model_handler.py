
import json
from ollama import Client
from pydantic import BaseModel

def model_fn(model_dir):
    client = Client(host='localhost:11434')
    return client

def predict_fn(input_data, model):
    prompt = input_data.pop('prompt')
    response = model.generate(
        model='deepseek',
        prompt=prompt,
        **input_data
    )
    return response

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        return json.loads(request_body)
    else:
        raise ValueError('Unsupported content type')

def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError('Unsupported content type')
        