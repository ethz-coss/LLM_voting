from typing import List
import json
import requests
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

headers = {'Content-Type': 'application/json'}

if OPENAI_API_KEY:
    headers["Authorization"] = f'Bearer {OPENAI_API_KEY}'
    API_URL = 'https://api.openai.com/v1' 
    using_openai_api = True
    print('Using OPENAI')
else:
    API_URL = 'http://localhost:8000'
    using_openai_api = False
    print('Using LOCALHOST')

class Message:
    def __init__(self, time: int, content: str, role: str):
        """

        Message object for the API.

        :param time: the time of the message
        :param content: the content of the message
        :param role: the role of the message sender

        """
        self.time = time
        self.content = content
        self.role = role

    def to_chat_completion_query(self):
        return {'content': self.content, 'role': self.role}

    def to_embedding_query(self):
        return self.content

    def __str__(self):
        return f"{self.time} -- {self.role:.1} -- {self.content}"

    def __repr__(self):
        return self.__str__()

def chat_request(messages: List[Message], max_tokens: int = 0, temperature: float = 0.8) -> List[Message]:
    assert 0 <= temperature <= 2, "temperature must be between 0 and 2"
    assert 0 <= max_tokens <= 2048, "max_tokens must be between 0 (unlimited) and 2048"
    assert len(messages) > 0, "messages must not be empty"
    # print('TEMP '+ str(temperature))
    if using_openai_api:
        response = requests.post(f'{API_URL}/chat/completions',
                                 headers=headers,
                                 json={
                                     "model": "gpt-4-1106-preview",  
                                     "messages": [message.to_chat_completion_query() for message in messages],
                                     "max_tokens": max_tokens,
                                     "temperature": temperature
                                 })
    else:
        response = requests.post(f'{API_URL}/v1/chat/completions',
                                 headers=headers,
                                 data=json.dumps({
                                     "messages": [message.to_chat_completion_query() for message in messages],
                                     "max_tokens": max_tokens,
                                     "temperature": temperature,
                                     "presence_penalty": 1,
                                     "frequency_penalty": 1,
                                     "repeat_penalty": 1,
                                     "top_k": 5,
                                     "mirostat_mode": 2
                                 }))

    if response.status_code != 200:
        print(f"Error in API call: {response.text}")
        return None

    response_data = response.json()

    try:
        answer = response_data['choices'][0]['message']['content']
    except KeyError:
        print("Error: Unexpected response format.")
        return None

    time = int(np.max([message.time for message in messages]) + 1)
    role = 'assistant'
    messages.append(Message(time=time, content=answer, role=role))

    return messages[-1]

def complete_request(messages: List[Message], max_tokens: int = 0, temperature: float = 0.8,
                     logprobs: int = 5) -> dict:
    assert 0 <= temperature <= 2, "temperature must be between 0 and 2"
    assert 0 <= max_tokens <= 2048, "max_tokens must be between 0 (unlimited) and 2048"
    assert len(messages) > 0, "messages must not be empty"

    response = requests.post(f'{API_URL}/v1/completions',
                             headers=headers,
                             data=json.dumps({
                                 "prompt": " ".join([message.content for message in messages]),
                                 "max_tokens": max_tokens,
                                 "echo": False,
                                 "stop": ["[/INST]"],
                                 "temperature": temperature,
                                 "presence_penalty": 1,
                                 "frequency_penalty": 1,
                                 "repeat_penalty": 1,
                                 "logprobs": logprobs,
                                 "mirostat_mode": 2
                             }))

    if 'error' in response.json().keys():
        print(response.json()['error'])

    response = response.json()

    answer = response['choices'][0]['text']
    logprobs = response["choices"][0]["logprobs"]["top_logprobs"]

    time = int(np.max([message.time for message in messages]) + 1)
    role = 'assistant'
    messages.append(Message(time=time, content=answer, role=role))
    
    return messages[-1], logprobs
        
