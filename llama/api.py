from typing import List
import json

import requests
import numpy as np


class Message:
    def __init__(self, time: int, content: str, role: str):
        """

        Message object for the API.

        :param time: the time of the message
        :param content: the content of the message
        :param role: the role of the message sender

        """
        self.time = time
        self.content = content.strip()
        self.role = role

    def to_chat_completion_query(self):
        return {'content': self.content, 'role': self.role}

    def to_embedding_query(self):
        return self.content

    def __str__(self):
        return f"{self.time} -- {self.role:.1} -- {self.content}"

    def __repr__(self):
        return self.__str__()


def chat_request(messages: List[Message], max_tokens: int = 16, temperature: float = 0.8) -> List[Message]:
    assert 0 <= temperature <= 2, "temperature must be between 0 and 2"
    assert 1 <= max_tokens <= 2048, "max_tokens must be between 1 and 2048"
    assert len(messages) > 0, "messages must not be empty"
    
    response = requests.post('http://10.249.72.2:8000/v1/chat/completions',
                             headers={'Content-Type': 'application/json'},
                             data=json.dumps({
                                 "messages": [message.to_chat_completion_query() for message in messages],
                                 "max_tokens": max_tokens,
                                 "echo": True,
                                 "stop": ["[/INST]"],
                                 "temperature": temperature
                             }))

    if 'error' in response.json().keys():
        print(response.json()['error'])
    answer = response.json()['choices'][0]['message']['content']
    time = int(np.max([message.time for message in messages]) + 1)
    role = 'assistant'
    messages.append(Message(time=time, content=answer, role=role))
    return messages
