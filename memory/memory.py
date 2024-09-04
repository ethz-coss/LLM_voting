from typing import List

import pandas as pd
import llm


class Memory:
    def __init__(self):
        self.memory = []

    def store(self, message: llm.Message):
        self.memory.append(message)

    def retrieve(self, time: int) -> List[llm.Message]:
        messages = [message for message in self.memory if message.time >= time]
        return messages
