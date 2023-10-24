from typing import List

import pandas as pd
import llama


class Memory:
    def __init__(self):
        self.memory = []

    def store(self, message: llama.Message):
        self.memory.append(message)

    def retrieve(self, time: int) -> List[llama.Message]:
        messages = [message for message in self.memory if message.time >= time]
        return messages
