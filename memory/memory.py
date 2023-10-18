from typing import List

import pandas as pd
import llama


class Memory:
    def __init__(self, context_message: llama.Message = None):
        self.memory = [] if context_message is None else [context_message]

    def store(self, message: llama.Message):
        self.memory.append(message)

    def retrieve(self, time: int) -> List[llama.Message]:
        messages = [message for message in self.memory if message.time >= time]
        return messages