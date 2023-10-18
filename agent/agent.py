import memory
import llama


class Agent:
    def __init__(self, aid: int, recall: int = 3, initial_context: llama.Message = None):
        self.id = aid
        self.recall = recall
        self.memory = memory.Memory(context_message=initial_context)

    def perceive(self, message: llama.Message, **kwargs) -> str:
        messages = self.memory.retrieve(time=message.time - self.recall)
        messages = [llama.Message(time=m.time, content=m.content, role='system') for m in messages] + [message]
        answer = llama.chat_request(messages=messages, **kwargs)[-1]
        self.memory.store(message=message)
        self.memory.store(message=answer)
        return answer