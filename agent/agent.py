import memory
import llama


class Agent:
    def __init__(self, aid: int, recall: int = 3, initial_context: llama.Message = None):
        self.id = aid
        self.recall = recall

        self.initial_context = initial_context
        self.memory = memory.Memory()

        
    def perceive(self, message: llama.Message, **kwargs) -> llama.Message:
        messages = self.memory.retrieve(time=message.time - self.recall)

        self.initial_context.content = "[INST]<<SYS>>\n" + self.initial_context.content + "\n<</SYS>>\n\n"
        message.content += " [/INST]"

        messages = [self.initial_context] + [llama.Message(time=m.time, content=m.content, role=m.role) for m in messages] + [message]
        answer = llama.chat_request(messages=messages, **kwargs)[-1]
        self.memory.store(message=message)
        self.memory.store(message=answer)
        return answer
