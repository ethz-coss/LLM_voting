import memory
import llama


class Agent:
    def __init__(self, aid: int, recall: int = 3, initial_context: llama.Message = None, temperature: float = 1.0):
        self.id = aid
        self.recall = recall

        self.initial_context = initial_context
        self.memory = memory.Memory()
        self.temperature = temperature

        
    def perceive(self, message: llama.Message, **kwargs) -> llama.Message:
        messages = self.memory.retrieve(time=message.time - self.recall)        
        messages = [self.initial_context] + [llama.Message(time=m.time, content=m.content, role=m.role) for m in messages] + [message]

        answer = llama.chat_request(messages=messages, temperature=self.temperature, **kwargs)

        self.memory.store(message=message)
        self.memory.store(message=answer)

        return answer


class Distribution:
    """
    Agent created to output distributions of tokens with logprobs
    """
    def __init__(self, aid: int, recall: int = 0, initial_context: llama.Message = None, temperature: float = 1.0):
        self.id = aid
        self.recall = recall
        self.initial_context = initial_context
        self.memory = memory.Memory()
        self.temperature = temperature

    def perceive(self, message: llama.Message, max_tokens: int = 10, logprobs: int = 5, **kwargs) -> tuple[dict, any]:
        messages = self.memory.retrieve(time=message.time - self.recall)
        messages = [self.initial_context] + messages + [message]

        answer, logprobs = llama.complete_request(messages=messages, max_tokens=max_tokens, temperature=self.temperature, logprobs=logprobs, **kwargs)
        
        self.memory.store(message=message)
        self.memory.store(message=answer)

        return answer, logprobs
