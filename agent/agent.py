import memory
import llm


class Agent:
    def __init__(self, aid: int, recall: int = 3, initial_context: llm.Message = None, temperature: float = 1.0):
        self.id = aid
        self.recall = recall

        self.initial_context = initial_context
        self.memory = memory.Memory()
        self.temperature = temperature

        
    def perceive(self, message: llm.Message, **kwargs) -> llm.Message:
        messages = self.memory.retrieve(time=message.time - self.recall)        
        messages = [self.initial_context] + [llm.Message(time=m.time, content=m.content, role=m.role) for m in messages] + [message]

        answer = llm.chat_request(messages=messages, temperature=self.temperature, **kwargs)

        self.memory.store(message=message)
        self.memory.store(message=answer)

        return answer


class Distribution:
    """
    Agent created to output distributions of tokens with logprobs
    """
    def __init__(self, aid: int, recall: int = 0, initial_context: llm.Message = None, temperature: float = 1.0):
        self.id = aid
        self.recall = recall
        self.initial_context = initial_context
        self.memory = memory.Memory()
        self.temperature = temperature

    def perceive(self, message: llm.Message, max_tokens: int = 10, logprobs: int = 5, **kwargs) -> tuple[dict, any]:
        messages = self.memory.retrieve(time=message.time - self.recall)
        messages = [self.initial_context] + messages + [message]

        answer, logprobs = llm.complete_request(messages=messages, max_tokens=max_tokens, temperature=self.temperature, logprobs=logprobs, **kwargs)
        
        self.memory.store(message=message)
        self.memory.store(message=answer)

        return answer, logprobs
