import memory
import llama


class Agent:
    def __init__(self, aid: int, recall: int = 3, initial_context: llama.Message = None):
        self.id = aid
        self.recall = recall

        self.initial_context = llama.Message(time=initial_context.time, content="<s>[INST]<<SYS>>\n" + initial_context.content + "\n<</SYS>>[/INST]\n\n", role=initial_context.role)
        self.memory = memory.Memory()

        
    def perceive(self, message: llama.Message, **kwargs) -> llama.Message:
        messages = self.memory.retrieve(time=message.time - self.recall)        
        # last_message = llama.Message(time=message.time, content=message.content + " [/INST]", role=message.role)
    
        # self.initial_context.content = "<s>[INST]<<SYS>>\n" + self.initial_context.content + "\n<</SYS>>\n\n"
        # message.content += " [/INST]"

        messages = [self.initial_context] + [llama.Message(time=m.time, content=m.content, role=m.role) for m in messages] + [message]
        answer = llama.chat_request(messages=messages, **kwargs)[-1]
        self.memory.store(message=message)
        self.memory.store(message=answer)
        return answer


class Distribution(Agent):
    """
    Agent created to output distributions of tokens with logprobs
    """
    def perceive(self, message: llama.Message, **kwargs) -> dict:
        messages = self.memory.retrieve(time=message.time - self.recall)

        messages = [self.initial_context] + [llama.Message(time=m.time, content=m.content, role=m.role) for m in
                                             messages] + [message]
        answer = llama.complete_request(messages=messages, **kwargs)
        # self.memory.store(message=message)
        # self.memory.store(message=answer)
        return answer
