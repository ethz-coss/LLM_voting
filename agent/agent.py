import memory
import llama


class Agent:
    def __init__(self, aid: int, recall: int = 3, initial_context: llama.Message = None):
        self.id = aid
        self.recall = recall

        self.initial_context = initial_context
        # self.initial_context = llama.Message(time=initial_context.time, content=initial_context.content, role=initial_context.role)
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


class Distribution:
    """
    Agent created to output distributions of tokens with logprobs
    """
    def __init__(self, aid: int, recall: int = 3, initial_context: llama.Message = None):
        self.id = aid
        self.recall = recall

        self.initial_context = initial_context
        self.memory = memory.Memory()

    def perceive(self, message: llama.Message, max_tokens: int = 10, temperature: float = 0.8, logprobs: int = 5, **kwargs) -> tuple[dict, any]:
        messages = self.memory.retrieve(time=message.time - self.recall)

        messages = [self.initial_context] + [llama.Message(time=m.time, content=m.content, role=m.role) for m in
                                             messages] + [message]
        response = llama.complete_request(messages=messages, max_tokens=max_tokens, temperature=temperature, logprobs=logprobs, **kwargs)
        # self.memory.store(message=message)
        # self.memory.store(message=answer)
        # print(response)
        top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][-1]
        answer = response['choices'][0]['text']
        return top_logprobs, answer
