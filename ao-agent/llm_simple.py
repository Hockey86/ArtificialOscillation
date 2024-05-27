import logging
from openai import OpenAI, RateLimitError
import backoff
from backoff._jitter import random_jitter


class MyLLM:
    def __init__(self, llm_model_name, instruction):
        self.llm_model_name = llm_model_name
        self.instruction = instruction
        self.client = OpenAI()
        self.history = []
        
    @backoff.on_exception(backoff.expo, RateLimitError, jitter=random_jitter)
    def ask(self, question, json=False):
        logging.info('Asking AI a question: '+question)

        msgs = [ {"role": "system", "content": self.instruction} ]
        for r, m in self.history:
            msgs.append( {'role':r, 'content':m} )
        msgs.append( {'role':'user', 'content':question} )

        if json:
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}
        ans = self.client.chat.completions.create(
          model=self.llm_model_name, messages=msgs,
          temperature=0, response_format=response_format)
        ans = ans.choices[0].message.content

        self.history.append( msgs[-1] )
        self.history.append( {'role':'assistant', 'content':ans} )
        logging.info(f'AI answers: {ans}\n\n')
        return ans

    def forget(self):
        self.history.clear()
        logging.info('AI memory has been erased.')
        return self
        
