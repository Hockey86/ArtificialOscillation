from openai import OpenAI, RateLimitError
import requests
import backoff
from backoff._jitter import random_jitter
from interfaces import CmdInterface


class MyLLM:
    def __init__(self, llm_model_name, instruction, interface=None):
        self.llm_model_name = llm_model_name
        self.instruction = instruction
        if interface is None:
            self.interface = CmdInterface()
        else:
            self.interface = interface

        self.client = OpenAI()
        self.history = []
        self.interface.says(f'Created an AI: model = {self.llm_model_name}, instruction = {instruction}', 'system')
        
    @backoff.on_exception(backoff.expo, (
            requests.exceptions.RequestException,
            RateLimitError),
        jitter=random_jitter, max_tries=10)
    def ask(self, question_, json=False, my_answer=None, role='human'):
        question = question_.strip()
        self.interface.says(question, role)

        msgs = [ {'role':'system', 'content':self.instruction} ] +\
               self.history +\
               [ {'role':'user', 'content':question} ]

        if json:
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}
        if my_answer is None:
            ans = self.client.chat.completions.create(
              model=self.llm_model_name.split('|')[0], messages=msgs,
              temperature=0, response_format=response_format)
            ans = ans.choices[0].message.content
        else:
            ans = my_answer

        self.history.append( msgs[-1] )
        self.history.append( {'role':'assistant', 'content':ans} )
        self.interface.says(ans, 'ai')
        return ans

    def forget(self):
        self.history.clear()
        self.interface.says('AI memory has been erased.', 'system')
        return self
        
