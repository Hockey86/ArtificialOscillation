import logging
from openai import OpenAI, RateLimitError
import requests
import backoff
from backoff._jitter import random_jitter


class MyLLM:
    def __init__(self, llm_model_name, instruction):
        self.llm_model_name = llm_model_name
        self.instruction = instruction
        self.client = OpenAI()
        self.history = []
        logging.info(f'Created an AI: model = {self.llm_model_name}, instruction = {instruction}')
        
    @backoff.on_exception(backoff.expo, (
            requests.exceptions.RequestException,
            RateLimitError),
        jitter=random_jitter, max_tries=10)
    def ask(self, question_, json=False, my_answer=None):
        question = question_.strip()
        logging.info('Asking AI a question: '+question)

        msgs = [ {'role':'system', 'content':self.instruction} ] +\
               self.history +\
               [ {'role':'user', 'content':question} ]

        if json:
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}
        if my_answer is None:
            ans = self.client.chat.completions.create(
              model=self.llm_model_name, messages=msgs,
              temperature=0, response_format=response_format)
            ans = ans.choices[0].message.content
        else:
            ans = my_answer

        self.history.append( msgs[-1] )
        self.history.append( {'role':'assistant', 'content':ans} )
        logging.info(f'AI answers: {ans}\n\n')
        return ans

    def forget(self):
        self.history.clear()
        logging.info('AI memory has been erased.')
        return self
        
