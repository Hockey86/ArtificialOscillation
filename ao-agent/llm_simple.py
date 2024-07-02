import os, json
from openai import OpenAI, RateLimitError
import requests
import backoff
from backoff._jitter import random_jitter
from interfaces import CmdInterface


class MyLLM:
    def __init__(self, llm_model_name, instruction, interface=None, human_verification=False, QA_path='llm_QAs.json'):
        self.llm_model_name = llm_model_name
        self.instruction = instruction
        self.human_verification = human_verification
        if interface is None:
            self.interface = CmdInterface()
        else:
            self.interface = interface
        self.QA_path = QA_path
        if os.path.exists(self.QA_path):
            with open(self.QA_path) as f:
                self.QA = json.load(f)
        else:
            self.QA = {}

        self.client = OpenAI()
        self.history = []
        self.interface.says('system', f'Created an AI: model = {self.llm_model_name}, instruction = {instruction}\n')
    
    @backoff.on_exception(backoff.expo, (
            requests.exceptions.RequestException,
            RateLimitError),
        jitter=random_jitter, max_tries=10)
    def ask(self, msg, reply_json=False, role='human'):
        msg = msg.strip()
        self.interface.says(role, msg)

        msgs = [ {'role':'system', 'content':self.instruction} ] +\
               self.history +\
               [ {'role':'user', 'content':msg} ]
        msgs2 = json.dumps(msgs)

        if msgs2 not in self.QA:
            ans = self.client.chat.completions.create(
              model=self.llm_model_name.split('|')[1],
              messages=msgs, temperature=0,
              response_format={'type': 'json_object' if reply_json else 'text'})
            ans = ans.choices[0].message.content
            if self.human_verification:
                #TODO human interacts with AI to refine answer
                self.interface.says('ai', ans+'\n\nIs my response ok? Type [enter] or Y to confirm, or give your response to replace mine.')
                res = self.interface.human_says().strip()
                if res.upper() not in ['Y', '']:
                    ans = res
            self.QA[msgs2] = ans
            with open(self.QA_path, 'w') as f:
                json.dump(self.QA, f, indent=2)
        else:
            ans = self.QA[msgs2]
            self.interface.says('ai', ans)

        self.history.append( msgs[-1] )
        self.history.append( {'role':'assistant', 'content':ans} )
        return ans

    def forget(self):
        self.history.clear()
        self.interface.says('system', 'AI memory has been erased.\n')
        return self
        
