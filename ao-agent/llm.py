import logging, datetime
import backoff
from backoff._jitter import random_jitter
from openai import RateLimitError
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages.ai import AIMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory#ChatMessageHistory
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
#pip3 install redis
#docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
REDIS_URL = "redis://localhost:6379/0"


class MyLLM:
    def __init__(self, model, instruction, session_id=None):
        self.model = model
        self.instruction = instruction
        if session_id is None:
            self.session_id = str(datetime.datetime.now())
        else:
            self.session_id = session_id
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.instruction),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        if model.startswith('llama'):
            model = Ollama(model=model)
        elif model.startswith('gpt'):
            model = ChatOpenAI(model_name=model, temperature=0)
        else:
            raise NotImplementedError(model)
        self.client = RunnableWithMessageHistory(
            prompt | model,
            lambda session_id: RedisChatMessageHistory(session_id, url=REDIS_URL),#ChatMessageHistory,
            input_messages_key="question",
            history_messages_key="history" )
        self.forget()
        
    def __del__(self):
        self.forget()

    @backoff.on_exception(backoff.expo, RateLimitError, jitter=random_jitter)
    def ask(self, question, json=False):
        if type(question) in [list, tuple]:
            assert question[-1][0]=='user', 'Last question\'s role should be user'
            for q in question[:-1]:
                assert q[0] in ['user', 'assistant'], f'Role can only be "user" or "assistant", got "{q[0]}".'
                if q[0]=='user':
                    self.client.get_session_history(self.session_id).add_user_message(q[1])
                elif q[0]=='assistant':
                    self.client.get_session_history(self.session_id).add_ai_message(q[1])
            question = question[-1][1]
        if json:
            question += ' Reply the JSON only.'
        logging.info('Asking AI a question: '+question)
        ans = self.client.invoke({'question':question},
            config={"configurable": {"session_id": self.session_id}})
        if type(ans)==AIMessage:
            ans = ans.content.strip()
        if json:
            start = max(0,ans.find('{'))
            end   = min(len(ans)-1,ans.rfind('}'))
            ans = ans[start:end+1]
        logging.info(f'AI answers: {ans}\n\n')
        return ans

    def forget(self):
        self.client.get_session_history(self.session_id).clear()
        #logging.info('AI memory has been erased.')
        return self
        
