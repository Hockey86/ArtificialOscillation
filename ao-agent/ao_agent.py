from memory import Memory
from planner import Planner
from actor import Actor
from explainer import Explainer

import datetime, logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log_{}.txt".format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f'))),
        logging.StreamHandler() ]
)

class AOAgent:
    """
AO AGI agent
    """
    def __init__(self, name='AO', llm_name='gpt-4o', profile_instruction=None, human_in_loop=False, circadian_period=24):
        self.name = name
        self.llm_name = llm_name
        self.profile_instruction = profile_instruction
        self.memory = Memory()
        self.planner = Planner(llm_name, profile_instruction)  # includes RL
        self.actor = Actor()
        self.explainer = Explainer(llm_name, profile_instruction)
        self.human_in_loop = human_in_loop
        self.circadian_period = circadian_period

    def __str__(self):
        return self.name

    def solve_task(self, task, env):
        """
        task: str
        env:  ?
        """
        #TODO each component run in Parallel, using ROS2?
        self.planner.init(task)
        timer = 0
        while True:
            timer += 1
            action = self.planner.next_step(task, self.memory, self.explainer)
            if action=='stop':
                break
            if self.human_in_loop:
                print('human takes over')
            self.actor.act(action, env)
            explanation = self.explainer.explain(action, env)
            self.memory.add(action, explanation)
            if timer%self.circadian_period==0:
                self.sleep()

    def sleep(self):
        print('zzz')
        self.planner.rl.fit(self.memory)

