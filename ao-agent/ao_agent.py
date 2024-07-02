from memory import Memory
from planner import Planner
from actor import Actor
from explainer import Explainer
from llm_simple import MyLLM

import datetime, logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log_{}.txt".format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f'))),]
    #    logging.StreamHandler() ]
)


"""
import torch as th
import torch.nn as nn
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MyNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, state2feat_func):
        super().__init__(observation_space, features_dim=2)
        self.state2feat_func = state2feat_func

    def forward(self, obs) -> th.Tensor:
        feat = th.cat([
            self.state2feat_func(obs), 
            obs['percent_completion_current_plan'],
            ])
        return feat
"""


class AOAgent:
    """ AO AGI agent
    action space: 0 is run current plan, # 1 is to come up with a new plan
    #TODO each component run in Parallel, using ROS2?
    """
    def __init__(self, name='AO', llm_name='gpt-4o', profile_instruction=None, human_in_loop=True, circadian_period=24, interface=None):
        self.name = name
        self.circadian_period = circadian_period
        self.memory = Memory()

        self.planner = Planner(
            MyLLM('planner|'+llm_name, profile_instruction,
            interface=interface, human_verification=human_in_loop),
            self.memory)

        self.actor = Actor(
            MyLLM('actor|'+llm_name, '',
            interface=interface, human_verification=human_in_loop))

        self.explainer = Explainer(
            MyLLM('explainer|'+llm_name, '',
            interface=interface, human_verification=human_in_loop),
            self.memory)

    def __str__(self):
        return self.name

    def solve_task(self, task):
        """
        task: str
        """
        self.planner.init(task)

        """
        self.rl = PPO("MultiInputPolicy", self.env, policy_kwargs=dict(
            features_extractor_class=MyNetwork,
            features_extractor_kwargs=dict(state2feat_func=self.env.reward_func),
            ), verbose=1)
        self.rl.set_logger(configure('sb3_tmp', ["stdout", "csv", "tensorboard"]))

        # a good policy is (overall state --> action):
        # 10: state is good, plan just started --> continue current plan
        # 11: state is good, plan close to end --> continue current plan
        # 00: state is bad, plan just started --> continue current plan
        # 01: state is bad, plan close to end --> come up with new (need memory)
        with th.no_grad():
            self.rl.policy_class.value_net.weight.copy_([-1., 1.])
            self.rl.policy_class.value_net.bias.copy_(0.)
            self.rl.policy_class.action_net.weight.copy_([-1., 1.])
            self.rl.policy_class.action_net.bias.copy_(0.)

        obs, _ = self.env.reset()
        timer = 0
        while True:
            timer += 1
            if timer%self.circadian_period==0:
                self.rl.learn(total_timesteps=self.circadian_period)
                self.memory.consolidate()
            action, _ = self.rl.predict(obs, deterministic=True)
            if self.human_in_loop:
                print('human takes over')
            obs, reward, done, info = self.env.step(action)
            explanation = self.explainer.explain(action, env)
            self.memory.add(explanation)
            if done:
                break
        """
        timer = 0
        while True:
            timer += 1
            if timer%self.circadian_period==0:
                self.memory.consolidate()
            step = self.planner.next_step()#TODO TODO TODO TODO!!
            if self.human_in_loop:
                print('human takes over')
            action_results = self.actor(step)
            explanation = self.explainer.explain(action_results)
            self.memory.add(explanation)
            if self.planner.success:
                break

        self.planner.clear()

