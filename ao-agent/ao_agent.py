import torch as th
import torch.nn as nn
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from memory import Memory
from planner import Planner
from actor import Actor
from explainer import Explainer

import datetime, logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log_{}.txt".format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S%f'))),]
    #    logging.StreamHandler() ]
)


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MyNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


class AOAgent:
    """ AO AGI agent
    action space: 0 is run current plan, # 1 is to come up with a new plan
    #TODO each component run in Parallel, using ROS2?
    """
    def __init__(self, name='AO', llm_name='gpt-4o', profile_instruction=None, human_in_loop=True, circadian_period=24, interface=None):
        self.name = name
        self.llm_name = llm_name
        self.profile_instruction = profile_instruction
        self.memory = Memory()
        self.planner = Planner(llm_name+'|planner', profile_instruction, interface=interface)
        self.actor = Actor()
        self.explainer = Explainer(llm_name+'|explainer', profile_instruction)
        self.human_in_loop = human_in_loop
        self.circadian_period = circadian_period

    def __str__(self):
        return self.name

    def solve_task(self, task):
        """
        task: str
        """
        self.env = self.planner.init_env(task)
        if self.human_in_loop:
            self.planner.human_verify_rl_setting()

        self.rl = PPO("MultiInputPolicy", self.env, policy_kwargs=dict(
            features_extractor_class=MyNetwork,
            #features_extractor_kwargs=dict(features_dim=128),
            ), verbose=1)
        self.rl.set_logger(configure('sb3_tmp', ["stdout", "csv", "tensorboard"]))

        #TODO pre-train
        # a good policy is (overall state --> action):
        # state is good, plan just started --> continue current plan
        # state is good, plan close to end --> stop
        # state is bad, plan just started --> continue current plan
        # state is bad, plan close to end --> come up with new (need memory)
        self.rl.collect_experience()
        self.rl.learn()

        obs = self.env.reset()
        timer = 0
        #TODO it's strange to have training in the Env class, move out?...
        while True:
            timer += 1
            if timer%self.circadian_period==0:
                rl.learn(total_timesteps=self.circadian_period)
            action, _ = rl.predict(obs, deterministic=True)
            if self.human_in_loop:
                print('human takes over')
            obs, reward, done, info = self.env.step(action)
            explanation = self.explainer.explain(action, env)
            self.memory.add(action, explanation)
            if done:
                break

