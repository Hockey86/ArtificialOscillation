from memory import Memory
from planner import Planner
from actor import Actor
from explainer import Explainer


class AOAgent:
    """
AO AGI agent
    """
    def __init__(self, name):
        self.name = name
        self.memory = Memory()
        self.planner = Planner()
        self.actor = Actor()
        self.explainer = Explainer()

    def __str__(self):
        return self.name