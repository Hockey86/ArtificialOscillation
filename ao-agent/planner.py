from stable_baselines3 import PPO
from llm import MyLLM

class Planner:
    """
    """
    def __init__(self, llm_name, profile_instruction):
        instruction = \
"""{profile_instruction}
Specifically here, you will use scientific method to plan the steps to solve a given task. Here, the scientific method is a metholology, logical inference, and a branch of philosophy.
The scientific method includes deduction (derive the consequence from known condition based on principle), abduction (infer the possible condition from the known consequence based on principle), and induction (summarize general principle from observed conditions and consequences)."""

        self.llm = MyLLM(llm_name, instruction)
        self.current_plan = []
        #self.rl = PPO()

    def next_step(task_txt, memory, explainer):
        if len(self.current_plan)==0: # make a plan:
            self.llm.ask(
f"""Here is a plan template:
1. make observations
2. make hypothesis (abductive reasoning)
3. if hypothesis is true, what to expect (deductive reasoning)
4. test the hypothesis
5. what general principle can we get (inductive reasoning)

Now, generate a plan for the given task: {task_txt}
"""
        else:
            last_plan = self.current_plan.pop()

        return action
