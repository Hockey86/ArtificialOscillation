from stable_baselines3 import PPO
from llm_simple import MyLLM


class Planner:
    """
    """
    def __init__(self, llm_name, profile_instruction):
        self.llm = MyLLM(llm_name, profile_instruction)
        self.current_plan = []
        #self.rl = PPO()
    
    def init(self, task):
        ans = self.llm.ask(
f"""You are given a task:
<task>
{task}
</task>

You need to formulate this task into a reinforcement learning (RL) problem. In RL, one needs to define state, action, and reward. How would you define the state for this task? The state is a vector, where each element represents a quantitative aspect. Give your answer in the format of a Python list to represent the state vector, where each element in the list is a string that describes one quantitative aspect of the state. In your answer, only include the list, do not assign this list to any variable, i.e., start with [ and end with ]. Do not include any other introductory text.
""")
        if ans.count('```')==2:
            if '```python' in ans:
                start = ans.index('```python')+9
            else:
                start = ans.index('```')+3
            stop = ans.rindex('```')
            ans = ans[start:stop]
        import pdb;pdb.set_trace()
        state_desc = eval(ans)

        for sd in state_desc:
            ans = self.llm.ask(f'Quantify {sd}')

#propose quantifiable criterion for goal accomplishment
        #TODO memory.add

    def next_step(self, task, memory, explainer):
        if len(self.current_plan)==0: # make a plan:
            self.llm.ask(
f"""Here is a plan template:
1. make observations
2. make hypothesis (abductive reasoning)
3. if hypothesis is true, what to expect (deductive reasoning)
4. test the hypothesis
5. what general principle can we get (inductive reasoning)

Now, generate a plan for the given task: {task}
""")
        else:
            last_plan = self.current_plan.pop()

        return action
