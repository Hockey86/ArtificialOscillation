import json, types
from treelib import Node, Tree
import backoff
#from stable_baselines3.common.env_checker import check_env
#from env import ToAskAIEnv


def _txt2code(txt):
    assert txt.count('```')==2
    if '```python' in txt.lower():
        start = txt.lower().index('```python')+9
    else:
        start = txt.index('```')+3
    stop = txt.rindex('```')
    return txt[start:stop].strip()    


class Planner:
    """
    """
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory
        self.plan = Tree()
        self.plan_outline = """
Here is the plan outline:

1. Collect relevant information to propose some directions. The directions should be targeted at ultimately achieving max goodness, i.e., the `get_goodness` function returns value of 1.
2. For each direction, use abductive reasoning to come up with possible proposal(s) (including generating hypotheses) that can explain the direction.
3. For each proposal, use deductive reasoning to infer what to expect if the proposal are fulfilled (such as hypotheses are unfalsified), in terms of the change in the state, and hence the change in the goodness.
4. Execute each proposal. Come up with ways to measure the change in the state.
5. Check if the expectation is met.
6. Using inductive reasoning: If met, what further proposal can we make to address the task? If not met, think about what could go wrong and make an alternative proposal. """.strip()
    self.plan_instructions = [
"""Now, you should materialize the 1st point in the plan outline into executable steps for the given task. Give your answer in Python code containing a list of strings representing the executable steps. The executable steps are numbered by "1.", "2.", ... Your answer is wrapped by ``` to indicate it's code.""",
"""?""",
"""?""",
"""?""",
"""?""",
"""?""", ]

    def set_success_func(self, code):
        self.success_func_code = code
        code_obj = compile(code, '<string>', 'exec')
        self.success_func = types.FunctionType(code_obj.co_consts[0], globals())

    def set_goodness_func(self, code):
        self.goodness_func_code = code
        code_obj = compile(code, '<string>', 'exec')
        self.goodness_func = types.FunctionType(code_obj.co_consts[0], globals()) 

    @backoff.on_exception(backoff.constant,
            SyntaxError, interval=1, max_tries=5)
    def init(self, task):
        """
        define state, and goodness(state)
        initialize the first step
        """
        self.llm.forget()
        #env = ToAskAIEnv(task, self, actor)
        self.memory.add(f"""
You are given a task:
{task}
""")

        # define state

        ans = self.llm.ask(f"""
You are given a task:
<task>
{task}
</task>

You need to formulate this task into a reinforcement learning (RL) problem. In RL, one needs to define state. How would you define the state for this task? The state is a vector, where each dimension represents a quantitative aspect.

Give your answer in JSON format, where each key-value pair represents a dimension in the state vector. The key is a brief python variable name. The value is an explanation of the state vector dimension.
""", reply_json=True, role='agent')
        self.state_desc = json.loads(ans)
        #self.state_desc['percent_completion_plan'] = 'The current step divided by the total steps number in the current plan'
        self.memory.add(f"""
The state is expressed in JSON format, where the value is the explanation:
{json.dumps(self.state_desc)}
""")

        ans = self.llm.ask(f"""
For each element in the state, what would be the initial value? Give your answer in JSON format, where the key is the state key name, the value is the initial value.""", role='agent')
        self.state_init = json.loads(ans)
        #self.state_init['percent_completion_plan'] = 0

        #env.define_state(state_desc, state_bounds, state_init)

        """
        # define success criteria for stopping
        ans = self.llm.ask(""
Propose a quantifiable criterion for task accomplishment based on the state you proposed.
Give your answer in Python code that defines a function. Do not include use case. The function is named "is_successful". The function is wrapped by ```. The input to the function is the state, represented by a dict as you proposed above. The keys of the dict should have the same name as in the state vector you proposed. The function should return a boolean.
"", role='agent')
        self.set_success_func(_txt2code(ans))
        self.memory.add(f""
The success of the task is defined using a function that maps the state to a boolea:
```python
{self.success_func_code}
```"")
        """

        # define goodness function

        ans = self.llm.ask("""
Propose a goodness for the state you proposed. The goodness should be a number between 0 and 1 as a function of the state. The goodness should be higher when closer to task accomplishment.
Give your answer in Python code that defines a function. Do not include use case. The function is named "get_goodness". The function is wrapped by ```.
The input to the function is the state, represented by a dict as you proposed above. The keys of the dict should have the same name as in the state vector you proposed. The function should return a float number between 0 and 1. There should be a docstring "The goodness is a number between 0 and 1 as a function of the state. The goodness is higher when closer to task accomplishment."
""", role='agent')
        self.set_goodness_func(_txt2code(ans))
        self.memory.add(f"""
The goodness of current state is defined by the following function that takes state as input:
```python
{self.goodness_func_code}
```""")

        self.memory.add(self.plan_outline, type_='long')
        self.plan = Tree()
        self.plan.create_node('root', 'root')  # root node
        self.current_step = 'root'
        #self.create_plan(task)
        #check_env(env)
        #return env

    def next_step(self, task):
        mem = self.memory.to_text()
        self.llm.forget()
        import pdb;pdb.set_trace()
        ans = self.llm.ask(
f"""
{mem}

{self.plan_instructions[0]}

There are only three types of executable steps: (1) use Internet, where the step description is "<number>. use internet: ..."; (2) coding, where the step description is "<number>. coding: ..."; and (3) ask a human, where the step description is "<number>. ask human: ..." The 3rd type has a lower priority. """, role='agent')

        directions = eval(_txt2code(ans))
        assert len(directions)>0
        for i, d in enumerate(directions):
            d_ = d.split('.')[1].split(':')[0].strip()
            assert d_ in ['use internet', 'coding', 'ask human']
            id_ = f'step 1/direction {i+1}'
            self.plan.create_node(f'{id_}: {d}', id_, parent=self.current_step)
        self.current_step = 'step 1/direction 1'
        return directions[0]

    def clear(self):
        self.llm.forget()
        self.plan.remove_node('root')
        self.plan.create_node('root', 'root')
        self.current_step = 'root'
        #TODO self.rl.clear()

