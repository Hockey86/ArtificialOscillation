import json, types
import backoff
from stable_baselines3 import PPO
from llm_simple import MyLLM
from env import AgentEnv


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
    def __init__(self, llm_name, profile_instruction):
        self.llm = MyLLM(llm_name, profile_instruction)
        self.current_plan = []

    def set_state_desc(self, state_json):
        self.state_desc_json = state_json
        self.state_desc = json.loads(state_json)
        self.state_desc['percent_completion_current_plan'] = 'The current step divided by the total steps number in the current plan'
    
    def set_success_func(self, code):
        self.success_func_code = code
        code_obj = compile(code, '<string>', 'exec')
        self.success_func = types.FunctionType(code_obj.co_consts[0], globals())

    def set_reward_func(self, code):
        self.reward_func_code = code
        code_obj = compile(code, '<string>', 'exec')
        self.reward_func = types.FunctionType(code_obj.co_consts[0], globals()) 
    
    @backoff.on_exception(backoff.constant,
            SyntaxError, interval=1, max_tries=5)
    def init(self, task):

        # define state

        ans = self.llm.ask(f"""
You are given a task:
<task>
{task}
</task>

You need to formulate this task into a reinforcement learning (RL) problem. In RL, one needs to define state, action, and reward. How would you define the state for this task? The state is a vector, where each dimension represents a quantitative aspect.

Give your answer in JSON format, where each key-value pair represents a dimension in the state vector. The key is a brief python variable name. The value is an explanation of the state vector dimension.
""", json=True, my_answer="""
{
  "literature_coverage": "The proportion of relevant academic literature that has been reviewed and understood. This can be quantified as a percentage of the total number of relevant papers identified.",
  "knowledge_structure_completeness": "A measure of how well the current understanding of the field is mapped out, including key concepts, theories, and findings. This can be quantified on a scale from 0 to 1, where 1 represents a complete and well-organized knowledge structure.",
  "identified_research_gaps": "The number of research gaps identified in the current literature. This can be a simple count of the gaps.",
  "proposed_directions_quality": "A qualitative measure of the novelty and potential impact of the proposed new research directions. This can be quantified on a scale from 0 to 1, where 1 represents highly novel and impactful directions.",
  "experiment_design_completeness": "A measure of how well the proposed experimental designs are developed, including hypotheses, methodologies, and expected outcomes. This can be quantified on a scale from 0 to 1, where 1 represents a fully developed and robust experimental design.",
  "collaboration_network_strength": "A measure of the strength and extent of the proposed collaboration network, including potential partners and their expertise. This can be quantified on a scale from 0 to 1, where 1 represents a strong and extensive network.",
  "conference_attendance_plan": "A measure of the completeness and relevance of the plan for attending conferences and workshops. This can be quantified on a scale from 0 to 1, where 1 represents a well-developed and relevant plan.",
  "funding_opportunities_identified": "The number of potential funding opportunities identified for the proposed research. This can be a simple count of the opportunities."
}""")
        self.set_state_desc(ans)

        #for state_name, desc in state_desc.items():
        #    ans = self.llm.ask(f'Quantify {desc}')
        #TODO self.state_func = state_func

        ans = self.llm.ask(f"""
For each element in the state, tell me the data type, lower bound, and upper bound. The data type should be "int" (keep the quotes in your answer) for integer or "float" (keep the quotes in your answer) for continuous number. In case the element is unbounded at either side (such as count, which is unbounded above), that bound should be Infinity (no quote). Give your answer in JSON format, where the key is the state key name, the value is a list: [data type, lower bound, upper bound].""")
        import pdb;pdb.set_trace()
        self.state_bounds = json.loads(ans)
        self.state_bounds = {k:[eval(v[0]),v[1],v[2]] for k,v in self.state_bounds.items()}
        self.state_bounds['percent_completion_current_plan'] = [float, 0, 1]

        ans = self.llm.ask(f"""
For each element in the state, what would be the initial value? Give your answer in JSON format, where the key is the state key name, the value is the initial value.""")
        self.state_init = json.loads(ans)
        self.state_init['percent_completion_current_plan'] = 0

        # define success criteria for stopping

        ans = self.llm.ask("""
Propose a quantifiable criterion for task accomplishment based on the state you proposed.
Give your answer in Python code that defines a function. Do not include use case. The function is named "is_successful". The function is wrapped by ```. The input to the function is the state, represented by a dict as you proposed above. The keys of the dict should have the same name as in the state vector you proposed. The function should return a boolean.
""", my_answer="""```python
def is_successful(state):
    # Define thresholds for each dimension
    thresholds = {
        "literature_coverage": 0.8,  # 80% of relevant literature reviewed
        "knowledge_structure_completeness": 0.7,  # 70% completeness in knowledge structure
        "identified_research_gaps": 5,  # At least 5 research gaps identified
        "proposed_directions_quality": 0.7,  # 70% quality in proposed directions
        "experiment_design_completeness": 0.7,  # 70% completeness in experimental design
        "collaboration_network_strength": 0.5,  # 50% strength in collaboration network
        "conference_attendance_plan": 0.6,  # 60% completeness in conference plan
        "funding_opportunities_identified": 3  # At least 3 funding opportunities identified
    }

    # Check if all criteria are met
    for key, threshold in thresholds.items():
        if state[key] < threshold:
            return False
    return True
```""")
        self.set_success_func(_txt2code(ans))

        # define reward function
        ans = self.llm.ask("""
Propose reward based on the state you proposed. The reward should be a single number as a function of the state. The reward should be higher when it is closer to task accomplishment.
Give your answer in Python code that defines a function. Do not include use case. The function is named "get_reward". The function is wrapped by ```.
The input to the function is the state, represented by a dict as you proposed above. The keys of the dict should have the same name as in the state vector you proposed. The function should return a float number.
""", my_answer="""```python
def get_reward(state):
    # Define weights for each dimension
    weights = {
        "literature_coverage": 0.15,
        "knowledge_structure_completeness": 0.15,
        "identified_research_gaps": 0.1,
        "proposed_directions_quality": 0.2,
        "experiment_design_completeness": 0.2,
        "collaboration_network_strength": 0.1,
        "conference_attendance_plan": 0.05,
        "funding_opportunities_identified": 0.05
    }

    # Normalize the identified_research_gaps and funding_opportunities_identified
    max_gaps = 10  # Assume a maximum of 10 gaps for normalization
    max_funding = 5  # Assume a maximum of 5 funding opportunities for normalization

    normalized_state = {
        "literature_coverage": state["literature_coverage"],
        "knowledge_structure_completeness": state["knowledge_structure_completeness"],
        "identified_research_gaps": min(state["identified_research_gaps"] / max_gaps, 1.0),
        "proposed_directions_quality": state["proposed_directions_quality"],
        "experiment_design_completeness": state["experiment_design_completeness"],
        "collaboration_network_strength": state["collaboration_network_strength"],
        "conference_attendance_plan": state["conference_attendance_plan"],
        "funding_opportunities_identified": min(state["funding_opportunities_identified"] / max_funding, 1.0)
    }

    # Calculate the weighted sum of the normalized state values
    reward = sum(normalized_state[key] * weight for key, weight in weights.items())

    return reward
```""")
        self.set_reward_func(_txt2code(ans))

        #TODO memory.add

    def human_verify_rl_setting(self):
        msg = f"""
I have figured out the following RL setting:

state:

{self.state_desc_json}

reward:

{self.reward_func_code}

success:

{self.success_func_code}

Is this ok? Type Y (default) or N.
"""
        ans = input(msg)
        while ans.lower().strip() not in ['', 'yes', 'no', 'y', 'n']:
            ans = input(f"I don't understand your answer \"{ans}\". Type Y (default) or N.")
        ans = ans.lower().strip()
        is_ok = ans in ['', 'yes', 'y']

        if is_ok:
            logging.info('Human check passed')
        else:
            self.clear()
            self.set_state_desc(input('Input state description in JSON format:'))
            self.set_success_func(input('Input success criteria function:'))
            self.set_reward_func(input('Input reward function:'))

    def next_step(self, task, memory, explainer):
        """
        """
        action = self.rl.predict(self.state)

        if action=='run current plan':
            if len(self.current_plan)==0: # make a plan:
                reward = -10
            else:
                last_step = self.current_plan.pop()
        elif action=='create new plan':
            ans = self.llm.ask(
f"""Here is a plan template:
1. make observations
2. make hypothesis (abductive reasoning)
3. if hypothesis is true, what to expect (deductive reasoning)
4. test the hypothesis
5. what general principle can we get (inductive reasoning)

Now, generate a plan for the given task: {task}
""")
            self.current_plan = eval(ans)

        self.rl.rollout_buffer.add()
        return action

    def clear(self):
        self.llm.forget()
        self.current_plan.clear()
        #TODO self.rl.clear()

