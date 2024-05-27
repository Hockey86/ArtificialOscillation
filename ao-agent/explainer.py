from llm_simple import MyLLM


class Explainer:
    """
    """
    def __init__(self, llm_name, profile_instruction):
        self.llm = MyLLM(llm_name, profile_instruction)
        self.env_observer = None
