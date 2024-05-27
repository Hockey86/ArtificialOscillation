from ao_agent import AOAgent
from env import WebEnvironment#TODO, LabEnvironment


def main():
    ao = AOAgent(name='AO-Sleep-AD-Researcher', llm_name='gpt-4o',
    human_in_loop=True,
    profile_instruction='Your role is an acamedic researcher. You are good at understanding the scientific literature and scientific thinking. You are curious about the nature and the underlying mechanisms.')

    env = WebEnvironment()

    ao.solve_task('Do scientific research in the field of sleep and dementia. Here, scientific research refers to scientific practice, including searching for academic literature, understanding knowledge structure, identifying research gaps, proposing new directions, etc. However, you are a text-based AI, therefore you cannot actually perform experiments or conduct trials, but you can propose how to do them.', env)



if __name__=='__main__':
    main()
