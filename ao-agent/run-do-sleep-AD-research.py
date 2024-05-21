from ao_agent import AOAgent
from env import WebEnvironment#TODO, LabEnvironment

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler() ]
)


def main():
    instruction = 'In general, your role is an acamedic researcher. You are good at understanding the scientific literature and scientific thinking. You are curious about the nature and the underlying mechanisms.'

    env = WebEnvironment()

    ao = AOAgent(name='AO-Sleep-AD-Researcher', llm_name='gpt-4o',
        profile_instruction=instruction, human_in_loop=True)
    ao.solve_task('Do research in the field of sleep and dementia.', env)



if __name__=='__main__':
    main()
