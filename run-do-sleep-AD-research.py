import sys
sys.path.insert(0, 'ao-agent')
from ao_agent import AOAgent
#TODO from interfaces import FlaskInterface


def main():
    profile = 'Your role is an academic researcher. You are good at understanding the scientific literature and scientific thinking. You are curious about the nature and the underlying mechanisms.'

    task = 'Do scientific research in the field of sleep and dementia. Here, scientific research refers to scientific practice, including searching for academic literature, understanding knowledge structure, identifying research gaps, proposing new directions, etc. However, you are a text-based AI, therefore you cannot actually perform experiments, conduct trials, attend conferences, or form collaborations. However, you can propose how to do them.'

    # define interface
    #TODO flask_app = FlaskInterface(template_folder='../flask_templates')

    # define agent
    ao = AOAgent(name='AO-Sleep-AD-Researcher', llm_name='gpt-4o',
        human_in_loop=True, profile_instruction=profile,)
        #TODO interface=flask_app)

    # run
    ao.solve_task(task)



if __name__=='__main__':
    main()
