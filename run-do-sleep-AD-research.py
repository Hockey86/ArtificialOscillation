import sys
sys.path.insert(0, 'ao-agent')
from ao_agent import AOAgent
#from interfaces import FlaskInterface


def main():
    profile = 'Your role is an academic researcher. You are curious about the nature and the underlying mechanisms. You are good at understanding the scientific literature, applying scientific methods (such as hypothesis testing), identifying research gaps, analyzing data, deductive/abductive/inductive reasoning, writing scientific manuscripts and grants, and forming collaborations.'

    task = "Identify research topics in the field of sleep and Alzheimer's disease, and research the topics. You can access a computer via function calls, including Internet, programming software, and text editing software. Since you are not an embodied AI, you cannot actually perform experiments, conduct trials, or attend conferences. Instead, you can propose to do them."

    #TODO define interface
    # flask_app = FlaskInterface(template_folder='../flask_templates')

    # define agent
    ao = AOAgent(name='AO-Sleep-AD-Researcher', llm_name='gpt-4o',
        human_in_loop=True, profile_instruction=profile,)
        #TODO interface=flask_app)

    # run
    ao.solve_task(task)



if __name__=='__main__':
    main()
