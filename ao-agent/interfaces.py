import sys, logging
from flask import Flask, request, jsonify, render_template
import colorama
from colorama import Fore, Back, Style
colorama.init()


class CmdInterface:
    def says(self, role, msg=None):
        if role=='ai':
            print(Fore.YELLOW + 'AI says:\n' + msg + Style.RESET_ALL)
        elif role=='agent':
            print(Fore.GREEN + 'Agent says:\n' + msg + Style.RESET_ALL)
        elif role=='system':
            print(Fore.RED + 'System says:\n' + msg + Style.RESET_ALL)
        elif role=='human':
            if msg is None:
                print(Fore.CYAN + 'Human says: ')
                msg = ''.join(sys.stdin.readlines()).strip()
                print(Style.RESET_ALL)
            else:
                print(Fore.CYAN + 'Human says:\n' + msg + Style.RESET_ALL)
        else:
            raise NotImplementedError(role)
        logging.info(msg)
        return msg
    
    def human_says(self):
        return self.says('human')

    def run(self, debug=False):
        pass


"""
class FlaskInterface(Flask):
    def __init__(self, template_folder='templates'):
        super().__init__('name', template_folder=template_folder)

        def index():
            return render_template('index.html')
        self.add_url_rule('/', 'index', index)

        def says(self, msg, role):
            return jsonify(response=msg)
        flask_app.add_url_rule('/ai_says', 'ai_says', ai_says, methods=['POST'])

        def human_input(self, msg):
            ??
        flask_app.add_url_rule('/human_says', 'human_says', human_says, methods=['POST'])
"""

