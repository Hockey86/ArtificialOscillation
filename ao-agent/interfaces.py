import logging
from flask import Flask, request, jsonify, render_template
import colorama
from colorama import Fore, Back, Style
colorama.init()


class CmdInterface:
    def says(self, msg, role):
        logging.info(msg)
        if role=='ai':
            print(Fore.YELLOW + msg + Style.RESET_ALL)
        elif role=='agent':
            print(Fore.GREEN + msg + Style.RESET_ALL)
        elif role=='human':
            print(Fore.CYAN + msg + Style.RESET_ALL)
        elif role=='system':
            print(Fore.RED + msg + Style.RESET_ALL)
        else:
            raise NotImplementedError(role)
    def run(self, debug=False):
        pass


"""
class FlaskInterface(Flask):
    def __init__(self, template_folder='templates'):
        super().__init__('name', template_folder=template_folder)

        def index():
            return render_template('index.html')
        self.add_url_rule('/', 'index', index)

        def ai_says(self, msg):
            return jsonify(response=msg)
        flask_app.add_url_rule('/ai_says', 'ai_says', ai_says, methods=['POST'])

        def human_says(self, msg):
            ??
        flask_app.add_url_rule('/human_says', 'human_says', human_says, methods=['POST'])
"""

