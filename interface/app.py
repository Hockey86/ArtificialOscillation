from flask import Flask, request, jsonify, render_template
import sys
sys.path.insert(0, '../ao-agent')
from llm import MyLLM

app = Flask(__name__)
llm = MyLLM('gpt-4o', 'You are a helpful assistant', session_id=None)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    response = llm.ask(request.json['message'])
    return jsonify(response=response)

if __name__ == '__main__':
    app.run(debug=True)

