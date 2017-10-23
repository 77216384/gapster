from __future__ import unicode_literals, division, print_function
import helpers
import pickle
import json
from flask import Flask, render_template, request, redirect, jsonify


app = Flask(__name__)

with open('gbc_optimal.pickle', 'rb') as f:
    gbc = pickle.load(f)

@app.route("/")
def index():
    return "Hello World!"

@app.route("/question", methods=['POST'])
def make_question():
	if request.is_json:
		data = request.get_json() # data will be a python dict
		text = data['text']
		sentences = helpers.get_best_sentences(text, num=5)
		questions = [] 
		
		for sentence in sentences:
			questions += helpers.make_all_questions(str(sentence), helpers.get_top_patterns())
		
		best_questions = helpers.predict_best_question(questions, gbc, top_n=5)
		distractors = helpers.make_distractors(text, best_questions[3][0]['pattern'], best_questions[3][0]['answer'])
		question = best_questions[3][0]['question']
		answer = best_questions[3][0]['answer']
		output = {'question': question, 'distractors': distractors, 'answer': answer}
		return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)