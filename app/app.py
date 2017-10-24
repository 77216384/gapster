from __future__ import unicode_literals, division, print_function
from semantic import Blanker
import helpers
import pickle
import json
from flask import Flask, render_template, request, redirect, jsonify


app = Flask(__name__)

with open('lr_opt2.pickle', 'rb') as f:
    lr = pickle.load(f)

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
			b = Blanker(sentence)
			questions += b.blanks
		
		best_questions = helpers.predict_best_question(questions, lr, top_n=5)
		distractors = DistractorSet(text).make_distractors(best_questions[0][0]['spacy_answer'])
		question = best_questions[0][0]['question']
		answer = best_questions[0][0]['answer']
		output = {'question': question, 'distractors': distractors, 'answer': answer}
		output = {'question': question, 'answer': answer}
		return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)