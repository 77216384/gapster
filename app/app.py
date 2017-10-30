from __future__ import unicode_literals, division, print_function
from semantic import Blanker, DistractorSet
import helpers
import pickle
import json
import spacy
import requests
from bs4 import BeautifulSoup as bs
import json
import re
from flask import Flask, render_template, request, redirect, jsonify, url_for


app = Flask(__name__)

with open('lr_opt2.pickle', 'rb') as f:
    lr = pickle.load(f)

nlp = spacy.load('en_core_web_md')

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/fetch_article/<topic>")
def get_demo_article(topic):
	article_dict = {
		'napoleon': 'https://en.wikipedia.org/wiki/Napoleonic_Wars',
		'snoop': 'https://en.wikipedia.org/wiki/Snoop_Dogg',
		'lincoln': 'https://en.wikipedia.org/wiki/Abraham_Lincoln',
		'buddhism': 'https://en.wikipedia.org/wiki/Buddhism',
		'wormhole': 'https://en.wikipedia.org/wiki/Wormhole'
	}
	url = article_dict[topic]
	res = requests.get(url)
	html = bs(res.content, 'lxml')
	article = " ".join([p.text for p in html.select('#mw-content-text p')])
	data = json={'text': re.sub('\[\d+\]', '', article)}
	return jsonify(data)

@app.route("/question", methods=['POST'])
def make_question():
	if request.is_json:
		data = request.get_json() # data will be a python dict
		text = data['text']
		spacy_text = nlp(text)
		sentences = helpers.get_best_sentences(text, num=5)
		
		sentences_with_srl = []
		for s in sentences:
			srl = helpers.get_srl(s)
			sentences_with_srl.append({'sentence':s, 'srl':srl})

		questions = [] 
		
		for sentence in sentences_with_srl:
			b = Blanker(sentence, nlp)
			questions += b.blanks
		
		predicted_best_questions = helpers.predict_best_question(questions, lr, nlp, top_n=5)
		best_questions = []

		for question in predicted_best_questions:
			question = helpers.check_question_quality(question, spacy_text)
			if question['quality']:
				best_questions.append(question)

		question = best_questions[0]['question']
		answer = best_questions[0]['answer']
		distractors = DistractorSet(best_questions[0], text, spacy_text, nlp).distractors
		output = {'question': question, 'answer': answer.capitalize(), 'distractors': distractors}

		return jsonify(output)
	else:
		return "not json"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)