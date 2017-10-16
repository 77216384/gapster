import helpers
import pickle
from flask import Flask, render_template, request, redirect


app = Flask(__name__)

with open('gapGBCmodel.pickle', 'rb') as f:
    gbc = pickle.load(f)

@app.route("/")
def index():
    return "Hello World!"

@app.route("/question", methods=['POST'])
def make_question():
	if request.is_json:
		data = request.get_json() # data will be a python dict
		text = data['text']
		sentence = str(helpers.get_best_sentences(text)[0])
		questions = helpers.make_all_questions(sentence, helpers.get_top_patterns())
		best_question = helpers.predict_best_question(questions, gbc)
		return best_question[0]['question']
	else:
		return "Error: Request not in JSON format."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
