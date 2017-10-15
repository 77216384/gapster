from flask import Flask, render_template, request, redirect
app = Flask(__name__)

with open('gapGBCmodel.pickle', 'rb') as f:
    gbc = pickle.load(f)

@app.route("/")
def index():
    return "Hello World!"

@app.route("/gapify", methods=['POST'])
def make_question():
    data = request.json() # data will be a python dict


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
