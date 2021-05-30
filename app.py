from flask import Flask, request, render_template
import pickle
from liberary import *

app = Flask(__name__)
modelx = pickle.load(open('Chat.pkl', "rb"))


@app.route('/', methods=["POST", 'GET'])
def home():
    if request.method == "POST":
        talk = [[request.form['talk']]]
        predict = modelx.predict(talk)
        return render_template('answer.html', predict=predict)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
