from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/hello')
def greeting():
	f = open('run.py', 'r')
	exec(f.read())
	return 'finish'