from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)  # {{ }} 중괄호 2개로

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
