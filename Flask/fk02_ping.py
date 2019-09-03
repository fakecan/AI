from flask import Flask

app = Flask(__name__)

@app.route('/') # app.route 사용 후, 바로 아래 함수와 연결
def hello333():
    return "<h1>hello world</h1>"

@app.route('/ping', methods=['GET'])
def ping():
    return "<h1>pong</h1>"

if __name__ == '__main__':
    # app.run(host="127.0.0.1", port=5000, debug=False)    
    app.run(host="127.0.0.1", port=8888, debug=False)
