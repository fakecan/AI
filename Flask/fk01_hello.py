from flask import Flask

app = Flask(__name__)

@app.route('/') # app.route 사용 후, 바로 아래 함수와 연결
def hello333():
    return "<h1>hello youngsun world</h1>"

@app.route('/bit') # app.route 사용 후, 바로 아래 함수와 연결
def hello334():
    return "<h1>hello bit computer world</h1>"

if __name__ == '__main__':
    # app.run(host="IP address", port=?, debug=False) 웹 접속 시 주소:   IP address:port number
    # 200일 시, 정상 상태   127.0.0.1 - - [03/Sep/2019 10:06:45] "?[37mGET / HTTP/1.1?[0m" 200 -
    # app.run(host="127.0.0.1", port=5000, debug=False)
    # app.run(host="192.168.0.173", port=8888, debug=False)
    app.run(host="127.0.0.1", port=8888, debug=False)
