# 모듈을 불러옵니다.
import pyodbc as pyo

# 연결 문자열을 설정합니다.
server = 'localhost'
database = 'bitdb'
username = 'bit'
password = '1234'

# 데이터베이스를 연결합니다.
cnxn = pyo.connect('DRIVER={ODBC Driver 13 for SQL Server}; SERVER=' +server+
                   '; PORT=1433; DATABASE=' +database+
                   ';UID=' +username+
                   ';PWD=' +password
                  )

# 커서를 만듭니다.
cursor = cnxn.cursor()

# 커서에 쿼리를 입력해 실행시킵니다.
tsql = "SELECT * FROM iris2;"

# flask 웹서버를 실행합니다.
from flask import Flask, render_template
app = Flask(__name__)

@app.route('/sqltable')
def showsql():
    cursor.execute(tsql)
    return render_template('myweb.html', rows=cursor.fetchall())

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
