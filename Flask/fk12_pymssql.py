# module load
import pymssql as ms
import numpy as np

# database connect
conn = ms.connect(server='localhost', user='bit',   # sever='localhost' or '127.0.0.1'
                  password='1234', database='bitdb')

# cursor create
cursor = conn.cursor()

# cursor query insert and execute
cursor.execute('SELECT TOP 1 * FROM train;')

# one row get
row = cursor.fetchone()
# print(type(row))    # tuple

while row:
    # print("첫 컬럼=%s, 둘 컬럼=%s" %(row[0], row[1]))
    print(row)
    row = cursor.fetchone()

d = np.array(conn)
cursor = conn.cursor()
cursor.execute('SELECT TOP 1 * FROM train;')
row = cursor.fetchone()
print(type(d))
print(d)
# connect close
conn.close()
