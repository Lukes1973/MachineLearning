import mysql.connector
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="qaz!@wsx",
    database="work"
)
print(mydb)

mycursor=mydb.cursor()
mycursor.execute("SELECT * FROM stock_age_cal_8 ")
stockdata=mycursor.fetchall()

