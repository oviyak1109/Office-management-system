import mysql.connector

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "mksomkso",
    "port": 3306,
    "database": "office_management"
}

def connect_database():
    connection = mysql.connector.connect(**DB_CONFIG)
    return connection
