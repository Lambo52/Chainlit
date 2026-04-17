import psycopg2
import sys
import sqlalchemy
import bcrypt
from dotenv import load_dotenv
import os
from database.utils import connect_db

load_dotenv()

def verify_password(stored_hash: str, password: str) -> bool:

    auth_secret = os.getenv('CHAINLIT_AUTH_SECRET', 'default-secret').encode()
    
    salted_password = password.encode() + auth_secret
    return bcrypt.checkpw(salted_password, stored_hash.encode())

def auth(username: str, password: str):

    connection = connect_db()

    cur = connection.cursor()

    cur.execute("SELECT * FROM auth_user WHERE identifier = %s", (username,))
    user = cur.fetchone()
    
    print(user)
    cur.close()
    connection.close()

    if user and verify_password(user[1], password):
    
        return user[0] + "-" + user[2]
    else:
    
        return None