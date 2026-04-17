import psycopg2
import sys
import bcrypt
from datetime import datetime
from dotenv import load_dotenv
import os
from utils import connect_db

load_dotenv()

def hash_password(password: str) -> str:

    auth_secret = os.getenv('CHAINLIT_AUTH_SECRET', 'default-secret').encode()

    salt = bcrypt.gensalt(rounds=12)
    
    salted_password = password.encode() + auth_secret
    hashed = bcrypt.hashpw(salted_password, salt)
    return hashed.decode()

def insert_user(identifier, password, gruppo):

    try:
    
        connection = connect_db()

        cur = connection.cursor()
        print("Connesso!")
        
        
        hashed_password = hash_password(password)
        cur.execute("""
        INSERT INTO auth_user (identifier, password, gruppo) 
        VALUES (%s, %s, %s)
        ON CONFLICT (identifier) 
        DO UPDATE SET 
            password = EXCLUDED.password,
            gruppo = EXCLUDED.gruppo;
        """, (identifier, hashed_password, gruppo))


        
    except Exception as e:
        print(f"Errore durante l'inserimento dell'utente: {e}")
    finally:
        cur.close()
        connection.close()


#passati come argomenti identifier, password e gruppo
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python new_user.py <identifier> <password> <gruppo>")
    else:
        identifier = sys.argv[1]
        password = sys.argv[2]
        gruppo = sys.argv[3]
        insert_user(identifier, password, gruppo)