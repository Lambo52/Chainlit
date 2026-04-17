import psycopg2
import sys
import bcrypt
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()


def connect_db():
    try:
    
        connection = psycopg2.connect(
            host="host.docker.internal", 
            port="5488",                 
            user="admin",
            password="password_segreta",
            database="chainlit_db"
        )
        print("Connesso")
        connection.autocommit = True

        return connection

    except Exception as e:
        print(f"Errore di connessione: {e}")
        sys.exit(1)

