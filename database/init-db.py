import psycopg2
import sys
from database.utils import connect_db


try:
    
    connection = connect_db()


    cur = connection.cursor()
    print("Connesso")



    cur.execute("""DROP INDEX IF EXISTS idx_users_identifier;""")
 
    print("Dati inseriti e salvati")


    cur.close()
    connection.close()
    
except Exception as e:
    print(f" Errore: {e}")




"""

psql -U postgres -p 5532

-- Terminate active connections (needed for DROP)
REVOKE CONNECT ON DATABASE chainlit_db FROM public;
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'chainlit_db';

-- Drop the database
DROP DATABASE IF EXISTS chainlit_db;

CREATE DATABASE chainlit_db; NON FARE QUESTA STRINGA DATO CHE IL DB ESISTERà GIà, E NEANCHE QUELLE SOTTO SUGLI USER (DOVREBBE ESSERE GIà CONNESSO AD DB)
CREATE USER chainlit_user WITH PASSWORD 'securepassword';
GRANT ALL PRIVILEGES ON DATABASE chainlit_db TO chainlit_user;



-- USERS
CREATE TABLE users (
    id UUID PRIMARY KEY,
    identifier TEXT NOT NULL,
    "createdAt" TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- THREADS
CREATE TABLE threads (
    id UUID PRIMARY KEY,
    name TEXT,
    "createdAt" TEXT ,
    "userId" UUID,
    "userIdentifier" TEXT,
    tags TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb
);

-- FEEDBACKS
CREATE TABLE feedbacks (
    id UUID PRIMARY KEY,
    "forId" UUID,
    value TEXT,
    comment TEXT
);

-- ELEMENTS
CREATE TABLE elements (
    id UUID PRIMARY KEY,
    "threadId" UUID,
    type TEXT,
    "chainlitKey" TEXT,
    url TEXT,
    "objectKey" TEXT,
    name TEXT,
    display TEXT,
    size TEXT,
    language TEXT,
    page TEXT,
    "forId" UUID,
    mime TEXT,
    props JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE steps (
    id UUID PRIMARY KEY,
    "threadId" UUID REFERENCES threads(id) ON DELETE CASCADE,
    "parentId" UUID,
    name TEXT,
    type TEXT,
    input TEXT,
    output TEXT,
    "isError" BOOLEAN,
    streaming BOOLEAN,
    "waitForAnswer" BOOLEAN,
    "showInput" TEXT,
    "defaultOpen" BOOLEAN,
    "createdAt" TEXT NOT NULL,
    start TEXT,
    "end" TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    generation TEXT,
    tags TEXT[],
    language TEXT
);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO chainlit_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO chainlit_user;

ALTER TABLE users OWNER TO chainlit_user;
ALTER TABLE threads OWNER TO chainlit_user;
ALTER TABLE steps OWNER TO chainlit_user;
ALTER TABLE feedbacks OWNER TO chainlit_user;
ALTER TABLE elements OWNER TO chainlit_user;

-- Optional: Grant default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO chainlit_user;

CREATE TABLE IF NOT EXISTS auth_user (identifier TEXT PRIMARY KEY, password TEXT NOT NULL, gruppo TEXT NOT NULL);




"""