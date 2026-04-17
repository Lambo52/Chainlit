import psycopg2
import sys
from utils import connect_db

try:
    
    connection = connect_db()


    cur = connection.cursor()
    print("Connesso")


    cur.execute("""DELETE FROM threads 
WHERE "createdAt" IS NULL;""")

#     CANCELLO I THREAD PIù VECCHI DI 5 GIORNI
    cur.execute("""
    DELETE FROM threads
    WHERE ("createdAt")::timestamptz < NOW() - INTERVAL '5 days';
""")
    
#   CANCELLO I THREAD OLTRE I 5 PIù RECENTI PER OGNI UTENTE
    cur.execute("""
                
    WITH RankedThreads AS (
    SELECT 
        id,
        -- Assegna un numero progressivo ai thread di ogni utente (1 = più recente)
        ROW_NUMBER() OVER (
            PARTITION BY threads."userId" 
            ORDER BY threads."createdAt" DESC, threads."id" DESC
        ) as numero_riga
    FROM threads
    )
    DELETE FROM threads
    WHERE id IN (
        SELECT id 
        FROM RankedThreads 
        WHERE numero_riga > 5
    );
                
""")

#   CANCELLO I THREADS PIù LUNGHI DI 5 MESSAGGI (USO >15 PERCHé OGNI ITERAZIONE AGGIUNGE UNA RIGA IN PIù OLTRE ALLE 2 STANDARD)

    cur.execute("""
    
    DELETE FROM threads
WHERE id IN (
    SELECT steps."threadId"
    FROM steps
    GROUP BY steps."threadId"
    HAVING COUNT(*) > 15
);

    """)




except Exception as e:
    print(f" Errore: {e}")