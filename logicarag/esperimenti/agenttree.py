import os
import qdrant_client
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.retrievers import QueryFusionRetriever, AutoMergingRetriever
from llama_index.core import QueryBundle
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.llms.google_genai import GoogleGenAI
import sqlite3
import pandas as pd
import numpy as np
from typing import Optional
from pydantic import BaseModel, Field
from logicarag.rerank import reranka
from logicarag.esperimenti.utils import filtra_con_deviazione_standard
from concurrent.futures import ThreadPoolExecutor, as_completed



VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL")
QDRANT_URL = os.getenv("QDRANT_URL")
collectionname = "WAMASRAGAGENT"

client = qdrant_client.QdrantClient(url=QDRANT_URL)

embed_model = OpenAIEmbedding(
    api_base=VLLM_API_BASE_URL,
    model_name="BAAI/bge-m3",
    api_key="null",
)
Settings.embed_model = embed_model




vector_store = QdrantVectorStore(
    client=client,
    collection_name=collectionname,
    enable_hybrid=True,
    dense_vector_name="bge_m3",
    sparse_vector_name="bm25",
    fastembed_sparse_model="Qdrant/bm25"
)


storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)

class RispostaParziale(BaseModel):
    ragionamento: str = Field(
        description="Ragionamento Chain of Thought che spiega perché il contesto è (o non è) pertinente."
    )
    titolo: Optional[str] = Field(
        default=None,
        description="Un titolo ultra-specifico che cattura l'argomento esatto del contesto per distinguerlo da altri simili. Ad esempio, 'Creazione UDC Primaria da Terminale'."
    )
    risposta: Optional[str] = Field(
        default=None,
        description="Risposta dettagliata basata SOLO sul contesto, che inizia specificando il proprio ambito di applicazione."
    )
    valido: bool = Field(
        description="True se il contesto contiene informazioni direttamente pertinenti e utili per rispondere alla domanda, altrimenti False."
    )

AGENT_PROMPT = """
Sei un agente esperto del sistema WMS WAMAS. Ricevi un estratto di un documento tecnico (contesto) e una domanda. La tua missione è analizzare il contesto e determinare se è utile per rispondere alla domanda, estraendo le informazioni in modo molto specifico.

**Istruzioni Rigorose:**

1.  **Analisi di Pertinenza**: Leggi la domanda e il contesto. Se il contesto NON contiene informazioni utili per rispondere, imposta "valido": False e interrompi.
2.  **Creazione Titolo Specifico**: Se il contesto è valido, crea un `titolo` che descriva l'argomento esatto del testo. NON usare titoli generici.
    *   NO: "Creazione di una UDC"
    *   SÌ: "Procedura per la Creazione di UDC Primaria e Secondaria"
    *   SÌ: "Creazione di Unità di Carico per un Camion (UDC Camion)"
3.  **Formulazione della Risposta**: Se il contesto è valido, scrivi una `risposta`. La risposta deve:
    *   **Iniziare sempre dichiarando il suo ambito specifico**, usando il titolo che hai creato. In questo modo la risposta si auto-contestualizza.
    *   Basarsi **esclusivamente** sulle informazioni presenti nel contesto.
    *   NO: "Per creare una UDC devi..."
    *   SÌ: "**Riguardo alla creazione di UDC Primarie e Secondarie**, la procedura è..."
    *   SÌ: "**Nel caso specifico della creazione di un'Unità di Carico Camion**, i passaggi sono..."

Contesto:
{context}
----------
Domanda:
{question}
"""


agent_prompt = PromptTemplate(AGENT_PROMPT)


# llm = OpenAILike(
#     model='Qwen/Qwen2.5-32B-Instruct-AWQ',
#     api_base=VLLM_API_BASE_URL, 
#     api_key="null",
#     is_chat_model=True,
#     is_function_calling_model=True,
#     timeout=60.0,
#     streaming=False,
#     context_window=8192, 
#     temperature=0,
#     max_tokens=1024 
# )

llm = GoogleGenAI(
    model="models/gemini-2.5-pro",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,        
    max_tokens=8192
)


TOPK_EMBEDDER_NODES = 20
TOPK_RERANKED_NODES = 10
TOPK_SIMILARITY_NODES = 5
OVERLAP_THRESHOLD = 3
CHUNK_EXPANSION_RANGE = 10

# Path del database
DB_PATH = os.path.join(os.path.dirname(__file__), f"{collectionname}.db")


def get_overlap_nodes(nodes, top_n=5):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    
    new_nodes = []

    for node in nodes:
        if len(new_nodes) >= top_n:
            break
        if len(new_nodes) == 0:
            new_nodes.append(node)
            continue
        aggiungo_nodi = True
        for existing_node in new_nodes:
            if node.node.metadata["origin_filename"] == existing_node.node.metadata["origin_filename"]:
                if node.node.metadata["chunk_index"] < existing_node.node.metadata["chunk_index"] + OVERLAP_THRESHOLD and node.node.metadata["chunk_index"] > existing_node.node.metadata["chunk_index"] - OVERLAP_THRESHOLD:
                    aggiungo_nodi = False
                    break

        if aggiungo_nodi:
            new_nodes.append(node)

    conn.close()
    return new_nodes


def expand_nodes(nodes):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    expanded_nodes = []

    for node in nodes:
        chunk_index = node.node.metadata["chunk_index"]
        origin_filename = node.node.metadata["origin_filename"]

        cursor = conn.cursor()
        cursor.execute("SELECT * FROM document_chunks WHERE filename = ? AND chunk_index BETWEEN ? AND ?", 
                       (origin_filename, chunk_index-CHUNK_EXPANSION_RANGE, chunk_index+CHUNK_EXPANSION_RANGE))
        rows = cursor.fetchall()

        testo_totale = ""
        
        for row in rows:
            testo_totale += "chunk index: " + str(row[1]) + "\n" + row[2] + "\n"
        
        new_text_node = TextNode(
            text=testo_totale,
            metadata={
                "origin_filename": origin_filename,
            }
        )
        # Preserva lo score originale del nodo
        new_node = NodeWithScore(node=new_text_node, score=node.score)
        expanded_nodes.append(new_node)

    conn.close()
    return expanded_nodes



def risposta_llm_parziale(nodo_espanso, query_text):


    nodo_espanso_text = "NOME FILE: " + nodo_espanso.node.metadata["origin_filename"] + "\n\nTESTO:\n" + nodo_espanso.node.get_content()

    #print(nodo_espanso)


    risultato_parziale = llm.structured_predict(
            RispostaParziale,
            prompt=agent_prompt,
            context=nodo_espanso_text,
            question=query_text
        )
    
    return risultato_parziale 


def agent_tree(query_text, user_role, hydeaugmented=None, queryaugmentation=None):

    
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="groups",  
                value=user_role,    
                operator=FilterOperator.EQ 
            )
        ]
    )

    retriever_dense = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT, 
        similarity_top_k=TOPK_EMBEDDER_NODES,
        filters=filters if user_role != "admin" else None
    )
    
    retriever_sparse = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.SPARSE,
        sparse_top_k=TOPK_EMBEDDER_NODES,
        filters=filters if user_role != "admin" else None
    )


    retriever = QueryFusionRetriever(
    retrievers=[retriever_dense, retriever_sparse],
    similarity_top_k=2*TOPK_EMBEDDER_NODES,
    num_queries=1,          
    mode="simple",                                       
    use_async=False
    )

    query_bundle = QueryBundle(
    query_str=query_text,
    custom_embedding_strs=hydeaugmented if hydeaugmented else (queryaugmentation if queryaugmentation else None)
    )

    nodes = retriever.retrieve(query_bundle) #prima scrematura
    print("finito retrieve, ritornati: ", len(nodes))

    nodes = get_overlap_nodes(nodes, top_n=TOPK_EMBEDDER_NODES) #scrematura per overlap
    print("tolti i nodi sovrapposti, ora sono:", len(nodes))

    # Estraiamo i TextNode dai NodeWithScore per passarli al reranker
    text_nodes = [node.node for node in nodes]
    nodes = reranka(text_nodes, query_text, top_n=TOPK_RERANKED_NODES) #rerank alto
    print("finito rerank, ritornati: ", len(nodes))
    
    #tengo quelli con similarità alta
    nodes = filtra_con_deviazione_standard(nodes, 2, TOPK_SIMILARITY_NODES)
    print("numero nodi dopo filtro similarità: ", len(nodes))

    nodes = expand_nodes(nodes)
    print("finito di espandere i nodi")

    chunk_rilevanti = []
    
    # Esecuzione parallela delle chiamate all'LLM
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Sottometti tutti i task
        future_to_node = {executor.submit(risposta_llm_parziale, nodo, query_text): nodo for nodo in nodes}
        
        # Raccogli i risultati man mano che vengono completati
        for future in as_completed(future_to_node):
            result = future.result()
            if result.valido:
                singolo_file = [result.titolo, result.risposta]
                chunk_rilevanti.append(singolo_file)
        
        print("ottenute risposte parziali valide:", len(chunk_rilevanti))

    testo_post_context = ""
    for t in chunk_rilevanti:
        testo_post_context += "Titolo: " + (t[0] if t[0] else "N/A") + "\n"
        testo_post_context += "Risposta: " + (t[1] if t[1] else "N/A") + "\n\n"



    return testo_post_context, None


if __name__ == "__main__":
    test_query = "ho un disallineamento fisico logico di un pallet, come posso risolvere?"
    test_role = "admin"
    context, parts = agent_tree(test_query, test_role)
    print(context)
