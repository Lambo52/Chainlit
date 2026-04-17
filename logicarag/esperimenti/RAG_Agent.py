import os
import qdrant_client
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# LlamaIndex Core
from llama_index.core import (
    VectorStoreIndex, 
    Settings, 
    StorageContext, 
    PromptTemplate, 
    QueryBundle
)
from llama_index.core.schema import NodeWithScore, TextNode

# LlamaIndex Vector Stores & Retrievers
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import (
    MetadataFilters, 
    MetadataFilter, 
    FilterOperator, 
    VectorStoreQueryMode
)
from llama_index.core.retrievers import QueryFusionRetriever

# LlamaIndex Models
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# Pydantic
from pydantic import BaseModel, Field

# Moduli Custom (Assicurati che questo file esista)
from logicarag.rerank import reranka

# =============================================
# 1. CONFIGURAZIONE E INIZIALIZZAZIONE
# =============================================

VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL")
QDRANT_URL = os.getenv("QDRANT_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

collectionname = "WAMASRAGOVERLAP"

# Setup Client Qdrant
client = qdrant_client.QdrantClient(url=QDRANT_URL)

# Setup Embedding
embed_model = OpenAIEmbedding(
    api_base=VLLM_API_BASE_URL,
    model_name="BAAI/bge-m3",
    api_key="null",
)
Settings.embed_model = embed_model

# Setup Vector Store
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

# Setup LLM
# Nota: Temperatura abbassata a 0.1 per garantire output JSON consistenti
llm = GoogleGenAI(
    model="models/gemini-2.5-pro",
    api_key=GOOGLE_API_KEY,
    temperature=0.1,        
    max_tokens=8192
)

# =============================================
# 2. CLASSI E PROMPT AGENTE
# =============================================

class AnalisiContestoRAG(BaseModel):
    ragionamento: str = Field(
        description="Un ragionamento passo-passo che spiega l'analisi del contesto, cosa manca e perché."
    )
    indici_chunk_utili: List[int] = Field(
        default=[],
        description="Lista degli indici (a partire da 0) dei chunk nel contesto attuale che sono rilevanti."
    )
    query_ricerca_aggiuntive: Optional[List[str]] = Field(
        default=None,
        description="Lista di NUOVE query di ricerca precise per trovare i pezzi mancanti. Null se completo."
    )
    informazioni_complete: bool = Field(
        description="True se hai TUTTE le informazioni necessarie per la risposta definitiva. False altrimenti."
    )

AGENT_PROMPT_TEXT = """
# RUOLO E OBIETTIVO
Sei un agente IA analista per un sistema RAG. Il tuo scopo è determinare se i documenti recuperati sono sufficienti per rispondere alla domanda dell'utente. NON rispondere alla domanda, valuta solo il contesto.

# PROCESSO DECISIONALE
1.  **Analisi:** Valuta criticamente ogni chunk fornito rispetto alla domanda.
2.  **Valutazione:**
    *   **INCOMPLETEZZA:** Se mancano passaggi, dettagli o se una procedura è troncata (es. finisce con "Step 1..."), imposta `informazioni_complete` a `False`. Spiega cosa manca nel `ragionamento` e crea `query_ricerca_aggiuntive` precise per trovarlo.
    *   **COMPLETEZZA:** Se hai tutto, imposta `informazioni_complete` a `True`. Elenca gli `indici_chunk_utili` (0, 1, 2...) che servono per la risposta.

# REGOLE FONDAMENTALI
1.  **Query Precise:** Non chiedere "più info". Chiedi "procedura creazione UDC passo 2", "elenco stati magazzino", etc.
2.  **Indici:** Riferisciti ai chunk usando SOLO il numero indice fornito nel contesto (es. [Indice: 0]).

**Contesto Attuale:**
{context}
----------
**Domanda Utente:**
{question}
"""

agent_prompt = PromptTemplate(AGENT_PROMPT_TEXT)

# =============================================
# 3. FUNZIONI HELPER
# =============================================

TOPK_NODES = 25
TOPK_RERANKER = 5
MAX_ITERATIONS = 5  # Limite di sicurezza per evitare loop infiniti

def contexter(nodes: List[NodeWithScore]) -> str:
    """Crea una stringa formattata con indici espliciti per l'LLM."""
    if not nodes:
        return "Nessun contesto disponibile."
    
    context_str = ""
    for i, node in enumerate(nodes):
        filename = node.node.metadata.get("origin_filename", "sconosciuto")
        # chunk_index = node.node.metadata.get("chunk_index", "N/A") # Opzionale visualizzarlo
        contenuto = node.node.get_content()
        context_str += f"[Indice: {i}] File: {filename}\n{contenuto}\n---\n"
    return context_str

def unique_nodi(nodi: List[NodeWithScore]) -> List[NodeWithScore]:
    """Rimuove duplicati basandosi sul node_id univoco."""
    seen = set()
    unique = []
    for nodo in nodi:
        if nodo.node.node_id not in seen:
            seen.add(nodo.node.node_id)
            unique.append(nodo)
    return unique

def retrieve_helper(retriever, query: str, top_n: int) -> List[NodeWithScore]:
    """Helper per recuperare e rerankare."""
    nodes = retriever.retrieve(query)
    # Applica reranking se ci sono nodi
    if nodes:
        nodes = reranka(nodes, query, top_n=top_n)
    return nodes

# =============================================
# 4. LOGICA CORE AGENTE (RAG ITERATIVO)
# =============================================

def agent_rag(query_text, user_role, hydeaugmented=None, queryaugmentation=None):
    
    # 1. Configurazione Retriever con Filtri
    filters = MetadataFilters(
        filters=[MetadataFilter(key="groups", value=user_role, operator=FilterOperator.EQ)]
    ) if user_role != "admin" else None

    # Retriever Ibrido (Densa + Sparsa)
    retriever_dense = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT, 
        similarity_top_k=TOPK_NODES,
        filters=filters
    )
    retriever_sparse = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.SPARSE,
        sparse_top_k=TOPK_NODES,
        filters=filters
    )
    
    # Query Fusion per unire i risultati
    retriever = QueryFusionRetriever(
        retrievers=[retriever_dense, retriever_sparse],
        similarity_top_k=TOPK_NODES, # Fusion gestirà il ranking
        num_queries=1,
        mode="reciprocal_rerank", # Migliore di 'simple'
        use_async=False
    )

    # 2. Recupero Iniziale
    print(f"--- Avvio RAG per: '{query_text}' ---")
    
    # Usiamo augmentation se presente per la prima query
    q_str = hydeaugmented if hydeaugmented else (queryaugmentation if queryaugmentation else query_text)
    
    current_nodes = retrieve_helper(retriever, q_str, TOPK_RERANKER)

    # 3. Ciclo Iterativo dell'Agente
    for i in range(MAX_ITERATIONS):
        print(f"\n--- Iterazione {i+1} ---")
        
        # Prepara il contesto numerato per l'LLM
        contesto_str = contexter(current_nodes)
        
        # Chiamata LLM strutturata
        try:
            analisi = llm.structured_predict(
                AnalisiContestoRAG,
                prompt=agent_prompt,
                context=contesto_str,
                question=query_text
            )
        except Exception as e:
            print(f"Errore nella chiamata LLM: {e}")
            break # Esce se l'LLM fallisce

        print(f"Ragionamento: {analisi.ragionamento}")

        # CASO A: Informazioni Complete
        if analisi.informazioni_complete:
            print(">> STATUS: Informazioni COMPLETE.")
            # Filtra solo i nodi che l'agente ha marcato come utili
            nodi_finali = []
            for idx in analisi.indici_chunk_utili:
                if 0 <= idx < len(current_nodes):
                    nodi_finali.append(current_nodes[idx])
            
            return contexter(nodi_finali), None

        # CASO B: Informazioni Incomplete -> Nuove Ricerche
        print(">> STATUS: Informazioni INCOMPLETE. Richiesta nuova ricerca.")
        
        if not analisi.query_ricerca_aggiuntive:
            print("Warning: L'agente non ha fornito query aggiuntive nonostante l'incompletezza.")
            break

        print(f"Query generate: {analisi.query_ricerca_aggiuntive}")

        # Mantieni i nodi "buoni" trovati finora
        nodi_utili_correnti = []
        for idx in analisi.indici_chunk_utili:
            if 0 <= idx < len(current_nodes):
                nodi_utili_correnti.append(current_nodes[idx])

        # Esegui le nuove ricerche in PARALLELO
        nuovi_nodi_trovati = []
        with ThreadPoolExecutor() as executor:
            # Mappa ogni query alla funzione di retrieve
            future_to_query = {
                executor.submit(retrieve_helper, retriever, q, TOPK_RERANKER): q 
                for q in analisi.query_ricerca_aggiuntive
            }
            
            for future in as_completed(future_to_query):
                try:
                    risultati = future.result()
                    nuovi_nodi_trovati.extend(risultati)
                except Exception as exc:
                    print(f"Errore ricerca parallela: {exc}")

        # Aggiorna il pool di nodi: (Vecchi Utili) + (Nuovi Trovati) - Duplicati
        current_nodes = unique_nodi(nodi_utili_correnti + nuovi_nodi_trovati)
        print(f"Nuovo contesto: {len(current_nodes)} nodi totali.")

        if not current_nodes:
            print("Nessun nodo trovato nelle ricerche successive. Stop.")
            break

    print("Raggiunto limite iterazioni o impossibile completare.")
    return contexter(current_nodes), None

# =============================================
# 5. MAIN ESECUZIONE TEST
# =============================================
if __name__ == "__main__":
    test_query = "Come cambio la lampadina al camion?"
    test_role = "admin"
    
    # Esegui la funzione
    context, _ = agent_rag(test_query, test_role)
    
    print("\n" + "="*30)
    print("CONTESTO FINALE PER LLM GENERATIVO")
    print("="*30)
    print(context)