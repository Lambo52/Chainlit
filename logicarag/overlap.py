import os
import qdrant_client
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.retrievers import QueryFusionRetriever
from logicarag.rerank import reranka
from llama_index.core import QueryBundle


VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL")
QDRANT_URL = os.getenv("QDRANT_URL")
collectionname = "WAMASRAGOVERLAP"
#collectionname = "WAMASRAGRIASSUNTO" 

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


index = VectorStoreIndex.from_vector_store(vector_store=vector_store)



def get_context_overlap(query_text, user_role, hydeaugmented=None, queryaugmentation=None):

    print(f"Ricerca in corso per: '{query_text}' con ruolo utente: '{user_role}'")

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="groups",
                value=user_role,
                operator=FilterOperator.ANY #MESSO ANY PERCHé CONTROLLA CHE ALMENO UN ELEMENTO DELLA LISTA DOCUMENTO (CHE è SEMPRE 1) SIA PRESENTE NELLA LSTA USER_GROUPS
            )
        ]
    )

    topk = 15

    retriever_dense = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
        similarity_top_k=topk,
        filters=filters if "admin" not in user_role else None
    )
    retriever_sparse = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.SPARSE,
        sparse_top_k=topk,
        filters=filters if "admin" not in user_role else None
    )

    retriever = QueryFusionRetriever(
    retrievers=[retriever_dense, retriever_sparse],
    similarity_top_k=2*topk,
    num_queries=1,          
    mode="simple",                                       
    use_async=False
    )

    query_bundle = QueryBundle(
    query_str=query_text,
    custom_embedding_strs=hydeaugmented if hydeaugmented else (queryaugmentation if queryaugmentation else None)
    )

    nodes = retriever.retrieve(query_bundle)



    
    #nodes = retriever.retrieve(hydeaugmented if hydeaugmented else query_text)
    #nodes = retriever.retrieve(queryaugmentation)
    if not nodes:
        return "Nessuna informazione rilevante trovata nel database.", " "

    # Estrai solo i nodi (senza score) per passarli al reranker
    nodes = reranka([node.node for node in nodes], query_text, top_n=5)
    
    context_parts = []
    
    docs_dict = {}
    
    for node_ws in nodes:
        filename = node_ws.node.metadata.get('origin_filename', 'Unknown File')
        page = node_ws.node.metadata.get('pages', 'N/A')
        content = node_ws.node.get_content().strip()
        context_parts.append(f"--- Documento: {filename} (Pag. {page}) ---\n{content}\n")
        
        if filename not in docs_dict:
            docs_dict[filename] = []
        
        docs_dict[filename].extend(page)
    
    
    context_parz_list = []
    for filename, pages in docs_dict.items():
        
        unique_pages = sorted(set(pages))
        
        clean_pages = [str(p).strip('[]') for p in unique_pages]
        pages_str = ", ".join(clean_pages)
        context_parz_list.append(f"--- Documento: {filename} (Pag. {pages_str}) ---")
    
    full_context = "\n".join(context_parts)
    context_parz = "\n".join(context_parz_list)
    #print(full_context)
    return full_context, context_parz