import os
import qdrant_client
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.retrievers import QueryFusionRetriever, AutoMergingRetriever
from llama_index.core import QueryBundle
from logicarag.rerank import reranka



VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL")
QDRANT_URL = os.getenv("QDRANT_URL")
collectionname = "WAMASHIERARCHICAL"

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


try:
    storage_context = StorageContext.from_defaults(persist_dir="./storage_hierarchical")
    print("Docstore caricato da disco locale.")
except Exception as e:

    print(f"Attenzione: Impossibile caricare StorageContext locale: {e}")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)


index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)



def get_hierarchical_context(query_text, user_role, hydeaugmented=None, queryaugmentation=None):

    print(f"Ricerca Gerarchica in corso per: '{query_text}' con ruolo: '{user_role}'")

    
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="groups",  
                value=user_role,    
                operator=FilterOperator.EQ 
            )
        ]
    )


    leaf_topk = 43 
    

    retriever_dense = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT, 
        similarity_top_k=leaf_topk,
        filters=filters if user_role != "admin" else None
    )
    
    retriever_sparse = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.SPARSE,
        sparse_top_k=leaf_topk,
        filters=filters if user_role != "admin" else None
    )


    base_retriever = QueryFusionRetriever(
        retrievers=[retriever_dense, retriever_sparse],
        similarity_top_k=2*leaf_topk, 
        num_queries=1,           
        mode="simple",                                      
        use_async=False
    )

    automerging_retriever = AutoMergingRetriever(
        base_retriever, 
        storage_context=storage_context, 
        verbose=True,
    )

    query_bundle = QueryBundle(
        query_str=query_text,
        custom_embedding_strs=hydeaugmented if hydeaugmented else (queryaugmentation if queryaugmentation else None)
    )

    
    merged_nodes = automerging_retriever.retrieve(query_bundle)

    merged_nodes = reranka(merged_nodes, query_text, top_n=3)

    
    # print(merged_nodes[0])
    merged_content = "\n\n".join(["-"*15 + "\n" + node.node.metadata["origin_filename"] + "\n" + node.node.get_content()  for node in merged_nodes]) #stringa oscena


    return merged_content