import os
import qdrant_client
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.llms.google_genai import GoogleGenAI
from logicarag.rerank import reranka
from llama_index.core.retrievers import QueryFusionRetriever
from logicarag.HYDE import generate_hypothetical_doc
from logicarag.queryaugmentation import generate_augmentation
from llama_index.core import QueryBundle
from logicarag.embeddings import get_context_from_knowledge_base
from logicarag.overlap import get_context_overlap
from logicarag.hierarchicalembeddings import get_hierarchical_context
from logicarag.esperimenti.agenttree import agent_tree




load_dotenv()

VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL")


# funzione di prima
def rispostarag(messaggio_attuale, gruppo, history):
    
    CHAT_RECENTI_TENUTE = 6 #2 è una conversazione domanda risposta

    #hydeaugmented = [generate_hypothetical_doc(messaggio_attuale)]    
    #queryaugmented = generate_augmentation(messaggio_attuale)
    #context_str, context_parz = get_context_from_knowledge_base(messaggio_attuale,gruppo)
    #context_str = get_hierarchical_context(messaggio_attuale, gruppo)#, hydeaugmented=hydeaugmented)
    #context_str = windowretrieve(messaggio_attuale, gruppo)#, augmented=hydeaugmented)
    context_str, context_parz = get_context_overlap(messaggio_attuale, gruppo)#, hydeaugmented=hydeaugmented)
    # context_str, context_parz = agent_tree(messaggio_attuale,gruppo)
    system_prompt_content = (
        "Sei un assistente tecnico esperto del sistema WMS WAMAS. "
        "Il tuo obiettivo è fornire risposte precise, professionali e basate esclusivamente sui dati forniti.\n\n"
        
        "### REGOLE DI COMPORTAMENTO:\n"
        "1. Usa SOLTANTO il contesto fornito nei tag <context> per rispondere.\n"
        "2. Se le informazioni necessarie per rispondere non sono presenti nel contesto, rispondi testualmente: "
        "'Mi dispiace, ma non ho informazioni sufficienti nel manuale WAMAS per rispondere a questa domanda.'\n"
        "3. Non utilizzare conoscenze esterne al di fuori del contesto fornito.\n"
        "4. Mantieni un tono tecnico, asciutto e professionale.\n"
        "5. Se possibile, cita la sezione o il capitolo specifico se presente nel testo, cita anche il nome del documento.\n\n"
        
        f"<context>\n{context_str}\n</context>\n\n"
        
        "### ISTRUZIONE FINALE:\n"
        "Rispondi alla domanda dell'utente in modo strutturato (usa elenchi puntati se la procedura è complessa)."
    )

    #prompt per farmi dire in quale/quali documenti si trova la risposta
    # system_prompt_content = (
    #     "Sei un assistente che dice in quali documenti si trova la risposta alla domanda dell'utente dato il contesto fornito, non rispondere direttamente alla domanda, ma indica solo i documenti.\n"
    #     f"### CONTESTO RECUPERATO:\n{context_str}\n\n"
    #     "### FINE CONTESTO"
    # )

    if CHAT_RECENTI_TENUTE > 0:    
        if len(history) > CHAT_RECENTI_TENUTE:
            history = history[-CHAT_RECENTI_TENUTE:]
    else:
        history = []

    
    
    llm = OpenAILike(
        model='Qwen/Qwen3-32B-AWQ',
        api_base=VLLM_API_BASE_URL,
        api_key="null",
        is_chat_model=True,
        is_function_calling_model=True,
        timeout=60.0,
        streaming=True,
        context_window=8192, # CHECK
        temperature=0,
        #max_tokens=1024, #CHECK totale
        model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    )
    # llm = GoogleGenAI(
    #     model="models/gemini-2.5-pro",
    #     api_key=os.getenv("GOOGLE_API_KEY"),
    #     temperature=0.8,        
    #     max_tokens=8192,
    #     streaming=True
    # )
    
    messages = []
    
    
    messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt_content))

    
    for msg in history:
        messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

    
    messages.append(ChatMessage(role=MessageRole.USER, content=messaggio_attuale))
    
    
    
    response_stream = llm.stream_chat(messages)
    
    full_response = ""
    for chunk in response_stream:
        delta = chunk.delta
        full_response += delta
        yield delta

    yield f"\n\n *DEBUG*\n(Risposta generata per il gruppo: {gruppo})"

    context_str = context_str.replace("#", "")

    yield "\nDOCUMENTI USATI:\n" + context_str + "\n"
    
    print("="*30)
    print(f"History length usata: {len(history)}")


# test
if __name__ == "__main__":
    
    domanda_test = "come faccio per arrestare una baia di picking?"
    history_finta = [] 

    print(f"--- Test User Question: {domanda_test} ---")
    
    generatore = rispostarag(domanda_test, "admin", history_finta)
    
    print("\n--- Risposta LLM ---")
    for partial in generatore:
        print(partial, end="", flush=True)
    print("\n")