
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage, MessageRole
from dotenv import load_dotenv
import os

load_dotenv()

VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL")

def rispostarag(messaggio_attuale, gruppo, history):
    
    CHAT_RECENTI_TENUTE = 2

    
    if len(history) > CHAT_RECENTI_TENUTE:
        history = history[-CHAT_RECENTI_TENUTE:]
    
    
    llm = OpenAILike(
        model='meta-llama/Llama-3.1-8B-Instruct',
        api_base=VLLM_API_BASE_URL, 
        api_key="null",
        is_chat_model=True,
        is_function_calling_model=True,    
        timeout=60.0,
        streaming=True,
        context_window=4096,
    )

    messages = [
        ChatMessage(role=msg["role"], content=msg["content"]) 
        for msg in history
    ]

    messages.append(ChatMessage(role=MessageRole.USER, content=messaggio_attuale))

    
    
    #TENERE STREAM_CHAT
    response_stream = llm.stream_chat(messages)
    
    
    
    for chunk in response_stream:
        yield chunk.delta

    
    yield f"\n\n (Risposta generata per il gruppo: {gruppo})"

    
    print("="*30)
    print(f"History length: {len(history)}")


