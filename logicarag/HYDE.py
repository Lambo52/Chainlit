import os
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import PromptTemplate
from pydantic import BaseModel, Field
 
load_dotenv()

url_llm = os.getenv("VLLM_API_BASE_URL")

llm = OpenAILike(
    model='Qwen/Qwen2.5-32B-Instruct-AWQ',
    api_base=url_llm,
    api_key="null",
    is_chat_model=True,
    is_function_calling_model=True,
    timeout=60.0,
    temperature=0.1,
    context_window=8192
)


class DocumentoIpotetico(BaseModel):
    contenuto_tecnico: str = Field(
        description="Un breve paragrafo tecnico che descrive la soluzione procedurale al problema."
    )

#Scrivi un paragrafo tecnico tratto da un manuale di gestione di magazzino (WMS) o procedure ERP che spieghi come risolvere la seguente discrepanza...
HYDE_TEMPLATE_STR = (
    "Sei un tecnico esperto del sistema WAMAS.\n"
    "Genera una soluzione procedurale ipotetica per il seguente problema utente.\n"
    "Scrivi un paragrafo tecnico tratto da un manuale di gestione di magazzino (WMS) o procedure ERP che spieghi come risolvere il problema"
    "Usa terminologia specifica di magazzino, nomi di maschere o processi standard.\n\n"
    "Domanda utente: {query}"
)
hyde_prompt = PromptTemplate(HYDE_TEMPLATE_STR)

def generate_hypothetical_doc(query_text):
    print(f"\n--- Generazione Documento Ipotetico Strutturato per: '{query_text}' ---")
    

    try:
        risultato = llm.structured_predict(
            DocumentoIpotetico,
            prompt=hyde_prompt,
            query=query_text
        )
        
        doc_text = risultato.contenuto_tecnico
        
        print(f"\n[CONTENUTO GENERATO]:\n{doc_text}")
        
        
        return doc_text

    except Exception as e:
        print(f"Errore nella generazione strutturata: {e}")
        
        return None

#test inutile
if __name__ == "__main__":
    domanda = "come gestisco i vuoti in baia nel qweekend?"
    generate_hypothetical_doc(domanda)