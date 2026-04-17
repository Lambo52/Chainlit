import os
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

url_llm = os.getenv("VLLM_API_BASE_URL")

llm = OpenAILike(
    model="Qwen/Qwen2.5-32B-Instruct-AWQ",
    api_base=url_llm,
    api_key="null",
    is_chat_model=True,
    is_function_calling_model=True,
    timeout=60.0,
    temperature=0.1,
    context_window=8192
)


class QueryAugmentation(BaseModel):
    questions: list[str] = Field(
        description=(
            "Lista di query alternative semanticamente rilevanti rispetto alla domanda originale. "
            "Le query devono essere pensate per il retrieval di documentazione tecnica WAMAS. "
            "Massimo 5 elementi."
        )
    )


QUERY_AUGMENTATION_TEMPLATE = """
Sei un esperto tecnico del sistema WAMAS e un agente specializzato in Retrieval-Augmented Generation (RAG).

Il tuo compito è generare query alternative da usare ESCLUSIVAMENTE per il recupero di documentazione tecnica.
NON devi rispondere alla domanda dell’utente.

Linee guida fondamentali:
- Mantieni lo stesso intento informativo della domanda originale
- Usa terminologia tecnica coerente con WAMAS
- Esplora angolazioni diverse utili al retrieval, come:
  - procedure operative
  - configurazioni di sistema
  - gestione errori o casi limite
  - moduli o funzionalità coinvolte
- Non introdurre concetti, moduli o funzionalità non impliciti nella domanda
- Evita parafrasi banali o troppo simili tra loro
- Genera al massimo 5 query (meno se non necessario)

Domanda originale:
{query}

Fornisci SOLO le query alternative nel formato strutturato richiesto.
"""

query_augmentation_prompt = PromptTemplate(QUERY_AUGMENTATION_TEMPLATE)


def generate_augmentation(query_text: str) -> list[str] | None:
    try:
        result = llm.structured_predict(
            QueryAugmentation,
            prompt=query_augmentation_prompt,
            query=query_text
        )
        return result.questions

    except Exception as e:
        print(f"Errore nella query augmentation: {e}")
        return None


# Test
if __name__ == "__main__":
    domanda = "come gestisco i vuoti in baia nel qweekend?"
    domande = generate_augmentation(domanda)
    print(domande)
