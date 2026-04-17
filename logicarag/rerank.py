import requests
from dotenv import load_dotenv
import os
from typing import List
from llama_index.core.schema import TextNode, NodeWithScore


load_dotenv()
url = os.getenv("RERANKER_API_URL")

def reranka(nodes: List[TextNode], query_text: str, top_n: int = 5) -> List[NodeWithScore]:
    
    if not nodes:
        return []

    top_n = min(top_n, len(nodes))
    
    document_texts = [node.get_content() for node in nodes]

    payload = {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": f"{query_text}",
        "documents": document_texts,
        "top_n": top_n
    }
    headers = {"Content-Type": "application/json"}
    
    results_with_scores = []

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        results_json = response.json()
           
        for result_item in results_json["results"]:
            original_index = result_item["index"]
            relevance_score = result_item["relevance_score"]
            
            original_node = nodes[original_index]

            node_with_score = NodeWithScore(node=original_node, score=relevance_score)
            
            results_with_scores.append(node_with_score)

    except requests.exceptions.RequestException as e:
        print(f"Errore durante la chiamata API: {e}")
        return []
    except Exception as e:
        print(f"Errore imprevisto durante il processing: {e}")
        return []

    return results_with_scores


if __name__ == "__main__":
    lista_nodi_originali = [
        TextNode(
            text="""per cambiare la lampadina al camion bisogna seguire i seguenti passaggi: ...""",
            metadata={"source": "manuale_camion.pdf"}
        ),
        TextNode(
            text="""3.4. Incongruenza logica tra WAMAS e Lighthouse\nPuò  verificarsi  la  situazione  in  cui  su  WAMAS  vedo  un  UDC  in  una  certa  locazione  che  però  su Lighthouse non viene visto. Questo causa un disallinenamento dei sistemi a gestire.\nPer ovviare a questo problema bisogna selezionare l'opzione 'Forza' e poi refresh e reset per forza il sistema a rilevare l'UDC e sistemare la questione.""",
            metadata={"source": "guida_wamas.pdf"}
        )
    ]
    
    query = "ho un disallinenamento fisico logico di un pallet, come posso risolvere?"

    risultati_finali = reranka(lista_nodi_originali, query)

    print("--- Risultati Finali (oggetti NodeWithScore) ---")
    if risultati_finali:
        for result in risultati_finali:
            print(f"Score: {result.score:.6f}")
            print(f"Node Text: \"{result.node.get_content()[:80]}...\"")
            print(f"Node Metadata: {result.node.metadata}")
            print("-" * 20)
    else:
        print("Nessun risultato ottenuto.")