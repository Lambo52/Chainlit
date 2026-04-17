import numpy as np
from typing import List

# Assumiamo che 'nodi' sia una lista di oggetti con un attributo '.score'
# Esempio: NodeWithScore di LlamaIndex

def trova_punto_di_rottura(nodi: List, max_nodi: int = 10) -> List:
    """
    Trova il punto in cui il "gap" di score tra nodi consecutivi è massimo.
    Restituisce i nodi fino a quel punto, senza mai superare max_nodi.
    """
    # Gestisce i casi in cui non ha senso calcolare i gap
    if len(nodi) <= 2:
        return nodi[:max_nodi]

    gaps = [nodi[i].score - nodi[i+1].score for i in range(len(nodi) - 1)]

    # Se per qualche motivo non ci sono gap, restituisce la lista limitata
    if not gaps:
        return nodi[:max_nodi]

    # Il numero di nodi da tenere è l'indice del gap più grande + 1
    nodi_da_tenere = gaps.index(max(gaps)) + 1
    
    # Applica il limite massimo imposto
    numero_nodi_finale = min(nodi_da_tenere, max_nodi)

    return nodi[:numero_nodi_finale]


def filtra_con_deviazione_standard(
    nodi: List, 
    fattore_deviazione: float = 2.0, 
    max_nodi: int = 5
) -> List:
    """
    Filtra i nodi mantenendo quelli fino a un calo di score statisticamente significativo.
    Il numero di nodi restituiti non supererà mai max_nodi.
    """
    # Gestisce i casi troppo piccoli per un'analisi statistica
    if len(nodi) <= 2:
        return nodi[:max_nodi]

    gaps = [nodi[i].score - nodi[i+1].score for i in range(len(nodi) - 1)]
    
    if not gaps:
        return nodi[:max_nodi]

    media_gap = np.mean(gaps)
    std_dev_gap = np.std(gaps)
    
    # Se la deviazione è quasi zero (i cali sono uniformi), usa il metodo più semplice
    if std_dev_gap < 1e-4:
        # --- BUG CORRETTO ---
        # Passiamo correttamente il parametro max_nodi alla funzione di fallback
        return trova_punto_di_rottura(nodi, max_nodi=max_nodi)

    soglia_drastica = media_gap + fattore_deviazione * std_dev_gap
    
    # Di default, si tengono tutti i nodi (se non si trova un calo drastico)
    nodi_da_tenere = len(nodi) 
    for i, gap in enumerate(gaps):
        if gap > soglia_drastica:
            nodi_da_tenere = i + 1
            break

    # --- BUG CORRETTO ---
    # Applica il limite massimo confrontando il risultato con max_nodi
    numero_nodi_finale = min(nodi_da_tenere, max_nodi)
            
    return nodi[:numero_nodi_finale]