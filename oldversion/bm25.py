from fastembed import SparseTextEmbedding

documents = [
    """3. Come si arresta una baia di picking
Prima di eseguire un'operazione sulla baia di picking , bisogna assicurarsi che questa sia in stato di ARRESTO. Se così non fosse, non sarebbe possibile eseguire nessuna attività sulla suddetta baia.
Per arrestare una baia, ci si connette alla pagina di WAMAS WH083 e si preme 'cerca' senza inserire nessun parametro di ricerca per mostrare l'elenco completo delle baie.
<tagimmagine> This is a screenshot of a webpage. <finetagimmagine>
Premere tasto destro sulla stazione desiderata ,  poi 'Attiva processo' per mostrare una finestra di WH086 dove la prima opzione dell'elenco è 'Arresta'. Cliccarci sopra per mandare in arresto la baia.""",
    """1. Introduzione procedura per l'arresto dalle baie di picking
La procedura per l'arresto delle baie di picking ha lo scopo di permettere il regolare svolgimento delle attività che andranno svolte direttamente sulla stazione stessa. Infatti, molteplici sono le tasks che possono essere svolte solo quando la baia è in stato di arresto. Gli esempi più comuni sono il cambio pinza, la creazione, modifica o eliminazione di un terminale dalla baia, ecc…
In questa procedura verrà spiegato anche come accelerare l'arresto di una baia in caso un'urgenza.""",
]

model = SparseTextEmbedding(model_name="Qdrant/bm25")
embeddings = list(model.embed(documents))

print(embeddings)
