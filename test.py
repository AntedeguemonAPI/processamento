from keybert import KeyBERT
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

texto = "O serviço de internet está com problemas de conexão e lentidão."
keywords = kw_model.extract_keywords(
    texto,
    keyphrase_ngram_range=(1, 2),
    stop_words=None,  # <- Teste com isso
    use_mmr=False,
    top_n=5
)
print(keywords)