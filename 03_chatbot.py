# Busca semântica
docs = vectorstore.similarity_search(
    "Como funciona compensação de cheques?",
    k=3
)

for d in docs:
    print(d.page_content)