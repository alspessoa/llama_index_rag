from rag import RAG


if __name__ == "__main__":
    query = 'Quais os principais desafios?'
    comment = 'descreva em bullet points'
    raw_documents_path = '/home/andre/llama_index_rag/data'

    try:
        rag = RAG(raw_documents_path=raw_documents_path)
        answer = rag.get_rag_answer(query=query, comment=comment)

        print(answer)

    except Exception as e:
        print(f"Ocorreu um erro: {e}")
