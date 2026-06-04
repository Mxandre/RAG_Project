from data_process import  jsonl_to_documents, build_chroma_vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(
        model = "BAAI/bge-m3"
    )   
    vectorstore = build_chroma_vectorstore(
        jsonl_path = "data/train.jsonl",
        embeddings = embeddings,
        persist_directory="./chroma_db",
        collection_name = "recipes"
    ) 
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name = "recipes"
    )

    queries =  [
        "risotto aux fruits de mer",
        "quels ingrédients pour un risotto aux champignons ?",
        "comment préparer un risotto aux asperges et citron ?",
    ]
    for query in queries:
        print("="*80)
        print("QUERY:", query)
        results = vectorstore.similarity_search_with_score(query, k=3)

        for doc, score in results:
            print(f"SCORE: {score:.4f}")
            print("CONTENT:", doc.page_content)
            print("METADATA:", doc.metadata)
            print("-"*80)