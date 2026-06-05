from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from data_process import build_chroma_vectorstore


if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    build_chroma_vectorstore(
        jsonl_path="data/train.jsonl",
        embeddings=embeddings,
        persist_directory="./chroma_db",
        collection_name="recipes",
    )

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="recipes",
    )

    queries = [
        "海鲜烩饭需要哪些材料？",
        "怎么做芦笋柠檬烩饭？",
        "有没有适合烧烤的猪肉串食谱？",
        "怎么做法式圣诞饼干？",
        "鸭胸肉用平底锅怎么煎？",
        "我想找一个含有布拉塔和烟熏鳟鱼的华夫饼食谱",
    ]

    for query in queries:
        print("=" * 80)
        print("QUERY:", query)
        results = vectorstore.similarity_search_with_score(query, k=3)

        for doc, score in results:
            print(f"SCORE: {score:.4f}")
            print("CONTENT:", doc.page_content)
            print("METADATA:", doc.metadata)
            print("-" * 80)
