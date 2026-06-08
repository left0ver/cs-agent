import os

import tiktoken
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_openai import OpenAIEmbeddings

from config import get_config

load_dotenv()


embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    base_url=os.getenv("EMBEDDING_BASE_URL"),
    api_key=os.getenv("EMBEDDING_API_KEY"),
)
DEFAULT_MILVUS_CONNECTION = {
    "host": os.getenv("MILVUS_HOST", "127.0.0.1"),
    "port": os.getenv("MILVUS_PORT", "19530"),
    "db_name": os.getenv("MILVUS_DB_NAME", "default"),
}
multi_analyzer_params = {
    "analyzers": {
        "english": {"type": "english"},
        "chinese": {"type": "chinese"},
        "default": {"tokenizer": "icu"},
    },
    "by_field": "language",
}

# dense_store = Milvus(
#     embedding_function=embedding_model,
#     # collection_name=os.getenv("MILVUS_COLLECTION_NAME", "handbook_knowledge_bank"),
#     collection_name=get_config()["MILVUS_COLLECTION_NAME"],
#     text_field="text",
#     vector_field="dense",
#     auto_id=True,
#     drop_old=False,
#     enable_dynamic_field=True,
#     connection_args=DEFAULT_MILVUS_CONNECTION,
#     index_params=[
#         {
#             "index_type": "HNSW",
#             "metric_type": "COSINE",
#         },
#     ],
# )


# sparse_store = Milvus(
#     embedding_function=None,
#     # collection_name=os.getenv("MILVUS_COLLECTION_NAME", "handbook_knowledge_bank"),
#     collection_name=get_config()["MILVUS_COLLECTION_NAME"],
#     text_field="text",
#     vector_field="sparse",
#     auto_id=True,
#     drop_old=False,
#     enable_dynamic_field=True,
#     connection_args=DEFAULT_MILVUS_CONNECTION,
#     index_params=[
#         {
#             "index_type": "AUTOINDEX",
#             "metric_type": "BM25",
#             "params": {},
#         },
#     ],
#     builtin_function=BM25BuiltInFunction(
#         input_field_names="text",
#         output_field_names="sparse",
#         multi_analyzer_params=multi_analyzer_params,
#         # analyzer_params={"type": "chinese"},
#         # enable_match=True,
#     ),
# )


async def retriever(
    query: str,
    query_classification: dict[str, str],
    top_k: int = 10,
    use_source: bool = True,
    top_token: int = 2000,  # <=0 means dont use top token
    min_top_k: int = 5,
    max_top_k: int = 30,
) -> list[Document]:
    """根据query检索出相关的上下文"""
    milvus = Milvus(
        embedding_function=embedding_model,
        # collection_name=os.getenv("MILVUS_COLLECTION_NAME", "handbook_knowledge_bank"),
        collection_name=get_config()["MILVUS_COLLECTION_NAME"],
        connection_args=DEFAULT_MILVUS_CONNECTION,
        auto_id=True,
        drop_old=False,
        enable_dynamic_field=False,
        vector_field=["dense", "sparse"],
        index_params=[
            {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {
                    "M": 16,
                    "efConstruction": 64,
                },
            },
            {
                "index_type": "AUTOINDEX",
                "metric_type": "BM25",
                "params": {},
            },
        ],
        builtin_function=BM25BuiltInFunction(
            input_field_names="text",
            output_field_names="sparse",
            multi_analyzer_params=multi_analyzer_params,
            # analyzer_params={"type": "chinese"},
            # enable_match=True,
        ),
    )

    # TODO: 根据分词长度来动态地选择最终的topk，而不是固定的top_k=10
    top_k = top_k if top_token <= 0 else max_top_k
    use_query_cls = get_config()["USE_QUERY_CLS"]
    language, llm_predict_source = (
        query_classification["language"],
        query_classification["source"],
    )
    expr = f"language == '{language}'"
    if llm_predict_source is not None and use_source and use_query_cls:
        expr = f"source == '{llm_predict_source}' AND language == '{language}'"
    # # 不使用查询分类
    # if not use_query_cls:
    #     expr = f"language == '{language}'"
    # dense_result = dense_store.similarity_search(
    #     query,
    #     k=top_k,
    #     fetch_k=top_k,
    #     expr=expr,
    # )
    # sparse_result = sparse_store.similarity_search(
    #     query,
    #     k=top_k,
    #     fetch_k=top_k,
    #     expr=expr,
    #     # expr=f"language == '{language}'",
    #     param={
    #         "metric_type": "BM25",
    #         "analyzer_name": language,
    #         "params": {},
    #     },
    # )
    # if len(sparse_result) == 0:
    #     print(
    #         f"query: {query}, language: {language}, source: {llm_predict_source},该source的sparse_result为空"
    #     )

    # 根据source和language进行文档的筛选，再混合检索
    results = await milvus.asimilarity_search(
        query,
        k=top_k,
        fetch_k=top_k,
        expr=expr,
        param=[
            {
                "metric_type": "COSINE",
            },
            {
                "metric_type": "BM25",
                "analyzer_name": language,
                "params": {},
            },
        ],
        ranker_type="rrf",
    )
    if top_token > 0:
        # gpt-5的分词
        enc = tiktoken.get_encoding("o200k_base")
        token_nums = [len(enc.encode(result.page_content)) for result in results]
        # 前x个文档的token数不超过top_token
        x = 0
        while x < len(results) and sum(token_nums[:x]) <= top_token:
            x += 1
        x = max(x, min_top_k)
        results = results[:x]

    if llm_predict_source is not None and use_source and use_query_cls:
        # 根据index字段，对results进行排序，确保每个chunk的相对顺序和原文的顺序一致
        sorted_results = sorted(
            results,
            key=lambda x: x.metadata["index"],
        )
        return sorted_results
    return results


if __name__ == "__main__":
    import asyncio

    asyncio.run(
        retriever(
            "What should I pay attention to in order to ensure my safety when using this fax?",
            {"language": "english", "source": "Multi-Function Printer User Manual.txt"},
            top_k=10,
            top_token=-1,
            min_top_k=5,
            max_top_k=20,
        )
    )
