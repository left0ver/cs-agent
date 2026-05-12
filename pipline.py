import logging
from typing import Literal

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from answer_general_query import answer_general_query
from answer_product_query import answer_product_query
from query_classification import ensembles_query_classification
from utils import language_detect

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_answer_language(
    answer: str, image_names: list[str], query_language: Literal["chinese", "english"]
) -> tuple[str, list[str]]:
    """
    确保answer的语言和query_language的语言一致
    """
    answer_language = language_detect(answer)
    if answer_language != query_language:
        raise Exception(f"模型回答的的语言不是{query_language}")

    return answer, image_names


def llm_can_answer_the_question(answer: str) -> bool:
    """
    检查LLM是否能够回答用户的问题
    """
    if "我不能回答这个问题" in answer or "I cannot answer the question" in answer:
        return False
    return True


async def wrap_ensembles_query_classification(x: dict):
    return await ensembles_query_classification(x["query"])


def router_by_query_cls(x: dict) -> str:
    """
    根据问题分类的结果进行路由, 同时问题通过answer_general_query, 产品问题则使用answer_product_query
    """
    query_cls = x["query_cls"]
    query = x["query"]
    question_type = query_cls["question_type"]
    top_k = x["top_k"]
    use_source = x["use_source"]
    if query_cls["source"] is None and question_type == "general":
        queries = query.split(",\n")
        return answer_general_query(queries, "1")
    else:
        return answer_product_query(
            query, "1", query_cls, top_k=top_k, use_source=use_source
        )


async def pipeline(query: str) -> str | None:
    top_k = 10
    pipeline_chain = RunnablePassthrough.assign(
        query_cls=RunnableLambda(wrap_ensembles_query_classification)
    ) | RunnableLambda(router_by_query_cls)

    answer = await pipeline_chain.with_retry().ainvoke(
        {"query": query.strip('"'), "top_k": top_k, "use_source": True}
    )
    return answer


if __name__ == "__main__":
    import asyncio

    async def main():

        # query = "使用和操作VR头显时应采取哪些安全预防措施，以确保用户安全和设备使用寿命？"
        query = "我收到的商品和图片不一样，颜色偏差很大，我要投诉！"
        answer = await pipeline(query)
        print(answer)
    asyncio.run(main())
