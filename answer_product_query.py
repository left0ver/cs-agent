import logging
import os
import re
from typing import Literal, cast

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from retriever import retriever
from utils import language_detect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


def parse_answer(answer: str) -> tuple[str, list[str]]:
    """
    解析LLM生成的内容,提取图片名称和回答内容
    """
    match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
    if match:
        content = match.group(1).strip()
        soup = BeautifulSoup(content, "html.parser")
        pics = soup.find_all("pic")
        image_names = cast(list[str], [pic.get("image_name") for pic in pics])

        for pic in pics:
            content = content.replace(str(pic), "<PIC>")

        pic_count = content.count("<PIC>")
        if pic_count != len(image_names):
            raise Exception("替换之后的PIC的数量和解析出来的图片数量不一致")
        return content, image_names
    else:
        raise Exception("LLM生成的格式有问题")


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
    检查LLM是否能够回答用户的问题,如果不能回答，则可能是source的预测有问题，之后重新不使用source进行检索生成
    """
    if "我不能回答这个问题" in answer or "I cannot answer the question" in answer:
        return False
    return True

# TODO: 多理多轮对话的情况,但是目前初赛中product的题目没有多轮对话的情况
def answer_product_query(
    query: str,
    thread_id: str,
    query_cls: dict,
    top_k: int = 10,
    use_source: bool = True,
):
    llm = ChatOpenAI(
        model="gpt-5.5",
        base_url=os.getenv("OPEANAI_BASE_URL"),
        api_key=os.getenv("OPEANAI_API_KEY"),
    )
    generate_answer_prompt_template = """你是一个智能客服，你需要根据用户的问题，以及知识库中检索出的相关上下文来回答用户的问题，上下文内容中<picture-description></picture-description>中的内容是插图的内容，属性image_name是对应的图片名称。
# 任务
你需要以图文互补的形式来回答用户的问题，需要插入图片的地方你需要使用PIC标签来表示图片，image_name的属性是图片名。例如：<pic image_name="Manual16_01"></pic>

# 要求
1. 上下文可能会存在冗余，你的回答必须要简洁，针对用户的问题来进行回答，**不需要回答与用户问题无关的内容，也不需要对用户提问**
2. 你的回答不必过于冗余，针对用户问题回答即可，结构严谨连贯，图片与文本完美互补，帮助用户更好地理解答案
3. 图片必须和答案相关，能够解决用户的问题，并且有助于用户更好地理解答案
4. 用户的问题是什么语言，你的答案也必须是什么语言
5. 如果所给的上下文不能回答用户的问题，你需要回答“我不能回答这个问题”,如果用户的语言是英文，则回答“I cannot answer the question”
6. 如果所给的上下文中有能够直接回答用户问题的内容，**优先采用上下文中的原本的内容来回答**

# 输出格式
你必须使用answer标签来包裹你的答案，如下:
<answer>
</answer>
# 用户的问题为
{{query}}

# 相关的上下文为
<context>
{{context}}
</context>
"""

    generate_answer_prompt = PromptTemplate.from_template(
        generate_answer_prompt_template, template_format="mustache"
    )

    product_answer_chain = (
        RunnablePassthrough.assign(
            context=lambda x: "\n\n".join(
                [
                    result.page_content
                    for result in retriever(
                        x["query"], x["query_cls"], x["top_k"], x["use_source"]
                    )
                ]
            )
        )
        | RunnablePassthrough.assign(
            parsed_answer=(
                generate_answer_prompt
                | llm
                | StrOutputParser()
                | RunnableLambda(parse_answer)
            )
        )
        | RunnableLambda(
            lambda x: ensure_answer_language(
                x["parsed_answer"][0],
                x["parsed_answer"][1],
                x["query_cls"]["language"],
            )
        )
    )
    # 第一次回答
    answer, image_names = product_answer_chain.invoke(
        {
            "query": query.strip('"'),
            "query_cls": query_cls,
            "top_k": top_k,
            "use_source": use_source,
        }
    )

    if not llm_can_answer_the_question(answer):
        logger.info(f"first answer: {query} -> {answer}")
        # 第二次回答
        answer, image_names = product_answer_chain.with_retry().invoke(
            {
                "query": query.strip('"'),
                "query_cls": query_cls,
                "top_k": top_k,
                "use_source": False,
            }
        )

    ret = answer
    if image_names and len(image_names) > 0:
        ret += "," + str(image_names)
    return ret
