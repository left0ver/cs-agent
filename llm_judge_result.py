import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from utils import encode_image, get_image_name

load_dotenv()

IMAGE_ROOT_DIR = os.getenv("IMAGE_ROOT_DIR", "data/KownledgeBase/手册/插图")

judge_result_prompt_template = """你是一个客服专家，你需要根据用户的问题对这段客服回答进行打分。
# 打分规则
1 分，质量差：回答未回应问题，结构混乱或缺失，图片无关或无帮助。
2 分，质量一般：回答部分回应问题，但不完整；结构较弱，图文结合较差或仅部分有帮助。
3 分，质量中等：回答回应了问题，但缺乏深度；结构清晰但可优化；图片有一定帮助，但未充分提升理解。
4 分，质量良好：回答清晰、较为全面；结构逻辑清晰、组织合理，图片有助于理解文本。
5 分，质量优秀：回答详细、有深度；结构严谨连贯，图片与文本完美互补，显著提升理解效果。

# TIPS
图中的<PIC>表示这个位置该放置图片，图片放置的顺序和输入的图片顺序一致

# 输出格式
你需要以json的格式输出大纲，json的结构如下：
{   
    "reason": str,  # 打分原因
    "score": float  # 对于这个回答的质量打分,取值范围为0-5的浮点数
}

# 用户问题
{{query}}
# 客服回答
{{answer}}
"""


async def judge_result_by_llm(query: str, ret: str) -> tuple[str, float]:
    """使用LLM对生成的答案进行打分，用来找出哪些答案分比较低,有问题,再找出原因进行优化"""
    data = ret.split(",[")
    description = data[0]
    if len(data) > 1:
        image_list = eval(f"[{data[1]}")
    else:
        image_list = []

    image_list = [
        get_image_name(os.path.join(IMAGE_ROOT_DIR, image)) for image in image_list
    ]
    image_messages = []
    # llm = ChatOpenAI(
    #     # model="gemini-3.1-pro-preview",
    #     model="gemini-3-flash-preview",
    #     base_url=os.getenv("GEMINI_BASE_URL"),
    #     api_key=os.getenv("GEMINI_API_KEY"),
    # )
    llm = ChatOpenAI(
        model="gpt-5.4",
        base_url=os.getenv("OPEANAI_BASE_URL"),
        api_key=os.getenv("OPEANAI_API_KEY"),
    )
    for image in image_list:
        if not Path(image).exists():
            raise FileNotFoundError(f"图片路径不存在：{image}")
        image_base64, mime_type = encode_image(image)
        image_messages.append(
            {
                "type": "image",
                "base64": image_base64,
                "mime_type": mime_type,
            },
        )

    messages = [
        {
            "role": "human",
            "content": [
                *image_messages,
                {"type": "text", "text": judge_result_prompt_template},
            ],
        }
    ]
    prompt = ChatPromptTemplate.from_messages(messages, template_format="mustache")
    chain = prompt | llm | JsonOutputParser()
    res = await chain.with_retry().ainvoke({"query": query, "answer": description})
    reason, score = res["reason"], res["score"]
    return reason, score


async def judge_all_result(
    submit_csv_path: str, question_csv_path: str = "data/question_public.csv"
):
    """对所有生成的答案进行打分，用来找出哪些答案分比较低,有问题,再找出原因进行优化"""
    answer_df = pd.read_csv(submit_csv_path, index_col="id")
    question_df = pd.read_csv(question_csv_path, index_col="id")
    judge_save_file = Path(Path(submit_csv_path).stem + "_judge_result.json")
    results = []
    exist_last_id = -1
    placeholder_answer = "您好，您的问题已收到，请您耐心等待处理结果，谢谢。"
    max_concurrency = 8
    tasks = []
    batch_ids = []
    end_id = 436

    if judge_save_file.exists():
        results = json.load(open(judge_save_file, "r"))
        exist_last_id = max([result["id"] for result in results])

    for answer_row, question_row in tqdm(
        zip(answer_df.iterrows(), question_df.iterrows()), total=len(answer_df)
    ):
        assert answer_row[0] == answer_row[0]
        id = answer_row[0]
        ret = answer_row[1]["ret"]
        query = question_row[1]["question"]
        if id <= exist_last_id:
            continue

        if ret == placeholder_answer:
            continue

        tasks.append(asyncio.create_task(judge_result_by_llm(query, ret)))
        batch_ids.append(id)
        max_concurrency -= 1

        if max_concurrency == 0 or id == end_id:
            batch_results = await asyncio.gather(*tasks)
            for id, (reason, score) in zip(batch_ids, batch_results):
                results.append(
                    {
                        "id": id,
                        "reason": reason,
                        "score": score,
                    }
                )
            tasks = []
            batch_ids = []
            max_concurrency = 8

            with open(judge_save_file, "w") as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    # 修改文件名
    asyncio.run(
        judge_all_result(
            "submit_gpt_5_5_product_top_k_10.csv", "data/question_public.csv"
        )
    )
