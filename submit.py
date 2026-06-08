import asyncio
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from config import get_config
from pipline import pipeline

load_dotenv()


async def submit():
    """
    对所有问题生成回答并提交
    """
    top_k = 19
    top_token = -1
    min_top_k = 5
    max_top_k = 30
    collection_name = get_config()["MILVUS_COLLECTION_NAME"]
    use_query_cls = get_config()["USE_QUERY_CLS"]
    submit_file = f"submission/submit_all_top_k={top_k}_use_query_cls={use_query_cls}_{collection_name}.csv"
    question_file = "data/question_public.csv"
    df = pd.read_csv(question_file, index_col="id")
    product_questions_start_id = 64
    product_questions_end_id = 436

    exist_last_id = -1
    submit_path = Path(submit_file)
    results = []
    if submit_path.exists():
        results = pd.read_csv(submit_file).to_dict(orient="records")
        exist_last_id = max([result["id"] for result in results])

    max_concurrency = 8
    tasks = []
    batch_ids = []
    placeholder_answer = "您好，您的问题已收到，请您耐心等待处理结果，谢谢。"

    for row in tqdm(df.iterrows(), total=len(df)):
        if row[0] <= exist_last_id:
            continue
        if row[0] >= product_questions_start_id and row[0] <= product_questions_end_id:
            # for _ in range(max_concurrency):
            question = row[1]["question"].strip('"')
            tasks.append(
                asyncio.create_task(
                    pipeline(
                        question,
                        top_k,
                        top_token,
                        min_top_k,
                        max_top_k,
                    )
                )
            )
            batch_ids.append(row[0])
            max_concurrency -= 1
        else:
            results.append(
                {
                    "id": row[0],
                    "ret": placeholder_answer,
                }
            )

        if max_concurrency == 0 or row[0] == product_questions_end_id:
            rets = await asyncio.gather(*tasks)

            for id, ret in zip(batch_ids, rets):
                if ret is None:
                    ret = placeholder_answer
                results.append(
                    {
                        "id": id,
                        "ret": ret,
                    }
                )
            tasks = []
            batch_ids = []
            max_concurrency = 8
            pd.DataFrame(results).to_csv(submit_file, index=False)


if __name__ == "__main__":
    # 修改文件名
    # submit_file = "submit_gpt_5_5_all_top_k_15_ensembles_query_cls.csv"
    asyncio.run(submit())
