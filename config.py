import os
from dotenv import load_dotenv

load_dotenv()


def get_config() -> dict:
    MILVUS_COLLECTION_NAME_DEFAULT = "handbook_knowledge_bank_test"
    IMAGE_DESCRIPTION_RESULTS_JSON_FILE = (
        f"experiment/上下文增强的实验/{MILVUS_COLLECTION_NAME_DEFAULT}.json"
        if os.getenv("USE_CONTEXTUAL_AUGMENTATION") == "1"
        else f"experiment/上下文增强的实验/{MILVUS_COLLECTION_NAME_DEFAULT}_without_context.json"
    )
    USE_QUERY_CLS = os.getenv("USE_QUERY_CLS") == "1"

    return {
        "USE_QUERY_CLS": USE_QUERY_CLS,
        "USE_CONTEXTUAL_AUGMENTATION": os.getenv("USE_CONTEXTUAL_AUGMENTATION") == "1",
        "IMAGE_DESCRIPTION_RESULTS_JSON_FILE": IMAGE_DESCRIPTION_RESULTS_JSON_FILE,
        "MILVUS_COLLECTION_NAME": MILVUS_COLLECTION_NAME_DEFAULT
        if os.getenv("USE_CONTEXTUAL_AUGMENTATION") == "1"
        else f"{MILVUS_COLLECTION_NAME_DEFAULT}_without_context",
    }
