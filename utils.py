# 一些工具函数
import base64
import mimetypes
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional

from langchain_core.documents import Document
from lingua import Language, LanguageDetectorBuilder


def get_image_name(image_name: str) -> str:
    """
    根据图片路径的stem，或者图片名称，原本的数据集中的图片是没有扩展名的，该函数返回带有扩展名的图片名称
    """
    image_root_dir = Path(os.getenv("IMAGE_ROOT_DIR", "data/KownledgeBase/手册/插图"))
    extensions = [".png", ".jpg", ".jpeg"]
    for ext in extensions:
        image_path = Path(image_root_dir, image_name + ext)
        if image_path.exists():
            return image_name + ext
    raise FileNotFoundError(f"该图片不存在,图片名为{image_name}")


def encode_image(image_path: str | Path) -> tuple[str, str]:
    """
    将图片编码为 base64 字符串，同时返回图片的 MIME 类型,用来生成图片描述
    """
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path)

    if mime_type is None:
        mime_type = "image/png"

    with open(path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    return image_base64, mime_type


def language_detect(text: str) -> Literal["chinese", "english"]:
    """
    对文本进行语言检测，返回检测到的语言（"chinese" 或 "english"）
    """
    languages = [Language.ENGLISH, Language.CHINESE]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    language = detector.detect_language_of(text)

    if language is None:
        raise Exception("语言检测失败")

    return language.name.lower()
