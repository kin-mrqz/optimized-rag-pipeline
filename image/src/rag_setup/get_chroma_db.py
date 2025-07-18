import os
import sys
import shutil
from langchain_chroma import Chroma
from rag_setup.get_embedding import get_embedding_model

CHROMA_PATH = os.environ.get("CHROMA_PATH", "src/chroma_db")
IS_USING_IMAGE_RUNTIME = os.environ.get("IS_USING_IMAGE_RUNTIME", "False").lower() == "true"
CHROMA_DB_INSTANCE = None


def load_vectorstore():
    global CHROMA_DB_INSTANCE
    if not CHROMA_DB_INSTANCE:
        if IS_USING_IMAGE_RUNTIME:
            __import__("pysqlite3")
            sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
            copy_to_temp()

        CHROMA_DB_INSTANCE = Chroma(
        persist_directory=get_runtime_chroma_path(),
        embedding_function=get_embedding_model(),
        )
    return CHROMA_DB_INSTANCE


def copy_to_temp():
    dst_chroma_path = get_runtime_chroma_path()
    if not os.path.exists(dst_chroma_path):
        os.makedirs(dst_chroma_path)

    tmp_contents = os.listdir(dst_chroma_path)
    if len(tmp_contents) > 0:
        os.makedirs(dst_chroma_path, exist_ok=True)
        shutil.copytree(CHROMA_PATH, dst_chroma_path, dirs_exist_ok=True)
    else:
        print(f"Chroma DB already exists in {dst_chroma_path}, skipping copy.")


def get_runtime_chroma_path():
    if IS_USING_IMAGE_RUNTIME:
        return f"/tmp/{CHROMA_PATH}"
    else:
        return CHROMA_PATH