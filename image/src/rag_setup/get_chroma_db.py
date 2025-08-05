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

        # print(f"✅ Init ChromaDB {CHROMA_DB_INSTANCE} from {get_runtime_chroma_path()}")
    return CHROMA_DB_INSTANCE


def copy_to_temp():
    dst_chroma_path = get_runtime_chroma_path()
    # print(f"Copying ChromaDB from {CHROMA_PATH} to {dst_chroma_path}")

    # note: this will overwrite the existing directory in /tmp
    if os.path.exists(dst_chroma_path):
        shutil.rmtree(dst_chroma_path)
    shutil.copytree(CHROMA_PATH, dst_chroma_path)
    # print("✅ ChromaDB copied successfully.")


def get_runtime_chroma_path():
    if IS_USING_IMAGE_RUNTIME:
        return f"/tmp/{CHROMA_PATH}"
    else:
        return CHROMA_PATH