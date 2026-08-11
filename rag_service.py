"""
RAG 知识问答服务 - 基于 ChromaDB 的本地向量检索。

架构：
  知识文档(.md) → 分段 Chunk → Embedding API → ChromaDB 存储
  用户提问 → Embedding → ChromaDB 检索 Top-K → 返回相关知识段落

Embedding 方案：
  使用 SiliconFlow Embedding API（BAAI/bge-large-zh-v1.5），无需本地安装模型
  如需切换为本地模型，替换 _get_embedding() 函数即可

知识库初始化：
  将 ./data/knowledge/ 目录下的 .md 文件按段落切分后写入 ChromaDB
  运行 python rag_service.py init 即可初始化
"""
import os
import re
import json
from typing import List, Dict, Any, Optional

import httpx

from config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL

# ==================== 配置 ====================
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "chroma")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "data", "knowledge")
COLLECTION_NAME = "smart_home_kb"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
CHUNK_SIZE = 300  # 每段最大字符数
CHUNK_OVERLAP = 50  # 段落重叠字符数

# ==================== ChromaDB 实例（延迟初始化） ====================
_chroma_client = None
_collection = None


def _get_chroma_client():
    """延迟初始化 ChromaDB 客户端"""
    global _chroma_client, _collection
    if _chroma_client is not None:
        return _chroma_client

    try:
        import chromadb
    except ImportError:
        print("[RAG] chromadb 未安装，请运行: pip install chromadb")
        return None

    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "智能家居知识库"},
    )
    print(f"[RAG] ChromaDB 已初始化: {CHROMA_DB_PATH}")
    return _chroma_client


def _get_embedding(text: str) -> Optional[List[float]]:
    """调用 SiliconFlow Embedding API 获取文本向量"""
    if not SILICONFLOW_API_KEY:
        print("[RAG] SILICONFLOW_API_KEY 未配置")
        return None

    url = f"{SILICONFLOW_BASE_URL.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text,
    }

    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        print(f"[RAG] Embedding 调用失败: {e}")
        return None


def _get_embeddings_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """批量获取文本向量"""
    if not SILICONFLOW_API_KEY:
        return None

    url = f"{SILICONFLOW_BASE_URL.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }

    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # 按 index 排序确保顺序正确
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]
    except Exception as e:
        print(f"[RAG] 批量 Embedding 调用失败: {e}")
        return None


def _chunk_text(text: str, source: str = "") -> List[Dict[str, str]]:
    """
    将文本按段落切分为 Chunk。

    策略：
    1. 先按双换行符切分段落
    2. 超长段落按 CHUNK_SIZE 二次切分，保留 CHUNK_OVERLAP 重叠
    3. 每个 Chunk 保留 source 信息
    """
    chunks = []
    # 按空行切分段落
    paragraphs = re.split(r'\n\s*\n', text.strip())

    chunk_id = 0
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 10:
            continue

        # 段落未超长，直接作为一个 chunk
        if len(para) <= CHUNK_SIZE:
            chunks.append({
                "id": f"{source}_{chunk_id}" if source else str(chunk_id),
                "content": para,
                "source": source,
            })
            chunk_id += 1
        else:
            # 超长段落滑动窗口切分
            start = 0
            while start < len(para):
                end = start + CHUNK_SIZE
                chunk_text = para[start:end]
                if len(chunk_text) > 20:  # 过滤过短的碎片
                    chunks.append({
                        "id": f"{source}_{chunk_id}" if source else str(chunk_id),
                        "content": chunk_text,
                        "source": source,
                    })
                    chunk_id += 1
                start = end - CHUNK_OVERLAP

    return chunks


def init_knowledge_base(knowledge_dir: str = "") -> int:
    """
    初始化知识库：读取目录下的 .md/.txt 文件，切分后写入 ChromaDB。

    Args:
        knowledge_dir: 知识文件目录，默认使用 KNOWLEDGE_DIR

    Returns:
        写入的文档段落数
    """
    knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
    if not os.path.exists(knowledge_dir):
        print(f"[RAG] 知识目录不存在: {knowledge_dir}")
        os.makedirs(knowledge_dir, exist_ok=True)
        print(f"[RAG] 已创建知识目录: {knowledge_dir}，请将 .md 文件放入其中")
        return 0

    client = _get_chroma_client()
    if not client:
        return 0

    # 收集所有知识文件
    all_chunks = []
    for filename in os.listdir(knowledge_dir):
        if not filename.endswith((".md", ".txt")):
            continue
        filepath = os.path.join(knowledge_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = _chunk_text(content, source=filename)
        all_chunks.extend(chunks)
        print(f"[RAG] {filename} → {len(chunks)} 段落")

    if not all_chunks:
        print("[RAG] 未找到知识文件，请在 data/knowledge/ 目录放入 .md 文件")
        return 0

    # 清空旧数据
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "智能家居知识库"},
    )

    # 批量获取 Embedding
    texts = [c["content"] for c in all_chunks]
    print(f"[RAG] 正在获取 {len(texts)} 段落的 Embedding...")

    # 分批处理（API 可能限制批量大小）
    BATCH_SIZE = 20
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = _get_embeddings_batch(batch)
        if embeddings:
            all_embeddings.extend(embeddings)
            print(f"[RAG] Embedding 进度: {len(all_embeddings)}/{len(texts)}")
        else:
            print(f"[RAG] 批次 {i//BATCH_SIZE} 失败，跳过")
            return 0

    # 写入 ChromaDB
    _collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=texts,
        embeddings=all_embeddings,
        metadatas=[{"source": c["source"]} for c in all_chunks],
    )

    print(f"[RAG] 知识库初始化完成: {len(all_chunks)} 段落已写入 ChromaDB")
    return len(all_chunks)


def search_knowledge(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """
    从知识库检索与查询最相关的 Top-K 段落。

    Args:
        query: 用户查询文本
        top_k: 返回的最大段落数

    Returns:
        [{"content": str, "source": str, "distance": float}, ...]
    """
    if not _get_chroma_client():
        return []

    query_embedding = _get_embedding(query)
    if not query_embedding:
        return []

    try:
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
    except Exception as e:
        print(f"[RAG] 检索失败: {e}")
        return []

    if not results or not results.get("documents"):
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0] if results.get("distances") else [0] * len(documents)

    return [
        {
            "content": doc,
            "source": meta.get("source", "") if meta else "",
            "distance": dist,
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def is_available() -> bool:
    """检查 RAG 服务是否可用（ChromaDB 已安装且有数据）"""
    if not _get_chroma_client():
        return False
    try:
        return _collection.count() > 0
    except Exception:
        return False


# ==================== 命令行入口：初始化知识库 ====================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        count = init_knowledge_base()
        print(f"\n知识库初始化完成，共 {count} 段落")
    else:
        print("用法: python rag_service.py init")
        print("  将 .md 知识文件放入 data/knowledge/ 目录后运行初始化")
