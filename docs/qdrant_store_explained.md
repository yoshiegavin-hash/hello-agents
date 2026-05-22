# Qdrant 向量数据库存储实现 — 逐行代码解析

## 一、整体概览

**文件用途：** 为记忆系统提供基于 [Qdrant](https://qdrant.tech/) 向量数据库的向量存储能力。

**解决什么问题：** 记忆系统需要将文本 embedding 后的向量持久化，并支持按语义相似度检索。Qdrant 是一个高性能的向量搜索引擎，支持 HNSW 近似最近邻搜索、payload 过滤、分布式部署等能力。

**文件包含三个类：**

| 类名 | 职责 |
|---|---|
| `QdrantConnectionManager` | 连接级单例管理器，避免重复连接和集合初始化 |
| `QdrantVectorStore` | 核心实现类，封装 CRUD + 搜索 + 统计 |

**核心工作流程：**
```
文本 → embedding(向量) → QdrantVectorStore.add_vectors() → 持久化到 Qdrant 集合
                                                    ↓
查询文本 → embedding(查询向量) → QdrantVectorStore.search_similar() → 返回 Top-K 相似结果
```

---

## 二、逐行代码解析

### 1. 导入与依赖检测（第 1-27 行）

```python
"""
Qdrant向量数据库存储实现
使用专业的Qdrant向量数据库替代ChromaDB
"""

import logging
import os
import uuid
import threading
from typing import Dict, List, Optional, Any, Union
import numpy as np
from datetime import datetime
```

| 行 | 说明 |
|---|---|
| 1-4 | 模块文档字符串，说明用途 |
| 6 | `logging` — 记录运行日志 |
| 7 | `os` — 读取环境变量（HNSW 配置、API Key 等） |
| 8 | `uuid` — 生成全局唯一的点 ID |
| 9 | `threading` — 单例模式需要线程锁 |
| 10 | 类型注解 — 方法签名使用 |
| 11 | `numpy` — 向量运算（虽然主要是类型兼容） |
| 12 | `datetime` — 写入时自动生成时间戳 |

```python
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import(
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue, SearchRequest
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    models = None

logger = logging.getLogger(__name__)
```

| 行 | 说明 |
|---|---|
| 14-21 | **延迟导入 + 可用性检测**：`try/except` 包裹 qdrant-client 导入。如果没安装，设 `QDRANT_AVAILABLE = False` 而非直接崩溃，让调用方在运行时才报错 |
| 22-25 | `ImportError` 分支 — 设 sentinel 值，防止后续引用 `QdrantClient` 时触发 `NameError` |
| 27 | 创建模块级 logger，所有日志通过 `logger.info/debug/error` 输出 |

---

### 2. QdrantConnectionManager — 连接级单例管理器（第 30-69 行）

```python
class QdrantConnectionManager:
    """Qdrant连接管理器 - 防止重复连接和初始化"""
    _instances = {}  # key: (url, collection_name) -> QdrantVectorStore instance
    _lock = threading.Lock()
```

| 行 | 说明 |
|---|---|
| 30-31 | 类定义 + 文档字符串。这是一个 **类级别的注册表** |
| 32 | `_instances` — 字典，key 是 `(url, collection_name)` 元组，value 是 `QdrantVectorStore` 实例。不同 URL 或不同集合名会有不同实例 |
| 33 | `_lock` — 线程锁，保证多线程下只创建一个实例 |

```python
@classmethod
def get_instance(
    cls,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    collection_name: str = "hello_agents_vector",
    vector_size: int = 384,
    distance: str = "cosine",
    timeout: int = 30,
    **kwargs
) -> 'QdrantVectorStore':
    """获取或创建Qdrant实例（单例模式）"""
```

| 行 | 说明 |
|---|---|
| 35 | `@classmethod` — 通过类名直接调用，无需实例化 |
| 36-45 | 方法签名，参数与 `QdrantVectorStore.__init__` 基本一致，多了默认值 |

```python
# 创建唯一键
key = (url or "local", collection_name)

if key not in cls._instances:
    with cls._lock:
        # 双重检查锁定
        if key not in cls._instances:
            logger.debug(f"🔄 创建新的Qdrant连接: {collection_name}")
            cls._instances[key] = QdrantVectorStore(
                url=url, api_key=api_key, collection_name=collection_name,
                vector_size=vector_size, distance=distance,
                timeout=timeout, **kwargs
            )
        else:
            logger.debug(f"♻️ 复用现有Qdrant连接: {collection_name}")
else:
    logger.debug(f"♻️ 复用现有Qdrant连接: {collection_name}")

return cls._instances[key]
```

| 行 | 说明 |
|---|---|
| 48 | 生成唯一 key — `url` 为空时用 `"local"` 替代，与 `collection_name` 组合。同 URL + 同集合 → 共享同一实例 |
| 50 | **第一次检查**（无锁）— 快速路径，已存在时直接跳过加锁 |
| 51 | 获取锁 — 只有第一个线程进入创建逻辑 |
| 53 | **第二次检查**（有锁）— 双重检查锁定（Double-Checked Locking），防止多个线程同时通过第一次检查后重复创建 |
| 55-63 | 创建新 `QdrantVectorStore` 实例并注册到 `_instances` |
| 64-67 | 已存在时记录日志，复用连接 |
| 69 | 返回实例 |

> **注意：** 这个单例是按 `(url, collection_name)` 粒度的，不是全局单例。不同集合会创建不同实例。

---

### 3. QdrantVectorStore — 核心实现类

#### 3.1 `__init__` 构造函数（第 74-129 行）

```python
def __init__(
    self,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    collection_name: str = "hello_agents_vectors",
    vector_size: int = 384,
    distance: str = "cosine",
    timeout: int = 30,
    **kwargs
):
```

| 参数 | 说明 |
|---|---|
| `url` | Qdrant 云服务 URL。为 `None` 时使用本地 `localhost:6333` |
| `api_key` | 云服务认证密钥 |
| `collection_name` | 集合名称（Qdrant 的"表"概念） |
| `vector_size` | 向量维度，默认 384（all-MiniLM-L6-v2 的维度） |
| `distance` | 距离度量：`cosine`（余弦）、`dot`（点积）、`euclidean`（欧氏） |
| `timeout` | 连接超时秒数 |

```python
if not QDRANT_AVAILABLE:
    raise ImportError("qdrant-client未安装。请运行pip install qdrant-client>=1.6.0")
```

| 行 | 说明 |
|---|---|
| 94-97 | 运行时检查 — 如果 import 阶段没装上 qdrant-client，构造时立即报错 |

```python
self.url = url
self.api_key = api_key
self.collection_name = collection_name
self.vector_size = vector_size
self.timeout = timeout
```

| 行 | 说明 |
|---|---|
| 99-103 | 保存构造参数为实例属性 |

```python
# HNSW/Query params via env
try:
    self.hnsw_m = int(os.getenv("QDRANT_HNSW_M", "32"))
except Exception:
    self.hnsw_m = 32
try:
    self.hnsw_ef_construct = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "256"))
except Exception:
    self.hnsw_ef_construct = 256
try:
    self.search_ef = int(os.getenv("QDRANT_SEARCH_EF", "128"))
except Exception:
    self.search_ef = 128
self.search_exact = os.getenv("QDRANT_SEARCH_EXACT", "0") == "1"
```

| 行 | 说明 |
|---|---|
| 104-117 | **HNSW（Hierarchical Navigable Small World）参数** — 从环境变量读取，控制向量索引的精度和速度。这三个参数是 Qdrant 性能调优的核心：<br><br>• `hnsw_m`（默认 32）— 每个节点的最大连接数。越大索引越精确，但构建越慢、占用内存越多<br>• `hnsw_ef_construct`（默认 256）— 构建索引时的候选集大小。越大索引质量越高<br>• `search_ef`（默认 128）— 搜索时的候选集大小。越大搜索结果越精确，但越慢<br>• `search_exact`（默认 false）— 是否精确搜索。设为 1 时关闭 HNSW 近似，走暴力搜索，结果最准但最慢<br><br>`try/except` 包裹防止环境变量值非法导致崩溃 |

```python
# 距离度量映射
distance_map = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclidean": Distance.EUCLID,
}
self.distance = distance_map.get(distance.lower(), Distance.COSINE)
```

| 行 | 说明 |
|---|---|
| 119-125 | 将用户传入的字符串映射为 Qdrant SDK 的枚举值。`cosine` 最常用，衡量向量方向相似度（不受长度影响） |

```python
# 初始化客户端
self.client = None
self._initialize_client()
```

| 行 | 说明 |
|---|---|
| 127-129 | 调 `_initialize_client()` 建立连接并创建/获取集合 |

---

#### 3.2 `_initialize_client` — 客户端连接（第 131-172 行）

```python
def _initialize_client(self):
    """初始化Qdrant客户端和集合"""
    try:
        if self.url and self.api_key:
            # 使用云服务API
            self.client = QdrantClient(
                url=self.url, api_key=self.api_key, timeout=self.timeout
            )
            logger.info(f"✅ 成功连接到Qdrant云服务: {self.url}")
        elif self.url:
            # 使用自定义URL（无API密钥）
            self.client = QdrantClient(
                url=self.url, timeout=self.timeout
            )
            logger.info(f"✅ 成功连接到Qdrant服务: {self.url}")
        else:
            # 使用本地服务（默认）
            self.client = QdrantClient(
                host="localhost", port=6333, timeout=self.timeout
            )
            logger.info("✅ 成功连接到本地Qdrant服务: localhost:6333")
```

| 行 | 说明 |
|---|---|
| 135-142 | **云服务 + API Key** — 有 URL 有 key，走认证连接（Qdrant Cloud） |
| 143-149 | **自定义 URL（无 key）** — 可能是自部署的 Qdrant 服务，不走认证 |
| 150-157 | **本地服务** — 默认 `localhost:6333`。这是 Qdrant Docker 容器的默认端口 |

```python
# 检查连接
collections = self.client.get_collections()

# 创建或获取集合
self._ensure_collection()
```

| 行 | 说明 |
|---|---|
| 160 | `get_collections()` 是连通性测试 — 能拿到列表说明连接成功 |
| 163 | 调 `_ensure_collection()` 确保目标集合存在 |

```python
except Exception as e:
    logger.error(f"❌ Qdrant连接失败: {e}")
    if not self.url:
        logger.info("💡 本地连接失败，可以考虑使用Qdrant云服务")
        logger.info("💡 或启动本地服务: docker run -p 6333:6333 qdrant/qdrant")
    else:
        logger.info("💡 请检查URL和API密钥是否正确")
    raise
```

| 行 | 说明 |
|---|---|
| 165-172 | 连接失败时记录详细日志，并给出排查建议（本地/Docker 命令或检查 URL/key）。最后 `raise` 向上层抛出异常 |

---

#### 3.3 `_ensure_collection` — 集合初始化（第 174-212 行）

```python
def _ensure_collection(self):
    """确保集合存在，不存在则创建"""
    try:
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
```

| 行 | 说明 |
|---|---|
| 178-179 | 获取所有已有集合的名称列表 |

```python
if self.collection_name not in collection_names:
    # 创建新集合
    hnsw_cfg = None
    try:
        hnsw_cfg = models.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct)
    except Exception:
        hnsw_cfg = None
    self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config=VectorParams(
            size=self.vector_size,
            distance=self.distance
        ),
        hnsw_config=hnsw_cfg
    )
    logger.info(f"✅ 创建Qdrant集合: {self.collection_name}")
```

| 行 | 说明 |
|---|---|
| 181-196 | 集合不存在时**创建它**：<br><br>• 先构建 `HnswConfigDiff` — HNSW 索引配置（`m` 和 `ef_construct` 来自环境变量）<br>• `VectorParams` — 定义向量空间的维度（`vector_size`）和距离度量（`distance`）<br>• `create_collection` — 向 Qdrant 发起创建请求<br><br>如果 SDK 版本不支持 `HnswConfigDiff`（`try/except`），则不传 HNSW 配置，使用 Qdrant 默认值 |

```python
else:
    logger.info(f"✅ 使用现有Qdrant集合: {self.collection_name}")
    # 尝试更新 HNSW 配置
    try:
        self.client.update_collection(
            collection_name=self.collection_name,
            hnsw_config=models.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct)
        )
    except Exception as ie:
        logger.debug(f"跳过更新HNSW配置: {ie}")
```

| 行 | 说明 |
|---|---|
| 197-206 | 集合已存在时，**尝试更新 HNSW 配置**（如果新参数和已有不同）。更新失败通常是权限或版本限制，用 `debug` 级别忽略 |

```python
# 确保必要的payload索引
self._ensure_payload_indexes()
```

| 行 | 说明 |
|---|---|
| 208 | 为常用过滤字段创建索引，加速查询时的条件过滤 |

```python
except Exception as e:
    logger.error(f"❌ 集合初始化失败: {e}")
    raise
```

| 行 | 说明 |
|---|---|
| 210-212 | 异常处理，向上抛出 |

---

#### 3.4 `_ensure_payload_indexes` — Payload 索引（第 214-242 行）

```python
def _ensure_payload_indexes(self):
    """为常用过滤字段创建payload索引"""
    try:
        index_fields = [
            ("memory_type", models.PayloadSchemaType.KEYWORD),
            ("user_id", models.PayloadSchemaType.KEYWORD),
            ("memory_id", models.PayloadSchemaType.KEYWORD),
            ("timestamp", models.PayloadSchemaType.INTEGER),
            ("modality", models.PayloadSchemaType.KEYWORD),
            ("source", models.PayloadSchemaType.KEYWORD),
            ("external", models.PayloadSchemaType.BOOL),
            ("namespace", models.PayloadSchemaType.KEYWORD),
            ("is_rag_data", models.PayloadSchemaType.BOOL),
            ("rag_namespace", models.PayloadSchemaType.KEYWORD),
            ("data_source", models.PayloadSchemaType.KEYWORD),
        ]
```

| 行 | 说明 |
|---|---|
| 217-229 | 定义需要索引的字段列表。每个字段是 `(字段名, 类型)` 元组：<br><br>• `KEYWORD` — 精确匹配的字符串字段（如类型、ID、命名空间）<br>• `INTEGER` — 数值型字段（如时间戳，支持范围查询）<br>• `BOOL` — 布尔型字段（如是否是外部数据、是否是 RAG 数据） |

```python
for field_name, schema_type in index_fields:
    try:
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )
    except Exception as ie:
        # 索引已存在会报错，忽略
        logger.debug(f"索引 {field_name} 已存在或创建失败: {ie}")
except Exception as e:
    logger.debug(f"创建payload索引时出错: {e}")
```

| 行 | 说明 |
|---|---|
| 231-240 | 逐个字段调用 `create_payload_index`。索引已存在时 Qdrant 会报错，被内层 `try/except` 吞掉（`debug` 级别）。外层 `try/except` 兜底防止意外崩溃 |

> **为什么需要 Payload 索引？** Qdrant 的向量搜索（ANN）是主路径，但当用户加过滤条件（如"只搜用户A的记忆"）时，payload 索引让过滤从 O(n) 降到近似 O(1)，避免全表扫描。

---

#### 3.5 `add_vectors` — 写入向量（第 244-331 行）

```python
def add_vectors(
    self, 
    vectors: List[List[float]], 
    metadata: List[Dict[str, Any]], 
    ids: Optional[List[str]] = None
) -> bool:
```

| 参数 | 说明 |
|---|---|
| `vectors` | 向量列表，每个向量是 `List[float]`，维度必须等于 `vector_size` |
| `metadata` | 每个向量附带的元数据字典（payload），如 `memory_type`, `user_id` 等 |
| `ids` | 可选的点 ID 列表。不提供时自动生成 |

```python
if not vectors:
    logger.warning("⚠️ 向量列表为空")
    return False
```

| 行 | 说明 |
|---|---|
| 262-264 | 空列表直接返回 `False`，快速退出 |

```python
if ids is None:
    ids = [f"vec_{i}_{int(datetime.now().timestamp() * 1000000)}" 
           for i in range(len(vectors))]
```

| 行 | 说明 |
|---|---|
| 267-269 | 自动生成 ID — 格式：`vec_{序号}_{微秒级时间戳}`。例如 `vec_0_1715678901234567` |

```python
points = []
for i, (vector, meta, point_id) in enumerate(zip(vectors, metadata, ids)):
    # 确保向量是正确的维度
    try:
        vlen = len(vector)
    except Exception:
        logger.error(f"[Qdrant] 非法向量类型: index={i} type={type(vector)} value={vector}")
        continue
    if vlen != self.vector_size:
        logger.warning(f"⚠️ 向量维度不匹配: 期望{self.vector_size}, 实际{len(vector)}")
        continue
```

| 行 | 说明 |
|---|---|
| 274 | 三路 `zip` 遍历 — 同时拿到向量、元数据、ID |
| 276-279 | **防御性编程**：`len(vector)` 可能抛异常（如果 vector 不是 iterable），记录错误后 `continue` 跳过该条 |
| 281-283 | **维度检查**：向量维度必须等于 `vector_size`。不匹配时跳过该条，防止写入后搜索出错 |

```python
# 添加时间戳到元数据
meta_with_timestamp = meta.copy()
meta_with_timestamp["timestamp"] = int(datetime.now().timestamp())
meta_with_timestamp["added_at"] = int(datetime.now().timestamp())
if "external" in meta_with_timestamp and not isinstance(meta_with_timestamp.get("external"), bool):
    val = meta_with_timestamp.get("external")
    meta_with_timestamp["external"] = True if str(val).lower() in ("1", "true", "yes") else False
```

| 行 | 说明 |
|---|---|
| 286-288 | 给元数据注入两个时间戳字段：<br>• `timestamp` — 当前时间（秒级）<br>• `added_at` — 写入时间（秒级）<br><br>两者在写入瞬间相同，但语义不同：`timestamp` 可被后续 `update` 修改，`added_at` 永远不变 |
| 289-292 | **规范化 `external` 字段为 bool** — Qdrant 的 BOOL 类型 payload 索引要求值是真正的 `bool`。如果传入的是字符串 `"true"` / `"1"` / `"yes"`，转为 `True`，否则 `False` |

```python
safe_id: Any
if isinstance(point_id, int):
    safe_id = point_id
elif isinstance(point_id, str):
    try:
        uuid.UUID(point_id)
        safe_id = point_id
    except Exception:
        safe_id = str(uuid.uuid4())
else:
    safe_id = str(uuid.uuid4())
```

| 行 | 说明 |
|---|---|
| 294-304 | **点 ID 安全化处理** — Qdrant 只接受无符号整数或 UUID 格式字符串作为点 ID：<br>• 整数 → 直接用<br>• 字符串 → 尝试解析为 UUID，合法则用，否则生成新的 UUID<br>• 其他类型 → 生成新的 UUID |

```python
point = PointStruct(
    id=safe_id,
    vector=vector,
    payload=meta_with_timestamp
)
points.append(point)
```

| 行 | 说明 |
|---|---|
| 306-311 | 构建 Qdrant 的 `PointStruct` 对象并加入列表。一个 point = 一个向量数据单元 |

```python
if not points:
    logger.warning("⚠️ 没有有效的向量点")
    return False
```

| 行 | 说明 |
|---|---|
| 313-315 | 全部向量都被跳过的情况（比如全部维度不匹配），返回 `False` |

```python
operation_info = self.client.upsert(
    collection_name=self.collection_name,
    points=points,
    wait=True
)
```

| 行 | 说明 |
|---|---|
| 319-323 | **`upsert`（upsert = update + insert）** — Qdrant 的批量写入操作。如果点 ID 已存在则更新，否则插入。`wait=True` 表示同步等待写入完成再返回 |

---

#### 3.6 `search_similar` — 向量相似度搜索（第 333-406 行）

```python
def search_similar(
    self, 
    query_vector: List[float], 
    limit: int = 10, 
    score_threshold: Optional[float] = None,
    where: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
```

| 参数 | 说明 |
|---|---|
| `query_vector` | 查询向量 — 由 embedding 模型将查询文本转换而来 |
| `limit` | 返回结果数量上限 |
| `score_threshold` | 相似度阈值 — 低于此分数的结果被过滤 |
| `where` | 过滤条件字典 — 如 `{"memory_type": "episodic", "user_id": "user_1"}` |

```python
if len(query_vector) != self.vector_size:
    logger.error(f"❌ 查询向量维度错误: 期望{self.vector_size}, 实际{len(query_vector)}")
    return []
```

| 行 | 说明 |
|---|---|
| 353-355 | 维度校验 — 查询向量维度必须等于集合的向量维度，否则搜索无意义 |

```python
query_filter = None
if where:
    conditions = []
    for key, value in where.items():
        if isinstance(value, (str, int, float, bool)):
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )
    if conditions:
        query_filter = Filter(must=conditions)
```

| 行 | 说明 |
|---|---|
| 358-371 | **构建 Qdrant 过滤条件**：<br><br>• 遍历 `where` 字典，每个 key-value 对转为一个 `FieldCondition`<br>• `MatchValue` — 精确匹配条件<br>• `Filter(must=conditions)` — 所有条件必须满足（AND 逻辑）<br><br>例如 `where={"memory_type": "episodic", "user_id": "u1"}` 转为：`memory_type == "episodic" AND user_id == "u1"` |

```python
search_params = None
try:
    search_params = models.SearchParams(hnsw_ef=self.search_ef, exact=self.search_exact)
except Exception:
    search_params = None
```

| 行 | 说明 |
|---|---|
| 375-379 | **搜索参数** — 控制搜索精度：<br>• `hnsw_ef` — 搜索时的候选集大小（来自环境变量 `QDRANT_SEARCH_EF`）<br>• `exact` — 是否精确搜索（`QDRANT_SEARCH_EXACT`） |

```python
search_result = self.client.search(
    collection_name=self.collection_name,
    query_vector=query_vector,
    query_filter=query_filter,
    limit=limit,
    score_threshold=score_threshold,
    with_payload=True,
    with_vectors=False,
    search_params=search_params
)
```

| 行 | 说明 |
|---|---|
| 380-389 | **执行搜索** — Qdrant SDK 的 `search` 方法：<br><br>• `query_vector` — 查询向量<br>• `query_filter` — 可选的 payload 过滤<br>• `limit` — 返回数量<br>• `score_threshold` — 最低相似度分数<br>• `with_payload=True` — 返回 payload（元数据）<br>• `with_vectors=False` — 不返回原始向量（节省网络传输） |

```python
results = []
for hit in search_result:
    result = {
        "id": hit.id,
        "score": hit.score,
        "metadata": hit.payload or {}
    }
    results.append(result)
```

| 行 | 说明 |
|---|---|
| 392-399 | **格式化结果** — 将 Qdrant 的 `ScoredPoint` 对象转为普通字典：<br>• `id` — 点 ID<br>• `score` — 相似度分数（cosine 相似度范围 [-1, 1]）<br>• `metadata` — 写入时附加的 payload |

---

#### 3.7 `delete_vectors` — 按点 ID 删除（第 408-435 行）

```python
def delete_vectors(self, ids: List[str]) -> bool:
    try:
        if not ids:
            return True
        operation_info = self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids),
            wait=True
        )
        logger.info(f"✅ 成功删除 {len(ids)} 个向量")
        return True
    except Exception as e:
        logger.error(f"❌ 删除向量失败: {e}")
        return False
```

| 行 | 说明 |
|---|---|
| 408-435 | 通过点 ID 批量删除向量。`PointIdsList` 是按 ID 列表删除的选择器。`wait=True` 同步等待删除完成 |

---

#### 3.8 `delete_memories` — 按 memory_id 删除（第 456-480 行）

```python
def delete_memories(self, memory_ids: List[str]):
    """删除指定记忆（通过payload中的 memory_id 过滤删除）"""
    try:
        if not memory_ids:
            return
        conditions = [
            FieldCondition(key="memory_id", match=MatchValue(value=mid))
            for mid in memory_ids
        ]
        query_filter = Filter(should=conditions)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=query_filter),
            wait=True,
        )
        logger.info(f"✅ 成功按memory_id删除 {len(memory_ids)} 个Qdrant向量")
    except Exception as e:
        logger.error(f"❌ 删除记忆失败: {e}")
        raise
```

| 行 | 说明 |
|---|---|
| 456-480 | **与 `delete_vectors` 的区别：** 这里不是按 Qdrant 的点 ID 删除，而是按 payload 中的 `memory_id` 字段删除。因为写入时可能将非 UUID 的点 ID 转换为 UUID（见 `add_vectors` 的 `safe_id` 逻辑），所以上层持有的 memory_id 可能与 Qdrant 点 ID 不一致。<br><br>• `Filter(should=conditions)` — `should` = OR 逻辑，匹配任一 `memory_id` 即删除<br>• `FilterSelector` — 按过滤条件选择要删除的点<br>• 失败时 `raise` 而非返回 `False`，因为上层可能需要区分"没找到"和"连接失败" |

---

#### 3.9 `clear_collection` — 清空集合（第 437-454 行）

```python
def clear_collection(self) -> bool:
    try:
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
        logger.info(f"✅ 成功清空Qdrant集合: {self.collection_name}")
        return True
    except Exception as e:
        logger.error(f"❌ 清空集合失败: {e}")
        return False
```

| 行 | 说明 |
|---|---|
| 446-447 | **先删除再重建** — `delete_collection` 彻底删除集合，然后 `_ensure_collection()` 创建一个空集合。比逐个删除点更高效 |

---

#### 3.10 `get_collection_info` / `get_collection_stats` — 统计信息（第 482-518 行）

```python
def get_collection_info(self) -> Dict[str, Any]:
    try:
        collection_info = self.client.get_collection(self.collection_name)
        info = {
            "name": self.collection_name,
            "vectors_count": collection_info.vectors_count,
            "indexed_vectors_count": collection_info.indexed_vectors_count,
            "points_count": collection_info.points_count,
            "segments_count": collection_info.segments_count,
            "config": {
                "vector_size": self.vector_size,
                "distance": self.distance.value,
            }
        }
        return info
    except Exception as e:
        logger.error(f"❌ 获取集合信息失败: {e}")
        return {}
```

| 行 | 说明 |
|---|---|
| 490-508 | 从 Qdrant 获取集合的运行状态信息：<br>• `vectors_count` — 总向量数<br>• `indexed_vectors_count` — 已索引的向量数（异步索引，可能暂时小于总数）<br>• `points_count` — 总点数<br>• `segments_count` — 底层存储段数（Qdrant 的存储结构） |

```python
def get_collection_stats(self) -> Dict[str, Any]:
    """获取集合统计信息（兼容抽象接口）"""
    info = self.get_collection_info()
    if not info:
        return {"store_type": "qdrant", "name": self.collection_name}
    info["store_type"] = "qdrant"
    return info
```

| 行 | 说明 |
|---|---|
| 510-518 | 兼容上层抽象接口的统计方法，额外加了 `store_type` 标识。如果获取失败返回最小可用信息 |

---

#### 3.11 `health_check` — 健康检查（第 520-533 行）

```python
def health_check(self) -> bool:
    try:
        collections = self.client.get_collections()
        return True
    except Exception as e:
        logger.error(f"❌ Qdrant健康检查失败: {e}")
        return False
```

| 行 | 说明 |
|---|---|
| 527-530 | 最简单的存活探针 — 能列出集合列表就认为服务健康。供运维监控系统调用 |

---

#### 3.12 `__del__` — 析构函数（第 535-541 行）

```python
def __del__(self):
    """析构函数，清理资源"""
    if hasattr(self, 'client') and self.client:
        try:
            self.client.close()
        except:
            pass
```

| 行 | 说明 |
|---|---|
| 535-541 | Python 对象被 GC 回收前调用，关闭网络连接。裸 `except` 防止 `close` 本身抛异常导致析构失败（Python 中 `__del__` 抛异常会被静默吞掉，但会污染 stderr） |

---

## 三、关键设计决策总结

### 1. 单例管理器的粒度

`QdrantConnectionManager` 按 `(url, collection_name)` 组合做 key，不是全局单例。这意味着：
- 不同集合名 → 不同实例 → 互不影响
- 同一集合被多处引用 → 共享连接 → 节省资源

### 2. HNSW 参数通过环境变量控制

三个核心性能参数（`QDRANT_HNSW_M`、`QDRANT_HNSW_EF_CONSTRUCT`、`QDRANT_SEARCH_EF`）通过环境变量注入，而非硬编码：
- **开发环境**：可以用默认值快速迭代
- **生产环境**：调大参数提高精度
- **无需改代码**：改环境变量即可

### 3. 写入时的防御性编程

`add_vectors` 中有三层防御：
- `len(vector)` 异常保护 → 防非法类型
- 维度不匹配 → 跳过该条
- 点 ID 格式校验 → 非 UUID 自动生成

这些保护保证批量写入时不会因为单条脏数据导致整个批次失败。

### 4. `delete_memories` 与 `delete_vectors` 的分离

- `delete_vectors`：按 Qdrant **点 ID** 删除（快速，直接定位）
- `delete_memories`：按业务 **memory_id**（payload 字段）删除（安全，不依赖点 ID 格式）

这是为了应对 `add_vectors` 中 `safe_id` 可能替换原始 ID 的情况，确保删除操作的可靠性。

### 5. Payload 索引的预先创建

在集合初始化时就为 11 个常用字段创建 payload 索引，而不是等到查询时才发现慢：
- `KEYWORD` 类型用于精确匹配（类型、ID、命名空间）
- `INTEGER` 类型用于范围查询（时间戳）
- `BOOL` 类型用于开关过滤（外部数据标记、RAG 标记）

---

## 四、与其他组件的关系

```
记忆类型 (working/episodic/semantic/perceptual)
    ↓
embedding.py → encode(text) → 向量 (List[float])
    ↓
qdrant_store.py → add_vectors(向量, metadata) → 持久化
    ↓
检索时：query → encode(query) → search_similar(查询向量) → 返回 Top-K 记忆
```

`qdrant_store.py` 处于记忆系统的**持久化层**，`embedding.py` 处于**向量化层**，两者配合实现"文本 → 向量 → 存储 → 检索"的完整链路。
