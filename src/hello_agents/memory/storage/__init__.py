"""存储层模块

按照第8章架构设计的存储层：
- DocumentStore: 文档存储
- QdrantVectorStore: Qdrant向量存储
- Neo4jGraphStore: Neo4j图存储
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from memory.storage.qdrant_store import QdrantVectorStore, QdrantConnectionManager

# Neo4j graph store and document store are optional - import only if available
try:
    from memory.storage.neo4j_store import Neo4jGraphStore
except ImportError:
    Neo4jGraphStore = None

try:
    from memory.storage.document_store import DocumentStore, SQLiteDocumentStore
except ImportError:
    DocumentStore = None
    SQLiteDocumentStore = None

__all__ = [
    "QdrantVectorStore",
    "QdrantConnectionManager",
    "Neo4jGraphStore",
    "DocumentStore",
    "SQLiteDocumentStore",
]
