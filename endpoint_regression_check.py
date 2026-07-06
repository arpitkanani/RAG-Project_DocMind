import io
import shutil
import sys
import tempfile
import types
from pathlib import Path


def install_dependency_stubs():
    langchain_chroma = types.ModuleType("langchain_chroma")

    class DummyCollection:
        def count(self):
            return 0

    class DummyRetriever:
        def invoke(self, query):
            return []

    class DummyChroma:
        def __init__(self, *args, **kwargs):
            self._collection = DummyCollection()

        @classmethod
        def from_documents(cls, *args, **kwargs):
            return cls()

        def as_retriever(self, *args, **kwargs):
            return DummyRetriever()

        def similarity_search(self, *args, **kwargs):
            return []

        def similarity_search_with_score(self, *args, **kwargs):
            return []

        def get(self):
            return {"documents": [], "metadatas": []}

    langchain_chroma.Chroma = DummyChroma
    sys.modules["langchain_chroma"] = langchain_chroma

    documents_mod = types.ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    documents_mod.Document = Document
    sys.modules["langchain_core.documents"] = documents_mod

    messages_mod = types.ModuleType("langchain_core.messages")

    class BaseMessage:
        def __init__(self, content=""):
            self.content = content

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    messages_mod.BaseMessage = BaseMessage
    messages_mod.HumanMessage = HumanMessage
    messages_mod.AIMessage = AIMessage
    sys.modules["langchain_core.messages"] = messages_mod

    output_parsers_mod = types.ModuleType("langchain_core.output_parsers")

    class StrOutputParser:
        def __ror__(self, other):
            return self

    output_parsers_mod.StrOutputParser = StrOutputParser
    sys.modules["langchain_core.output_parsers"] = output_parsers_mod

    prompts_mod = types.ModuleType("langchain_core.prompts")

    class ChatPromptTemplate:
        @classmethod
        def from_messages(cls, messages):
            return cls()

        def __or__(self, other):
            return other

    class MessagesPlaceholder:
        def __init__(self, variable_name):
            self.variable_name = variable_name

    prompts_mod.ChatPromptTemplate = ChatPromptTemplate
    prompts_mod.MessagesPlaceholder = MessagesPlaceholder
    sys.modules["langchain_core.prompts"] = prompts_mod

    runnables_mod = types.ModuleType("langchain_core.runnables")

    class RunnableLambda:
        def __init__(self, func):
            self.func = func

        def __or__(self, other):
            return other

    class RunnableParallel(dict):
        def __or__(self, other):
            return other

    runnables_mod.RunnableLambda = RunnableLambda
    runnables_mod.RunnableParallel = RunnableParallel
    sys.modules["langchain_core.runnables"] = runnables_mod

    ollama_mod = types.ModuleType("langchain_ollama")

    class ChatOllama:
        def __init__(self, *args, **kwargs):
            pass

        def __ror__(self, other):
            return self

        def __or__(self, other):
            return other

    ollama_mod.ChatOllama = ChatOllama
    sys.modules["langchain_ollama"] = ollama_mod

    loaders_mod = types.ModuleType("langchain_community.document_loaders")

    class DummyLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return []

    loaders_mod.PyPDFLoader = DummyLoader
    loaders_mod.TextLoader = DummyLoader
    loaders_mod.Docx2txtLoader = DummyLoader
    loaders_mod.UnstructuredMarkdownLoader = DummyLoader
    loaders_mod.CSVLoader = DummyLoader
    sys.modules["langchain_community.document_loaders"] = loaders_mod

    chroma_mod = types.ModuleType("chromadb")

    class DummyPersistentClient:
        def __init__(self, *args, **kwargs):
            pass

        def list_collections(self):
            return []

        def delete_collection(self, name):
            return None

    chroma_mod.PersistentClient = DummyPersistentClient
    sys.modules["chromadb"] = chroma_mod

    youtube_mod = types.ModuleType("youtube_transcript_api")

    class YouTubeTranscriptApi:
        def fetch(self, *args, **kwargs):
            return []

    youtube_mod.YouTubeTranscriptApi = YouTubeTranscriptApi
    sys.modules["youtube_transcript_api"] = youtube_mod


install_dependency_stubs()

from fastapi.testclient import TestClient

import app as app_module
import src.components.memory_manager as memory_manager_module
from src.components.memory_manager import MemoryManager
from src.exception import CollectionNotFoundError, KnowledgeBaseEmptyError


temp_memory_dir = tempfile.mkdtemp(prefix="docmind_memory_", dir=str(Path.cwd()))
memory_manager_module.config["memory"]["persist_directory"] = temp_memory_dir

mock_collections = set()
upload_counter = {"file": 0, "youtube": 0}


def fake_run_from_bytes(self, file_bytes, filename):
    upload_counter["file"] += 1
    collection_name = f"doc_{upload_counter['file']}"
    mock_collections.add(collection_name)
    return {
        "success": True,
        "collection_name": collection_name,
        "documents_loaded": 1,
        "chunks_stored": 2,
        "is_youtube": False,
    }


def fake_run(self, source, collection_name=None, clear_existing=True):
    upload_counter["youtube"] += 1
    collection_name = collection_name or f"youtube_{upload_counter['youtube']}"
    mock_collections.add(collection_name)
    return {
        "success": True,
        "collection_name": collection_name,
        "documents_loaded": 1,
        "chunks_stored": 3,
        "is_youtube": True,
    }


def fake_query_run(self, query):
    if not mock_collections:
        raise KnowledgeBaseEmptyError("No documents found.")

    missing = [name for name in self.collection_names if name not in mock_collections]
    if missing:
        raise CollectionNotFoundError(missing)

    return {
        "answer": f"Answer for: {query}",
        "collection_scope": self.collection_names or "all",
        "query": query,
        "session_id": self.session_id,
    }


def fake_list_collections(self):
    return sorted(mock_collections)


def fake_delete_collection(self):
    if self.collection_name in mock_collections:
        mock_collections.remove(self.collection_name)
        return True
    return False


app_module.IngestionPipeline.run_from_bytes = fake_run_from_bytes
app_module.IngestionPipeline.run = fake_run
app_module.QAPipeline.run = fake_query_run
app_module.VectorStore.list_collections = fake_list_collections
app_module.VectorStore.delete_collection = fake_delete_collection

client = TestClient(app_module.app)
checks = []


def record(name, passed, details):
    checks.append((name, passed, details))


try:
    response = client.get("/health")
    record("GET /health", response.status_code == 200 and response.json()["status"] == "ok", response.json())

    response = client.post("/sessions")
    session_id = response.json()["session_id"]
    record("POST /sessions", response.status_code == 200 and bool(session_id), response.json())

    MemoryManager(session_id=session_id).save_message("human", "hello")
    response = client.get("/sessions")
    session_ids = [session["session_id"] for session in response.json()["sessions"]]
    record("GET /sessions", response.status_code == 200 and session_id in session_ids, response.json())

    response = client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": ("doc_a.txt", io.BytesIO(b"alpha"), "text/plain")},
    )
    record("POST /upload", response.status_code == 200 and response.json()["success"], response.json())

    response = client.post(
        "/query",
        json={"query": "What is in doc A?", "collection_names": ["doc_1"], "session_id": session_id},
    )
    record("POST /query single collection", response.status_code == 200 and response.json()["collection_scope"] == ["doc_1"], response.json())

    response = client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": ("doc_b.txt", io.BytesIO(b"beta"), "text/plain")},
    )
    record("POST /upload second document", response.status_code == 200 and response.json()["collection_name"] == "doc_2", response.json())

    response = client.post(
        "/query",
        json={"query": "Compare docs", "collection_names": ["doc_1", "doc_2"], "session_id": session_id},
    )
    record("POST /query multi collection", response.status_code == 200 and response.json()["collection_scope"] == ["doc_1", "doc_2"], response.json())

    response = client.post(
        "/youtube",
        json={"url": "https://youtube.com/watch?v=demo123", "session_id": session_id},
    )
    record("POST /youtube", response.status_code == 200 and response.json()["is_youtube"], response.json())

    response = client.get("/collections")
    record("GET /collections", response.status_code == 200 and set(response.json()["collections"]) == mock_collections, response.json())

    response = client.get(f"/sessions/{session_id}")
    attachments = response.json().get("attachments", [])
    record("GET /sessions/{id}", response.status_code == 200 and len(attachments) == 3, response.json())

    response = client.delete("/collections/doc_1")
    record("DELETE /collections/{name}", response.status_code == 200 and response.json()["deleted_collection"] == "doc_1", response.json())

    response = client.post(
        "/query",
        json={"query": "Ask removed doc", "collection_names": ["doc_1"], "session_id": session_id},
    )
    record(
        "POST /query deleted collection",
        response.status_code == 404 and response.json()["error_code"] == "collection_not_found",
        response.json(),
    )

    stale_scope = [item["collection"] for item in attachments]
    response = client.post(
        "/query",
        json={"query": "Ask stale session", "collection_names": stale_scope, "session_id": session_id},
    )
    record(
        "POST /query stale session scope",
        response.status_code == 404 and response.json()["error_code"] == "collection_not_found",
        response.json(),
    )

    response = client.delete(f"/sessions/{session_id}")
    record("DELETE /sessions/{id}", response.status_code == 200 and response.json()["success"], response.json())

    response = client.delete("/sessions")
    record("DELETE /sessions", response.status_code == 200 and response.json()["success"], response.json())

    MemoryManager(session_id="temp1234").save_message("human", "seed")
    response = client.delete("/memory")
    record("DELETE /memory", response.status_code == 200 and response.json()["success"], response.json())

finally:
    shutil.rmtree(temp_memory_dir, ignore_errors=True)


failed = [item for item in checks if not item[1]]
for name, passed, details in checks:
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {name} -> {details}")

if failed:
    raise SystemExit(1)
