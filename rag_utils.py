import hashlib
import streamlit as st
import fitz  # PyMuPDF
import bleach
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore as LlamaQdrantVectorStore
from llama_index.core import PromptTemplate
from llm_utils import LLMWrapper, LLMProvider, CustomLLM
from typing import List, Optional, Dict, Any
from apconfig import AppConfig

KNOWLEDGE_HUB_COLLECTION = "ustaad-jee-knowledge-hub"
CONVERSATION_CONTEXT_COLLECTION = "ustaad-jee-conversation-context"

def initialize_qdrant() -> bool:
    """Initialize Qdrant client and create collections"""
    try:
        api_key = st.secrets.get("QDRANT_API_KEY")
        url = st.secrets.get("QDRANT_URL")
        if not api_key or not url:
            st.error("Qdrant API key or URL not found in secrets")
            return False

        client = QdrantClient(url=url, api_key=api_key)
        st.session_state["qdrant_client"] = client
        collections = client.get_collections()
        existing_collections = [c.name for c in collections.collections]

        # Create knowledge hub collection
        if KNOWLEDGE_HUB_COLLECTION not in existing_collections:
            client.create_collection(
                collection_name=KNOWLEDGE_HUB_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print(f"Created collection: {KNOWLEDGE_HUB_COLLECTION}")

        # Create conversation context collection
        if CONVERSATION_CONTEXT_COLLECTION not in existing_collections:
            client.create_collection(
                collection_name=CONVERSATION_CONTEXT_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print(f"Created collection: {CONVERSATION_CONTEXT_COLLECTION}")

        time.sleep(0.5)
        return True
    except Exception as e:
        st.error(f"Qdrant initialization failed: {str(e)}")
        return False

def initialize_embeddings() -> bool:
    """Initialize HuggingFace embeddings"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        st.session_state["embeddings"] = embeddings
        Settings.embed_model = embeddings
        return True
    except Exception as e:
        st.error(f"Embeddings initialization failed: {str(e)}")
        return False

def initialize_llm() -> bool:
    """Initialize LLM for LlamaIndex"""
    try:
        if "llm" in st.session_state and st.session_state["llm"]:
            Settings.llm = CustomLLM(st.session_state["llm"])
            return True
        st.error("LLM not initialized. Configure Ustaad Jee's brain first.")
        return False
    except Exception as e:
        st.error(f"LLM initialization failed: {str(e)}")
        return False

def initialize_components():
    """Initialize all RAG components with proper storage context"""
    try:
        if not initialize_qdrant():
            return
        if not initialize_embeddings():
            return
        if not initialize_llm():
            return

        client = st.session_state["qdrant_client"]
        embeddings = st.session_state["embeddings"]

        # Initialize storage context with docstore
        docstore = SimpleDocumentStore()
        storage_context = StorageContext.from_defaults(docstore=docstore)
        st.session_state["storage_context"] = storage_context

        # Knowledge hub vector store
        knowledge_vector_store = LlamaQdrantVectorStore(
            client=client,
            collection_name=KNOWLEDGE_HUB_COLLECTION
        )

        # Conversation context vector store
        context_vector_store = LlamaQdrantVectorStore(
            client=client,
            collection_name=CONVERSATION_CONTEXT_COLLECTION
        )

        st.session_state["knowledge_vectorstore"] = knowledge_vector_store
        st.session_state["context_vectorstore"] = context_vector_store

        # Initialize node parsers
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
        automerging_parser = HierarchicalNodeParser.from_defaults()

        # Create indexes with storage context
        knowledge_sentence_window_index = VectorStoreIndex.from_vector_store(
            vector_store=knowledge_vector_store,
            embed_model=embeddings,
            storage_context=storage_context
        )
        knowledge_automerging_index = VectorStoreIndex.from_vector_store(
            vector_store=knowledge_vector_store,
            embed_model=embeddings,
            storage_context=storage_context
        )

        # Context indexes
        context_sentence_window_index = VectorStoreIndex.from_vector_store(
            vector_store=context_vector_store,
            embed_model=embeddings,
            storage_context=storage_context
        )
        context_automerging_index = VectorStoreIndex.from_vector_store(
            vector_store=context_vector_store,
            embed_model=embeddings,
            storage_context=storage_context
        )

        st.session_state["knowledge_sentence_window_index"] = knowledge_sentence_window_index
        st.session_state["knowledge_automerging_index"] = knowledge_automerging_index
        st.session_state["context_sentence_window_index"] = context_sentence_window_index
        st.session_state["context_automerging_index"] = context_automerging_index
        st.session_state["node_parser"] = node_parser
        st.session_state["automerging_parser"] = automerging_parser
        st.session_state["rag_initialized"] = True
        print("Components initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.session_state["rag_initialized"] = False

def get_conversation_id() -> str:
    """Get or create conversation ID for current session"""
    if "conversation_id" not in st.session_state:
        user_id = st.session_state.get("user_info", {}).get("localId", "anonymous")
        timestamp = str(int(time.time()))
        st.session_state["conversation_id"] = f"{user_id}_{timestamp}"
    return st.session_state["conversation_id"]

def parse_document(document: any) -> str:
    """Parse document content from various file types"""
    try:
        if isinstance(document, str):
            return document

        document.seek(0)
        if document.name.endswith(".pdf"):
            doc = fitz.open(stream=document.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            if not text.strip():
                raise ValueError("No extractable text in PDF")
            return bleach.clean(text, tags=[], strip=True)
        else:
            text = document.read().decode("utf-8")
            if not text.strip():
                raise ValueError("No extractable text in document")
            return bleach.clean(text, tags=[], strip=True)
    except Exception as e:
        st.error(f"Error parsing document: {str(e)}")
        return ""

def index_document(document_text: str) -> bool:
    """Index document text into knowledge base with proper storage context handling"""
    try:
        if not document_text.strip():
            st.error("No text provided for indexing")
            return False
        if not st.session_state.get("rag_initialized", False):
            initialize_components()

        documents = [Document(text=document_text)]
        node_parser = st.session_state.get("node_parser")
        automerging_parser = st.session_state.get("automerging_parser")
        knowledge_sentence_window_index = st.session_state.get("knowledge_sentence_window_index")
        knowledge_automerging_index = st.session_state.get("knowledge_automerging_index")
        storage_context = st.session_state.get("storage_context")

        if not all([node_parser, automerging_parser, knowledge_sentence_window_index, knowledge_automerging_index, storage_context]):
            st.error("Indexing components not initialized")
            return False

        # Parse documents into nodes
        sentence_nodes = node_parser.get_nodes_from_documents(documents)
        automerging_nodes = automerging_parser.get_nodes_from_documents(documents)

        # Add nodes to docstore so AutoMergingRetriever can access them
        storage_context.docstore.add_documents(sentence_nodes + automerging_nodes)

        # Insert nodes into indexes
        knowledge_sentence_window_index.insert_nodes(sentence_nodes)
        knowledge_automerging_index.insert_nodes(automerging_nodes)

        return True
    except Exception as e:
        st.error(f"Error indexing document: {str(e)}")
        return False

def index_knowledge_hub_document(document: any) -> bool:
    """Index a document into the knowledge hub"""
    try:
        document_text = parse_document(document)
        if not document_text.strip():
            st.error("No extractable text in document")
            return False
        return index_document(document_text)
    except Exception as e:
        st.error(f"Error indexing knowledge hub document: {str(e)}")
        return False

def index_conversation_context(query: str, response: str, conversation_id: str) -> bool:
    """Index conversation exchanges for better context retrieval"""
    try:
        if not st.session_state.get("rag_initialized", False):
            initialize_components()

        # Create a combined context document
        context_text = f"User Query: {query}\nAssistant Response: {response}"

        # Add metadata
        context_document = Document(
            text=context_text,
            metadata={
                "conversation_id": conversation_id,
                "timestamp": time.time(),
                "type": "conversation_exchange"
            }
        )

        context_automerging_index = st.session_state.get("context_automerging_index")
        automerging_parser = st.session_state.get("automerging_parser")
        storage_context = st.session_state.get("storage_context")

        if context_automerging_index and automerging_parser and storage_context:
            nodes = automerging_parser.get_nodes_from_documents([context_document])

            # Add nodes to docstore
            storage_context.docstore.add_documents(nodes)

            # Insert nodes into index
            context_automerging_index.insert_nodes(nodes)
            return True

        return False
    except Exception as e:
        print(f"Error indexing conversation context: {str(e)}")
        return False

def retrieve_relevant_chunks(query: str, max_chunks: int = 5) -> Dict[str, List[str]]:
    """Retrieve relevant chunks from both knowledge and conversation context"""
    try:
        if not st.session_state.get("rag_initialized", False):
            return {"knowledge": [], "context": []}

        knowledge_chunks = []
        context_chunks = []

        # Retrieve from knowledge base
        knowledge_automerging_index = st.session_state.get("knowledge_automerging_index")
        if knowledge_automerging_index:
            try:
                # Create base retriever from index
                base_knowledge_retriever = knowledge_automerging_index.as_retriever(
                    similarity_top_k=max_chunks
                )

                # Create storage context if not exists
                if "storage_context" not in st.session_state:
                    docstore = SimpleDocumentStore()
                    st.session_state["storage_context"] = StorageContext.from_defaults(docstore=docstore)

                # Create AutoMergingRetriever with correct parameters
                knowledge_retriever = AutoMergingRetriever(
                    base_knowledge_retriever,
                    st.session_state["storage_context"],
                    verbose=False
                )

                knowledge_nodes = knowledge_retriever.retrieve(query)
                knowledge_chunks = [node.text for node in knowledge_nodes]
            except Exception as e:
                print(f"Error retrieving knowledge chunks: {str(e)}")
                # Fallback to basic retrieval without auto-merging
                try:
                    base_knowledge_retriever = knowledge_automerging_index.as_retriever(
                        similarity_top_k=max_chunks
                    )
                    knowledge_nodes = base_knowledge_retriever.retrieve(query)
                    knowledge_chunks = [node.text for node in knowledge_nodes]
                except Exception as e2:
                    print(f"Error with fallback knowledge retrieval: {str(e2)}")

        # Retrieve from conversation context
        context_automerging_index = st.session_state.get("context_automerging_index")
        if context_automerging_index:
            try:
                # Create base retriever from index
                base_context_retriever = context_automerging_index.as_retriever(
                    similarity_top_k=max_chunks
                )

                # Create AutoMergingRetriever with correct parameters
                context_retriever = AutoMergingRetriever(
                    base_context_retriever,
                    st.session_state["storage_context"],
                    verbose=False
                )

                context_nodes = context_retriever.retrieve(query)
                context_chunks = [node.text for node in context_nodes]
            except Exception as e:
                print(f"Error retrieving context chunks: {str(e)}")
                # Fallback to basic retrieval without auto-merging
                try:
                    base_context_retriever = context_automerging_index.as_retriever(
                        similarity_top_k=max_chunks
                    )
                    context_nodes = base_context_retriever.retrieve(query)
                    context_chunks = [node.text for node in context_nodes]
                except Exception as e2:
                    print(f"Error with fallback context retrieval: {str(e2)}")

        print(f"Knowledge chunks found: {len(knowledge_chunks)}")
        print(f"Context chunks found: {len(context_chunks)}")

        # Debug: show first few chunks
        for i, chunk in enumerate(knowledge_chunks[:2]):
            print(f"Knowledge chunk {i}: {chunk[:100]}...")

        return {"knowledge": knowledge_chunks, "context": context_chunks}
    except Exception as e:
        print(f"Error retrieving chunks: {str(e)}")
        return {"knowledge": [], "context": []}

def clear_indexes():
    """Clear all indexes and reinitialize"""
    try:
        client = st.session_state.get("qdrant_client")
        if client:
            # Clear knowledge hub collection
            client.delete_collection(KNOWLEDGE_HUB_COLLECTION)
            client.create_collection(
                collection_name=KNOWLEDGE_HUB_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

            # Clear conversation context collection
            client.delete_collection(CONVERSATION_CONTEXT_COLLECTION)
            client.create_collection(
                collection_name=CONVERSATION_CONTEXT_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

        # Reinitialize storage context
        docstore = SimpleDocumentStore()
        storage_context = StorageContext.from_defaults(docstore=docstore)
        st.session_state["storage_context"] = storage_context

        # Reinitialize indexes
        if "knowledge_sentence_window_index" in st.session_state:
            st.session_state["knowledge_sentence_window_index"] = VectorStoreIndex.from_vector_store(
                vector_store=st.session_state["knowledge_vectorstore"],
                embed_model=st.session_state["embeddings"],
                storage_context=storage_context
            )

        if "knowledge_automerging_index" in st.session_state:
            st.session_state["knowledge_automerging_index"] = VectorStoreIndex.from_vector_store(
                vector_store=st.session_state["knowledge_vectorstore"],
                embed_model=st.session_state["embeddings"],
                storage_context=storage_context
            )

        if "context_sentence_window_index" in st.session_state:
            st.session_state["context_sentence_window_index"] = VectorStoreIndex.from_vector_store(
                vector_store=st.session_state["context_vectorstore"],
                embed_model=st.session_state["embeddings"],
                storage_context=storage_context
            )

        if "context_automerging_index" in st.session_state:
            st.session_state["context_automerging_index"] = VectorStoreIndex.from_vector_store(
                vector_store=st.session_state["context_vectorstore"],
                embed_model=st.session_state["embeddings"],
                storage_context=storage_context
            )

        st.session_state.index_version = st.session_state.get("index_version", 0) + 1
        # Clear conversation ID to start fresh
        if "conversation_id" in st.session_state:
            del st.session_state["conversation_id"]

        print("Indexes cleared and reinitialized successfully")
    except Exception as e:
        st.error(f"Error clearing indexes: {str(e)}")

@st.cache_data(show_spinner=False, ttl=3600, max_entries=100, hash_funcs={LLMWrapper: lambda x: id(x)})
def generate_response(chat_language: str, query: str, context_text: Optional[str], document_text: Optional[str], _glossary_version: int = 0) -> str:
    """Generate response using RAG with proper context retrieval"""
    try:
        if not st.session_state.get("llm"):
            return "Error: LLM not initialized."

        glossary = st.session_state.get("glossary", {})
        glossary_section = "\n".join([
            f"English: {translations.get('English', term)}, Urdu: {translations.get('Urdu', '')}, Roman Urdu: {translations.get('Roman Urdu', '')}"
            for term, translations in glossary.items()
        ]) if glossary else "No glossary terms available."

        # Get recent chat history
        last_6_exchanges = "\n".join(
            f"Q: {chat['query']}\nA: {chat['response']}"
            for chat in st.session_state.get("chat_history", [])[-6:]
        ) if st.session_state.get("chat_history") else ""

        # Retrieve relevant chunks from both knowledge and context
        retrieved_chunks = retrieve_relevant_chunks(query, max_chunks=5)
        knowledge_context = "\n".join(retrieved_chunks["knowledge"]) if retrieved_chunks["knowledge"] else "No relevant knowledge chunks found."
        conversation_context = "\n".join(retrieved_chunks["context"]) if retrieved_chunks["context"] else "No relevant conversation context found."

        # Enhanced context section
        context_section = f"""
Recent Conversation History:
{last_6_exchanges}

Relevant Previous Conversation Context:
{conversation_context}

Supplementary Information:
{context_text or 'No supplementary info provided.'}

Document Knowledge Base:
{knowledge_context}
"""

        prompt_template = (
            AppConfig.URDU_CHAT_PROMPT if chat_language == "Urdu" else
            AppConfig.ROMAN_URDU_CHAT_PROMPT if chat_language == "Roman Urdu" else
            AppConfig.ENGLISH_CHAT_PROMPT
        )

        prompt = prompt_template.format(
            language=chat_language,
            conversation_history=last_6_exchanges,
            document_text=document_text or "No document provided.",
            rag_context=knowledge_context,
            supplementary_info=context_section,
            glossary_section=glossary_section,
            glossary_translation_rules=AppConfig.GLOSSARY_TRANSLATION_RULES,
            question=query,
            allow_llm_knowledge="False",
            confidence_score="0.9"
        )

        response = st.session_state.llm.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )

        if chat_language == "Urdu":
            response = f"\u200F{response}"

        # Index this conversation exchange for future reference
        conversation_id = get_conversation_id()
        index_conversation_context(query, response, conversation_id)

        return response
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        return error_message