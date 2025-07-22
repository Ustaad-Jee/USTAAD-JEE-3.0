import streamlit as st
st.set_page_config(
    page_title="Ustaad Jee's Knowledge Hub",
    page_icon="dude.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
import openai
import hashlib
import requests
import json
from apconfig import AppConfig
from typing import Optional, Dict, Tuple, List
from rag_utils import index_document, generate_response, clear_indexes, initialize_components, parse_document, index_knowledge_hub_document
from enum import Enum
import os
from abc import ABC, abstractmethod
import pandas as pd
from llm_utils import LLMWrapper, LLMProvider
import io
import time
import bleach
import firebase_admin
from firebase_admin import credentials, auth, firestore
from auth import (
    sign_in_with_email_and_password,
    get_account_info,
    send_email_verification,
    send_password_reset_email,
    create_user_with_email_and_password,
    delete_user_account,
    raise_detailed_error
)
from firestore_utils import log_user_activity, store_chat_history, set_admin_user

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    try:
        service_account_json = st.secrets["firebase"]["SERVICE_ACCOUNT"]
        if service_account_json:
            try:
                service_account_dict = json.loads(service_account_json)
                cred = credentials.Certificate(service_account_dict)
                firebase_admin.initialize_app(cred)
                print("✅ Firebase initialized successfully with Streamlit Secrets")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in firebase.SERVICE_ACCOUNT: {str(e)}")
        else:
            raise ValueError("SERVICE_ACCOUNT not found in firebase secrets")
    except KeyError as e:
        service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
        if service_account_path and os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
            print("✅ Firebase initialized with file path")
        else:
            raise ValueError(
                "Firebase service account not found. Ensure 'firebase.SERVICE_ACCOUNT' "
                "is properly set in Streamlit Secrets with valid JSON content."
            )
    except Exception as e:
        raise ValueError(f"Firebase initialization failed: {str(e)}")

# Get Firestore client
db = firestore.client()

@st.cache_data(show_spinner="Parsing document...", ttl=3600)
def cached_parse_document(document: any) -> list:
    return parse_document(document)

@st.cache_data(show_spinner="Indexing document...", ttl=3600)
def cached_index_document(document_text: str) -> bool:
    if not st.session_state.get("rag_initialized", False):
        initialize_components()
    return index_document(document_text)

@st.cache_data()
def cached_retrieve_and_generate(query: str, context_text: str, document_texts: List[str], index_version: int) -> str:
    return generate_response(query, context_text, document_texts)

def init_session_state():
    if 'llm' not in st.session_state:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
            if not api_key:
                st.error("OpenAI API key not found!")
                st.session_state.connection_status = "Failed"
                return
            llm = LLMWrapper(
                provider=LLMProvider.OPENAI,
                api_key=api_key,
                model="gpt-4o-mini"
            )
            st.session_state.llm = llm
            st.session_state.connection_status = "Connected"
        except Exception as e:
            st.session_state.connection_status = "Failed"
            st.error(f"Failed to initialize LLM: {str(e)}")

    if 'embeddings' not in st.session_state or 'qdrant_client' not in st.session_state:
        initialize_components()

    default_state = {
        'glossary': {},
        'connection_status': "Not Connected",
        'results': {},
        'glossary_updated': False,
        'chat_history': [],
        'uploaded_documents': [],
        'document_texts': [],
        'context_text': "",
        'auth_warning': "",
        'auth_success': "",
        'is_admin': False,
        'index_version': 0
    }

    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False

def create_auth_interface():
    st.markdown("""
    <style>
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 0.5rem 1rem;
        margin-top: -38rem;
    }
    .auth-title {
        text-align: center;
        color: #1f2937;
        font-size: 3rem;
        font-weight: 600;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.2s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        transition: all 0.2s;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .auth-links {
        text-align: center;
        margin-top: 1.5rem;
    }
    .auth-links .stButton {
        display: inline-block;
        margin: 0 0.5rem;
    }
    .auth-links .stButton > button {
        background: none !important;
        border: none !important;
        color: #3b82f6 !important;
        text-decoration: none !important;
        font-weight: 500 !important;
        padding: 0.25rem 0.5rem !important;
        font-size: 0.95rem !important;
        cursor: pointer !important;
        transition: color 0.2s, text-decoration 0.2s !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        min-height: auto !important;
        height: auto !important;
        width: auto !important;
        margin: 0 !important;
    }
    .auth-links .stButton > button:hover {
        color: #1d4ed8 !important;
        text-decoration: underline !important;
        background: none !important;
        transform: none !important;
        box-shadow: none !important;
    }
    .auth-links .stButton > button:focus {
        outline: 2px solid #3b82f6 !important;
        outline-offset: 2px !important;
        box-shadow: none !important;
    }
    .auth-divider {
        margin: 1rem 0;
        color: #6b7280;
        font-size: 0.9rem;
    }
    .back-link {
        text-align: center;
        margin-bottom: 1rem;
    }
    .back-link .stButton > button {
        background: none !important;
        border: none !important;
        color: #6b7280 !important;
        text-decoration: none !important;
        font-size: 0.9rem !important;
        padding: 0.25rem 0.5rem !important;
        transition: color 0.2s !important;
        cursor: pointer !important;
        min-height: auto !important;
        height: auto !important;
        width: auto !important;
        margin: 0 !important;
        box-shadow: none !important;
    }
    .back-link .stButton > button:hover {
        color: #3b82f6 !important;
        text-decoration: underline !important;
        background: none !important;
        transform: none !important;
        box-shadow: none !important;
    }
    .form-description {
        text-align: center;
        color: #6b7280;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
    .stAlert {
        border-radius: 8px;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        if 'auth_mode' not in st.session_state:
            st.session_state.auth_mode = 'signin'

        if st.session_state.auth_mode != 'signin':
            st.markdown('<div class="back-link">', unsafe_allow_html=True)
            if st.button("← Back to Sign In", key="back_to_signin"):
                st.session_state.auth_mode = 'signin'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.auth_mode == 'signin':
            st.markdown('<h1 class="auth-title">Sign In</h1>', unsafe_allow_html=True)
            st.markdown('<p class="form-description">Welcome back! Please sign in to your account.</p>',
                        unsafe_allow_html=True)
            with st.form("sign_in_form", clear_on_submit=False):
                email = st.text_input("Email Address", placeholder="Enter your email", label_visibility="collapsed")
                password = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed")
                submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                if submit:
                    if not email or not password:
                        st.error("Please fill in all fields")
                    else:
                        try:
                            with st.spinner("Signing you in..."):
                                id_token = sign_in_with_email_and_password(email, password)['idToken']
                                user_info = get_account_info(id_token)["users"][0]
                                if not user_info["emailVerified"]:
                                    send_email_verification(id_token)
                                    st.session_state.auth_warning = 'Check your email to verify your account'
                                else:
                                    st.session_state.user_info = user_info
                                    st.session_state.id_token = id_token
                                    user = auth.get_user(user_info['localId'])
                                    st.session_state.is_admin = user.custom_claims.get('admin', False) if user.custom_claims else False
                                    log_user_activity(user_info['localId'], "login", {"email": user_info['email']})
                                    st.success("Welcome! Redirecting...")
                                    st.rerun()
                        except requests.exceptions.HTTPError as error:
                            error_message = json.loads(error.args[1])['error']['message']
                            if error_message in {"INVALID_EMAIL", "EMAIL_NOT_FOUND", "INVALID_PASSWORD", "MISSING_PASSWORD"}:
                                st.session_state.auth_warning = 'Invalid email or password. Please try again.'
                            else:
                                st.session_state.auth_warning = 'Something went wrong. Please try again later.'
                        except Exception as error:
                            print(error)
                            st.session_state.auth_warning = 'Connection error. Please try again later.'

            st.markdown('<div class="auth-links">', unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Forgot Password?", key="forgot_password_link"):
                    st.session_state.auth_mode = 'reset'
                    st.rerun()
            with col_b:
                if st.button("Create Account", key="create_account_link"):
                    st.session_state.auth_mode = 'signup'
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        elif st.session_state.auth_mode == 'signup':
            st.markdown('<h1 class="auth-title">Create Account</h1>', unsafe_allow_html=True)
            st.markdown('<p class="form-description">Join us today! Create your account in just a few steps.</p>',
                        unsafe_allow_html=True)
            with st.form("sign_up_form", clear_on_submit=False):
                email = st.text_input("Email Address", placeholder="Enter your email", label_visibility="collapsed")
                password = st.text_input("Password", type="password", placeholder="Choose a strong password (min 6 characters)", label_visibility="collapsed")
                submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                if submit:
                    if not email or not password:
                        st.error("Please fill in all fields")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters long")
                    else:
                        try:
                            with st.spinner("Creating your account..."):
                                id_token = create_user_with_email_and_password(email, password)['idToken']
                                user_info = get_account_info(id_token)["users"][0]
                                send_email_verification(id_token)
                                st.session_state.auth_success = 'Account created! Check your inbox to verify your email.'
                                log_user_activity(user_info['localId'], "account_creation", {"email": user_info['email']})
                        except requests.exceptions.HTTPError as error:
                            error_message = json.loads(error.args[1])['error']['message']
                            if error_message == "EMAIL_EXISTS":
                                st.session_state.auth_warning = 'This email is already registered. Try signing in instead.'
                            elif error_message in {"INVALID_EMAIL", "INVALID_PASSWORD", "MISSING_PASSWORD", "MISSING_EMAIL", "WEAK_PASSWORD"}:
                                st.session_state.auth_warning = 'Please check your email format and password strength.'
                            else:
                                st.session_state.auth_warning = 'Something went wrong. Please try again later.'
                        except Exception as error:
                            print(error)
                            st.session_state.auth_warning = 'Connection error. Please try again later.'

        elif st.session_state.auth_mode == 'reset':
            st.markdown('<h1 class="auth-title">Reset Password</h1>', unsafe_allow_html=True)
            st.markdown('<p class="form-description">Enter your email address and we\'ll send you a link to reset your password.</p>',
                        unsafe_allow_html=True)
            with st.form("reset_password_form", clear_on_submit=False):
                email = st.text_input("Email Address", placeholder="Enter your registered email", label_visibility="collapsed")
                submit = st.form_submit_button("Send Reset Link", use_container_width=True, type="primary")
                if submit:
                    if not email:
                        st.error("Please enter your email address")
                    else:
                        try:
                            with st.spinner("Sending reset link..."):
                                send_password_reset_email(email)
                                st.session_state.auth_success = 'Password reset link sent! Check your email inbox.'
                        except requests.exceptions.HTTPError as error:
                            error_message = json.loads(error.args[1])['error']['message']
                            if error_message in {"MISSING_EMAIL", "INVALID_EMAIL", "EMAIL_NOT_FOUND"}:
                                st.session_state.auth_warning = 'Please enter a valid registered email address.'
                            else:
                                st.session_state.auth_warning = 'Something went wrong. Please try again later.'
                        except Exception:
                            st.session_state.auth_warning = 'Connection error. Please try again later.'

        if st.session_state.get('auth_warning'):
            st.error(st.session_state.auth_warning)
            st.session_state.auth_warning = ""
        if st.session_state.get('auth_success'):
            st.success(st.session_state.auth_success)
            st.session_state.auth_success = ""
        st.markdown('</div>', unsafe_allow_html=True)

def create_admin_interface():
    if not st.session_state.get('user_info') or not st.session_state.get('id_token'):
        st.error("Session expired. Please log in again.")
        return
    if not st.session_state.get('is_admin', False):
        st.warning("You do not have admin access.")
        return

    try:
        db = firestore.client()
    except Exception as e:
        st.error(f"Failed to initialize Firestore client: {str(e)}")
        return

    st.markdown("### Admin Panel")
    tab1, tab2, tab3, tab4 = st.tabs(["User Management", "Document Management", "Activity Logs", "Chat History"])

    with tab1:
        st.markdown("#### User Management")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Grant Admin Access**")
            admin_email = st.text_input("Enter email to grant admin access", key="admin_email_input")
            if st.button("Grant Admin Access", key="grant_admin_btn"):
                if admin_email.strip():
                    try:
                        set_admin_user(admin_email.strip())
                        st.success(f"Admin access granted to {admin_email}")
                        log_user_activity(
                            st.session_state.user_info['localId'],
                            "grant_admin",
                            {"email": admin_email}
                        )
                    except Exception as e:
                        st.error(f"Error granting admin access: {str(e)}")
                else:
                    st.warning("Please enter a valid email address.")
        with col2:
            st.markdown("**Revoke Admin Access**")
            revoke_email = st.text_input("Enter email to revoke admin access", key="revoke_email_input")
            if st.button("Revoke Admin Access", key="revoke_admin_btn"):
                if revoke_email.strip():
                    try:
                        user = auth.get_user_by_email(revoke_email.strip())
                        auth.set_custom_user_claims(user.uid, {"admin": False})
                        st.success(f"Admin access revoked for {revoke_email}.")
                        log_user_activity(
                            st.session_state.user_info['localId'],
                            "revoke_admin",
                            {"email": revoke_email}
                        )
                    except auth.UserNotFoundError:
                        st.error("Error: User not found or invalid email.")
                    except Exception as e:
                        st.error(f"Error revoking admin access: {str(e)}")
                else:
                    st.warning("Please enter a valid email address.")

    with tab2:
        st.markdown("#### Knowledge Hub Management")
        st.info(
            "📈 Documents are processed with Sentence Window, Auto-merging, and Qdrant-hosted embeddings in the Knowledge Hub.")
        document_text, _ = create_admin_input_interface()
        has_documents = "sentence_window_index" in st.session_state or "automerging_index" in st.session_state
        if has_documents:
            st.success("📄 Knowledge Hub indexed!")
            col1, col2 = st.columns(2)
            with col1:
                st.info("Current indexes: Sentence Window, Auto-merging, Qdrant")
            with col2:
                if st.button("Kill Switch", key="admin_clear_indexes"):
                    try:
                        clear_indexes()
                        st.session_state.admin_uploaded_document = ''
                        st.session_state.admin_context_text = ''
                        st.success("✅ Knowledge Hub cleared.")
                        log_user_activity(
                            st.session_state.user_info['localId'],
                            "clear_knowledge_hub",
                            {"action": "cleared_knowledge_hub"}
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error clearing Knowledge Hub: {str(e)}")

        st.markdown("---")
        st.markdown("#### Knowledge Hub")
        st.info("This contains supplemental information used when questions go beyond the main document")
        hub_doc = st.file_uploader(
            "Upload to Knowledge Hub",
            type=["txt", "pdf"],
            key="knowledge_hub_upload"
        )
        if hub_doc and st.button("Add to Knowledge Hub"):
            try:
                with st.spinner("Adding to knowledge hub..."):
                    if index_knowledge_hub_document(hub_doc):
                        st.success("✅ Added to knowledge hub!")
                        log_user_activity(
                            st.session_state.user_info['localId'],
                            "index_knowledge_hub",
                            {"document_name": hub_doc.name}
                        )
                    else:
                        st.error("Failed to add to knowledge hub")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with tab3:
        st.markdown("#### Activity Logs")
        try:
            filter_email = st.text_input("Filter by Email", key="log_filter_email")
            limit = st.slider("Number of logs to fetch", min_value=10, max_value=200, value=50, key="admin_log_limit")
            if st.button("Refresh Admin Logs", key="refresh_admin_logs"):
                query = db.collection("admin_logs").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)
                logs = query.get()
                log_data = []
                for log in logs:
                    log_dict = log.to_dict()
                    email = log_dict.get('email', 'Unknown')
                    if not filter_email or filter_email.lower() in email.lower():
                        log_data.append({
                            "User ID": log_dict.get('user_id', 'Unknown'),
                            "Email": email,
                            "Action": log_dict.get('action', 'Unknown'),
                            "Details": str(log_dict.get('details', '')),
                            "Timestamp": log_dict.get('timestamp', 'Unknown')
                        })
                if log_data:
                    df = pd.DataFrame(log_data)
                    st.dataframe(df, use_container_width=True, height=400)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Admin Logs as CSV",
                        data=csv,
                        file_name=f"admin_logs_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No admin logs found.")
        except Exception as e:
            st.error(f"Error fetching admin logs: {str(e)}")

        st.markdown("**User Activity Logs**")
        try:
            user_log_filter_email = st.text_input("Filter User Logs by Email", key="user_log_filter_email")
            user_limit = st.slider("Number of user logs to fetch", min_value=10, max_value=200, value=50, key="user_log_limit")
            if st.button("Refresh User Logs", key="refresh_user_logs"):
                user_log_data = []
                all_users = db.collection("user_logs").stream()
                for user in all_users:
                    user_id = user.id
                    user_logs = db.collection("user_logs").document(user_id).collection("logs").order_by(
                        "timestamp", direction=firestore.Query.DESCENDING).limit(user_limit).get()
                    for log in user_logs:
                        log_dict = log.to_dict()
                        email = log_dict.get('email', 'Unknown')
                        if not user_log_filter_email or user_log_filter_email.lower() in email.lower():
                            user_log_data.append({
                                "User ID": user_id,
                                "Email": email,
                                "Action": log_dict.get('action', 'Unknown'),
                                "Details": str(log_dict.get('details', '')),
                                "Timestamp": log_dict.get('timestamp', 'Unknown')
                            })
                if user_log_data:
                    df = pd.DataFrame(user_log_data)
                    st.dataframe(df, use_container_width=True, height=400)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download User Logs as CSV",
                        data=csv,
                        file_name=f"user_logs_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No user logs found.")
        except Exception as e:
            st.error(f"Error fetching user logs: {str(e)}")

    with tab4:
        st.markdown("#### Chat History")
        try:
            chat_filter_email = st.text_input("Filter Chat History by Email", key="chat_filter_email")
            chat_limit = st.slider("Number of chats to fetch", min_value=10, max_value=200, value=50, key="chat_limit")
            if st.button("Refresh Chat History", key="refresh_chat_history"):
                chat_data = []
                all_users = db.collection("chat_history").stream()
                for user in all_users:
                    user_id = user.id
                    user_chats = db.collection("chat_history").document(user_id).collection("chats").order_by(
                        "timestamp", direction=firestore.Query.DESCENDING).limit(chat_limit).get()
                    for chat in user_chats:
                        chat_dict = chat.to_dict()
                        email = chat_dict.get('email', 'Unknown')
                        if not chat_filter_email or chat_filter_email.lower() in email.lower():
                            chat_data.append({
                                "User ID": user_id,
                                "Email": email,
                                "Query": chat_dict.get('query', 'Unknown'),
                                "Response": chat_dict.get('response', 'Unknown'),
                                "Language": chat_dict.get('language', 'Unknown'),
                                "Type": chat_dict.get('type', 'Unknown'),
                                "Timestamp": chat_dict.get('timestamp', 'Unknown')
                            })
                if chat_data:
                    df = pd.DataFrame(chat_data)
                    st.dataframe(df, use_container_width=True, height=400)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Chat History as CSV",
                        data=csv,
                        file_name=f"chat_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No chat history found.")
        except Exception as e:
            st.error(f"Error fetching chat history: {str(e)}")

def create_admin_input_interface() -> Tuple[Optional[str], Optional[str]]:
    if not st.session_state.get('is_admin', False):
        st.warning("Only admins can access this section.")
        return None, None

    st.markdown("### Knowledge Hub Management (Admin Only)")
    st.markdown("Upload documents to the Knowledge Hub for Ustaad Jee's RAG functionality.")
    uploaded_document = st.file_uploader(
        "Upload Document to Knowledge Hub",
        type=["txt", "pdf"],
        help="Upload a text or PDF file to be indexed in the Knowledge Hub.",
        key="admin_knowledge_hub_upload"
    )
    document_text = None
    if uploaded_document:
        try:
            with st.spinner("Processing document..."):
                document_chunks = cached_parse_document(uploaded_document)
                document_text = "\n".join(document_chunks) if isinstance(document_chunks, list) else document_chunks
                document_text = bleach.clean(document_text, tags=[], strip=True)
                st.session_state.admin_uploaded_document = document_text
                if st.session_state.get('user_info'):
                    log_user_activity(
                        st.session_state.user_info['localId'],
                        "upload_document",
                        {"document_length": len(document_text), "document_name": uploaded_document.name}
                    )
                st.success(f"Document '{uploaded_document.name}' processed successfully!")
        except Exception as e:
            st.error(f"Failed to process document: {str(e)}")
            return None, None

    st.markdown("#### Or Enter Text Directly")
    manual_text = st.text_area(
        "Paste Document Text for Knowledge Hub",
        height=300,
        placeholder="Paste your document text here to add to the Knowledge Hub...",
        help="Type or paste the document text for indexing.",
        key="admin_knowledge_hub_text_input"
    )
    if manual_text.strip():
        document_text = bleach.clean(manual_text.strip(), tags=[], strip=True)
        st.session_state.admin_uploaded_document = document_text

    final_document_text = st.session_state.get('admin_uploaded_document', '')
    if final_document_text:
        with st.expander("Document Preview"):
            st.text_area(
                "Document Content",
                value=final_document_text[:1000] + "..." if len(final_document_text) > 1000 else final_document_text,
                height=200,
                disabled=True
            )
            st.caption(f"Document length: {len(final_document_text)} characters")

    if final_document_text and st.button("Index Document to Knowledge Hub", key="admin_index_doc_btn", type="primary"):
        try:
            success = cached_index_document(final_document_text)
            if success:
                st.success("✅ Document indexed successfully to Knowledge Hub!")
                st.session_state.index_version += 1
                if st.session_state.get('user_info'):
                    log_user_activity(
                        st.session_state.user_info['localId'],
                        "index_document",
                        {"document_length": len(final_document_text)}
                    )
                st.rerun()
        except Exception as e:
            st.error(f"Error during indexing: {str(e)}")

    st.markdown("#### Knowledge Hub Status")
    if st.session_state.get("vectorstore"):
        st.success("✅ Knowledge Hub is active and ready for RAG queries")
        if st.session_state.get("sentence_window_index") and st.session_state.get("automerging_index"):
            st.info("🔍 Advanced RAG indexes are available (Sentence Window + Auto-merging)")
        else:
            st.info("📊 Basic Knowledge Hub vector store is available")
    else:
        st.warning("⚠️ No Knowledge Hub found. Upload and index documents to enable RAG functionality.")

    if st.session_state.get("vectorstore") and st.button("Clear Knowledge Hub", key="clear_knowledge_hub", type="secondary"):
        try:
            clear_indexes()
            st.session_state.admin_uploaded_document = ''
            st.session_state.admin_context_text = ''
            st.success("Knowledge Hub cleared successfully!")
            log_user_activity(
                st.session_state.user_info['localId'],
                "clear_knowledge_hub",
                {"action": "cleared_knowledge_hub"}
            )
            st.session_state.index_version += 1
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing Knowledge Hub: {str(e)}")

    return document_text, None

def create_llm_configuration_section():
    st.markdown("### Configure Ustaad Jee's Brain 🧠")
    if not st.session_state.get('is_admin', False):
        status_color = {"Connected": "🟢", "Not Connected": "🟡", "Failed": "🔴"}
        st.info(f"Status: {status_color.get(st.session_state.connection_status, '🟡')} {st.session_state.connection_status}")
        st.markdown("Using OpenAI GPT-4o-mini (default configuration).")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        provider_options = {
            "OpenAI (GPT)": LLMProvider.OPENAI,
            "Claude": LLMProvider.CLAUDE,
            "DeepSeek": LLMProvider.DEEPSEEK,
            "OpenRouter": LLMProvider.OPENROUTER,
            "Local LLM": LLMProvider.LOCAL
        }
        selected_provider = st.selectbox("Select Brain", list(provider_options.keys()), help="Choose the AI provider for Ustaad Jee.")
        provider_enum = provider_options[selected_provider]
        available_models = AppConfig.MODELS.get(selected_provider, [])
        selected_model = st.selectbox("Select Model", available_models, help="Pick a model for Ustaad Jee's brain.")
    with col2:
        if selected_provider != "Local LLM":
            api_key = st.text_input("API Key", type="password", placeholder="Enter your API key", help="Required to connect to the AI provider.")
        else:
            api_key = None
            base_url = st.text_input("Local Server URL", value="http://localhost:11434", help="URL where the local LLM is running.")
        if st.button("Test Connection", type="secondary", key="test_connection_btn"):
            try:
                config = {"model": selected_model}
                if selected_provider != "Local LLM":
                    if not api_key:
                        st.error("API key is required!")
                        return None
                    config["api_key"] = api_key
                else:
                    config["base_url"] = base_url
                with st.spinner("Testing connection..."):
                    test_llm = LLMWrapper(provider_enum, **config)
                    if test_llm.client:
                        test_response = test_llm.generate("Hello", max_tokens=10)
                        if test_response:
                            st.success("Connection successful!")
                            st.session_state.connection_status = "Connected"
                            st.session_state.llm = test_llm
                            log_user_activity(st.session_state.user_info['localId'], "llm_connection",
                                              {"provider": selected_provider, "model": selected_model})
                        else:
                            st.error("Connection failed!")
                            st.session_state.connection_status = "Failed"
                    else:
                        st.session_state.connection_status = "Failed"
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.connection_status = "Failed"

    if st.session_state.llm is None and st.session_state.connection_status != "Failed":
        try:
            config = {"model": selected_model}
            if selected_provider != "Local LLM":
                if api_key:
                    config["api_key"] = api_key
                    st.session_state.llm = LLMWrapper(provider_enum, **config)
            else:
                config["base_url"] = base_url
                st.session_state.llm = LLMWrapper(provider_enum, **config)
        except:
            pass

    status_color = {"Connected": "🟢", "Not Connected": "🟡", "Failed": "🔴"}
    st.info(f"Status: {status_color.get(st.session_state.connection_status, '🟡')} {st.session_state.connection_status}")

def create_glossary_section():
    st.markdown("### Ustaad Jee's Glossary 📚")
    st.markdown("Add technical terms for Ustaad Jee to use!")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        english_term = st.text_input("English Term", key="eng_term_input")
        english_term = bleach.clean(english_term) if english_term else ""
    with col2:
        urdu_term = st.text_input("Urdu/Roman Urdu Term", key="urdu_term_input")
        urdu_term = bleach.clean(urdu_term) if urdu_term else ""
    with col3:
        st.write("")
        language_type = st.selectbox("Term Language", ["Urdu", "Roman Urdu"], key="term_language")
        if st.button("+ Add", key="add_term_btn"):
            if english_term and urdu_term:
                # Create a dictionary for the term with all language variants
                term_dict = {
                    "English": english_term,
                    "Urdu": urdu_term if language_type == "Urdu" else "",
                    "Roman Urdu": urdu_term if language_type == "Roman Urdu" else ""
                }
                st.session_state.glossary[english_term] = term_dict
                st.success(f"Added: {english_term} → {urdu_term} ({language_type})")
                log_user_activity(st.session_state.user_info['localId'], "add_glossary_term",
                                  {"english_term": english_term, "urdu_term": urdu_term, "language": language_type})
                st.rerun()
            else:
                st.warning("Both terms are required!")

    if st.session_state.glossary:
        # Convert any legacy string entries to dictionary format
        for eng, term in list(st.session_state.glossary.items()):
            if isinstance(term, str):
                st.session_state.glossary[eng] = {
                    "English": eng,
                    "Urdu": term if term else "",
                    "Roman Urdu": ""
                }

        glossary_df = pd.DataFrame([
            {
                "English Term": eng,
                "Urdu Term": term_dict.get("Urdu", term_dict.get("Roman Urdu", ""))
            }
            for eng, term_dict in st.session_state.glossary.items()
        ])
        csv_buffer = io.StringIO()
        glossary_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Glossary",
            data=csv_buffer.getvalue(),
            file_name="ustaad_jee_glossary.csv",
            mime="text/csv",
            key="download_glossary"
        )

    st.markdown("**Upload Glossary**")
    merge_option = st.checkbox("Merge with existing terms (uncheck to replace)", value=True)
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        key=f"glossary_upload_{len(st.session_state.glossary)}",
        help="CSV must have 'English Term' and 'Urdu Term' columns."
    )
    if uploaded_file is not None:
        try:
            file_size = uploaded_file.size if hasattr(uploaded_file, 'size') else 0
            if file_size > 1024 * 1024:
                st.error("File too large (max 1MB)!")
                return
            uploaded_df = pd.read_csv(uploaded_file)
            required_columns = ["English Term", "Urdu Term"]
            if not all(col in uploaded_df.columns for col in required_columns):
                st.error(f"CSV must have columns: {', '.join(required_columns)}")
                st.info(f"Found columns: {', '.join(uploaded_df.columns)}")
                return
            if len(uploaded_df) > 1000:
                st.error("Too many terms (max 1000)!")
                return
            if st.button("Upload Terms", key="process_upload_btn", type="primary"):
                try:
                    if not merge_option:
                        st.session_state.glossary = {}
                    added_count = 0
                    skipped_count = 0
                    for _, row in uploaded_df.iterrows():
                        english_term = str(row["English Term"]).strip()
                        urdu_term = str(row["Urdu Term"]).strip()
                        english_term = bleach.clean(english_term)
                        urdu_term = bleach.clean(urdu_term)
                        if english_term and urdu_term and english_term != 'nan' and urdu_term != 'nan':
                            term_dict = {
                                "English": english_term,
                                "Urdu": urdu_term,
                                "Roman Urdu": ""
                            }
                            if english_term not in st.session_state.glossary or not merge_option:
                                st.session_state.glossary[english_term] = term_dict
                                added_count += 1
                            else:
                                skipped_count += 1
                    if added_count > 0:
                        st.success(f"Added {added_count} new terms!")
                        log_user_activity(st.session_state.user_info['localId'], "upload_glossary",
                                          {"terms_added": added_count, "terms_skipped": skipped_count})
                    if skipped_count > 0:
                        st.info(f"Skipped {skipped_count} existing terms.")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.info(f"Ready to upload {len(uploaded_df)} terms. Click 'Upload Terms'!")
                with st.expander("Preview first 5 terms"):
                    st.dataframe(uploaded_df.head())
        except Exception as e:
            st.error(f"Failed to read CSV: {str(e)}")
            st.info("Ensure CSV has 'English Term' and 'Urdu Term' columns!")

    if st.session_state.glossary:
        st.markdown("**Current Glossary**")
        for idx, (eng, term_dict) in enumerate(st.session_state.glossary.items()):
            col_eng, col_urdu, col_action = st.columns([2, 2, 1])
            with col_eng:
                st.write(f"**{eng}**")
            with col_urdu:
                term_display = term_dict.get("Urdu", term_dict.get("Roman Urdu", ""))
                st.write(term_display)
            with col_action:
                if st.button("Remove", key=f"remove_{idx}", help="Remove this term"):
                    del st.session_state.glossary[eng]
                    st.success(f"Removed {eng}!")
                    log_user_activity(st.session_state.user_info['localId'], "remove_glossary_term",
                                      {"english_term": eng})
                    st.rerun()
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("Clear All", type="secondary", key="clear_all_btn"):
                st.session_state.glossary = {}
                st.info("Glossary cleared!")
                log_user_activity(st.session_state.user_info['localId'], "clear_glossary", {})
                st.rerun()
        with col2:
            st.write(f"**Total: {len(st.session_state.glossary)} terms**")

class FeedbackDB:
    def __init__(self):
        pass

    def store_feedback(self, question: str, response: str, language: str, rating: str):
        feedback_entry = {
            "question": bleach.clean(question),
            "response": bleach.clean(response),
            "language": bleach.clean(language),
            "rating": bleach.clean(rating),
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        try:
            user_id = st.session_state.user_info['localId']
            db.collection("feedback").document(user_id).collection("entries").add(feedback_entry)
            log_user_activity(user_id, "submit_feedback", {"rating": rating})
        except Exception as e:
            st.error(f"Error saving feedback: {str(e)}")


def create_input_interface(admin_only: bool = False) -> Tuple[Optional[str], str]:
    MAX_DOCUMENTS = 3  # Maximum number of documents a user can upload

    # Initialize session state for documents
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    if 'document_texts' not in st.session_state:
        st.session_state.document_texts = []

    # Document upload
    st.markdown("### Upload Documents")
    uploaded_documents = st.file_uploader(
        "Upload Documents (Max 3)",
        type=["txt", "pdf"],
        help="Upload up to 3 text or PDF files",
        key="main_document_upload",
        accept_multiple_files=True
    )

    # Process new uploads
    new_documents = []
    if uploaded_documents:
        # Limit to first 3 documents
        for doc in uploaded_documents[:MAX_DOCUMENTS]:
            if doc not in st.session_state.uploaded_documents:
                try:
                    doc.seek(0)
                    document_chunks = cached_parse_document(doc)
                    document_text = "\n".join(document_chunks) if isinstance(document_chunks, list) else document_chunks
                    if not document_text.strip():
                        st.error(f"No extractable text in '{doc.name}'")
                    else:
                        document_text = bleach.clean(document_text, tags=[], strip=True)
                        new_documents.append((doc, document_text))
                except Exception as e:
                    st.error(f"Failed to parse '{doc.name}': {str(e)}")

    # Update session state with new documents
    for doc, doc_text in new_documents:
        st.session_state.uploaded_documents.append(doc)
        st.session_state.document_texts.append(doc_text)
        if st.session_state.get('user_info'):
            log_user_activity(
                st.session_state.user_info['localId'],
                "upload_document",
                {"document_length": len(doc_text), "document_name": doc.name}
            )
        st.success(f"Uploaded '{doc.name}' successfully!")

    # Display uploaded documents
    if st.session_state.uploaded_documents:
        st.markdown("**Uploaded Documents**")
        for i, (doc, doc_text) in enumerate(zip(st.session_state.uploaded_documents, st.session_state.document_texts)):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f"Document {i+1}: {doc.name}")
                with st.expander(f"View Document {i+1} Content"):
                    preview = doc_text[:1000] + "..." if len(doc_text) > 1000 else doc_text
                    st.text_area(f"Content {i+1}", value=preview, height=200, disabled=True)
            with col2:
                if st.button("❌", key=f"remove_doc_{i}"):
                    del st.session_state.uploaded_documents[i]
                    del st.session_state.document_texts[i]
                    st.rerun()

    # Text area for manual document input
    st.markdown("### Or Enter Text Directly")
    manual_text = st.text_area(
        "Paste Document Text",
        height=200,
        placeholder="Type or paste document text here...",
        help="Manually add document content",
        key="main_document_text_input"
    )
    if manual_text.strip():
        clean_text = bleach.clean(manual_text.strip(), tags=[], strip=True)
        if clean_text and clean_text not in st.session_state.document_texts:
            if len(st.session_state.document_texts) < MAX_DOCUMENTS:
                st.session_state.document_texts.append(clean_text)
                st.success("Added manual document text!")
                log_user_activity(
                    st.session_state.user_info['localId'],
                    "add_manual_document",
                    {"document_length": len(clean_text)}
                )
            else:
                st.warning(f"Maximum {MAX_DOCUMENTS} documents reached")

    # Clear Documents Button
    if st.session_state.document_texts or st.session_state.context_text:
        if st.button("🧹 Clear Documents", key="clear_docs_btn", type="secondary"):
            st.session_state.uploaded_documents = []
            st.session_state.document_texts = []
            st.session_state.context_text = ""
            st.success("Documents cleared!")
            st.rerun()

    # Context upload
    st.markdown("### Additional Context (Optional)")
    uploaded_context = st.file_uploader(
        "Upload Context",
        type=["txt"],
        help="Upload a text file with additional context.",
        key="main_context_upload"
    )
    if uploaded_context:
        try:
            uploaded_context.seek(0)
            context_chunks = cached_parse_document(uploaded_context)
            context_text = "\n".join(context_chunks) if isinstance(context_chunks, list) else context_chunks
            context_text = bleach.clean(context_text, tags=[], strip=True)
            st.session_state.context_text = context_text
            if st.session_state.get('user_info'):
                log_user_activity(
                    st.session_state.user_info['localId'],
                    "upload_context",
                    {"context_length": len(context_text)}
                )
            st.success("Context uploaded successfully!")
        except Exception as e:
            st.error(f"Failed to parse context: {str(e)}")
            context_text = ""

    # Context text area
    context_text = st.text_area(
        "Paste or Edit Context",
        value=st.session_state.context_text,
        height=200,
        placeholder="Type or paste additional context...",
        help="Extra info for Ustaad Jee.",
        key="main_context_text_input"
    )
    context_text = bleach.clean(context_text.strip(), tags=[], strip=True) if context_text else ""
    st.session_state.context_text = context_text

    return st.session_state.document_texts, context_text

def create_translation_and_chat_interface(document_texts: List[str], context_text: Optional[str] = None) -> None:
    if not st.session_state.get('user_info'):
        st.warning("Please sign in to use Ustaad Jee!")
        return

    # Add CSS for chat container scrolling and thinner buttons
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"] > div[style*="height: 550px"] {
        max-height: 550px;
        overflow-y: auto;
        padding: 10px;
    }
    .thinner-button button {
        max-width: 200px;  /* Limit button width */
        width: 100%;
        margin: 0 auto 10px auto;  /* Center with vertical spacing */
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h3>Interact with Ustaad Jee</h3>", unsafe_allow_html=True)

    # Initialize feedback system if not exists
    if 'feedback_db' not in st.session_state:
        st.session_state.feedback_db = FeedbackDB()

    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        try:
            # Load existing chat history from Firestore
            user_id = st.session_state.user_info.get('localId', 'unknown_user')
            db = firestore.client()
            chats_ref = db.collection("chat_history").document(user_id).collection("chats")
            docs = chats_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20).stream()

            for doc in docs:
                chat_data = doc.to_dict()
                # Convert Firestore timestamp to Python timestamp if needed
                if hasattr(chat_data['timestamp'], 'timestamp'):
                    chat_data['timestamp'] = chat_data['timestamp'].timestamp()
                st.session_state.chat_history.append(chat_data)

            # Sort by timestamp (oldest first)
            st.session_state.chat_history.sort(key=lambda x: x.get('timestamp', 0))
        except Exception as e:
            st.error(f"Error loading chat history: {str(e)}")
            st.session_state.chat_history = []

    # Three-column layout for select boxes and buttons
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        translate_language = st.selectbox(
            label="Translation Language",
            options=["Urdu", "English", "Roman Urdu"],
            key="translate_language",
            help="Select the language for translation."
        )
    with col2:
        chat_language = st.selectbox(
            label="Chat Language",
            options=["English", "Urdu", "Roman Urdu"],
            key="chat_language",
            help="Select the language for responses."
        )
    with col3:
        st.markdown('<div class="thinner-button">', unsafe_allow_html=True)
        if st.button("Translate Document", type="primary", help="Translate the entire document", key="translate_button", use_container_width=True):
            if document_texts and st.session_state.llm:
                with st.spinner("Translating..."):
                    try:
                        # Combine all document texts
                        combined_text = "\n\n".join(document_texts)

                        user_id = st.session_state.user_info.get('localId', 'unknown_user')
                        glossary = st.session_state.get("glossary", {})
                        glossary_section = "\n".join([
                            f"English: {translations.get('English', term)}, Urdu: {translations.get('Urdu', '')}, Roman Urdu: {translations.get('Roman Urdu', '')}"
                            for term, translations in glossary.items()
                        ]) if glossary else "No glossary terms available."

                        last_6_exchanges = "\n".join(
                            f"Q: {chat['query']}\nA: {chat['response']}"
                            for chat in st.session_state.chat_history[-6:]
                        ) if st.session_state.chat_history else ""

                        context_section = f"Conversation Context:\n{last_6_exchanges}\nSupplementary Info:\n{context_text or ''}"

                        prompt_template = (
                            AppConfig.URDU_TRANSLATION_PROMPT if translate_language == "Urdu" else
                            AppConfig.ROMAN_URDU_TRANSLATION_PROMPT if translate_language == "Roman Urdu" else
                            AppConfig.DIRECT_PROMPT
                        )

                        translation_prompt = prompt_template.format(
                            language=translate_language,
                            conversation_history=last_6_exchanges,
                            glossary_section=glossary_section,
                            glossary_translation_rules=AppConfig.GLOSSARY_TRANSLATION_RULES,
                            context_section=context_section,
                            text=combined_text
                        )

                        translation = st.session_state.llm.generate(
                            prompt=translation_prompt,
                            max_tokens=1000,
                            temperature=0.3
                        )

                        if translate_language == "Urdu":
                            translation = f"\u200F{translation}"

                        chat_entry = {
                            "query": "Translation Request",
                            "response": translation,
                            "language": translate_language,
                            "type": "translation",
                            "timestamp": time.time()
                        }

                        st.session_state.chat_history.append(chat_entry)
                        store_chat_history(user_id, chat_entry)
                        log_user_activity(user_id, "translation",
                                          {"language": translate_language, "document_length": len(combined_text)})
                        st.rerun()
                    except Exception as e:
                        error_message = f"Error: {str(e)}"
                        chat_entry = {
                            "query": "Translation Request",
                            "response": error_message,
                            "language": translate_language,
                            "type": "translation",
                            "timestamp": time.time()
                        }
                        user_id = st.session_state.user_info.get('localId', 'unknown_user')
                        store_chat_history(user_id, chat_entry)
                        st.error(error_message)

        # Single download button for direct download
        translation_text = next(
            (chat['response'] for chat in reversed(st.session_state.chat_history)
             if chat['type'] == 'translation'), ""
        )
        if translation_text:
            try:
                # Apply RTL mark for Urdu text
                if translate_language == "Urdu":
                    translation_text = f"\u200F{translation_text}"

                # Encode as UTF-8 to handle Unicode characters
                txt_bytes = translation_text.encode('utf-8')
                st.download_button(
                    label="download",
                    data=txt_bytes,
                    file_name=f"ustaad_translation_{translate_language}.txt",
                    mime="text/plain",
                    key="download_translation",
                    help="Download the latest translation as a text file",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating TXT: {str(e)}")
        else:
            st.button(
                "📥 Download Translation as TXT",
                key="download_translation",
                help="No translation available to download",
                disabled=True,
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Ustaad Jee's Chat**")

    # Track chat history length for auto-scroll
    if 'last_chat_length' not in st.session_state:
        st.session_state.last_chat_length = 0

    current_chat_length = len(st.session_state.chat_history)
    should_scroll = current_chat_length > st.session_state.last_chat_length
    st.session_state.last_chat_length = current_chat_length

    def scroll_to_bottom():
        # Generate a unique dummy variable that changes each time
        dummy_var = int(time.time() * 1000)  # Use timestamp in milliseconds

        st.markdown(
            f"""
            <div id="end-of-chat-{dummy_var}" style="height: 1px;"></div>
            <script>
                // The dummy variable MUST be inside the JavaScript
                var dummy = {dummy_var};
                
                // Find and scroll the chat container
                var chatContainer = window.parent.document.querySelector('div[data-testid="stVerticalBlock"] > div[style*="height: 550px"]');
                if (chatContainer) {{
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }}
            </script>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.chat_history:
        with st.container(height=550, border=True):
            for i, chat in enumerate(st.session_state.chat_history):
                # Generate a unique key for each chat message
                chat_key = f"chat_{i}_{chat.get('timestamp', time.time())}"

                # Format timestamp
                chat_time = time.strftime("%H:%M", time.localtime(chat.get('timestamp', time.time())))

                # User message
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 5px;">
                        <div style="margin-right: 8px; font-size: 0.8rem; color: #555;">You at {chat_time}</div>
                        <div style="width: 24px; height: 24px; border-radius: 50%; background: #40C4FF; 
                                    display: flex; align-items: center; justify-content: center; color: white;">🤗</div>
                    </div>
                    <div style="background: linear-gradient(135deg, #40C4FF, #80D8FF); color: white; 
                                padding: 10px 15px; border-radius: 15px 5px 15px 15px; 
                                margin-left: auto; max-width: 70%; margin-bottom: 15px; float: right; clear: both;">
                        {chat['query']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Bot response
                bot_name = "Ustaad Jee" if chat['language'] in ["Urdu", "Roman Urdu"] else "Ustaad Jee (English)"
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="width: 24px; height: 24px; border-radius: 50%; background: #50E3C2; 
                                    display: flex; align-items: center; justify-content: center; color: white; margin-right: 8px;">🪄</div>
                        <div style="font-size: 0.8rem; color: #555;">{bot_name} at {chat_time}</div>
                    </div>
                    <div style="background: linear-gradient(135deg, #00B7EB, #50E3C2); color: white; 
                                padding: 10px 15px; border-radius: 5px 15px 15px 15px; 
                                margin-right: auto; max-width: 70%; margin-bottom: 15px; float: left; clear: both;">
                        {chat['response']}
                    </div>
                    <div style="clear: both;"></div>
                    """,
                    unsafe_allow_html=True
                )

                # Feedback system
                feedback_key = f"feedback_{chat_key}"
                if feedback_key not in st.session_state:
                    st.session_state[feedback_key] = 0

                with st.expander("Rate this response", expanded=False):
                    current_rating = st.session_state.get(feedback_key, 0) or 0
                    cols = st.columns(5)

                    for star_num in range(1, 6):
                        with cols[star_num - 1]:
                            if st.button(
                                    "⭐" if star_num <= current_rating else "☆",
                                    key=f"{feedback_key}_star_{star_num}",
                                    help=f"{star_num} star{'s' if star_num > 1 else ''}",
                                    use_container_width=True
                            ):
                                st.session_state[feedback_key] = star_num
                                st.rerun()

                    if current_rating > 0:
                        if st.button("Submit Rating", key=f"{feedback_key}_submit", type="primary"):
                            try:
                                st.session_state.feedback_db.store_feedback(
                                    question=chat['query'],
                                    response=chat['response'],
                                    language=chat['language'],
                                    rating=str(current_rating)
                                )
                                st.success("Thanks for your feedback!")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error submitting rating: {str(e)}")

                        if st.button("Clear Rating", key=f"{feedback_key}_clear"):
                            st.session_state[feedback_key] = 0
                            st.rerun()

                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("<hr style='border-top: 1px dashed #50E3C2; margin: 15px 0;'>", unsafe_allow_html=True)

            # Auto-scroll to bottom only when new messages are added
            if should_scroll:
                scroll_to_bottom()
    else:
        st.info("Ustaad Jee's Chat is empty. Ask a question to start!")

    st.markdown("**Ask Ustaad Jee**")
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([5, 2, 1])
        with col1:
            chat_query = st.text_input(
                "Type your question...",
                placeholder="Ask about your document or request an explanation...",
                label_visibility="collapsed",
                key="chat_input"
            )
        with col2:
            quick_action = st.selectbox(
                "Quick Actions",
                ["Quick Action", "Summarize", "Key Points", "Simplify", "Technical Terms"],
                key="quick_action_dropdown",
                label_visibility="collapsed"
            )
        with col3:
            send_btn = st.form_submit_button("Send", type="primary", use_container_width=True)

        if send_btn:
            if quick_action != "Quick Action":
                quick_queries = {
                    "Summarize": "Provide a concise summary of the key points in the user entered document",
                    "Key Points": "List the main concepts and key points in the user entered document",
                    "Simplify": "Explain this in simple terms without adding new information in the user entered document",
                    "Technical Terms": "List and explain the technical terms used in the user entered document"
                }
                processed_query = quick_queries[quick_action]
            else:
                processed_query = chat_query

            if processed_query:
                with st.spinner("Ustaad Jee is thinking..."):
                    try:
                        user_id = st.session_state.user_info.get('localId', 'unknown_user')
                        response = generate_response(
                            chat_language=chat_language,
                            query=processed_query,
                            context_text=context_text,
                            document_text=document_texts,
                            _glossary_version=st.session_state.get("glossary_version", 0)
                        )

                        chat_entry = {
                            "query": quick_action if quick_action != "Quick Action" else chat_query,
                            "response": response,
                            "language": chat_language,
                            "type": "quick_action" if quick_action != "Quick Action" else "question",
                            "timestamp": time.time()
                        }

                        st.session_state.chat_history.append(chat_entry)
                        store_chat_history(user_id, chat_entry)
                        log_user_activity(user_id, "chat_query",
                                          {"query": processed_query, "language": chat_language})
                        st.rerun()
                    except Exception as e:
                        error_message = f"Error: {str(e)}"
                        chat_entry = {
                            "query": quick_action if quick_action != "Quick Action" else chat_query,
                            "response": error_message,
                            "language": chat_language,
                            "type": "quick_action" if quick_action != "Quick Action" else "question",
                            "timestamp": time.time()
                        }
                        user_id = st.session_state.user_info.get('localId', 'unknown_user')
                        store_chat_history(user_id, chat_entry)
                        st.error(error_message)
            else:
                st.warning("Please enter a question or select an action!")
def create_usage_tips():
    with st.expander("How to Use Ustaad Jee"):
        st.markdown("""
        ### Learn with Ustaad Jee:
        - **Document**: Break long documents into smaller chunks (1000-2000 words).
        - **Glossary**: Add technical terms (e.g., computer terms). Save or load terms via CSV!
        - **Brain Selection**: Use GPT-4 or Claude for complex documents, DeepSeek for code, or Local Models for offline use.
        - **Chat**:
          - Use quick action buttons for summaries, key points, etc.
          - Scroll through Ustaad Jee's chat history.
          - Rate each response with 1–5 stars.
          - Save your chat as a text file.
          - Clear chat or reset everything.
        - **Languages**: Ustaad Jee can teach in English, Urdu, or Roman Urdu!
        - **Translation**: Click 'Start Translation' to translate the entire document.
        """)

def create_sample_data():
    with st.expander("Sample Data"):
        st.markdown("### Sample Document:")
        sample_doc = """
        Authentication System Overview
        This system implements OAuth 2.0 authentication with JWT tokens. The authentication flow begins when a user attempts to access a protected resource. The system validates the user's credentials against the database and generates a JSON Web Token (JWT) upon successful authentication.
        The JWT contains encoded user information and has an expiration time of 24 hours. When making API requests, the client must include the JWT in the Authorization header using the Bearer token format.
        Security measures include password hashing using bcrypt, rate limiting to prevent brute force attacks, and HTTPS encryption for all communications.
        """
        if st.button("Use Sample Document", key="use_sample_doc_btn"):
            if len(st.session_state.document_texts) < 3:
                st.session_state.document_texts.append(sample_doc)
                st.success("Sample document loaded!")
                log_user_activity(st.session_state.user_info['localId'], "load_sample_document", {})
                st.rerun()
            else:
                st.warning("Maximum 3 documents reached")

        st.markdown("### Sample Glossary:")
        st.write("Add these terms to Ustaad Jee's glossary:")
        sample_terms = {
            "Authentication": "تصديق",
            "JWT": "JSON ویب ٹوکن",
            "API": "ایپلیکیشن پروگرامنگ انٹرفیس",
            "Database": "ڈیٹابیس",
            "Encryption": "خفیہ کاری"
        }
        for eng, urdu in sample_terms.items():
            if st.button(f"+ Add: {eng} → {urdu}", key=f"sample_{eng}"):
                st.session_state.glossary[eng] = urdu
                st.success(f"Added: {eng}")
                log_user_activity(st.session_state.user_info['localId'], "add_sample_glossary_term",
                                  {"english_term": eng, "urdu_term": urdu})
                st.rerun()

def create_footer():
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #333333; font-family: Roboto, sans-serif; font-size: 14px;'>Ustaad Jee's Knowledge Hub - RSM IS TESTING!</div>",
        unsafe_allow_html=True
    )

def main():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Aladin&display=swap');
        .stApp {
            background: linear-gradient(90deg, #ACECFF, #B0E0E6) !important;
            font-family: 'Roboto', sans-serif !important;
            color: #333333 !important;
            font-size: 16px !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Aladin', 'Matura MT Script Capitals', cursive !important;
            color: #00A0CF !important;
            font-weight: bold !important;
            text-shadow: none !important;
            -webkit-text-stroke: none !important;
        }
        .title-text {
            font-family: 'Aladin', 'Matura MT Script Capitals', cursive !important;
            font-size: 6rem !important;
            font-weight: bold !important;
            background: linear-gradient(90deg, #00B7EB, #50E3C2) !important;
            -webkit-background-clip: text !important;
            background-clip: text !important;
            color: transparent !important;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5) !important;
            -webkit-text-stroke: 1px rgba(255, 255, 255, 0.2) !important;
        }
        .sidebar .sidebar-content {
            background: rgba(255, 255, 255, 0.9) !important;
            border-right: 2px dashed #00B7EB !important;
            padding: 20px !important;
            box-shadow: 0 2px 10px rgba(0, 183, 235, 0.3) !important;
            color: #000000 !important;
        }
        .stButton>button {
            border-radius: 10px !important;
            border: 2px solid transparent !important;
            background: linear-gradient(135deg, #66B2FF, #99CCFF) !important;
            color: white !important;
            box-shadow: none !important;
            transition: all 0.3s ease !important;
            font-weight: 500 !important;
            padding: 8px 16px !important;
            position: relative !important;
            overflow: hidden !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(102, 178, 255, 0.3) !important;
        }
        .stButton>button:active {
            transform: translateY(0) !important;
        }
        .stTextInput>div>input, .stTextArea>div>textarea {
            border-radius: 10px !important;
            border: 2px solid #00B7EB !important;
            background: rgba(255, 255, 255, 0.1) !important;
            color: #000000 !important;
            transition: border-color 0.3s ease !important;
        }
        .stTextInput>div>input:focus, .stTextArea>div>textarea:focus {
            border-color: #50E3C2 !important;
            box-shadow: 0 0 8px rgba(80, 227, 194, 0.5) !important;
        }
        .stSelectbox>div>div {
            border-radius: 10px !important;
            border: 2px solid #00B7EB !important;
            background: rgba(255, 255, 255, 0.1) !important;
            color: #000000 !important;
            transition: border-color 0.3s ease !important;
        }
        .stSelectbox>div>div:focus-within {
            border-color: #50E3C2 !important;
            box-shadow: 0 0 8px rgba(80, 227, 194, 0.5) !important;
        }
        .stDownloadButton>button {
            background: linear-gradient(135deg, #66B2FF, #99CCFF) !important;
            border: 2px solid transparent !important;
        }
        .stDownloadButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(102, 178, 255, 0.3) !important;
        }
        .stMarkdown, .stText, .stInfo, .stSuccess, .stWarning, .stError {
            color: #00A0CF !important;
        }
        .stInfo, .stSuccess, .stWarning, .stError {
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 10px !important;
        }
        .chat-container {
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 15px !important;
            padding: 20px !important;
            margin-bottom: 20px !important;
            border: 2px dashed #00B7EB !important;
            box-shadow: 0 4px 12px rgba(0, 183, 235, 0.3) !important;
            color: #000000 !important;
            display: flex !important;
            flex-direction: column !important;
            overflow-y: auto !important;
        }
        .user-bubble, .bot-bubble {
            margin: 10px 10px !important;
            padding: 12px 18px !important;
            border-radius: 15px !important;
            word-wrap: break-word !important;
            max-width: 70% !important;
            font-size: 14px !important;
            color: #333333 !important;
            transition: all 0.3s ease !important;
            border: 1px solid rgba(0, 183, 235, 0.5) !important;
            box-shadow: 0 2px 5px rgba(0, 183, 235, 0.1) !important;
            clear: both !important;
        }
        .user-bubble {
            background: linear-gradient(135deg, #40C4FF, #80D8FF) !important;
            float: right !important;
            margin-left: auto !important;
            border-radius: 15px 5px 15px 15px !important;
        }
        .user-bubble:hover {
            transform: translateY(-3px) scale(1.01) !important;
            box-shadow: 0 4px 8px rgba(64, 196, 255, 0.3) !important;
        }
        .bot-bubble {
            background: linear-gradient(135deg, #00B7EB, #50E3C2) !important;
            float: left !important;
            margin-right: auto !important;
            border-radius: 5px 15px 15px 15px !important;
        }
        .bot-bubble:hover {
            transform: translateY(-3px) scale(1.01) !important;
            box-shadow: 0 4px 8px rgba(80, 227, 194, 0.3) !important;
        }
        .chat-header {
            font-size: 12px !important;
            font-weight: 600 !important;
            color: #555555 !important;
            margin: 5px 10px !important;
            display: flex !important;
            align-items: center !important;
            clear: both !important;
        }
        .avatar {
            width: 24px !important;
            height: 24px !important;
            border-radius: 50% !important;
            margin-right: 8px !important;
            background-size: cover !important;
            background-position: center !important;
            border: 2px solid #00B7EB !important;
            font-size: 16px !important;
            line-height: 24px !important;
            text-align: center !important;
        }
        .user-avatar { background: #40C4FF !important; }
        .bot { background: #50E3C2 !important; }
        hr {
            border: none !important;
            border-top: 2px dashed #50E3C2 !important;
            opacity: 0.7 !important;
            margin: 10px 0 !important;
            clear: both !important;
        }
        .cute-rating .streamlit-expanderHeader {
            font-size: 13px !important;
            padding: 0.2rem 0.5rem !important;
            background: transparent !important;
            border: none !important;
            border-radius: 15px !important;
        }
        .cute-rating .streamlit-expanderContent {
            padding: 0.3rem !important;
            border: none !important;
            background: transparent !important;
        }
        .cute-rating {
            border: none !important;
            background: transparent !important;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 5, 2])
    with col1:
        st.image("dude.png")
    with col2:
        st.markdown(
            '<h1 class="title-text" style="margin-top: 2rem; margin-bottom: -1rem;">Ustaad Jee\'s Knowledge Hub</h1>',
            unsafe_allow_html=True)
        st.markdown("----")
    init_session_state()

    if 'user_info' not in st.session_state:
        create_auth_interface()
        create_usage_tips()
        create_footer()
        return

    with st.sidebar:
        st.markdown(f"Welcome, {st.session_state.user_info['email']}")
        if st.button("Sign Out"):
            log_user_activity(st.session_state.user_info['localId'], "logout",
                              {"email": st.session_state.user_info['email']})
            st.session_state.clear()
            st.session_state.auth_success = 'You have successfully signed out'
            st.rerun()
        if st.button("Delete Account"):
            with st.form("delete_account_form"):
                password = st.text_input("Confirm Password", type="password")
                if st.form_submit_button("Delete", use_container_width=True, type="primary"):
                    try:
                        id_token = sign_in_with_email_and_password(st.session_state.user_info['email'], password)['idToken']
                        user_id = st.session_state.user_info['localId']
                        delete_user_account(id_token)
                        log_user_activity(user_id, "account_deletion", {"email": st.session_state.user_info['email']})
                        st.session_state.clear()
                        st.session_state.auth_success = 'You have successfully deleted your account'
                        st.rerun()
                    except requests.exceptions.HTTPError as error:
                        error_message = json.loads(error.args[1])['error']['message']
                        st.session_state.auth_warning = 'Error: Invalid password or try again later'
                    except Exception as error:
                        print(error)
                        st.session_state.auth_warning = 'Error: Please try again later'
        st.divider()
        create_admin_interface()
        st.divider()
        st.markdown("### Settings ⚙️")
        create_llm_configuration_section()
        st.divider()
        st.markdown("### Glossary")
        create_glossary_section()
        st.divider()
        st.markdown("### Sample Data")
        create_sample_data()

    if st.session_state.llm is None or st.session_state.connection_status == "Failed":
        st.warning("Please configure Ustaad Jee's brain in the sidebar!")
        create_usage_tips()
        create_footer()
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        document_texts, context_text = create_input_interface()
    with col2:
        create_translation_and_chat_interface(document_texts, context_text)
    create_usage_tips()
    create_footer()

if __name__ == "__main__":
    main()