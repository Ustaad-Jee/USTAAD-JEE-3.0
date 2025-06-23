import streamlit as st
import openai
import requests
import json
from typing import Optional, Dict, Any, List
from enum import Enum
import os
from abc import ABC, abstractmethod
import pandas as pd
import io
import time
import bleach
from typing import Tuple
import base64
import firebase_admin
from firebase_admin import credentials, auth, firestore
from dotenv import load_dotenv
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

from firebase_admin import auth, firestore




# Load environment variables
load_dotenv()

# Initialize Firebase Admin SDK
# Load environment variables (optional for local fallback)
load_dotenv()

# Load environment variables (optional for local fallback)
load_dotenv()
# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    try:
        # Get service account from Streamlit Secrets as JSON content
        service_account_json = st.secrets["firebase"]["SERVICE_ACCOUNT"]  # Fixed: Use proper nested access

        if service_account_json:
            try:
                # Parse the JSON string from secrets
                service_account_dict = json.loads(service_account_json)
                cred = credentials.Certificate(service_account_dict)
                firebase_admin.initialize_app(cred)
                print("✅ Firebase initialized successfully with Streamlit Secrets")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in firebase.SERVICE_ACCOUNT: {str(e)}")
        else:
            raise ValueError("SERVICE_ACCOUNT not found in firebase secrets")

    except KeyError as e:
        # Fallback to environment variable or file path
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
# Configuration for easy maintenance
class AppConfig:
    """Ustaad Jee's Knowledge Hub Configuration"""
    MODELS = {
        "OpenAI (GPT)": ["gpt-4o-mini", "gpt-4.1", "gpt-4-turbo", "gpt-4o"],
        "Claude": ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307",
                   "claude-sonnet-4-20250514"],
        "DeepSeek": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        "OpenRouter": ["anthropic/claude-3-sonnet", "openai/gpt-4", "meta-llama/llama-2-70b-chat",
                       "deepseek/deepseek-chat", "google/gemini-pro"],
        "Local LLM": ["llama3.2:3b", "llama3.2:1b", "llama3.1:8b", "llama3.1:70b", "deepseek-coder:6.7b",
                      "deepseek-coder:33b", "deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b",
                      "deepseek-r1:70b", "codellama:7b", "codellama:13b", "codellama:34b", "mistral:7b",
                      "mistral:instruct", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "phi3:mini", "phi3:medium",
                      "gemma2:2b", "gemma2:9b", "gemma2:27b"]
    }

    URDU_TRANSLATION_PROMPT = """You are Ustaad Jee, an expert teacher translating complex technical documents from {source_lang} to friendly, easy {target_lang}. 🧑‍🏫
GUIDELINES:
1. Use simple, natural {target_lang} like you're chatting with a friend
2. Break big sentences into small, clear ones
3. Explain techy stuff with everyday examples
4. Keep the technical meaning correct but super easy to read
5. Use the glossary terms exactly as given
6. Keep English tech terms in English but explain them in {target_lang}
7. Add short notes in brackets for tricky concepts
8. Sound warm and friendly, like explaining to a curious student
9. Use common Urdu words people love
{glossary_section}{context_section}
DOCUMENT TO TRANSLATE:
{text}
TRANSLATION (in fun, easy {target_lang}):"""

    ROMAN_URDU_TRANSLATION_PROMPT = """You are Ustaad Jee, an expert teacher translating complex technical documents from {source_lang} to friendly, easy Roman Urdu.
GUIDELINES:
1. Use simple, natural Roman Urdu like you're chatting with a friend
2. Break big sentences into small, clear ones
3. Explain techy stuff with everyday examples
4. Keep the technical meaning correct but super easy to read
5. Use the glossary terms exactly as given
6. Keep English tech terms in English but explain them in Roman Urdu
7. Add short notes in brackets for tricky concepts
8. Sound warm and friendly, like explaining to a curious student
9. Use fun Roman Urdu phrases like "Yeh basically...", "Is ka matlab hai ke...", "Jab aap..."
{glossary_section}{context_section}
DOCUMENT TO TRANSLATE:
{text}
TRANSLATION (in fun, easy Roman Urdu):"""

    URDU_CHAT_PROMPT = """You are Ustaad Jee, a friendly teacher explaining technical stuff from a document in simple Urdu.
INSTRUCTIONS:
1. Answer in clear, friendly Urdu like you're talking to a student
2. Use only the document for answers
3. Break big ideas into small, easy sentences
4. Use everyday examples to explain techy stuff
5. Keep the meaning correct but super clear
6. Use glossary terms as given
7. Keep English tech terms but explain them in Urdu
8. Add short notes in brackets for hard concepts
9. Sound like a kind, patient teacher
10. If the question isn't in the document, say so politely
{glossary_section}
DOCUMENT:
{document_text}
QUESTION:
{question}
ANSWER (in friendly Urdu):"""

    ROMAN_URDU_CHAT_PROMPT = """You are Ustaad Jee, a friendly teacher explaining technical stuff from a document in simple Roman Urdu.
INSTRUCTIONS:
1. Answer in clear, friendly Roman Urdu like you're talking to a student
2. Use only the document for answers
3. Break big ideas into small, easy sentences
4. Use everyday examples to explain techy stuff
5. Keep the meaning correct but super easy to read
6. Use glossary terms as given
7. Keep English tech terms but explain them in Roman Urdu
8. Add short notes in brackets for hard concepts
9. Sound like a kind, patient teacher
10. Use fun phrases like "Yeh basically...", "Is ka matlab hai ke..."
11. If the question isn't in the document, say so politely
{glossary_section}
DOCUMENT:
{document_text}
QUESTION:
{question}
ANSWER (in friendly Roman Urdu):"""

    ENGLISH_CHAT_PROMPT = """You are Ustaad Jee, a friendly teacher explaining technical stuff from a document in simple English.
INSTRUCTIONS:
1. Answer in clear, friendly English like you're talking to a student
2. Use only the document for answers
3. Break big ideas into small, easy sentences
4. Use everyday examples to explain techy stuff
5. Keep the meaning correct but super clear
6. Use glossary terms as given
7. Add short notes in brackets for hard concepts
8. Sound like a kind, patient teacher
9. If the question isn't in the document, say so politely
{glossary_section}
DOCUMENT:
{document_text}
QUESTION:
{question}
ANSWER (in friendly English):"""


class LLMProvider(Enum):
    OPENAI = "openai"
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"
    LOCAL = "local"


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        try:
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
        except Exception as e:
            raise Exception(f"Error connecting to OpenAI: {str(e)}")

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 2000),
                temperature=kwargs.get("temperature", 0.3)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI error: {str(e)}")


class ClaudeClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            data = {
                "model": self.model,
                "max_tokens": kwargs.get("max_tokens", 2000),
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"].strip()
        except Exception as e:
            raise Exception(f"Claude error: {str(e)}")


class DeepSeekClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 2000),
                "temperature": kwargs.get("temperature", 0.3)
            }
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise Exception(f"DeepSeek error: {str(e)}")


class OpenRouterClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "anthropic/claude-3-sonnet"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": kwargs.get("app_url", "https://localhost:8501"),
                "X-Title": kwargs.get("app_name", "Ustaad Jee")
            }
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 2000),
                "temperature": kwargs.get("temperature", 0.3)
            }
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise Exception(f"OpenRouter error: {str(e)}")


class LocalLLMClient(BaseLLMClient):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.base_url = base_url.rstrip('/')
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            url = f"{self.base_url}/api/generate"
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.3),
                    "num_predict": kwargs.get("max_tokens", 2000)
                }
            }
            response = requests.post(url, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            raise Exception(f"Local LLM error: {str(e)}")

class LLMWrapper:
    def __init__(self, provider: LLMProvider = LLMProvider.OPENAI, **config):
        self.provider = provider
        self.config = config
        self.client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        try:
            if self.provider == LLMProvider.OPENAI:
                # Check st.secrets first, then config, then environment variable
                api_key = self.config.get("api_key") or st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise Exception("OpenAI API key required! Please set it in Streamlit secrets or provide it manually.")
                model = self.config.get("model", "gpt-4o-mini")
                self.client = OpenAIClient(api_key, model)
            elif self.provider == LLMProvider.CLAUDE:
                api_key = self.config.get("api_key") or st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise Exception("Claude API key required!")
                model = self.config.get("model", "claude-3-sonnet-20240229")
                self.client = ClaudeClient(api_key, model)
            elif self.provider == LLMProvider.DEEPSEEK:
                api_key = self.config.get("api_key") or st.secrets.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
                if not api_key:
                    raise Exception("DeepSeek API key required!")
                model = self.config.get("model", "deepseek-chat")
                self.client = DeepSeekClient(api_key, model)
            elif self.provider == LLMProvider.OPENROUTER:
                api_key = self.config.get("api_key") or st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise Exception("OpenRouter API key required!")
                model = self.config.get("model", "anthropic/claude-3-sonnet")
                self.client = OpenRouterClient(api_key, model)
            elif self.provider == LLMProvider.LOCAL:
                base_url = self.config.get("base_url", "http://localhost:11434")
                model = self.config.get("model", "llama3.2:3b")
                self.client = LocalLLMClient(base_url, model)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            self.client = None

    def switch_provider(self, provider: LLMProvider, **config):
        self.provider = provider
        self.config.update(config)
        self._initialize_client()

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.client:
            raise Exception("LLM not initialized!")
        return self.client.generate(prompt, **kwargs)

    def translate_to_urdu(self, text: str, glossary: Optional[Dict[str, str]] = None, context: Optional[str] = None,
                          **kwargs) -> str:
        glossary_section = ""
        if glossary:
            glossary_section = "\n\nGLOSSARY:\n"
            for english_term, urdu_term in glossary.items():
                glossary_section += f"- {english_term} → {urdu_term}\n"
        context_section = ""
        if context:
            context_section = f"\n\nCONTEXT:\n{context}\n"
        prompt = AppConfig.URDU_TRANSLATION_PROMPT.format(
            source_lang="English",
            target_lang="Urdu",
            glossary_section=glossary_section,
            context_section=context_section,
            text=text
        )
        return self.generate(prompt, **kwargs)

    def translate_to_roman_urdu(self, text: str, glossary: Optional[Dict[str, str]] = None,
                                context: Optional[str] = None, **kwargs) -> str:
        glossary_section = ""
        if glossary:
            glossary_section = "\n\nGLOSSARY:\n"
            for english_term, urdu_term in glossary.items():
                glossary_section += f"- {english_term} → {urdu_term}\n"
        context_section = ""
        if context:
            context_section = f"\n\nCONTEXT:\n{context}\n"
        prompt = AppConfig.ROMAN_URDU_TRANSLATION_PROMPT.format(
            source_lang="English",
            target_lang="Roman Urdu",
            glossary_section=glossary_section,
            context_section=context_section,
            text=text
        )
        return self.generate(prompt, **kwargs)

    def document_chat(self, document_text: str, question: str, language: str = "English",
                      glossary: Optional[Dict[str, str]] = None, **kwargs) -> str:
        glossary_section = ""
        if glossary:
            glossary_section = "\n\nGLOSSARY:\n"
            for english_term, urdu_term in glossary.items():
                glossary_section += f"- {english_term} → {urdu_term}\n"

        if language == "Urdu":
            prompt = AppConfig.URDU_CHAT_PROMPT.format(
                glossary_section=glossary_section,
                document_text=document_text,
                question=question
            )
        elif language == "Roman Urdu":
            prompt = AppConfig.ROMAN_URDU_CHAT_PROMPT.format(
                glossary_section=glossary_section,
                document_text=document_text,
                question=question
            )
        else:
            prompt = AppConfig.ENGLISH_CHAT_PROMPT.format(
                glossary_section=glossary_section,
                document_text=document_text,
                question=question
            )

        return self.generate(prompt, temperature=0.3, max_tokens=2000, **kwargs)


class FeedbackDB:
    def __init__(self):
        # Initialize Firestore feedback collection (no in-memory storage needed)
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


def init_session_state():
    if 'llm' not in st.session_state:
        if st.session_state.get('is_admin', False):
            st.session_state.llm = None
            st.session_state.connection_status = "Not Connected"
        else:
            try:
                # Prioritize st.secrets, then fallback to environment variable
                api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    st.error("OpenAI API key not found in Streamlit secrets or environment variables!")
                    st.session_state.connection_status = "Failed"
                    return
                # Initialize LLMWrapper
                llm = LLMWrapper(
                    provider=LLMProvider.OPENAI,
                    api_key=api_key,
                    model="gpt-4o-mini"
                )
                # Test the connection with a simple prompt
                try:
                    test_response = llm.generate("Test connection", max_tokens=10)
                    if test_response and "error" not in test_response.lower():
                        st.session_state.llm = llm
                        st.session_state.connection_status = "Connected"
                    else:
                        st.session_state.connection_status = "Failed"
                        st.error("API test failed: Invalid response from OpenAI")
                except Exception as e:
                    st.session_state.connection_status = "Failed"
                    st.error(f"API connection test failed: {str(e)}")
            except Exception as e:
                st.session_state.connection_status = "Failed"
                st.error(f"Failed to initialize LLM: {str(e)}")

    # Initialize other session state variables
    if 'glossary' not in st.session_state:
        st.session_state.glossary = {}
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = "Not Connected"
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'glossary_updated' not in st.session_state:
        st.session_state.glossary_updated = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'feedback_db' not in st.session_state:
        st.session_state.feedback_db = FeedbackDB()
    if 'uploaded_document' not in st.session_state:
        st.session_state.uploaded_document = ""
    if 'context_text' not in st.session_state:
        st.session_state.context_text = ""
    if 'auth_warning' not in st.session_state:
        st.session_state.auth_warning = ""
    if 'auth_success' not in st.session_state:
        st.session_state.auth_success = ""
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False

def create_auth_interface():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 0.5rem 1rem;
        margin-top: -14rem;
    }

    .auth-title {
        text-align: center;
        color: #1f2937;
        font-size: 2rem;
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

    # Center the form using columns
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)

        # Initialize session state for auth mode
        if 'auth_mode' not in st.session_state:
            st.session_state.auth_mode = 'signin'

        # Back to Sign In link (for sign up and reset password modes)
        if st.session_state.auth_mode != 'signin':
            st.markdown('<div class="back-link">', unsafe_allow_html=True)
            if st.button("← Back to Sign In", key="back_to_signin"):
                st.session_state.auth_mode = 'signin'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Sign In Form
        if st.session_state.auth_mode == 'signin':
            st.markdown('<h1 class="auth-title">Sign In</h1>', unsafe_allow_html=True)
            st.markdown('<p class="form-description">Welcome back! Please sign in to your account.</p>',
                        unsafe_allow_html=True)

            with st.form("sign_in_form", clear_on_submit=False):
                email = st.text_input(
                    "Email Address",
                    placeholder="Enter your email",
                    label_visibility="collapsed"
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    label_visibility="collapsed"
                )

                submit = st.form_submit_button(
                    "Sign In",
                    use_container_width=True,
                    type="primary"
                )

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
                                    st.session_state.is_admin = user.custom_claims.get('admin',
                                                                                       False) if user.custom_claims else False
                                    log_user_activity(user_info['localId'], "login", {"email": user_info['email']})
                                    st.success("Welcome! Redirecting...")
                                    st.rerun()
                        except requests.exceptions.HTTPError as error:
                            error_message = json.loads(error.args[1])['error']['message']
                            if error_message in {"INVALID_EMAIL", "EMAIL_NOT_FOUND", "INVALID_PASSWORD",
                                                 "MISSING_PASSWORD"}:
                                st.session_state.auth_warning = 'Invalid email or password. Please try again.'
                            else:
                                st.session_state.auth_warning = 'Something went wrong. Please try again later.'
                        except Exception as error:
                            print(error)
                            st.session_state.auth_warning = 'Connection error. Please try again later.'

            # Links for Reset Password and Sign Up
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

        # Sign Up Form
        elif st.session_state.auth_mode == 'signup':
            st.markdown('<h1 class="auth-title">Create Account</h1>', unsafe_allow_html=True)
            st.markdown('<p class="form-description">Join us today! Create your account in just a few steps.</p>',
                        unsafe_allow_html=True)

            with st.form("sign_up_form", clear_on_submit=False):
                email = st.text_input(
                    "Email Address",
                    placeholder="Enter your email",
                    label_visibility="collapsed"
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Choose a strong password (min 6 characters)",
                    label_visibility="collapsed"
                )

                submit = st.form_submit_button(
                    "Create Account",
                    use_container_width=True,
                    type="primary"
                )

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
                                log_user_activity(user_info['localId'], "account_creation",
                                                  {"email": user_info['email']})
                        except requests.exceptions.HTTPError as error:
                            error_message = json.loads(error.args[1])['error']['message']
                            if error_message == "EMAIL_EXISTS":
                                st.session_state.auth_warning = 'This email is already registered. Try signing in instead.'
                            elif error_message in {"INVALID_EMAIL", "INVALID_PASSWORD", "MISSING_PASSWORD",
                                                   "MISSING_EMAIL", "WEAK_PASSWORD"}:
                                st.session_state.auth_warning = 'Please check your email format and password strength.'
                            else:
                                st.session_state.auth_warning = 'Something went wrong. Please try again later.'
                        except Exception as error:
                            print(error)
                            st.session_state.auth_warning = 'Connection error. Please try again later.'

        # Reset Password Form
        elif st.session_state.auth_mode == 'reset':
            st.markdown('<h1 class="auth-title">Reset Password</h1>', unsafe_allow_html=True)
            st.markdown(
                '<p class="form-description">Enter your email address and we\'ll send you a link to reset your password.</p>',
                unsafe_allow_html=True)

            with st.form("reset_password_form", clear_on_submit=False):
                email = st.text_input(
                    "Email Address",
                    placeholder="Enter your registered email",
                    label_visibility="collapsed"
                )

                submit = st.form_submit_button(
                    "Send Reset Link",
                    use_container_width=True,
                    type="primary"
                )

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

        # Display messages
        if st.session_state.get('auth_warning'):
            st.error(st.session_state.auth_warning)
            st.session_state.auth_warning = ""
        if st.session_state.get('auth_success'):
            st.success(st.session_state.auth_success)
            st.session_state.auth_success = ""

        st.markdown('</div>', unsafe_allow_html=True)


def create_admin_interface():
    """Create admin interface for managing users, viewing logs, and chat history."""
    # Validate session state
    if not st.session_state.get('user_info') or not st.session_state.get('id_token'):
        st.error("Session expired. Please log in again.")
        return
    if not st.session_state.get('is_admin', False):
        st.warning("You do not have admin access.")
        return

    db = firestore.client()
    st.markdown("### Admin Panel")
    with st.expander("Admin Actions", expanded=False):
        # Grant Admin Access
        st.markdown("**Grant Admin Access**")
        admin_email = st.text_input("Enter email to grant admin access", key="admin_email_input")
        if st.button("Grant Admin Access", key="grant_admin_btn"):
            try:
                set_admin_user(admin_email)
            except Exception as e:
                st.error(f"Error granting admin access: {str(e)}")

        # Revoke Admin Access
        st.markdown("**Revoke Admin Access**")
        revoke_email = st.text_input("Enter email to revoke admin access", key="revoke_email_input")
        if st.button("Revoke Admin Access", key="revoke_admin_btn"):
            try:
                user = auth.get_user_by_email(revoke_email)
                auth.set_custom_user_claims(user.uid, {"admin": False})
                st.success(f"Admin access revoked for {revoke_email}. They must log out and log back in.")
                log_user_activity(st.session_state.user_info['localId'], "revoke_admin", {"email": revoke_email})
            except auth.UserNotFoundError:
                st.error("Error: User not found or invalid email.")
            except Exception as e:
                st.error(f"Error revoking admin access: {str(e)}")

        # View Admin Logs
        st.markdown("**View All User Activities (Admin Logs)**")
        try:
            filter_email = st.text_input("Filter by Email", key="log_filter_email")
            query = db.collection("admin_logs").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(50)
            logs = query.get()
            log_data = []
            for log in logs:
                log_dict = log.to_dict()
                email = log_dict.get('email', 'Unknown')
                if not filter_email or email.lower().find(filter_email.lower()) != -1:
                    log_data.append({
                        "User ID": log_dict['user_id'],
                        "Email": email,
                        "Action": log_dict['action'],
                        "Details": str(log_dict['details']),
                        "Timestamp": log_dict['timestamp']
                    })
            if log_data:
                st.dataframe(pd.DataFrame(log_data), use_container_width=True)
            else:
                st.info("No admin logs available.")
        except Exception as e:
            st.error(f"Error fetching admin logs: {str(e)}")

        # View User-Specific Logs
        st.markdown("**View User-Specific Logs**")
        try:
            user_log_filter_email = st.text_input("Filter User Logs by Email", key="user_log_filter_email")
            user_log_data = []
            users_with_logs = db.collection_group("logs").limit(1).get()
            if users_with_logs:
                all_users = {user.uid: user.email for user in auth.list_users().iterate_all()}
                all_logs_query = db.collection_group("logs").order_by("timestamp",
                                                                      direction=firestore.Query.DESCENDING).limit(50)
                all_logs = all_logs_query.get()
                for log_doc in all_logs:
                    path_parts = log_doc.reference.path.split('/')
                    if len(path_parts) >= 2:
                        user_id = path_parts[1]
                        user_email = all_users.get(user_id, 'Unknown')
                        if not user_log_filter_email or user_email.lower().find(user_log_filter_email.lower()) != -1:
                            log_dict = log_doc.to_dict()
                            user_log_data.append({
                                "User ID": user_id,
                                "Email": user_email,
                                "Action": log_dict.get('action', ''),
                                "Details": str(log_dict.get('details', '')),
                                "Timestamp": log_dict.get('timestamp')
                            })
            if user_log_data:
                st.dataframe(pd.DataFrame(user_log_data), use_container_width=True)
            else:
                st.info("No user logs available.")
        except Exception as e:
            st.error(f"Error fetching user logs: {str(e)}")

        # View User Chat History
        st.markdown("**View User Chat History**")
        try:
            chat_filter_email = st.text_input("Filter Chat History by Email", key="chat_filter_email")
            chat_data = []
            chats_sample = db.collection_group("chats").limit(1).get()
            if chats_sample:
                all_users = {user.uid: user.email for user in auth.list_users().iterate_all()}
                all_chats_query = db.collection_group("chats").order_by("timestamp",
                                                                        direction=firestore.Query.DESCENDING).limit(50)
                all_chats = all_chats_query.get()
                for chat_doc in all_chats:
                    path_parts = chat_doc.reference.path.split('/')
                    if len(path_parts) >= 2:
                        user_id = path_parts[1]
                        user_email = all_users.get(user_id, 'Unknown')
                        if not chat_filter_email or user_email.lower().find(chat_filter_email.lower()) != -1:
                            chat_dict = chat_doc.to_dict()
                            chat_data.append({
                                "User ID": user_id,
                                "Email": user_email,
                                "Query": chat_dict.get('query', ''),
                                "Response": chat_dict.get('response', ''),
                                "Language": chat_dict.get('language', ''),
                                "Type": chat_dict.get('type', ''),
                                "Timestamp": chat_dict.get('timestamp')
                            })
            if chat_data:
                st.dataframe(pd.DataFrame(chat_data), use_container_width=True)
            else:
                st.info("No chat history available.")
        except Exception as e:
            st.error(f"Error fetching chat history: {str(e)}")

def create_llm_configuration_section():
    st.markdown("### Configure Ustaad Jee's Brain 🧠")

    if not st.session_state.get('is_admin', False):
        # Non-admin users: Display connection status only
        status_color = {"Connected": "🟢", "Not Connected": "🟡", "Failed": "🔴"}
        st.info(f"Status: {status_color.get(st.session_state.connection_status, '🟡')} {st.session_state.connection_status}")
        st.markdown("Using OpenAI GPT-4o-mini (default configuration).")
        return

    # Admin users: Full configuration options
    col1, col2 = st.columns([1, 1])
    with col1:
        provider_options = {
            "OpenAI (GPT)": LLMProvider.OPENAI,
            "Claude": LLMProvider.CLAUDE,
            "DeepSeek": LLMProvider.DEEPSEEK,
            "OpenRouter": LLMProvider.OPENROUTER,
            "Local LLM": LLMProvider.LOCAL
        }
        selected_provider = st.selectbox(
            "Select Brain",
            list(provider_options.keys()),
            help="Choose the AI provider for Ustaad Jee."
        )
        provider_enum = provider_options[selected_provider]
        available_models = AppConfig.MODELS.get(selected_provider, [])
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            help="Pick a model for Ustaad Jee's brain."
        )
    with col2:
        if selected_provider != "Local LLM":
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="Enter your API key",
                help="Required to connect to the AI provider."
            )
        else:
            api_key = None
            base_url = st.text_input(
                "Local Server URL",
                value="http://localhost:11434",
                help="URL where the local LLM is running."
            )
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
        if st.button("+ Add", key="add_term_btn"):
            if english_term and urdu_term:
                st.session_state.glossary[english_term] = urdu_term
                st.success(f"Added: {english_term} → {urdu_term}")
                log_user_activity(st.session_state.user_info['localId'], "add_glossary_term",
                                  {"english_term": english_term, "urdu_term": urdu_term})
                st.rerun()
            else:
                st.warning("Both terms are required!")

    if st.session_state.glossary:
        glossary_df = pd.DataFrame([
            {"English Term": eng, "Urdu Term": urdu}
            for eng, urdu in st.session_state.glossary.items()
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
                            if english_term not in st.session_state.glossary or not merge_option:
                                st.session_state.glossary[english_term] = urdu_term
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
        for idx, (eng, urdu) in enumerate(st.session_state.glossary.items()):
            col_eng, col_urdu, col_action = st.columns([2, 2, 1])
            with col_eng:
                st.write(f"**{eng}**")
            with col_urdu:
                st.write(urdu)
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


def create_input_interface() -> Tuple[str, str]:
    st.markdown("### Upload Your Document")
    uploaded_document = st.file_uploader(
        "Upload Document",
        type=["txt","pdf"],
        help="Upload a plain text file for Ustaad Jee to process.",
        key="document_upload"
    )
    document_text = st.session_state.uploaded_document
    if uploaded_document:
        try:
            document_text = uploaded_document.read().decode("utf-8").strip()
            document_text = bleach.clean(document_text)
            st.session_state.uploaded_document = document_text
            if st.session_state.get('user_info'):
                log_user_activity(
                    st.session_state.user_info['localId'],
                    "upload_document",
                    {"document_length": len(document_text)}
                )
            st.success("Document uploaded successfully!")
        except Exception as e:
            st.error(f"Failed to read document: {str(e)}")

    document_text = st.text_area(
        "Or Paste Your Document Here",
        value=document_text,
        height=300,
        placeholder="Paste your document text here...",
        help="Type or paste the document for Ustaad Jee.",
        key="document_text_input"
    )
    document_text = bleach.clean(document_text.strip()) if document_text else ""
    st.session_state.uploaded_document = document_text

    st.markdown("### Upload Additional Context")
    uploaded_context = st.file_uploader(
        "Upload Context",
        type=["txt"],
        help="Upload a plain text file containing additional context for Ustaad Jee.",
        key="context_upload"
    )
    context_text = st.session_state.context_text
    if uploaded_context:
        try:
            context_text = uploaded_context.read().decode("utf-8").strip()
            context_text = bleach.clean(context_text)
            st.session_state.context_text = context_text
            if st.session_state.get('user_info'):
                log_user_activity(
                    st.session_state.user_info['localId'],
                    "upload_context",
                    {"context_length": len(context_text)}
                )
            st.success("Context uploaded successfully!")
        except Exception as e:
            st.error(f"Failed to read context file: {str(e)}")

    context_text = st.text_area(
        "Additional Context",
        value=context_text,
        height=200,
        placeholder="Type or paste your context here...",
        help="Description about your document",
        key="context_text_input"
    )
    context_text = bleach.clean(context_text.strip()) if context_text else ""
    st.session_state.context_text = context_text

    return document_text, context_text


def create_translation_and_chat_interface(document_text, context_text):
    if not st.session_state.get('user_info'):
        st.warning("Please sign in to use Ustaad Jee!")
        return
    st.markdown("<h3>Interact with Ustaad Jee</h3>", unsafe_allow_html=True)
    if not document_text:
        st.warning("Please provide a document for Ustaad Jee to work with!")
        return
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
        st.empty()
        st.empty()
        st.empty()
        st.write("")
        st.write("\n")
        if st.button("Start Translation", type="primary",
                     help="Translate the entire document"):
            if document_text and st.session_state.llm:
                with st.spinner("Translating..."):
                    try:
                        if translate_language == "Urdu":
                            translation = st.session_state.llm.translate_to_urdu(
                                text=document_text,
                                glossary=st.session_state.glossary if st.session_state.glossary else None,
                                context=context_text,
                                temperature=0.3
                            )
                            chat_entry = {
                                "query": "Translation Request",
                                "response": translation,
                                "language": "Urdu",
                                "type": "translation",
                                "timestamp": time.time()
                            }
                        elif translate_language == "Roman Urdu":
                            translation = st.session_state.llm.translate_to_roman_urdu(
                                text=document_text,
                                glossary=st.session_state.glossary if st.session_state.glossary else None,
                                context=context_text,
                                temperature=0.3
                            )
                            chat_entry = {
                                "query": "Translation Request",
                                "response": translation,
                                "language": "Roman Urdu",
                                "type": "roman",
                                "timestamp": time.time()
                            }
                        else:
                            chat_entry = {
                                "query": "Translation Request",
                                "response": document_text,
                                "language": "English",
                                "type": "translation",
                                "timestamp": time.time()
                            }
                        st.session_state.chat_history.append(chat_entry)
                        store_chat_history(st.session_state.user_info['localId'], chat_entry)
                        log_user_activity(st.session_state.user_info['localId'], "translation",
                                          {"language": translate_language, "document_length": len(document_text)})
                        st.success(f"{translate_language} translation completed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    st.markdown("---")
    st.markdown("**Ustaad Jee's Chat**")
    if st.session_state.chat_history:
        with st.container(height=550, border=True):
            for i, chat in enumerate(st.session_state.chat_history):
                chat_time = time.strftime("%H:%M", time.localtime(chat.get('timestamp', time.time())))
                st.markdown(
                    f"""
                    <div class="chat-header" style="justify-content: flex-end;">
                        <div class="avatar user-avatar">🤗</div>
                        Student at {chat_time}
                    </div>
                    <div class="user-bubble">{chat['query']}</div>
                    """,
                    unsafe_allow_html=True
                )
                bot_name = "Ustaad Jee" if chat["language"] in ["Urdu", "Roman Urdu"] else "Ustaad Jee (English)"
                st.markdown(
                    f"""
                    <div class="chat-header">
                        <div class="avatar bot">🪄</div>
                        {bot_name} at {chat_time}
                    </div>
                    <div class="bot-bubble">{chat['response']}</div>
                    """,
                    unsafe_allow_html=True
                )

                # Cute Rating System with minimal styling
                feedback_key = f"feedback_{i}_{chat.get('timestamp', time.time())}"

                # Check if feedback has not been submitted
                if feedback_key not in st.session_state or not st.session_state.get(feedback_key, False):
                    # Custom CSS for cute, compact rating
                    st.markdown("""
                        <style>
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

                    with st.expander("Rate this response ", expanded=False):
                        # Store rating in session state
                        rating_session_key = f"temp_rating_{feedback_key}"
                        if rating_session_key not in st.session_state:
                            st.session_state[rating_session_key] = 0

                        # Inline star rating without extra text
                        star_text = ""
                        for star_num in range(1, 6):
                            star_icon = "⭐" if star_num <= st.session_state[rating_session_key] else "⭐️"
                            star_text += f'<span style="cursor: pointer; font-size: 18px; margin: 0 2px;" onclick="rate({star_num})">{star_icon}</span>'

                        # Create inline star buttons
                        cols = st.columns(7)  # 5 stars + submit + clear
                        for star_num in range(1, 6):
                            with cols[star_num - 1]:
                                star_icon = "⭐" if star_num <= st.session_state[rating_session_key] else "☆"
                                if st.button(
                                        f"{star_icon}",
                                        key=f"star_btn_{feedback_key}_{star_num}",
                                        help=f"{star_num} star{'s' if star_num > 1 else ''}",
                                        use_container_width=True
                                ):
                                    st.session_state[rating_session_key] = star_num
                                    st.rerun()

                        # Submit button
                        if st.session_state[rating_session_key] > 0:
                            with cols[5]:
                                if st.button("✨", key=f"submit_{feedback_key}", type="primary", help="Submit rating"):
                                    try:
                                        rating_value = str(st.session_state[rating_session_key])
                                        st.session_state.feedback_db.store_feedback(
                                            question=str(chat['query']),
                                            response=str(chat['response']),
                                            language=str(chat['language']),
                                            rating=rating_value
                                        )
                                        st.session_state[feedback_key] = True
                                        st.session_state[f"final_rating_{feedback_key}"] = st.session_state[
                                            rating_session_key]
                                        del st.session_state[rating_session_key]
                                        st.success("Thanks! ")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error submitting rating: {str(e)}")

                            # Clear button
                            with cols[6]:
                                if st.button("🧹", key=f"clear_{feedback_key}", help="Clear rating"):
                                    st.session_state[rating_session_key] = 0
                                    st.rerun()

                else:
                    # Show completed rating in a cute way
                    final_rating = st.session_state.get(f"final_rating_{feedback_key}", "N/A")
                    if final_rating != "N/A":
                        stars_display = "⭐" * final_rating


                        with st.expander(f"{stars_display}", expanded=False):
                            st.markdown("*Thank you for the feedback!*")
                    else:
                        with st.expander(" Rated", expanded=False):
                            st.markdown("*Thank you for the feedback!*")

                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("<hr>", unsafe_allow_html=True)
    else:
        st.info("Ustaad Jee's Chat is empty. Ask a question to start!")

    st.markdown("**Ask Ustaad Jee**")
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2, col3 = st.columns([5, 2, 1])  # Adjusted column ratios for layout
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
                if document_text and st.session_state.llm:
                    quick_queries = {
                        "Summarize": "Provide a brief summary of the document.",
                        "Key Points": "List the key points and main concepts of the document.",
                        "Simplify": "Explain the document in a very simple way.",
                        "Technical Terms": "List the key technical terms and their meanings."
                    }
                    with st.spinner(f"{quick_action}..."):
                        try:
                            response = st.session_state.llm.document_chat(
                                document_text=document_text,
                                question=quick_queries[quick_action],
                                language=chat_language,
                                glossary=st.session_state.glossary if st.session_state.glossary else None
                            )
                            chat_entry = {
                                "query": quick_action,
                                "response": response,
                                "language": chat_language,
                                "type": "quick_action",
                                "timestamp": time.time()
                            }
                            st.session_state.chat_history.append(chat_entry)
                            store_chat_history(st.session_state.user_info['localId'], chat_entry)
                            log_user_activity(st.session_state.user_info['localId'], "quick_action",
                                              {"action": quick_action.lower().replace(" ", "_")})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                elif not document_text:
                    st.warning("Please provide a document to use quick actions!")
                elif not st.session_state.llm:
                    st.error("Please configure Ustaad Jee's brain first!")
            elif chat_query:
                chat_query = bleach.clean(chat_query.strip())
                if st.session_state.llm:
                    with st.spinner("Ustaad Jee is thinking..."):
                        try:
                            response = st.session_state.llm.document_chat(
                                document_text=document_text,
                                question=chat_query,
                                language=chat_language,
                                glossary=st.session_state.glossary if st.session_state.glossary else None
                            )
                            chat_entry = {
                                "query": chat_query,
                                "response": response,
                                "language": chat_language,
                                "type": "question",
                                "timestamp": time.time()
                            }
                            st.session_state.chat_history.append(chat_entry)
                            store_chat_history(st.session_state.user_info['localId'], chat_entry)
                            log_user_activity(st.session_state.user_info['localId'], "chat",
                                              {"query": chat_query, "language": chat_language})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.error("Please configure Ustaad Jee's brain first!")
            else:
                st.warning("Please enter a question or select a quick action!")

    with st.expander("Chat Controls", expanded=False):
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.session_state.chat_history and st.button("Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                log_user_activity(st.session_state.user_info['localId'], "clear_chat", {})
                st.success("Chat cleared!")
                st.rerun()
        with col2:
            if st.button("Reset All", type="secondary"):
                st.session_state.chat_history = []
                st.session_state.uploaded_document = ""
                st.session_state.context_text = ""
                st.session_state.results = {}
                log_user_activity(st.session_state.user_info['localId'], "reset_all", {})
                st.success("Ready for a new session!")
                st.rerun()
        with col3:
            if st.session_state.chat_history:
                chat_export = ""
                for chat in st.session_state.chat_history:
                    chat_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(chat.get("timestamp")))
                    chat_export += f"[{chat_time}] Student: {chat['query']}\n"
                    chat_export += f"[{chat_time}] Ustaad Jee ({chat['language']}): {chat['response']}\n\n"
                st.download_button(
                    label="Download Chat",
                    data=chat_export,
                    file_name=f"ustaad_jee_chat_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

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
            st.session_state.uploaded_document = sample_doc
            st.success("Sample document loaded!")
            log_user_activity(st.session_state.user_info['localId'], "load_sample_document", {})
            st.rerun()
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
    st.set_page_config(
        page_title="Ustaad Jee's Knowledge Hub",
        page_icon="dude.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
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
            font-size: 6.5rem !important;
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
        .bot-avatar { background: #50E3C2 !important; }
        hr {
            border: none !important;
            border-top: 2px dashed #50E3C2 !important;
            opacity: 0.7 !important;
            margin: 10px 0 !important;
            clear: both !important;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 5, 2])
    with col1:
        st.image("dude.png")
    with col2:
        st.markdown('<h1 class="title-text" style =  margin-top: -8rem , margin-bottom : -5rem ; >Ustaad Jee\'s Knowledge Hub</h1>', unsafe_allow_html=True)
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
                        id_token = sign_in_with_email_and_password(st.session_state.user_info['email'], password)[
                            'idToken']
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
        document_text, context_text = create_input_interface()
    with col2:
        create_translation_and_chat_interface(document_text, context_text)

    create_usage_tips()
    create_footer()


if __name__ == "__main__":
    main()
