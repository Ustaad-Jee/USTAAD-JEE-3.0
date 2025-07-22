import streamlit as st
import openai
import requests
from enum import Enum
import json
from typing import Dict, Optional, Any, Generator
from llama_index.core.llms.custom import CustomLLM as BaseLLM
from llama_index.core.llms import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.callbacks import CallbackManager
from apconfig import AppConfig
from abc import ABC, abstractmethod
import os

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

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
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

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=kwargs.get("max_tokens", 2000),
                temperature=kwargs.get("temperature", 0.3)
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"OpenAI streaming error: {str(e)}"

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

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        # Claude streaming implementation (placeholder)
        response = self.generate(prompt, **kwargs)
        yield response

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

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 2000),
                "temperature": kwargs.get("temperature", 0.3),
                "stream": True
            }
            response = requests.post(self.base_url, headers=headers, json=data, stream=True, timeout=60)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        data = json.loads(decoded_line[6:])
                        if data.get('choices') and data['choices'][0].get('delta', {}).get('content'):
                            yield data['choices'][0]['delta']['content']
        except Exception as e:
            yield f"DeepSeek streaming error: {str(e)}"

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

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
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
                "temperature": kwargs.get("temperature", 0.3),
                "stream": True
            }
            response = requests.post(self.base_url, headers=headers, json=data, stream=True, timeout=60)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        data = json.loads(decoded_line[6:])
                        if data.get('choices') and data['choices'][0].get('delta', {}).get('content'):
                            yield data['choices'][0]['delta']['content']
        except Exception as e:
            yield f"OpenRouter streaming error: {str(e)}"

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

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        try:
            url = f"{self.base_url}/api/generate"
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": kwargs.get("temperature", 0.3),
                    "num_predict": kwargs.get("max_tokens", 2000)
                }
            }
            response = requests.post(url, json=data, stream=True, timeout=120)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    result = json.loads(line.decode('utf-8'))
                    if result.get("response"):
                        yield result["response"]
        except Exception as e:
            yield f"Local LLM streaming error: {str(e)}"

class LLMWrapper:
    def __init__(self, provider: LLMProvider = LLMProvider.OPENAI, **config):
        self.provider = provider
        self.config = config
        self.client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        try:
            if self.provider == LLMProvider.OPENAI:
                api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
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

    def generate_stream(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> Generator[str, None, None]:
        """Generator that streams tokens from the LLM"""
        if not self.client:
            raise Exception("LLM not initialized!")
        return self.client.generate_stream(prompt, max_tokens=max_tokens, temperature=temperature)

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

class CustomLLM(BaseLLM):
    def __init__(self, llm_wrapper: LLMWrapper, callback_manager: Optional[CallbackManager] = None):
        super().__init__(callback_manager=callback_manager)
        if not hasattr(llm_wrapper, 'generate'):
            raise ValueError(f"llm_wrapper must have a 'generate' method, got {type(llm_wrapper)}")
        self._llm_wrapper = llm_wrapper
        self._call_count = 0
        print(f"CustomLLM initialized with llm_wrapper: {type(self._llm_wrapper)}")

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=4096,
            num_output=2000,
            model_name="custom_llm"
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        self._call_count += 1
        if self._call_count > 10:
            raise Exception(f"Possible recursive call detected in CustomLLM.complete (count: {self._call_count})")

        if not isinstance(prompt, str) or not prompt.strip():
            print(f"Invalid prompt in CustomLLM.complete: {prompt}")
            return CompletionResponse(text="Error: Invalid or empty prompt provided.")

        print(f"CustomLLM.complete called (count: {self._call_count}) with prompt: {prompt[:50]}...")
        try:
            response_text = self._llm_wrapper.generate(prompt, **kwargs)
            if not isinstance(response_text, str) or not response_text.strip():
                print(f"Invalid response from llm_wrapper: {response_text}")
                return CompletionResponse(text="Error: LLM returned an invalid or empty response.")
            return CompletionResponse(text=response_text)
        except Exception as e:
            print(f"Error in CustomLLM.complete: {str(e)}")
            return CompletionResponse(text=f"Error: {str(e)}")

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        """Stream completion using the LLMWrapper's generate_stream method."""
        self._call_count += 1
        if self._call_count > 10:
            raise Exception(f"Possible recursive call detected in CustomLLM.stream_complete (count: {self._call_count})")

        if not isinstance(prompt, str) or not prompt.strip():
            print(f"Invalid prompt in CustomLLM.stream_complete: {prompt}")
            yield CompletionResponse(text="Error: Invalid or empty prompt provided.")
            return

        print(f"CustomLLM.stream_complete called (count: {self._call_count}) with prompt: {prompt[:50]}...")
        try:
            for token in self._llm_wrapper.generate_stream(prompt, **kwargs):
                yield CompletionResponse(text=token, delta=token)
        except Exception as e:
            print(f"Error in CustomLLM.stream_complete: {str(e)}")
            yield CompletionResponse(text=f"Error: {str(e)}")

    def chat(self, messages: list, **kwargs) -> str:
        self._call_count += 1
        if self._call_count > 10:
            raise Exception(f"Possible recursive call detected in CustomLLM.chat (count: {self._call_count})")

        if not messages:
            print("Empty messages list in CustomLLM.chat")
            return "Error: No messages provided."

        print(f"CustomLLM.chat called (count: {self._call_count}) with {len(messages)} messages")
        prompt = ""
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = str(msg)
            if not isinstance(content, str) or not content.strip():
                print(f"Invalid message content: {content}")
                return "Error: Invalid or empty message content."
            prompt += content + "\n"

        try:
            response_text = self._llm_wrapper.generate(prompt.strip(), **kwargs)
            if not isinstance(response_text, str) or not response_text.strip():
                print(f"Invalid response from llm_wrapper: {response_text}")
                return "Error: LLM returned an invalid or empty response."
            return response_text
        except Exception as e:
            print(f"Error in CustomLLM.chat: {str(e)}")
            return f"Error: {str(e)}"