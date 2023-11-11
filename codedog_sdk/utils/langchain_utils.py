try:
    from dashscope import Generation
except ImportError:
    pass

from functools import lru_cache

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import Tongyi
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel


class ModelLoader:
    def __init__(
        self,
        *,
        llm_type="azure",
        embedding_type="azure",
        api_key="",
        api_base="",
        api_version="2023-08-01-preview",
        cheap_llm_dep_id="",
        expensive_llm_dep_id="",
        embedding_dep_id="",
    ):
        self.llm_type = llm_type
        self.embedding_type = embedding_type
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.cheap_llm_dep_id = cheap_llm_dep_id
        self.expensive_llm_dep_id = expensive_llm_dep_id
        self.embedding_dep_id = embedding_dep_id

    @lru_cache
    def load_cheap_llm(self, temperature=0.0) -> BaseLanguageModel:
        """Load GPT 3.5 level model."""
        if self.llm_type == "tongyi":
            llm = Tongyi(
                client=Generation(),
                model_name="qwen-turbo",
            )
        elif self.llm_type == "azure":
            llm = AzureChatOpenAI(
                openai_api_type="azure",
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
                openai_api_version=self.api_version,
                deployment_name=self.cheap_llm_dep_id,
                model="gpt-3.5-turbo",
                temperature=temperature,
            )
        elif self.llm_type == "openai":
            llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model="gpt-3.5-turbo",
                temperature=temperature,
            )
        else:
            raise NotImplementedError(f"type {self.llm_type} not supported")
        return llm

    def load_expensive_llm(self, temperature: float = 0.0):
        """Load GPT4 level model."""
        if self.llm_type == "azure":
            llm = AzureChatOpenAI(
                openai_api_type="azure",
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
                openai_api_version=self.api_version,
                deployment_name=self.expensive_llm_dep_id,
                model="gpt-4",
                temperature=temperature,
            )
        elif self.llm_type == "openai":
            llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model="gpt-4",
                temperature=temperature,
            )
        else:
            raise NotImplementedError(f"type {self.llm_type} not supported")
        return llm

    def load_embeddings(self) -> Embeddings:
        if self.embedding_type == "azure":
            embeddings = OpenAIEmbeddings(
                client=None,
                openai_api_type="azure",
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
                openai_api_version=self.api_version,
                deployment=self.embedding_dep_id,
            )
        elif self.embedding_type == "openai":
            embeddings = OpenAIEmbeddings(
                client=None,
                openai_api_key=self.api_key,
            )
        else:
            raise NotImplementedError(f"type {self.embedding_type} not supported")

        return embeddings
