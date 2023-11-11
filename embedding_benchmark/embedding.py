import copy
from itertools import chain
from typing import Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from codedog_sdk.chains import ListChain
from embedding_benchmark import grimoire


class Case(BaseModel):
    topic: str
    content: str
    positive_queries: dict[str, str] = Field(default_factory=dict)
    negative_queries: dict[str, str] = Field(default_factory=dict)
    _doc: Optional[Document] = None
    _positive_query_docs: Optional[list[Document]] = None
    _negative_query_docs: Optional[list[Document]] = None

    def doc(self) -> Document:
        if not self._doc:
            self._doc = Document(page_content=self.content, metadata={"id": self.topic})

        return self._doc

    def positive_query_docs(self) -> list[Document]:
        if not self._positive_query_docs:
            self._positive_query_docs = [
                Document(
                    page_content=query,
                    metadata={
                        "id": self.topic,
                        "gt": "positive",
                        "q_type": query_type,
                    },
                )
                for query_type, query in self.positive_queries.items()
            ]

        return self._positive_query_docs

    def negative_query_docs(self) -> list[Document]:
        if not self._negative_query_docs:
            self._negative_query_docs = [
                Document(
                    page_content=query,
                    metadata={
                        "id": self.topic,
                        "gt": "negative",
                        "q_type": query_type,
                    },
                )
                for query_type, query in self.negative_queries.items()
            ]

        return self._negative_query_docs


class DocumentEmbeddingBuilder:
    def __init__(self, llm: BaseLanguageModel, embedding: Embeddings, *, splitter=None):
        self.llm: BaseLanguageModel = llm
        self.embedding: Embeddings = embedding
        self.text_splitter = (
            RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " "], chunk_size=500, chunk_overlap=0
            )
            if not splitter
            else splitter
        )
        self._summary_chain = LLMChain(
            llm=self.llm, prompt=PromptTemplate.from_template(grimoire.SUMMARIZE_TEXT)
        )
        self._predict_query_chain = ListChain.from_llm(
            llm=self.llm,
            question=grimoire.HYPOTHETICAL_QUERY,
            k=2,
        )
        self._fuse_chain = ListChain.from_llm(
            llm=self.llm, question=grimoire.FUSE_QUERY, k=2
        )
        self._predict_answer_chain = LLMChain(
            llm=self.llm, prompt=PromptTemplate.from_template(grimoire.PREDICT_ANSWER)
        )

        self.docs: dict[str, Document] = {}
        self.queries: dict[str, list[Document]] = {}

    def add_doc(self, key: str, doc: Document):
        self.docs[key] = doc

    def add_queries(self, key: str, queries: list[Document]):
        self.queries[key] = queries

    def run_doc_embedding(
        self,
    ) -> dict[str, dict[str, list[tuple[Document, list[float]]]]]:
        prepared_texts = []

        prepared_texts.extend(self._prepare_direct_text(self.docs))
        prepared_texts.extend(self._prepare_split_text(self.docs))
        prepared_texts.extend(self._prepare_summary_text(self.docs))
        prepared_texts.extend(self._prepare_predict_query(self.docs))

        text_and_embeddings = self._run_embedding(prepared_texts)
        embeddings_by_doc = self._group_by_doc(text_and_embeddings)
        return embeddings_by_doc

    def run_query_embedding(
        self,
    ) -> dict[str, dict[str, dict[str, list[tuple[Document, list[float]]]]]]:
        prepared_texts = []
        flat_queries = list(chain.from_iterable(self.queries.values()))
        prepared_texts.extend(self._prepare_direct_query(flat_queries))
        prepared_texts.extend(self._prepare_rag_fusion_queries(flat_queries))
        prepared_texts.extend(self._prepare_predict_answer(flat_queries))

        text_and_embeddings = self._run_embedding(prepared_texts)
        embeddings_by_doc = self._group_by_doc_for_query(text_and_embeddings)
        return embeddings_by_doc

    def _prepare_direct_text(self, docs: dict[str, Document]) -> list[Document]:
        for key, doc in docs.items():
            doc.metadata["type"] = "direct"
        return list(docs.values())

    def _prepare_split_text(self, docs: dict[str, Document]):
        sub_docs = self.text_splitter.split_documents(docs.values())

        for doc in sub_docs:
            doc.metadata["type"] = "split"

        return sub_docs

    def _prepare_summary_text(self, docs: dict[str, Document]):
        id_list = []
        doc_list = []
        for key, doc in docs.items():
            id_list.append(key)
            doc_list.append(doc)
        summaries = self._summary_chain.apply([{"doc": doc} for doc in doc_list])
        summary_docs = [
            Document(
                page_content=s["text"], metadata={"id": id_list[i], "type": "summary"}
            )
            for i, s in enumerate(summaries)
        ]

        return summary_docs

    def _prepare_predict_query(self, docs: dict[str, Document]):
        id_list = []
        doc_list = []
        for key, doc in docs.items():
            id_list.append(key)
            doc_list.append(doc)
        predict_querys = self._predict_query_chain.apply(
            [{"doc": doc} for doc in doc_list]
        )

        pq_docs = []
        for i, pq in enumerate(predict_querys):
            queries = pq["texts"]
            for q in queries:
                pq_docs.append(
                    Document(
                        page_content=q,
                        metadata={"id": id_list[i], "type": "predict_query"},
                    )
                )

        return pq_docs

    def _run_embedding(
        self, texts: list[Document]
    ) -> list[tuple[Document, list[float]]]:
        vectors = self.embedding.embed_documents([t.page_content for t in texts])
        text_and_embeddings = list(zip(texts, vectors))
        return text_and_embeddings

    def _group_by_doc(
        self, text_and_embeddings: list[tuple[Document, list[float]]]
    ) -> dict[str, dict[str, list[tuple[Document, list[float]]]]]:
        embeddings_by_doc: dict[str, dict[str, list[tuple[Document, list[float]]]]] = {}
        for text, embedding in text_and_embeddings:
            doc_id = text.metadata["id"]
            if doc_id not in embeddings_by_doc:
                embeddings_by_doc[doc_id] = {}
            doc_type = text.metadata["type"]
            if doc_type not in embeddings_by_doc[doc_id]:
                embeddings_by_doc[doc_id][doc_type] = []

            embeddings_by_doc[doc_id][doc_type].append((text, embedding))

        return embeddings_by_doc

    def _prepare_direct_query(self, queries: list[Document]):
        prepared_queries = []
        for query in queries:
            query.metadata["type"] = "direct"
            prepared_queries.append(query)

        return prepared_queries

    def _prepare_rag_fusion_queries(self, queries: list[Document]):
        prepared_queries = []

        fusioned_queries = self._fuse_chain.apply(
            [{"query": q.page_content} for q in queries]
        )
        for i, fusioned_query in enumerate(fusioned_queries):
            qs = fusioned_query["texts"]
            meta = copy.copy(queries[i].metadata)
            meta["type"] = "fusion"
            for q in qs:
                prepared_queries.append(Document(page_content=q, metadata=meta))

        return prepared_queries

    def _prepare_predict_answer(self, queries: list[Document]) -> list[Document]:
        prepared_queries = []

        predict_answers = self._predict_answer_chain.apply(
            [{"query": q.page_content} for q in queries]
        )

        for i, pa in enumerate(predict_answers):
            meta = copy.copy(queries[i].metadata)
            meta["type"] = "predict_answer"
            prepared_queries.append(Document(page_content=pa["text"], metadata=meta))

        return prepared_queries

    def _group_by_doc_for_query(
        self, text_and_embeddings: list[tuple[Document, list[float]]]
    ) -> dict[str, dict[str, dict[str, list[tuple[Document, list[float]]]]]]:
        embeddings_by_doc: dict[
            str, dict[str, dict[str, list[tuple[Document, list[float]]]]]
        ] = {}

        for text, embedding in text_and_embeddings:
            doc_id = text.metadata["id"]
            gt = text.metadata["gt"]
            q_type = text.metadata["q_type"]
            doc_type = text.metadata["type"]

            q_id = f"{gt}|{q_type}"
            if doc_id not in embeddings_by_doc:
                embeddings_by_doc[doc_id] = {}
            if q_id not in embeddings_by_doc[doc_id]:
                embeddings_by_doc[doc_id][q_id] = {}
            if doc_type not in embeddings_by_doc[doc_id][q_id]:
                embeddings_by_doc[doc_id][q_id][doc_type] = []

            embeddings_by_doc[doc_id][q_id][doc_type].append((text, embedding))

        return embeddings_by_doc
