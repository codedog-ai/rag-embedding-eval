from os import environ as env

from langchain.callbacks import get_openai_callback

from codedog_sdk.utils.langchain_utils import ModelLoader

loader = ModelLoader(
    api_key=env.get("AZURE_OPENAI_API_KEY", ""),
    api_base=env.get("AZURE_OPENAI_API_BASE", ""),
    api_version="2023-08-01-preview",
    cheap_llm_dep_id="gpt-35-turbo",
    embedding_dep_id="text-embedding-ada-002",
)


with get_openai_callback() as cb:
    llm = loader.load_cheap_llm()
    embeddings = loader.load_embeddings()
    print(llm.predict("hello world"))
    print(embeddings.embed_query("hello world"))
    print(cb)
