import json
import os
import time
from os import environ as env

import pandas as pd
import streamlit as st
from langchain.callbacks import get_openai_callback

from codedog_sdk.utils.langchain_utils import ModelLoader
from embedding_benchmark.analysis import analyze
from embedding_benchmark.benchmark import Benchmark
from embedding_benchmark.embedding import Case


# --- Functions ----------------------------------------------------------------
def load_case_file(f) -> list[Case]:
    cases = []
    for case_obj in json.loads(f.read()):
        case = Case(**case_obj)
        cases.append(case)

    return cases


def load_result_file(f) -> list:
    return json.loads(f.read())


def store_result(result: list):
    result_set[ds_select] = result
    with open(f"results/{ds_select}.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=4, ensure_ascii=False))


def show_result(result: list):
    df = pd.DataFrame(result)
    df.sort_values(
        by=["topic", "query_type", "doc_embedding_type", "query_embedding_type"]
    )
    summary_tab, record_tab = st.tabs(["summary", "record"])
    with summary_tab:
        df_stats = analyze(df)
        cols = df_stats.columns.tolist()
        col_config = dict(
            zip(
                cols,
                [st.column_config.ProgressColumn(min_value=0.0, max_value=0.1)]
                * len(cols),
            )
        )
        st.dataframe(df_stats, column_config=col_config)

    with record_tab:
        st.dataframe(df)


# --- Initialize ---------------------------------------------------------------


st.set_page_config(layout="wide", page_title="RAG Embedding Evaluation")
st.session_state.setdefault("data_set", {})
st.session_state.setdefault("result_set", {})
st.session_state.setdefault("enable_azure", True)
data_set = st.session_state.data_set
result_set = st.session_state.result_set

if not data_set:
    for filename in os.listdir("cases"):
        with open(f"cases/{filename}", encoding="utf-8") as f:
            if filename[-5:] == ".json":
                st.session_state.data_set[filename[:-5]] = load_case_file(f)

if not result_set:
    for filename in os.listdir("results"):
        with open(f"results/{filename}", encoding="utf-8") as f:
            if filename[-5:] == ".json":
                st.session_state.result_set[filename[:-5]] = load_result_file(f)


# --- UI -----------------------------------------------------------------------
with st.sidebar:
    ds_select = st.selectbox("dataset", list(data_set.keys()))
    st.divider()

    api_key = st.text_input("AZURE_API_KEY", value=env.get("AZURE_OPENAI_API_KEY", ""))
    api_base = st.text_input(
        "AZURE_API_BASE", value=env.get("AZURE_OPENAI_API_BASE", "")
    )
    openai_api_version = st.text_input("OPENAI_API_VERSION", value="2023-08-01-preview")
    llm_dep_id = st.text_input("AZURE_LLM_DEPLOYMENT_ID", value="gpt-35-turbo")
    ebd_dep_id = st.text_input(
        "AZURE_EMBEDDING_DEPLOYMENT_ID", value="text-embedding-ada-002"
    )
    st.divider()

    if ds_select in result_set:
        show_btn = st.button("show_result", type="primary", use_container_width=True)
    else:
        show_btn = False

    run_btn = st.button("run", use_container_width=True)
    show_cases_btn = st.button("show cases", use_container_width=True)

st.markdown(
    """### Embedding Benchmark
Github: [codedog-ai/embedding-benchmark]()

This project benchmark the performance of different text process techniques by evaluate it's ability to differentiate related and unrelated queries from
document during embedding search.

Currently tested document process techniques:

**Document side:**
- Summary: [MultiVector Retriever - Langchain](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)
- Split Document: [MultiVector Retriever - Langchain](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)
- Hypothesis Query Generation: [MultiVector Retriever - Langchain](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector)

**Query side:**
- Fuse Query: [Forget RAG, the Future is RAG-Fusion - Adrian H. Raudaschl](https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1)
- Predict Answer: [Three LLM tricks that boosted embeddings search accuracy by 37% - Alistair Pullen](https://www.buildt.ai/blog/3llmtricks)

#### Benchmark result

Columns are document process methods. rows are query process methods. Values are average difference from max score of related query and unrelated query of all the cases related to <direct, direct> score.
"""
)

st.divider()

if run_btn:
    loader = ModelLoader(
        api_key=api_key or env.get("AZURE_OPENAI_API_KEY", ""),
        api_base=api_base or env.get("AZURE_OPENAI_API_BASE", ""),
        api_version="2023-08-01-preview",
        cheap_llm_dep_id="gpt-35-turbo",
        embedding_dep_id="text-embedding-ada-002",
    )

    t = time.time()
    cases = data_set[ds_select]
    benchmark = Benchmark(
        cases=cases, llm=loader.load_cheap_llm(), embedding=loader.load_embeddings()
    )

    with get_openai_callback() as cb:
        benchmark.run_embedddings()
    benchmark.calculate(benchmark.doc_ebds, benchmark.query_ebds)

    st.text_area(
        label="log",
        value=f"Time Usage: {time.time() - t:.2f}s Cost: ${cb.total_cost:.4f} (Embedding Cost is not included.)",
    )

    result = benchmark.result

    store_result(benchmark.result)
    show_result(benchmark.result)


if show_btn:
    result = result_set[ds_select]
    show_result(result)


if show_cases_btn:
    st.dataframe([case.dict() for case in data_set[ds_select]])
