# RAG Embedding Evaluation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]()


## Overview

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


## Evaluation Method

- For each case, multiple embeddings will be generated with each doc side and query side methods. Then cosine similarity is calculated accross 2 side embeddings.

- Then Max similarity for each query of each <doc, query> side method pair is accumulated as a score for this 2 methods pair for this.

- Then we calculate the average score for each positive query and negative query. And get the diff of average positive query to doc similarity to average negative query to doc similarity as the overall score for this <doc, query> method pair.

- The score of origin doc embedding with origin query embedding, showed as <direct, direct> is reduced from each score. Since approach is expected to better differentiate related and unrelated queries.

## How to use:

- `poetry install`
- `poetry run streamlit run`
- add your own azure openai key and endpoint in streamlit app
- select dataset
- click run to run evaluation process. This will take a while. completed result will be store in `results/` folder
- click show cases to reveal current dataset
- click show result to reveal previous eval result

## Add or modify dataset

Each dataset is added to cases folder as a json file. The json file should contain a list of dict with following format:

```json
{
    "topic": "topic name should be unique for each case",
    "content": "document content",
    "positive_queries": {
            "pq1 name": "query content",
            "pq2 name": "query content"
        },
    "negative_queries": {
        "nq1 name": "query content"
    }
}
```
