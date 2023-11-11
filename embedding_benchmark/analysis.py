import pandas as pd


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    positive_avg_pos_max = (
        df[df["label"] == "positive"]
        .groupby(["doc_embedding_type", "query_embedding_type"])["max_score"]
        .mean()
        .reset_index(name="avg_pos_max")
    )

    # 计算label为negative的min_score的平均值
    negative_avg_neg_max = (
        df[df["label"] == "negative"]
        .groupby(["doc_embedding_type", "query_embedding_type"])["max_score"]
        .mean()
        .reset_index(name="avg_neg_max")
    )

    # 合并两个结果以计算差值
    merged = pd.merge(
        positive_avg_pos_max,
        negative_avg_neg_max,
        on=["doc_embedding_type", "query_embedding_type"],
    )
    merged["difference"] = merged["avg_pos_max"] - merged["avg_neg_max"]

    # 创建最终的pivot table
    analysis_table = merged.pivot_table(
        index="query_embedding_type",
        columns="doc_embedding_type",
        values="difference",
        aggfunc="mean",
    )

    # 找到query_embedding_type和doc_embedding_type都是'direct'的差值
    baseline_value = analysis_table.loc["direct", "direct"]

    # 从每一列中减去基准值
    analysis_table = analysis_table.apply(lambda col: col - baseline_value)

    # 计算所有列的平均值，添加为新的行 'over_all_avg'
    analysis_table.loc["overall_avg"] = analysis_table.mean()

    # 计算所有行的平均值，添加为新的列 'over_all_avg'
    analysis_table["overall_avg"] = analysis_table.mean(axis=1)

    return analysis_table
