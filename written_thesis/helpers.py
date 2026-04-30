import re

import pandas as pd


def rename_patient(patient: str) -> str:
    m = re.fullmatch(r"competition-(\d+)", patient)
    if m:
        return f"C{int(m.group(1)):02d}"

    m = re.fullmatch(r"U002-DE01-(\d+)", patient)
    if m:
        return f"U{int(m.group(1)):02d}"

    return patient


def boldify_index_and_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Bold column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples(
            [tuple(r'\textbf{{{}}}'.format(str(x)) for x in tup) for tup in df.columns],
            names=[r'\textbf{{{}}}'.format(str(n)) if n else None for n in df.columns.names]
        )
    else:
        df.columns = [r'\textbf{{{}}}'.format(str(x)) for x in df.columns]
        if df.columns.name:
            df.columns.name = r'\textbf{{{}}}'.format(str(df.columns.name))

    # Bold index names
    if isinstance(df.index, pd.MultiIndex):
        df.index.names = [r'\textbf{{{}}}'.format(str(n)) if n else None for n in df.index.names]
    else:
        if df.index.name:
            df.index.name = r'\textbf{{{}}}'.format(str(df.index.name))
    return df
