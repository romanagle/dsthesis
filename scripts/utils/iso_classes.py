"""
Canonical tRNA isotype class definitions.

Merges: Ile2 → Ile
Drops:  SeC, Undet, Sup
"""

ISOTYPE_ORDER = [
    "Ala", "Arg", "Asn", "Asp", "Cys", "fMet", "Gln", "Glu", "Gly", "His",
    "Ile", "iMet", "Leu", "Lys", "Met", "Phe", "Pro", "Ser",
    "Thr", "Trp", "Tyr", "Val",
]

DROP_ISOS  = frozenset({"SeC", "Undet", "Sup"})
MERGE_ISOS = {"Ile2": "Ile"}


def normalize_isotypes(df, col="Anticodon_predicted_isotype"):
    """Remap Ile2→Ile; drop SeC/Undet/Sup rows. Returns new df."""
    df = df.copy()
    df[col] = df[col].replace(MERGE_ISOS)
    return df[~df[col].isin(DROP_ISOS)]


def remap_isotypes(df, col="Anticodon_predicted_isotype"):
    """Remap Ile2→Ile without dropping any rows. Returns new df."""
    df = df.copy()
    df[col] = df[col].replace(MERGE_ISOS)
    return df
