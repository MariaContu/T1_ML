import pandas as pd
from sklearn.utils import resample


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def apply_undersampling(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Aplica undersampling na classe majoritária.

    Parameters:
        df (DataFrame): dataset original
        target (str): nome da variável alvo

    Returns:
        DataFrame balanceado
    """

    df_majority = df[df[target] == False]
    df_minority = df[df[target] == True]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )

    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    df_balanced = df_balanced.sample(frac=1, random_state=42)

    return df_balanced

def check_distribution(df: pd.DataFrame, target: str):
    print("\nDistribuição da variável alvo:")
    print(df[target].value_counts())
    print("\nProporção:")
    print(df[target].value_counts(normalize=True))