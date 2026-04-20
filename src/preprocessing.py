import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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

def basic_eda(df: pd.DataFrame):
    print("\n=== INFORMAÇÕES GERAIS ===")
    print(df.info())

    print("\n=== ESTATÍSTICAS ===")
    print(df.describe())

    print("\n=== VALORES NULOS ===")
    print(df.isnull().sum())

    print("\n=== TIPOS DE DADOS ===")
    print(df.dtypes)

def split_features_target(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def encode_categorical(X: pd.DataFrame):
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded

def normalize_data(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def split_train_test(X, y):
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )