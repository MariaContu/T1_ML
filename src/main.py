from models import train_and_evaluate
from preprocessing import (
    load_data,
    apply_undersampling,
    check_distribution,
    basic_eda,
    split_features_target,
    encode_categorical,
    normalize_data,
    split_train_test
)

def main():
    df = load_data("data/online_shoppers_intention.csv")

    basic_eda(df)

    print("\nANTES DO BALANCEAMENTO")
    check_distribution(df, "Revenue")

    df_balanced = apply_undersampling(df, "Revenue")

    print("\nDEPOIS DO BALANCEAMENTO")
    check_distribution(df_balanced, "Revenue")

    X, y = split_features_target(df_balanced, "Revenue")

    X = encode_categorical(X)

    X = normalize_data(X)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("\nShape treino:", X_train.shape)
    print("Shape teste:", X_test.shape)

    trained = train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()