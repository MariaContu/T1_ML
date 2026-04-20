import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_data, apply_undersampling, check_distribution


def main():
    df = load_data("data/online_shoppers_intention.csv")

    print("ANTES DO BALANCEAMENTO")
    check_distribution(df, "Revenue")

    df_balanced = apply_undersampling(df, "Revenue")

    print("\nDEPOIS DO BALANCEAMENTO")
    check_distribution(df_balanced, "Revenue")


if __name__ == "__main__":
    main()