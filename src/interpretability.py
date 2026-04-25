from pathlib import Path
import importlib
import importlib.util
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

def plot_decision_tree(dt_model, feature_names, output_dir):
    plt.figure(figsize=(30, 14))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=["Nao Compra", "Compra"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=7,
    )
    plt.title("Arvore de Decisao - Revenue")
    plt.tight_layout()
    plt.savefig(output_dir + "/arvore_decisao.png", dpi=300)
    plt.close()

def plot_tree_feature_importance(dt_model, feature_names, output_dir):
    importance = pd.Series(dt_model.feature_importances_, index=feature_names)
    top_importance = importance.sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    top_importance.sort_values().plot(kind="barh")
    plt.title("Top Features - Arvore de Decisao")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(output_dir + "/arvore_feature_importance.png", dpi=300)
    plt.close()

    top_importance.to_csv(output_dir + "/arvore_feature_importance.csv", header=["importance"])
    print("\nTop features da Arvore de Decisao:")
    print(top_importance.head(10))

def analyze_naive_bayes(nb_model, feature_names, output_dir):
    class_labels = [f"classe_{c}" for c in nb_model.classes_]
    theta = pd.DataFrame(nb_model.theta_, columns=feature_names, index=class_labels)
    var = pd.DataFrame(nb_model.var_, columns=feature_names, index=class_labels)

    theta.to_csv(output_dir + "/naive_bayes_theta.csv")
    var.to_csv(output_dir + "/naive_bayes_var.csv")

    if len(class_labels) >= 2:
        mean_gap = (theta.iloc[1] - theta.iloc[0]).abs().sort_values(ascending=False)
    else:
        mean_gap = pd.Series(0, index=feature_names)

    top_gap = mean_gap.head(12)

    plt.figure(figsize=(10, 6))
    top_gap.sort_values().plot(kind="barh")
    plt.title("Naive Bayes - Features com maior separacao entre classes")
    plt.xlabel("|media_classe_1 - media_classe_0|")
    plt.tight_layout()
    plt.savefig(output_dir + "/naive_bayes_mean_gap.png", dpi=300)
    plt.close()

    top_gap.to_csv(output_dir + "/naive_bayes_mean_gap.csv", header=["mean_gap"])
    print("\nTop features por separacao de medias (Naive Bayes):")
    print(top_gap.head(10))

def analyze_knn(knn_model, X_train, X_test, y_test, output_dir):
    perm = permutation_importance(
        knn_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="f1",
    )

    importances_mean = getattr(perm, "importances_mean", None)

    if importances_mean is not None:
        pass
    elif isinstance(perm, dict) and "importances_mean" in perm:
        importances_mean = perm["importances_mean"]
    elif isinstance(perm, dict):
        first_value = next(iter(perm.values()))
        nested_mean = getattr(first_value, "importances_mean", None)
        if nested_mean is not None:
            importances_mean = nested_mean
        else:
            importances_mean = first_value
    else:
        raise TypeError("Formato inesperado no retorno de permutation_importance")

    perm_series = pd.Series(importances_mean, index=X_test.columns).sort_values(ascending=False)
    top_perm = perm_series.head(15)

    plt.figure(figsize=(10, 6))
    top_perm.sort_values().plot(kind="barh")
    plt.title("KNN - Permutation Importance (F1)")
    plt.xlabel("Queda media no F1 ao permutar a feature")
    plt.tight_layout()
    plt.savefig(output_dir + "/knn_permutation_importance.png", dpi=300)
    plt.close()

    top_perm.to_csv(output_dir + "/knn_permutation_importance.csv", header=["importance"])
    print("\nTop features por Permutation Importance (KNN):")
    print(top_perm.head(10))

    try:
        if importlib.util.find_spec("shap") is None:
            raise ModuleNotFoundError("Pacote shap nao instalado")

        shap = importlib.import_module("shap")

        background_size = min(80, len(X_train))
        explain_size = min(20, len(X_test))

        background = shap.sample(X_train, background_size, random_state=42)
        explain_samples = X_test.sample(n=explain_size, random_state=42)

        explainer = shap.KernelExplainer(knn_model.predict_proba, background)
        shap_values = explainer.shap_values(explain_samples, nsamples=100)

        if isinstance(shap_values, list):
            target_shap = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            target_shap = shap_values

        plt.figure()
        shap.summary_plot(target_shap, explain_samples, show=False)
        plt.tight_layout()
        plt.savefig(output_dir + "/knn_shap_summary.png", dpi=300)
        plt.close()
    except Exception as exc:
        print("SHAP nao executado.")

def run_interpretability(trained_models, X_train, X_test, y_test, output_dir):
    print("\n=== INTERPRETABILIDADE ===")

    dt_model = trained_models["Árvore de Decisão"]
    nb_model = trained_models["Naïve Bayes"]
    knn_model = trained_models["KNN"]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plot_decision_tree(dt_model, X_train.columns, output_dir)
    plot_tree_feature_importance(dt_model, X_train.columns, output_dir)
    analyze_naive_bayes(nb_model, X_train.columns, output_dir)
    analyze_knn(knn_model, X_train, X_test, y_test, output_dir)