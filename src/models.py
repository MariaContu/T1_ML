from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Dicionário com os modelos
    # Usamos max_depth=5 na Árvore para evitar que ela fique gigante e ininterpretável
    models = {
        "Árvore de Decisão": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Naïve Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Treinando e Avaliando: {name} ---")
        
        # Treinamento
        model.fit(X_train, y_train)
        
        # Previsão
        y_pred = model.predict(X_test)
        
        # Métricas
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Acurácia: {acc:.4f}")
        print(f"Precisão: {prec:.4f}")
        print(f"Recall:   {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Matriz de Confusão:\n{cm}")
        
        # Salvando os modelos treinados para usar na interpretabilidade depois
        results[name] = model

    return results