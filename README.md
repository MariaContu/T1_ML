# 📊 Trabalho de Aprendizado de Máquina — Interpretabilidade

## 1. 📌 Descrição do Problema

Este trabalho tem como objetivo aplicar e interpretar modelos de aprendizado de máquina em um problema de classificação.

O dataset utilizado foi o **Online Shoppers Intention Dataset**, cujo objetivo é prever se um usuário realizará uma compra (Revenue).

---

## 2. 📂 Dataset

- Fonte: Kaggle
- Tipo: Classificação binária
- Variável alvo: `Revenue`
- Features: Numéricas e categóricas

### ⚠️ Observação importante

O dataset apresenta **desbalanceamento entre as classes**, o que pode impactar a performance dos modelos.

---

## 3. ⚙️ Pré-processamento

As seguintes etapas foram realizadas:

- Tratamento de valores ausentes
- Codificação de variáveis categóricas (One-Hot Encoding)
- Normalização dos dados (necessário para KNN)
- Divisão treino/teste (80/20)

### ⚖️ Balanceamento

Foi utilizada a técnica **Undersampling** para equilibrar as classes.

Justificativa:
O desbalanceamento poderia enviesar métricas como acurácia, tornando o modelo incapaz de identificar corretamente a classe minoritária.

---

## 4. 🤖 Modelos Utilizados

- KNN (K-Nearest Neighbors)
- Naïve Bayes
- Árvore de Decisão

---

## 5. 📊 Avaliação

As métricas utilizadas foram:

- Acurácia
- Precisão
- Recall
- F1-score

---

## 6. 🔍 Interpretabilidade

### Árvore de Decisão

- Análise da estrutura da árvore
- Importância das features

### Naïve Bayes

- Análise das probabilidades condicionais

### KNN

- Utilização de SHAP/LIME para interpretação

---

## 7. ⚖️ Comparação dos Modelos

- Diferenças de desempenho
- Diferenças de interpretabilidade
- Limitações de cada abordagem

---

## 8. 📌 Conclusão

Discussão sobre:

- Efetividade dos modelos
- Impacto do balanceamento
- Importância da interpretabilidade

---

## 9. ▶️ Execução do Projeto

```bash
pip install -r requirements.txt
python -m src.main
```
