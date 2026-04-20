# ✅ Checklist do Trabalho — Interpretabilidade em Machine Learning

## 🧩 1. Escolha do Dataset

- [x] Dataset escolhido  
- [x] Problema de classificação definido (`Revenue`)  
- [x] Dataset possui variáveis numéricas e categóricas  
- [x] Dataset possui quantidade suficiente de instâncias  
- [x] Dataset não é um dos proibidos (Iris, Titanic, etc.)  
- [x] Fonte do dataset documentada  

---

## 🔍 2. Análise Inicial (EDA)

- [x] Verificação da variável alvo  
- [x] Identificação de desbalanceamento  
- [x] Análise geral das features (distribuição, tipos)  
- [x] Verificação de valores nulos  

---

## ⚙️ 3. Pré-processamento

### 🧹 Limpeza e preparação
- [ ] Tratamento de valores ausentes  
- [x] Separação entre variáveis independentes (X) e target (y)  

### 🔤 Transformações
- [x] Codificação de variáveis categóricas (One-Hot Encoding)  
- [x] Normalização dos dados (necessário para KNN)  

---

## ⚖️ 4. Balanceamento dos Dados

- [x] Identificação do desbalanceamento  
- [x] Aplicação de undersampling  
- [x] Validação do balanceamento (antes/depois)  
- [x] Justificativa da técnica utilizada  
- [ ] Discussão da limitação (perda de dados)  

---

## 🔀 5. Divisão dos Dados

- [x] Separação treino/teste (80/20 ou 70/30)  
- [x] Garantir aleatoriedade (random_state)  

---

## 🤖 6. Treinamento dos Modelos

- [ ] Treinamento do KNN  
- [ ] Treinamento do Naïve Bayes  
- [ ] Treinamento da Árvore de Decisão  
- [ ] Ajuste de hiperparâmetros (se aplicável)  

---

## 📊 7. Avaliação dos Modelos

- [ ] Cálculo da acurácia  
- [ ] Cálculo da precisão  
- [ ] Cálculo do recall  
- [ ] Cálculo do F1-score  
- [ ] Geração da matriz de confusão  
- [ ] Justificativa das métricas escolhidas  

---

## 🔍 8. Interpretabilidade

### 🌳 Árvore de Decisão
- [ ] Visualização da árvore  
- [ ] Análise das features mais importantes  

### 📈 Naïve Bayes
- [ ] Análise das probabilidades condicionais  

### 👥 KNN
- [ ] Discussão da dificuldade de interpretabilidade  
- [ ] Aplicação de SHAP ou LIME  

---

## ⚖️ 9. Comparação dos Modelos

- [ ] Comparação de desempenho  
- [ ] Comparação de interpretabilidade  
- [ ] Identificação das variáveis mais relevantes  
- [ ] Discussão das limitações de cada modelo  
- [ ] Resposta às perguntas do enunciado  

---

## 🎥 10. Apresentação (Vídeo)

- [ ] Explicação do dataset  
- [ ] Explicação do pré-processamento  
- [ ] Explicação dos modelos  
- [ ] Explicação da interpretabilidade  
- [ ] Comparação final  
- [ ] Duração entre 10–15 minutos  

---

## 💻 11. Entregáveis Técnicos

- [x] Estrutura de projeto organizada (`src/`, `data/`)  
- [x] README estruturado  
- [ ] Código comentado  
- [ ] README com explicações adicionais  
- [ ] Link do vídeo incluído  
- [ ] Projeto disponível (GitHub ou zip)  

---

## 📌 Status Geral

### ✅ Concluído
- Dataset selecionado  
- Problema definido  
- Desbalanceamento identificado  
- Undersampling aplicado  
- Estrutura do projeto organizada  

### 🟡 Em andamento
- Pré-processamento completo  
- Implementação dos modelos  

### 🔴 Pendente
- Avaliação  
- Interpretabilidade  
- Comparação  
- Vídeo  