# 🌎🔥 Preditor de Risco de Queimadas

**Projeto completo com pipelines de coleta, processamento, NLP, previsão e dashboard.**

---

## 📌 Visão Geral

Este repositório contém um sistema de previsão de risco de queimadas baseado em:

* 📥 **Coleta automática** de dados mensais (BDQueimadas)
* 🧹 **Processamento e limpeza** dos dados
* 🤖 **Treinamento de modelo preditivo**
* 🗞️ **Pipeline NLP** para análise de manchetes jornalísticas
* 📊 **Dashboard interativo** (Streamlit)

As pastas `data/` são **geradas automaticamente** e **não são versionadas**.

---

## 📁 Estrutura do Projeto

```
projeto/
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── scripts/
│   └── download_data.sh
├── src/
│   ├── data_collection.py
│   ├── data_processing.py
│   ├── ml_pipeline.py
│   ├── nlp_pipeline.py
│   └── dashboard.py
├── notebooks/
│   └── Relatório.ipynb
├── requirements.txt
└── README.md
```

---

## 🚀 Como Rodar o Projeto

### 1️⃣ Criar ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3️⃣ Baixar dados (obrigatório)

```bash
bash scripts/download_data.sh
```

### 4️⃣ Processar os dados

```bash
python src/data_processing.py
```

### 5️⃣ Pipeline NLP (opcional)

```bash
python src/nlp_pipeline.py
```

### 6️⃣ Treinar modelo

```bash
python src/ml_pipeline.py
```

### 7️⃣ Rodar o dashboard 🚀

```bash
streamlit run src/dashboard.py
```

Acesse:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🧪 Teste rápido (Smoke Test)

```bash
python src/data_collection.py
python src/data_processing.py
python src/nlp_pipeline.py
python src/ml_pipeline.py
streamlit run src/dashboard.py
```

Se tudo rodar sem erro → instalação perfeita.

---

## 🛠 Dependências úteis (caso necessário)

```bash
pip install pyarrow
pip install beautifulsoup4
pip install certifi
```
