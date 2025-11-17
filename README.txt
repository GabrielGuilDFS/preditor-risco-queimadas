PREDITOR DE RISCO DE QUEIMADAS
==============================

Este projeto contém pipelines completos de:
- Coleta de dados
- Processamento e limpeza
- NLP (análise de textos)
- Treinamento de modelo de previsão
- Dashboard interativo (Streamlit)

As pastas data/ estão vazias porque os arquivos são grandes
e devem ser baixados pelo script scripts/download_data.sh.

-----------------------------------------
1) CRIAR AMBIENTE VIRTUAL
-----------------------------------------

python3 -m venv .venv
source .venv/bin/activate

-----------------------------------------
2) INSTALAR DEPENDÊNCIAS
-----------------------------------------

pip install --upgrade pip
pip install -r requirements.txt

-----------------------------------------
3) BAIXAR DADOS (OBRIGATÓRIO)
-----------------------------------------

bash scripts/download_data.sh

Isso vai preencher:

data/raw/
data/processed/

-----------------------------------------
4) EXECUTAR PIPELINE DE PROCESSAMENTO
-----------------------------------------

python src/data_processing.py

-----------------------------------------
5) EXECUTAR PIPELINE DE NLP (OPCIONAL)
-----------------------------------------

python src/nlp_pipeline.py

Se aparecer a mensagem:
"data/raw/texts.csv não encontrada"

Crie um arquivo texts.csv assim:

echo "id,date,text,source" > data/raw/texts.csv
echo "1,2025-01-01,Focos aumentam no estado,g1" >> data/raw/texts.csv

-----------------------------------------
6) TREINAR MODELO (OPCIONAL)
-----------------------------------------

python src/ml_pipeline.py

O modelo será salvo em:

data/models/

-----------------------------------------
7) INICIAR DASHBOARD (STREAMLIT)
-----------------------------------------

streamlit run src/dashboard.py

O dashboard abrirá no navegador local:

http://localhost:8501

-----------------------------------------
8) ESTRUTURA DO PROJETO
-----------------------------------------

data/
    raw/        -> dados brutos baixados
    processed/  -> dados processados
    models/     -> modelo treinado
src/
    data_collection.py
    data_processing.py
    ml_pipeline.py
    nlp_pipeline.py
    dashboard.py
scripts/
    download_data.sh
notebooks/
requirements.txt
README.txt

-----------------------------------------
9) AJUDA
-----------------------------------------

Se algum erro ocorrer ao salvar arquivos Parquet:
instale manualmente:

pip install pyarrow

Se erro for de BeautifulSoup:

pip install beautifulsoup4

Se erro for de certifi/SSL:

pip install certifi
