import os
import requests
from bs4 import BeautifulSoup
import zipfile
import pandas as pd
from io import BytesIO

# URL base do INPE (dados mensais Brasil)
BASE_URL = "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/mensal/Brasil/"
RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def listar_arquivos():
    """Lista todos os arquivos disponíveis no diretório mensal do INPE."""
    print("🔍 Buscando lista de arquivos no INPE...")
    resp = requests.get(BASE_URL, verify=False)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    arquivos = [
        a.get("href") for a in soup.find_all("a")
        if a.get("href") and a.get("href").endswith(".zip")
    ]
    print(f"📦 {len(arquivos)} arquivos encontrados.")
    return arquivos

def baixar_e_extrair(arquivo):
    """Baixa e extrai um arquivo .zip do INPE."""
    url = BASE_URL + arquivo
    zip_path = os.path.join(RAW_DIR, arquivo)
    csv_dir = os.path.join(RAW_DIR, arquivo.replace(".zip", ""))
    
    if os.path.exists(csv_dir):
        print(f"✅ {arquivo} já extraído. Pulando.")
        return
    
    print(f"⬇️ Baixando {arquivo}...")
    resp = requests.get(url, verify=False)
    resp.raise_for_status()
    
    with zipfile.ZipFile(BytesIO(resp.content)) as z:
        z.extractall(csv_dir)
    
    print(f"📂 Extraído em {csv_dir}")

def unificar_csvs():
    """Lê todos os CSVs extraídos e unifica em um único DataFrame."""
    print("🧩 Unificando todos os CSVs em um só arquivo...")
    all_dfs = []
    
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(root, file)
                try:
                    df = pd.read_csv(path, sep=";", encoding="latin1")
                    all_dfs.append(df)
                except Exception as e:
                    print(f"⚠️ Erro ao ler {file}: {e}")
    
    if not all_dfs:
        print("❌ Nenhum CSV encontrado.")
        return
    
    df_final = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(PROCESSED_DIR, "focos_mensal_brasil.csv")
    df_final.to_csv(out_path, index=False)
    print(f"✅ Arquivo consolidado salvo em: {out_path}")
    print(f"📊 Total de registros: {len(df_final)}")

if __name__ == "__main__":
    print("🚀 Iniciando coleta de dados de queimadas (INPE)...")
    arquivos = listar_arquivos()
    for arq in arquivos:
        baixar_e_extrair(arq)
    unificar_csvs()
    print("🏁 Coleta e unificação concluídas com sucesso!")
