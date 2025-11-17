import os
import pandas as pd
from datetime import datetime

RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def carregar_todos_csvs():
    """Lê todos os CSVs extraídos da pasta data/raw/."""
    print("📂 Lendo arquivos CSV de queimadas...")
    all_dfs = []

    for root, _, files in os.walk(RAW_DIR):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(root, file)
                try:
                    # Detectar separador automaticamente
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        first_line = f.readline()
                    sep = ";" if ";" in first_line else ","

                    # Ler CSV com encoding e separador detectados
                    df = pd.read_csv(path, sep=sep, encoding="utf-8", low_memory=False)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"⚠️ Erro ao ler {file}: {e}")

    if not all_dfs:
        raise ValueError("❌ Nenhum arquivo CSV encontrado em data/raw/")
    
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ {len(df):,} registros carregados.")
    print(f"📋 Colunas detectadas: {df.columns.tolist()[:10]}")
    return df

def limpar_e_processar(df):
    """Limpa colunas, ajusta tipos e gera colunas de data e mês."""
    print("🧹 Limpando e processando dados...")

    # Tentar detectar a coluna de data
    col_data = None
    for col in df.columns:
        if col.lower() in ["datahora", "data", "data_observacao", "data_detecta", "data_hora", "data_hora_gmt"]:
            col_data = col
            break
    # Converter para datetime
    df[col_data] = pd.to_datetime(df[col_data], errors='coerce', dayfirst=True)
    df['ano_mes'] = df[col_data].dt.to_period('M').astype(str)

    # Detectar coluna de estado
    col_estado = None
    for col in df.columns:
        if col.lower() in ["estado", "uf", "nomestado"]:
            col_estado = col
            break

    if not col_estado:
        raise KeyError(f"❌ Nenhuma coluna de estado encontrada. Colunas disponíveis: {list(df.columns)[:10]}")

    df['estado'] = df[col_estado].astype(str).str.upper().str.strip()

    # Agrupar por estado e mês
    df_proc = df[['estado', 'ano_mes']].copy()
    df_agg = df_proc.groupby(['estado', 'ano_mes']).size().reset_index(name='focos')

    print(f"📊 Dados agregados: {df_agg.shape[0]} linhas, {df_agg.shape[1]} colunas.")
    return df_agg


def criar_target(df):
    """Cria a coluna 'focos_next' (focos do próximo mês por estado)."""
    print("🧩 Criando variável alvo (focos_next)...")
    df['ano_mes'] = pd.to_datetime(df['ano_mes'])
    df = df.sort_values(['estado', 'ano_mes'])

    df['focos_next'] = df.groupby('estado')['focos'].shift(-1)
    df = df.dropna(subset=['focos_next'])
    df['ano_mes'] = df['ano_mes'].dt.to_period('M').astype(str)
    return df

def salvar_dataset(df):
    """Salva dataset final em Parquet."""
    out_path = os.path.join(PROCESSED_DIR, "features_state_month.parquet")
    df.to_parquet(out_path, index=False)
    print(f"💾 Dataset salvo em: {out_path}")
    print(df.head())

if __name__ == "__main__":
    print("🚀 Iniciando processamento dos dados de queimadas...")
    df_raw = carregar_todos_csvs()
    df_clean = limpar_e_processar(df_raw)
    df_final = criar_target(df_clean)
    salvar_dataset(df_final)
    print("🏁 Processamento concluído com sucesso!")
