# -------------------------
# imports e constantes
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import re
from datetime import datetime
from dateutil import parser as dateparser  # pip install python-dateutil
from difflib import get_close_matches

DATA_FEAT = "data/processed/features_state_month.parquet"
PRED_CSV = "data/processed/predictions.csv"
NLP_KW = "data/processed/nlp_keywords.csv"
MODEL_PATH = "data/models/rf_model.joblib"

# Load data FIRST so helpers can reference them safely
@st.cache_data
def load_data():
    df = None
    preds = None
    if os.path.exists(DATA_FEAT):
        df = pd.read_parquet(DATA_FEAT)
    if os.path.exists(PRED_CSV):
        preds = pd.read_csv(PRED_CSV)
    kw = pd.read_csv(NLP_KW) if os.path.exists(NLP_KW) else None
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    return df, preds, kw, model

df, preds, kw, model = load_data()


# -------------------------
# Helpers for improved chat
# -------------------------
# small mapping state name <-> UF (cleaner, priority: canonical full name mapping)
STATE_FULL = {
    "AC":"Acre","AL":"Alagoas","AP":"Amapá","AM":"Amazonas","BA":"Bahia","CE":"Ceará",
    "DF":"Distrito Federal","ES":"Espírito Santo","GO":"Goiás","MA":"Maranhão","MT":"Mato Grosso",
    "MS":"Mato Grosso do Sul","MG":"Minas Gerais","PA":"Pará","PB":"Paraíba","PR":"Paraná",
    "PE":"Pernambuco","PI":"Piauí","RJ":"Rio de Janeiro","RN":"Rio Grande do Norte","RS":"Rio Grande do Sul",
    "RO":"Rondônia","RR":"Roraima","SC":"Santa Catarina","SP":"São Paulo","SE":"Sergipe","TO":"Tocantins"
}
# lowercase name -> sigla
NAME_TO_SIGLA = {v.lower():k for k,v in STATE_FULL.items()}
# create aliases for common spellings/without accents
ALIAS_MAP = {
    "amapa":"AP","amapá":"AP","paraiba":"PB","paraíba":"PB","piaui":"PI","piauí":"PI",
    "sao paulo":"SP","são paulo":"SP","ceara":"CE","ceará":"CE","espirito santo":"ES","espírito santo":"ES",
    "mato grosso":"MT","mato grosso do sul":"MS","rondonia":"RO","rondônia":"RO","maranhao":"MA","maranhão":"MA"
}
# combined lookup
def normalize_state(text):
    txt = (text or "").lower()
    # check for sigla pattern
    mo = re.search(r'\b([A-Za-z]{2})\b', txt)
    if mo:
        cand = mo.group(1).upper()
        if cand in STATE_FULL:
            return cand
    # direct name match (full names)
    for name, sig in NAME_TO_SIGLA.items():
        if name in txt:
            return sig
    # alias map
    for alias, sig in ALIAS_MAP.items():
        if alias in txt:
            return sig
    # fuzzy match against names
    matches = get_close_matches(txt, list(NAME_TO_SIGLA.keys()), n=1, cutoff=0.8)
    if matches:
        return NAME_TO_SIGLA[matches[0]]
    return None

def parse_requested_month(msg, preds_local):
    msg_l = (msg or "").lower()
    # explicit yyyy-mm or yyyy/mm
    mo = re.search(r'(\d{4})[-/](\d{1,2})', msg_l)
    if mo:
        y = int(mo.group(1)); m = int(mo.group(2))
        return "single", f"{y}-{m:02d}"
    # month name + year
    mo2 = re.search(r'(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\s+(\d{4})', msg_l)
    months_map = {"jan":1,"fev":2,"mar":3,"abr":4,"mai":5,"jun":6,"jul":7,"ago":8,"set":9,"out":10,"nov":11,"dez":12}
    if mo2:
        mon = mo2.group(1)[:3]
        year = int(mo2.group(2))
        m = months_map.get(mon, None)
        if m:
            return "single", f"{year}-{m:02d}"
    # próximos N meses
    mo3 = re.search(r'pr[oó]ximos?\s+(\d+)\s+mes', msg_l) or re.search(r'next\s+(\d+)\s+month', msg_l)
    if mo3:
        n = int(mo3.group(1))
        if preds_local is None:
            return None, None
        last = sorted(preds_local["ano_mes"].unique())[-1]
        y0, m0 = map(int, last.split("-"))
        months = []
        for i in range(1, n+1):
            mm = m0 + i
            yy = y0 + (mm-1)//12
            mm = ((mm-1)%12)+1
            months.append(f"{yy}-{mm:02d}")
        return "range", months
    # próximo mês
    if "próximo mês" in msg_l or "proximo mes" in msg_l or "proximo mês" in msg_l:
        if preds_local is None:
            return None, None
        last = sorted(preds_local["ano_mes"].unique())[-1]
        y0, m0 = map(int, last.split("-"))
        mm = m0 + 1
        yy = y0 + (mm-1)//12
        mm = ((mm-1)%12)+1
        return "single", f"{yy}-{mm:02d}"
    return None, None

def top_n_for_month(ano_mes, n=5):
    if preds is None:
        return None
    if ano_mes not in preds["ano_mes"].values:
        return None
    return preds[preds["ano_mes"]==ano_mes].sort_values("predicted_focos_next", ascending=False).head(n)

def predict_for_state_month(state_sigla, ano_mes):
    if model is None or df is None or preds is None:
        return None
    try:
        y, m = map(int, ano_mes.split("-"))
    except Exception:
        return None
    mean_by_state = df.groupby("estado")["focos"].mean().to_dict()
    # map sigla to full name using STATE_FULL
    state_full = STATE_FULL.get(state_sigla)
    if state_full is None:
        # try to find by startswith in preds
        for s in preds["estado"].unique():
            if s.lower().startswith(state_sigla.lower()):
                state_full = s
                break
    if state_full is None:
        return None
    focos_val = mean_by_state.get(state_full, df["focos"].mean())
    # build a minimal X — *adapt if your model expects other features*
    try:
        if "estado_encoded" in df.columns:
            estado_encoded = int(df[df["estado"]==state_full]["estado_encoded"].mode().iloc[0])
        else:
            estado_encoded = list(sorted(preds["estado"].unique())).index(state_full)
        X = np.array([[estado_encoded, y, m, focos_val]])
        pred = model.predict(X)[0]
        return float(pred)
    except Exception:
        return None

CHAT_HISTORY_PATH = "data/processed/chat_history.csv"
def save_chat_history(history):
    try:
        import csv, os
        os.makedirs(os.path.dirname(CHAT_HISTORY_PATH), exist_ok=True)
        with open(CHAT_HISTORY_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for who,msg in history:
                writer.writerow([datetime.now().isoformat(), who, msg])
    except Exception:
        pass
    
st.set_page_config(layout="wide", page_title="Preditor de Risco de Queimadas")

st.title("🔥 Preditor de Risco de Queimadas — Dashboard")
st.markdown("Visualização de previsões mensais por estado. Chat simples para consultas rápidas.")
st.sidebar.header("Filtros")

# Construir lista de estados disponíveis
if preds is not None:
    state_list = sorted(preds["estado"].dropna().unique().tolist())
elif df is not None:
    state_list = sorted(df["estado"].dropna().unique().tolist())
else:
    state_list = ["BRASIL"]

# Adicionar opção de visão nacional no topo
if "BRASIL" not in state_list:
    state_list = ["BRASIL"] + state_list

# Construir lista de meses disponíveis (histórico + previsões futuras)
if df is not None and preds is not None:
    all_months = sorted(set(df["ano_mes"].unique()).union(set(preds["ano_mes"].unique())))
elif df is not None:
    all_months = sorted(df["ano_mes"].unique())
elif preds is not None:
    all_months = sorted(preds["ano_mes"].unique())
else:
    all_months = ["-"]

# Sidebar controls
select_yearmonth = st.sidebar.selectbox("Selecione mês (ano-mes):", all_months)
selected_state = st.sidebar.selectbox("Escolha o estado:", state_list, index=0)

st.sidebar.markdown("---")
st.sidebar.write("Executar NLP / Atualizar dados")
if st.sidebar.button("Rodar NLP (se existir texts.csv)"):
    st.sidebar.info("Rodando NLP... execute `python src/nlp_pipeline.py` no terminal para atualizar arquivos.")
st.sidebar.write("Observação: atualize via terminal se necessário.")

# Main: show time series
col1, col2 = st.columns((2, 1))

with col1:
    st.header("Série temporal: reais vs previstos")
    if preds is None:
        st.warning("Arquivo de previsões não encontrado. Rode `python src/ml_pipeline.py` primeiro.")
    else:
        display_df = preds.copy()
        if selected_state != "BRASIL":
            display_df = display_df[display_df["estado"] == selected_state]
        # aggregate by ano_mes summing actuals and predictions (if multiple states)
        agg = display_df.groupby("ano_mes")[["focos_next","predicted_focos_next"]].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(agg["ano_mes"], agg["focos_next"], marker='o', label="Real")
        ax.plot(agg["ano_mes"], agg["predicted_focos_next"], marker='x', label="Previsto")
        ax.set_xticks(agg["ano_mes"][::max(1,len(agg)//10)])
        ax.set_xticklabels(agg["ano_mes"][::max(1,len(agg)//10)], rotation=45)
        ax.set_ylabel("Número de focos")
        ax.set_title(f"Reais vs Previstos — {selected_state}")
        ax.legend()
        st.pyplot(fig)

with col2:
    st.header("Resumo / Erros")
    if preds is not None:
        if selected_state == "BRASIL":
            m_mae = preds["erro_absoluto"].mean()
            st.metric("Erro médio absoluto (Brasil)", f"{m_mae:,.0f} focos")
            st.write("Top 5 estados com maior erro médio:")
            top_err = preds.groupby("estado")["erro_absoluto"].mean().sort_values(ascending=False).head(5)
            st.table(top_err.reset_index().rename(columns={"erro_absoluto":"erro_medio"}))
        else:
            sub = preds[preds["estado"]==selected_state]
            if not sub.empty:
                st.metric("Erro médio absoluto (estado)", f"{sub['erro_absoluto'].mean():.0f} focos")
                st.write(sub.tail(5)[["ano_mes","focos_next","predicted_focos_next","erro_absoluto"]].set_index("ano_mes"))
            else:
                st.info("Sem dados para esse estado no arquivo de previsões.")

st.markdown("---")
# NLP keywords
st.header("NLP — principais palavras-chave")
if kw is None:
    st.info("Arquivo de keywords não encontrado. Rode `python src/nlp_pipeline.py` para gerar `data/processed/nlp_keywords.csv`.")
else:
    st.table(kw.head(30))

st.markdown("---")
# Chat simples baseado em regras
st.header("Chat (respostas rápidas)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def chat_response(msg):
    if not msg or preds is None:
        return "Ainda não há dados de previsão carregados. Rode o pipeline e atualize."

    msg_lower = msg.lower()
    if re.search(r'\b(ajuda|como usar|o que posso perguntar|exemplos)\b', msg_lower):
        return ("Você pode perguntar por exemplo:\n"
                "- 'Top 5 estados para 2025-06'\n"
                "- 'Qual o risco no MT 2025-06?'\n"
                "- 'Top 3 estados próximos 2 meses'\n"
                "- 'Mostre previsão para Pará 2025-12'\n")

    mo_topn = re.search(r'top\s+(\d+)', msg_lower)
    top_n = int(mo_topn.group(1)) if mo_topn else 5

    month_type, month_val = parse_requested_month(msg, preds)
    state_sigla = normalize_state(msg)

    # Top states
    if "top" in msg_lower and "estado" in msg_lower:
        if month_type == "range":
            lines = []
            for am in month_val:
                top = top_n_for_month(am, n=top_n)
                if top is None or top.empty:
                    lines.append(f"{am}: sem dados")
                else:
                    lines.append(f"{am}: " + "; ".join([f"{r['estado']} ({int(r['predicted_focos_next']):,})" for _,r in top.iterrows()]))
            return " | ".join(lines)
        elif month_type == "single":
            top = top_n_for_month(month_val, n=top_n)
            if top is None:
                last = sorted(preds['ano_mes'].unique())[-1]
                return f"Não há previsões para {month_val}. Último mês disponível: {last}"
            lines = [f"{r['estado']}: {int(r['predicted_focos_next']):,} focos" for _,r in top.iterrows()]
            return f"Top {top_n} previstos para {month_val}: " + "; ".join(lines)
        else:
            last = sorted(preds["ano_mes"].unique())[-1]
            top = top_n_for_month(last, n=top_n)
            lines = [f"{r['estado']}: {int(r['predicted_focos_next']):,} focos" for _,r in top.iterrows()]
            return f"Top {top_n} previstos para {last}: " + "; ".join(lines)

    # Risk for state
    if state_sigla:
        if month_type == "single":
            row = preds[(preds["estado"].str.upper()==STATE_FULL.get(state_sigla,"").upper()) | (preds["estado"].str.upper()==state_sigla)]
            row = row[row["ano_mes"]==month_val]
            if not row.empty:
                r = row.iloc[0]
                val = int(r["predicted_focos_next"])
                real = r["focos_next"]
                return (f"Previsão para {r['estado']} ({r['ano_mes']}): {val:,} focos. "
                        + (f"Valor real: {int(real):,}." if not np.isnan(real) else "") + f" [Fonte: predictions.csv]")
            pred_on_demand = predict_for_state_month(state_sigla, month_val)
            if pred_on_demand is not None:
                return f"Previsão (gerada on-demand) para {state_sigla} {month_val}: {int(pred_on_demand):,} focos."
            else:
                return "Não encontrei previsão para esse estado/mês."
        elif month_type == "range":
            totals = []
            for am in month_val:
                row = preds[(preds["estado"].str.upper()==STATE_FULL.get(state_sigla,"").upper()) | (preds["estado"].str.upper()==state_sigla)]
                row_this = row[row["ano_mes"]==am]
                if not row_this.empty:
                    totals.append(row_this.iloc[0]["predicted_focos_next"])
            if totals:
                return f"Soma de previsões para {state_sigla} nos meses solicitados: {int(sum(totals)):,} focos."
            else:
                return "Sem previsões para esse intervalo."
        else:
            row = preds[(preds["estado"].str.upper()==STATE_FULL.get(state_sigla,"").upper()) | (preds["estado"].str.upper()==state_sigla)]
            if not row.empty:
                r = row.sort_values("ano_mes").tail(1).iloc[0]
                return f"Última previsão disponível para {r['estado']} ({r['ano_mes']}): {int(r['predicted_focos_next']):,} focos."
            else:
                return "Sem dados para esse estado."

    # growth intent
    if any(k in msg_lower for k in ["crescimento","maior aumento","cresc"]):
        months = sorted(preds["ano_mes"].unique())
        if len(months) < 2:
            return "Não há meses suficientes para calcular crescimento."
        last, prev = months[-1], months[-2]
        df_change = preds[preds["ano_mes"].isin([prev,last])].pivot(index="estado", columns="ano_mes", values="predicted_focos_next").dropna()
        df_change["pct"] = (df_change[last] - df_change[prev]) / df_change[prev]
        top_growth = df_change["pct"].sort_values(ascending=False).head(5)
        lines = [f"{idx}: {pct*100:.1f}%" for idx,pct in top_growth.items()]
        return "Top 5 estados por aumento percentual (último vs anterior): " + "; ".join(lines)

    return ("Desculpe — não entendi. Exemplos válidos:\n"
            "- 'Top 5 estados para 2025-06'\n"
            "- 'Qual o risco no MT 2025-06?'\n")

# chat UI
user_input = st.text_input("Digite sua pergunta:", key="input")
if st.button("Enviar"):
    if user_input:
        resp = chat_response(user_input)
        st.session_state.chat_history.append(("Você", user_input))
        st.session_state.chat_history.append(("Sistema", resp))
        # salvamos as duas últimas mensagens
        save_chat_history(st.session_state.chat_history[-2:])

# show chat history
for who, msg in st.session_state.chat_history[-10:]:
    if who == "Você":
        st.markdown(f"**Você:** {msg}")
    else:
        st.markdown(f"**Sistema:** {msg}")

st.markdown("---")
st.caption("Observação: Chat é baseado em regras usando o arquivo de previsões (predictions.csv) — não é um modelo de linguagem grande.")
