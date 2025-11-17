"""
Funcionalidades:
- Se existir: lê data/raw/texts.csv (colunas: id, date, text, source)
- Caso contrário: tenta buscar manchetes/RSS (se houver internet)
- Classificação temática (keyword-based) em labels como: desmatamento, denuncia, controle, clima
- Extração de indicadores numéricos (ha, focos, pessoas) via regex
- Extração de entidades via spaCy (se o modelo pt_core_news_sm estiver instalado)
- Salva:
    - data/processed/nlp_indicators.csv  (id, date, indicator_type, value, unit, context)
    - data/processed/nlp_keywords.csv    (keyword, count)
    - data/processed/texts_classified.csv (id,date,text,theme)
"""

import os
import re
import csv
import requests
import pandas as pd
from collections import Counter, defaultdict

RAW_TEXTS_PATH = "data/raw/texts.csv"            # optional input (id,date,text,source)
PROCESSED_DIR = "data/processed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- helper: try importing spaCy model ---
use_spacy = False
try:
    import spacy
    try:
        nlp = spacy.load("pt_core_news_sm")
        use_spacy = True
    except Exception:
        # spaCy installed but model not found
        nlp = None
        use_spacy = False
except Exception:
    spacy = None
    nlp = None
    use_spacy = False

# --- Step 1: load texts ---
def load_texts():
    """
    Returns a dataframe with columns: id, date, text, source
    If data/raw/texts.csv not found, tries to fetch some RSS headlines (best-effort).
    """
    if os.path.exists(RAW_TEXTS_PATH):
        print("🔍 Carregando textos existentes em", RAW_TEXTS_PATH)
        df = pd.read_csv(RAW_TEXTS_PATH, encoding="utf-8", low_memory=False)
        # ensure columns
        if 'id' not in df.columns:
            df = df.reset_index().rename(columns={'index':'id'})
        if 'text' not in df.columns:
            raise ValueError("Arquivo texts.csv precisa conter coluna 'text'")
        return df[['id','date','text','source']] if 'date' in df.columns else df[['id','text']].assign(date="")
    else:
        print("⚠️ data/raw/texts.csv não encontrada — tentando coletar manchetes RSS (melhor usar texts.csv se possível).")
        # minimal RSS fetch: we'll try G1 busca por 'queimada' (caso internet/ssl funcione). This is best-effort.
        feeds = [
            "https://g1.globo.com/rss/g1/amazonia/",
            "https://g1.globo.com/rss/g1/educacao/",
            "https://feeds.folha.uol.com.br/emcimadahora/rss.xml"
        ]
        texts = []
        idx = 0
        for feed in feeds:
            try:
                r = requests.get(feed, timeout=10, verify=False)
                if r.status_code == 200:
                    # parse simple: extract <title> occurrences
                    titles = re.findall(r"<title>(.*?)</title>", r.text, flags=re.DOTALL|re.IGNORECASE)
                    for t in titles[:20]:
                        txt = re.sub("<.*?>","",t).strip()
                        if txt and ("queimad" in txt.lower() or "incêndio" in txt.lower() or "fogo" in txt.lower()):
                            texts.append({"id": idx, "date": "", "text": txt, "source": feed})
                            idx += 1
            except Exception as e:
                print("  ❌ Erro fetching feed", feed, e)
        if not texts:
            print("❌ Não foi possível coletar RSS automaticamente — crie data/raw/texts.csv com textos (id,date,text,source).")
            return pd.DataFrame(columns=['id','date','text','source'])
        return pd.DataFrame(texts)

# --- Step 2: simple thematic classification (keyword-based) ---
THEME_KEYWORDS = {
    "desmatamento": ["desmat", "desmatar", "desflorestamento"],
    "denuncia": ["denúncia", "denuncia", "alerta", "cidadao", "denunciar"],
    "controle": ["controle", "bombeiro", "combate", "contido", "contenção"],
    "clima": ["seca", "chuva", "temperatura", "meteorologia", "previsão", "precipitação"],
    "agronegocio": ["agronegócio", "queima controlada", "limpeza"]
}

def classify_text(text):
    t = text.lower()
    counts = {}
    for theme, kws in THEME_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                counts[theme] = counts.get(theme,0) + 1
    if counts:
        # choose theme with most hits
        return max(counts.items(), key=lambda x: x[1])[0]
    # fallback: neutral
    return "outros"

# --- Step 3: extract numeric indicators via regex ---
INDICATOR_PATTERN = re.compile(r'(\d+[.,]?\d*)\s*(ha|hectares|km2|km²|focos|pessoas|casas|m2|m²)', flags=re.IGNORECASE)

def extract_indicators(text):
    results = []
    for m in INDICATOR_PATTERN.finditer(text):
        val = m.group(1).replace(",",".")
        unit = m.group(2)
        context = text[max(0,m.start()-40):m.end()+40]
        results.append({"indicator":"numeric","value":float(val), "unit":unit, "context":context})
    return results

# --- Step 4: extract entities (spaCy NER) or fallback to capitalized-word heuristics ---
def extract_entities(text):
    ents = []
    if use_spacy and nlp:
        doc = nlp(text)
        for e in doc.ents:
            ents.append({"text": e.text, "label": e.label_})
    else:
        # fallback: find capitalized words sequences as possible LOC/ORG
        matches = re.findall(r'\b([A-Z][a-zà-ú]{2,}(?:\s+[A-Z][a-zà-ú]{2,})*)\b', text)
        for m in matches[:10]:
            ents.append({"text": m, "label": "PROPN"})
    return ents

# --- Main pipeline ---
def run():
    df = load_texts()
    if df.empty:
        print("Nada para processar. Saindo.")
        return

    indicators_rows = []
    keywords_counter = Counter()
    classified_rows = []

    for idx, row in df.iterrows():
        _id = row.get('id', idx)
        text = str(row.get('text',''))
        date = row.get('date','')
        theme = classify_text(text)
        classified_rows.append({"id": _id, "date": date, "text": text, "theme": theme})

        # keywords (simple tokenization)
        toks = re.findall(r'\b[a-zà-ú]{3,}\b', text.lower())
        for t in toks:
            keywords_counter[t] += 1

        # indicators
        inds = extract_indicators(text)
        for ind in inds:
            indicators_rows.append({
                "id": _id,
                "date": date,
                "indicator_type": ind["indicator"],
                "value": ind["value"],
                "unit": ind["unit"],
                "context": ind["context"],
                "source": row.get("source","")
            })

        # entities
        ents = extract_entities(text)
        # we can save top entities as keywords as well
        for e in ents:
            keywords_counter[e["text"].lower()] += 1

    # Save outputs
    df_class = pd.DataFrame(classified_rows)
    path_class = os.path.join(PROCESSED_DIR, "texts_classified.csv")
    df_class.to_csv(path_class, index=False, encoding="utf-8")
    print("✅ Texts classified ->", path_class)

    if indicators_rows:
        df_inds = pd.DataFrame(indicators_rows)
        path_inds = os.path.join(PROCESSED_DIR, "nlp_indicators.csv")
        df_inds.to_csv(path_inds, index=False, encoding="utf-8")
        print("✅ Indicators extracted ->", path_inds)
    else:
        print("ℹ️ Nenhum indicador numérico extraído.")

    # Save keywords
    most = keywords_counter.most_common(200)
    df_kw = pd.DataFrame(most, columns=["keyword","count"])
    path_kw = os.path.join(PROCESSED_DIR, "nlp_keywords.csv")
    df_kw.to_csv(path_kw, index=False, encoding="utf-8")
    print("✅ Keywords saved ->", path_kw)

    print("🏁 NLP pipeline concluído.")

if __name__ == "__main__":
    run()
