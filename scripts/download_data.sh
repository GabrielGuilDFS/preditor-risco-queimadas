#!/usr/bin/env bash

set -euo pipefail

echo "🔥 Iniciando download dos dados de queimadas..."

# Diretórios
RAW_DIR="data/raw"
PROC_DIR="data/processed"

mkdir -p "$RAW_DIR"
mkdir -p "$PROC_DIR"

# Lista de meses que você quer baixar (edite livremente)
ANO_ATUAL=$(date +"%Y")
ANOS=($((ANO_ATUAL-1)) $ANO_ATUAL)

MESES=()
for ANO in "${ANOS[@]}"; do
  for M in {01..12}; do
    MESES+=("${ANO}${M}")
  done
done


# URL base do BDQueimadas (exemplo real utilizado no original)
BASE_URL="https://queimadas.dgi.inpe.br/queimadas/bdqueimadas/v1/focos/csv"

for MES in "${MESES[@]}"; do
    ANO="${MES:0:4}"
    MES_NUM="${MES:4:2}"

    ZIP="$RAW_DIR/focos_mensal_br_${MES}.zip"
    OUT_DIR="$RAW_DIR/focos_mensal_br_${MES}"

    echo "⬇️  Baixando $MES..."

    # Baixar CSV comprimido ou ZIP
    wget -q --show-progress \
      "${BASE_URL}?dataInicial=${ANO}-${MES_NUM}-01&dataFinal=${ANO}-${MES_NUM}-31" \
      -O "$ZIP"

    echo "📂 Extraindo para $OUT_DIR..."
    mkdir -p "$OUT_DIR"
    unzip -o "$ZIP" -d "$OUT_DIR" >/dev/null 2>&1 || true
done

echo "🎉 Download concluído!"
echo "Execute agora: python src/data_processing.py"
