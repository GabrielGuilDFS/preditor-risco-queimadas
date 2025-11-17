#!/bin/bash
set -euo pipefail

SRC_ROOT="/root/backups/Project_IA_Queimada-backup-20251116"
DST="/home/GGabriel/Projetos/preditor-clean-clone"

# caminhos candidatos (edite aqui se quiser incluir/excluir algo)
paths=( "src" "scripts" "configs" "notebooks" "README.md" "README.txt" "requirements.txt" "requirements-dev.txt" )

echo "Destino: $DST"
mkdir -p "$DST"

for p in "${paths[@]}"; do
  if [ -e "$SRC_ROOT/$p" ]; then
    echo ">> Copiando $p ..."
    # se for arquivo (README/requirements), copia para o root do clone
    if [ -f "$SRC_ROOT/$p" ]; then
      sudo rsync -av --progress --omit-dir-times --no-perms --exclude='*.csv' --exclude='*.joblib' "$SRC_ROOT/$p" "$DST/"
    else
      sudo rsync -av --progress --omit-dir-times --no-perms --exclude='.git' --exclude='.venv*' --exclude='data' --exclude='*.csv' --exclude='*.joblib' --exclude='*.pkl' "$SRC_ROOT/$p" "$DST/"
    fi
  else
    echo "-- Pulando $p (não existe em backup)"
  fi
done

# ajustar permissões para seu usuário
sudo chown -R GGabriel:GGabriel "$DST"
echo "Pronto. Verifique com: cd $DST && git status"
