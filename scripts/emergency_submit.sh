#!/usr/bin/env bash
# Envio manual de emergencia del ciclo abierto, cuando GitHub Actions no esta entregando.
#
# Usa el campeon VIGENTE (el champion-package del ultimo reentrenamiento exitoso), nunca un
# paquete viejo que haya quedado en artifacts/. Requiere PULSO_API_KEY en .env y `gh`
# autenticado. SUPABASE_SERVICE_ROLE_KEY es opcional: sin ella se envia igual, pero el
# recibo no queda en Supabase (no aparece en el dashboard).
#
#   bash scripts/emergency_submit.sh            # primer envio del ciclo
#   bash scripts/emergency_submit.sh 2          # reemplaza la entrega del ciclo (intento 2 o 3)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${PULSO_REPO:-mjcastano29-tech/competition-MLOPS}"
ATTEMPT="${1:-0}"
BUNDLE="artifacts/emergency_champion"

set -a; [[ -f .env ]] && . ./.env; set +a
: "${PULSO_API_KEY:?Falta PULSO_API_KEY en .env}"

run_id=$(gh run list -R "$REPO" --workflow retrain_on_drift.yml --limit 20 \
  --json databaseId,conclusion --jq '[.[]|select(.conclusion=="success")][0].databaseId')
echo "Campeon del reentrenamiento $run_id"
rm -rf "$BUNDLE"
gh run download -R "$REPO" "$run_id" -n champion-package -D "$BUNDLE"

args=(--output artifacts/emergency_submission.json)
[[ "$ATTEMPT" != "0" ]] && args+=(--replace-attempt "$ATTEMPT")
PULSO_BUNDLE_DIR="$BUNDLE" PYTHONPATH=src PYTHONWARNINGS=ignore \
  python3 scripts/infer_and_submit.py "${args[@]}"
