#!/bin/bash
#SBATCH --job-name=ga-array-test
#SBATCH --output=test_fold_%a.out
#SBATCH --error=test_blad_fold_%a.err
#SBATCH --partition=lem-gpu-normal
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:hopper:4,storage:local:4G,storage:shm:16G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=36G
#SBATCH --array=0-1
#SBATCH --nodelist=r16-12,r16-14

# ===============================
#   Konfiguracja i informacje
# ===============================
echo "--- TESTOWE ZADANIE TABLICOWE ---"
echo "Zadanie tablicowe $SLURM_ARRAY_JOB_ID, fold $SLURM_ARRAY_TASK_ID"
echo "Węzeł: $SLURMD_NODENAME"
echo "Data: $(date)"
echo "Katalog roboczy: $(pwd)"
echo "=================================="

# ===============================
#   Moduły
# ===============================
source /usr/local/sbin/modules.sh
module load Python/3.11.5-GCCcore-13.2.0 CUDA/12.1.1

# ===============================
#   Środowisko wirtualne
# ===============================
PROJECT_DIR="$HOME/tmp/genetyk/HPO_Tuner"
VENV_PATH="$PROJECT_DIR/.venv"

cd "$PROJECT_DIR" || { echo "❌ Nie znaleziono katalogu projektu!"; exit 1; }

# Sprawdź, czy środowisko istnieje i działa
if [ ! -x "$VENV_PATH/bin/python3" ]; then
    echo "⚠️  Nie znaleziono poprawnego środowiska Python. Tworzę nowe..."
    rm -rf "$VENV_PATH"
    python3 -m venv "$VENV_PATH"
fi

# Aktywuj środowisko
source "$VENV_PATH/bin/activate"

# Sprawdź, czy pip działa
if ! "$VENV_PATH/bin/pip" --version >/dev/null 2>&1; then
    echo "⚠️  Pip nie działa — reinstalacja środowiska..."
    rm -rf "$VENV_PATH"
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
fi

# Upewnij się, że pip działa
python3 -m ensurepip --upgrade
pip install --upgrade pip setuptools wheel

# Zainstaluj wymagania, jeśli plik istnieje
if [ -f requirements.txt ]; then
    echo "📦 Instaluję wymagania..."
    pip install -r requirements.txt
else
    echo "⚠️  Brak pliku requirements.txt — pomijam instalację."
fi

export OMP_DYNAMIC="FALSE"
export KMP_BLOCKTIME="1"
export KMP_AFFINITY="granularity=fine,compact,1,0"
export ONEDNN_MAX_CPU_ISA="AVX512_CORE_AMX"
export DATA_DIR="$(pwd)"

ulimit -n 65535
# ===============================
#   Uruchomienie eksperymentu
# ===============================
echo "🚀 Uruchamiam eksperyment dla fold-u $SLURM_ARRAY_TASK_ID..."
python3 main.py 

echo "✅ Test fold $SLURM_ARRAY_TASK_ID zakończony pomyślnie."

