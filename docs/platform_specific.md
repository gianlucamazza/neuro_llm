# Configurazioni Specifiche per Piattaforma

Guide di installazione e configurazione per piattaforme specifiche.

## macOS (Apple Silicon M1/M2)

### PyTorch per ARM

```bash
# PyTorch per Apple Silicon
pip install torch torchvision

# Verifica architettura
python -c "import platform; print(platform.machine())"
# Output: arm64

# LNN funziona nativamente
pip install lnn
```

### Problemi Comuni

- **Xcode Command Line Tools**: Installa se mancano compilatori
  ```bash
  xcode-select --install
  ```

## Windows con WSL2

### Installazione WSL2

```bash
# Installa WSL2 (da PowerShell come Administrator)
wsl --install

# Riavvia sistema
# Apri WSL e configura Ubuntu
```

### Dentro WSL

```bash
# Aggiorna sistema
sudo apt update
sudo apt install python3.10 python3-pip

# Installa dipendenze
pip3 install -r requirements.txt
```

### Vantaggi WSL2

- Ambiente Linux completo
- Migliore compatibilità con LNN
- Accesso a GPU NVIDIA (se disponibile)

## Linux Server (no GUI)

### Backend Matplotlib

```bash
# Matplotlib richiede backend non-interattivo
export MPLBACKEND=Agg

# Oppure configura permanentemente
echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc

# Installa normalmente
pip install -r requirements.txt
```

### Jupyter su Server

```bash
# Installa Jupyter
pip install jupyter ipykernel

# Configura per accesso remoto
jupyter notebook --generate-config
# Modifica ~/.jupyter/jupyter_notebook_config.py
# c.NotebookApp.ip = '0.0.0.0'
# c.NotebookApp.open_browser = False

# Avvia
jupyter notebook --port=8889
```

## Google Colab

### Installazione Automatica

I notebook includono installazione automatica:

```python
!pip install -q lnn>=1.2.0 torch>=2.0.0 numpy>=1.24.0
```

### GPU su Colab

```python
# Verifica GPU
import torch
print(torch.cuda.is_available())  # True se GPU disponibile

# Sposta modello su GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)  # Per modelli PyTorch
```

### Limiti Colab

- **Tempo**: Sessione si disconnette dopo 12h inattività
- **RAM**: ~12GB gratuito, fino a 25GB Pro
- **GPU**: Limitata, può essere revocata

## Docker

### Dockerfile per LNN

```dockerfile
FROM python:3.10-slim

# Installa dipendenze sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crea utente non-root
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copia requirements
COPY requirements.txt .

# Installa Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copia codice
COPY . .

# Espone porta per Jupyter (opzionale)
EXPOSE 8888

CMD ["python", "examples/01_medical_expert/medical_diagnosis.py"]
```

### Build e Run

```bash
# Build
docker build -t lnn-app .

# Run
docker run -it lnn-app

# Con GPU (se disponibile)
docker run --gpus all -it lnn-app
```

## Cloud Platforms

### AWS EC2

#### Instance Type
- **CPU**: t3.medium (2 vCPU, 4GB RAM) - sufficiente per esempi base
- **GPU**: p3.2xlarge (1 GPU V100) - per training intensivo

#### Setup
```bash
# Aggiorna sistema
sudo yum update -y

# Installa Python 3.10
sudo yum install -y python310 python310-pip

# Installa CUDA (per GPU instances)
# Segui guida NVIDIA: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

# Installa LNN
pip3 install lnn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Google Cloud Platform

#### VM Instance
- **CPU**: e2-medium (2 vCPU, 4GB RAM)
- **GPU**: A100 instances per performance massima

#### Setup
```bash
# Installa CUDA (per GPU)
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py

# Installa LNN
pip install lnn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Azure VM

#### VM Size
- **CPU**: Standard_B2s (2 vCPU, 4GB RAM)
- **GPU**: NC6 (1 GPU K80) o NV12 (2 GPU M60)

#### Setup
```bash
# Installa CUDA
# Segui Microsoft docs: https://docs.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup

# Installa LNN
pip install lnn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## HPC Clusters

### SLURM

```bash
# Job script SLURM
#!/bin/bash
#SBATCH --job-name=lnn_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Carica moduli
module load python/3.10
module load cuda/11.8

# Attiva environment
source ~/venv/bin/activate

# Run training
python examples/04_learning/learning_weights.py
```

### PBS/Torque

```bash
# Job script PBS
#PBS -N lnn_job
#PBS -l nodes=1:ppn=8
#PBS -l mem=32gb
#PBS -l walltime=24:00:00
#PBS -q gpu

# Setup environment
module load python/3.10
module load cuda/11.8

cd $PBS_O_WORKDIR
source venv/bin/activate

python examples/01_medical_expert/medical_diagnosis.py
```

## Configurazioni Avanzate

### Ottimizzazione Performance

#### CPU Optimization
```bash
# Usa MKL per Intel CPUs
conda install mkl mkl-service
export MKL_NUM_THREADS=4  # Numero di core CPU
```

#### GPU Optimization
```python
# PyTorch GPU optimization
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
```

### Environment Variables

```bash
# PyTorch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# LNN
export LNN_LOG_LEVEL=INFO

# Jupyter
export JUPYTER_CONFIG_DIR=~/.jupyter
```

---

**Nota**: Per problemi specifici della tua piattaforma, consulta la [documentazione troubleshooting](troubleshooting.md).</content>
</xai:function_call<docs/platform_specific.md