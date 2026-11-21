"""Descarga Qwen3-0.6B para que inference.py pueda cargarlo desde el disco.

Ejecuta este script desde la raíz del proyecto. La primera descarga necesita
acceso a Hugging Face y guarda los archivos dentro de models/.
"""

from huggingface_hub import snapshot_download

# La copia local incluye los pesos y los archivos del tokenizador.
path = snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir="./models/qwen3-0.6b"
)

print(f"Modelo descargado en: {path}")
