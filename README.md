# Inferencia local con Qwen3-0.6B

Laboratorio en Python para ejecutar Qwen3-0.6B y entender cómo un mensaje se convierte en una respuesta. El ejemplo usa Vulkan en la GPU por defecto y permite ejecutar el mismo mensaje con PyTorch en CPU.

## Preparar el entorno

Desde la raíz del repositorio, con Python 3.15 (versión probada):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ejecutar en la GPU

Estos pasos se probaron en Linux x86-64 con una AMD Radeon RX 5600 XT y el controlador Vulkan RADV. Descarga y extrae la compilación Vulkan de llama.cpp desde su repositorio oficial. [Lanzamiento b12602](https://github.com/ggml-org/llama.cpp/releases/tag/b12602)

```bash
mkdir -p .cache/llama.cpp
curl --fail --location \
  --output .cache/llama.cpp-vulkan.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b12602/llama-b12602-bin-ubuntu-vulkan-x64.tar.gz
echo '670cc86ce5fff46d524eb929c9881fb3a9dbdd829be334b42888411b2d59f386  .cache/llama.cpp-vulkan.tar.gz' | sha256sum --check &&
  tar -xzf .cache/llama.cpp-vulkan.tar.gz -C .cache/llama.cpp --strip-components=1
.cache/llama.cpp/llama-cli --list-devices
```

Comprueba que la lista muestre la tarjeta como `Vulkan0`. El script selecciona ese dispositivo y le asigna todas las capas. La primera ejecución descarga el modelo GGUF a `models/`. [Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)

```bash
python inference.py
```

## Ejecutar en la CPU

Instala las versiones probadas de PyTorch para CPU y de Transformers. Descarga los pesos y el tokenizador con `download.py`. [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.15.0+cpu'
python -m pip install 'transformers==5.23.0' -r requirements.txt
python download.py
python inference.py --cpu
```

## Inspeccionar los pesos

```bash
python -m pip install -r requirements.txt
python inspect_weights.py
```

El script lee el GGUF local que usa la ruta Vulkan. Muestra cuántos valores se almacenan, el nombre, la forma y el tipo de los primeros 15 tensores, y la media y la desviación estándar del primero. La inspección del archivo se hace en CPU. En este GGUF, verás tipos como `Q8_0` (cuantizado) y `F32` (coma flotante de 32 bits).

Si ya preparaste la ruta de PyTorch, usa `python inspect_weights.py --cpu` para seguir el ejercicio con los pesos originales y los nombres de Transformers.

## Probar con otro mensaje

Cambia `PROMPT` en [inference.py](inference.py). En GPU, llama.cpp prepara el mensaje y ejecuta el modelo. Para estudiar esos pasos en Python, lee `generate_on_cpu()`. Allí la plantilla de chat prepara el mensaje, el modelo genera nuevos tokens y el tokenizador los convierte en texto.

## Archivos del repositorio

Git guarda los scripts, este README, las dependencias y las reglas de exclusión. `.gitignore` excluye los pesos de `models/`, el entorno `.venv/`, el ejecutable y los registros de `.cache/`, el bytecode y los archivos locales `.env`. Los pasos anteriores permiten recuperar las dependencias y los modelos tras clonar el proyecto.
