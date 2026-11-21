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

## Paso 6. Inspeccionar los pesos

```bash
python -m pip install -r requirements.txt
python inspect_weights.py
```

El script lee el GGUF local que usa la ruta Vulkan. Muestra cuántos valores se almacenan, el nombre, la forma y el tipo de los primeros 15 tensores, y la media y la desviación estándar del primero. La inspección del archivo se hace en CPU. En este GGUF, verás tipos como `Q8_0` (cuantizado) y `F32` (coma flotante de 32 bits).

Si ya preparaste la ruta de PyTorch, usa `python inspect_weights.py --cpu` para seguir el ejercicio con los pesos originales y los nombres de Transformers.

## Paso 7. Medir el efecto de cambiar un peso

Con la instalación de CPU preparada, ejecuta:

```bash
OMP_NUM_THREADS=4 python perturb_weight.py
```

El script guarda el valor original de `q_proj.weight[0, 0]` en la primera capa de atención, suma `0.1` en memoria y repite las mediciones con los mismos cuatro prompts. Usa generación voraz con un máximo de 48 tokens por respuesta. Al terminar, restaura el peso original.

El informe `.cache/weight-perturbation.json` compara los logits y las probabilidades del **siguiente token** para todo el vocabulario. Incluye el cambio máximo de logit, la distancia de variación total (la mitad de la suma de las diferencias absolutas de probabilidad) y los valores de los tokens más probables en cada condición. Las respuestas completas y sus IDs de token se comparan aparte, porque tras la primera divergencia cada generación puede estar condicionada por un prefijo distinto.

## Paso 8. Aprender un peso mediante gradientes

Con PyTorch instalado en el entorno virtual, ejecuta:

```bash
python train_linear.py
```

El ejercicio usa CPU y cuatro pares de entrenamiento, de `(1, 2)` a `(4, 8)`. Una capa lineal predice `Wx + b` y la pérdida MSE mide la media de los errores al cuadrado.

En cada iteración, `zero_grad()` limpia los gradientes acumulados, `backward()` calcula cuánto cambia la pérdida con respecto a cada parámetro y `step()` aplica la regla de SGD. Por ejemplo, el peso se actualiza como `W = W - 0.01 * gradiente_de_W`. El script muestra el cálculo de la primera actualización y el valor que aplicó PyTorch.

Tras 1000 iteraciones, compara el peso con 2, el sesgo con 0 y la predicción para la entrada 5 con su valor esperado de 10. El paso 7 fijaba manualmente cuánto cambiar un peso de Qwen3. Aquí el gradiente de la pérdida determina la actualización.

## Paso 9. Entrenar un adaptador LoRA

En el entorno virtual, con PyTorch instalado para tu equipo, ejecuta:

```bash
python -m pip install -r requirements-lora.txt
python download.py
python train_lora.py
```

El script prefiere una GPU compatible con PyTorch (CUDA, ROCm, XPU o MPS) y usa CPU si PyTorch no detecta ninguna. También puedes elegir CPU con `python train_lora.py --cpu`. Para limitar los hilos en CPU, ejecuta `OMP_NUM_THREADS=4 python train_lora.py --cpu`.

### PyTorch con ROCm en la RX 5600 XT

En Linux x86-64 con Python 3.15, la compilación de AMD TheRock probada en este equipo usa el extra `device-gfx1010`. Instálala dentro de `.venv` con `uv` desde la raíz del proyecto. Es una compilación nightly fijada a la versión probada. [Instrucciones de AMD TheRock](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#installing-multi-arch-pytorch-python-packages)

```bash
uv pip install --python .venv/bin/python \
  --index https://nightly.repo.amd.com/rocm/whl-next/ \
  --index-strategy unsafe-first-match \
  'torch[device-gfx1010]==2.16.0a0+rocm10.3.0a20261121'
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
python train_lora.py
```

PyTorch usa la interfaz `torch.cuda` también con ROCm. Por eso el script muestra `cuda` cuando entrena en la Radeon. La inferencia con llama.cpp sigue usando Vulkan.

PEFT congela los pesos originales y añade matrices entrenables a `q_proj` y `v_proj` con rango 4 y alfa 8. El entrenamiento lee [los tres ejemplos sintéticos](data/train.jsonl), realiza una pasada y usa un ejemplo por paso. El límite es de 128 tokens por ejemplo. Las etiquetas de la pregunta y la plantilla se ponen a `-100` para que la pérdida se calcule sobre la respuesta y su token de fin de turno.

Cada ejecución guarda el adaptador en `checkpoints/qwen3-0.6b-lora/`. El peso base sigue en `models/`. Los tres ejemplos comprueban que el entrenamiento y el guardado funcionan. Para medir si el adaptador generaliza, amplía los datos y reserva ejemplos independientes para validación. [Documentación de PEFT](https://huggingface.co/docs/transformers/peft)

Para adaptar una parte mayor del modelo, ejecuta `python train_lora.py --wide`. Este perfil aplica LoRA de rango 8 a las cuatro proyecciones de atención y a las tres proyecciones de la MLP en cada bloque. Los pesos base siguen congelados. El resultado se guarda en `checkpoints/qwen3-0.6b-lora-wide/`. La prueba usa los mismos tres ejemplos y una pasada, igual que el perfil inicial.

## Probar con otro mensaje

Cambia `PROMPT` en [inference.py](inference.py). En GPU, llama.cpp prepara el mensaje y ejecuta el modelo. Para estudiar esos pasos en Python, lee `generate_on_cpu()`. Allí la plantilla de chat prepara el mensaje, el modelo genera nuevos tokens y el tokenizador los convierte en texto.

## Archivos del repositorio

Git guarda los scripts, este README, las dependencias y las reglas de exclusión. `.gitignore` excluye los pesos de `models/`, el entorno `.venv/`, el ejecutable y los registros de `.cache/`, el bytecode y los archivos locales `.env`. Los pasos anteriores permiten recuperar las dependencias y los modelos tras clonar el proyecto.
