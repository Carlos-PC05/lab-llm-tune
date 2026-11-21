"""Genera una respuesta de Qwen3-0.6B en la Radeon mediante Vulkan.

Ejecuta `python inference.py` para usar la GPU. El ejecutable llama-cli
con soporte Vulkan se busca en .cache/llama.cpp, junto a este script.
La primera ejecución descarga el GGUF oficial de Qwen desde Hugging Face
si aún no está guardado en models/qwen3-0.6b-gguf.

Para usar la ruta de PyTorch en CPU, ejecuta download.py primero y luego
`python inference.py --cpu`.
"""

import argparse
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
GGUF_MODEL_PATH = PROJECT_DIR / "models/qwen3-0.6b-gguf/Qwen3-0.6B-Q8_0.gguf"
PROMPT = "Explica qué es un ataque de prompt injection."
MAX_NEW_TOKENS = 150


def generate_on_gpu():
    """Carga las capas del modelo en Vulkan y genera una sola respuesta."""
    llama_cli = PROJECT_DIR / ".cache/llama.cpp/llama-cli"
    if not llama_cli.is_file():
        raise SystemExit(f"Falta el ejecutable llama-cli con Vulkan en {llama_cli}")

    model_path = GGUF_MODEL_PATH
    if not model_path.is_file():
        from huggingface_hub import hf_hub_download

        model_path = Path(
            hf_hub_download(
                "Qwen/Qwen3-0.6B-GGUF", model_path.name, local_dir=model_path.parent
            )
        )

    # Seleccionar Vulkan0 de forma explícita hace que un dispositivo ausente
    # produzca un error. El contexto de 2048 tokens basta para este ejemplo.
    result = subprocess.run(
        [
            str(llama_cli),
            "--model", str(model_path),
            "--device", "Vulkan0",
            "--gpu-layers", "all",
            "--ctx-size", "2048",
            "--n-predict", str(MAX_NEW_TOKENS),
            "--temp", "0",
            "--chat-template-kwargs", '{"enable_thinking": false}',
            "--single-turn",
            "--no-display-prompt",
            "--simple-io",
            "--prompt", PROMPT,
        ],
        check=False,
    )
    if result.returncode:
        raise SystemExit(result.returncode)


def generate_on_cpu():
    """Carga el modelo original de Transformers en CPU e imprime la respuesta."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = PROJECT_DIR / "models/qwen3-0.6b"

    # El tokenizador convierte texto en identificadores numéricos y viceversa.
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Los pesos son los valores aprendidos por el modelo. Aquí se cargan como
    # números de 32 bits y se ejecuta el modelo en la CPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
    ).to("cpu")

    # Activa el comportamiento de evaluación de las capas del modelo.
    model.eval()

    messages = [
        {
            "role": "user",
            "content": PROMPT
        }
    ]

    # La plantilla marca el rol del mensaje y el punto donde empieza la respuesta
    # del asistente. Qwen admite desactivar su modo de razonamiento con este ajuste.
    # En la versión instalada, el resultado es un BatchEncoding con input_ids y
    # attention_mask. return_tensors="pt" convierte esos campos en tensores PyTorch.
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )

    # inference_mode evita el seguimiento de gradientes durante la generación.
    # Con do_sample=False y una sola secuencia, se elige el token más probable
    # en cada paso. max_new_tokens limita solo la parte generada.
    # Con **inputs, generate recibe los tensores input_ids y attention_mask
    # como argumentos nombrados.
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # La salida del modelo incluye los tokens del prompt al principio. Para mostrar
    # solo la respuesta, se omiten tantos tokens como contiene el prompt.
    # skip_special_tokens omite los marcadores de control al reconstruir el texto.
    prompt_length = inputs["input_ids"].shape[-1]
    answer = tokenizer.decode(
        outputs[0, prompt_length:],
        skip_special_tokens=True,
    )

    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true", help="Usa PyTorch en CPU.")
    if parser.parse_args().cpu:
        generate_on_cpu()
    else:
        generate_on_gpu()
