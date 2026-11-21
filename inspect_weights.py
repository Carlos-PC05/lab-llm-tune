"""Inspecciona los pesos locales de Qwen3-0.6B sin generar texto.

Por defecto lee el GGUF que usa llama.cpp con Vulkan. Con --cpu conserva
el ejercicio original sobre el modelo de Transformers.
"""

import argparse

from inference import GGUF_MODEL_PATH, PROJECT_DIR


def inspect_gguf():
    """Muestra las formas y tipos de los tensores del archivo GGUF."""
    from gguf import GGUFReader, dequantize

    if not GGUF_MODEL_PATH.is_file():
        raise SystemExit("Falta el GGUF local. Ejecuta primero python inference.py.")

    # El lector mapea el archivo en memoria y permite consultar las formas
    # sin descomprimir todos los pesos ni cargar el modelo en la GPU.
    tensors = GGUFReader(GGUF_MODEL_PATH).tensors
    total = sum(int(tensor.n_elements) for tensor in tensors)
    print(f"Valores almacenados en GGUF: {total:,}")
    print(f"Tensores: {len(tensors)}")

    for tensor in tensors[:15]:
        # GGUF guarda las dimensiones en orden inverso al de NumPy y PyTorch.
        shape = tuple(int(size) for size in reversed(tensor.shape))
        print(tensor.name, shape, tensor.tensor_type.name)

    first = tensors[0]
    # Si el tensor está cuantizado, recuperamos valores aproximados antes
    # de calcular las estadísticas. Los bytes codificados incluyen escalas.
    values = dequantize(first.data, first.tensor_type)
    print("\nPrimer tensor:", first.name)
    print("Forma:", values.shape)
    print("Tipo en GGUF:", first.tensor_type.name)
    print("Media:", values.mean(dtype="float64"))
    print("Desviación estándar muestral:", values.std(dtype="float64", ddof=1))


def inspect_transformers():
    """Reproduce el ejercicio original sobre los parámetros de PyTorch."""
    import torch
    from transformers import AutoModelForCausalLM

    model_path = PROJECT_DIR / "models/qwen3-0.6b"
    if not model_path.is_dir():
        raise SystemExit("Faltan los pesos originales. Ejecuta primero python download.py.")

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parámetros totales: {total:,}")

    for name, parameter in list(model.named_parameters())[:15]:
        print(name, tuple(parameter.shape))

    name, parameter = next(iter(model.named_parameters()))
    values = parameter.detach()
    print("\nPrimer tensor:", name)
    print("Forma:", tuple(values.shape))
    print("Media:", values.mean().item())
    print("Desviación estándar muestral:", values.std().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true", help="Inspecciona el modelo de Transformers.")
    if parser.parse_args().cpu:
        inspect_transformers()
    else:
        inspect_gguf()
