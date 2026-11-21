"""Compara respuestas de Qwen3 base y de los dos adaptadores LoRA guardados."""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora import ADAPTER_PATH, MODEL_PATH, WIDE_ADAPTER_PATH, select_device

DEFAULT_PROMPT = "Explica para qué sirve un log de sistema."
MAX_NEW_TOKENS = 80


@torch.inference_mode()
def generate_answer(model, tokenizer, inputs):
    """Genera sin muestreo y decodifica únicamente los tokens nuevos."""
    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    return tokenizer.decode(outputs[0, prompt_length:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Pregunta para los tres modelos.")
    parser.add_argument("--cpu", action="store_true", help="Fuerza la ejecución en CPU.")
    args = parser.parse_args()

    if not (MODEL_PATH / "config.json").is_file():
        raise SystemExit("Faltan los pesos originales. Ejecuta python download.py.")
    for path, command in (
        (ADAPTER_PATH, "python train_lora.py"),
        (WIDE_ADAPTER_PATH, "python train_lora.py --wide"),
    ):
        if not (path / "adapter_config.json").is_file() or not (
            path / "adapter_model.safetensors"
        ).is_file():
            raise SystemExit(f"Falta el adaptador en {path}. Ejecuta {command}.")

    device = select_device(args.cpu)
    print(f"Dispositivo: {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32).to(device)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH, adapter_name="original")
    model.load_adapter(WIDE_ADAPTER_PATH, adapter_name="wide")
    model.eval()

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    # Compartimos una copia de los pesos base y cambiamos el adaptador activo.
    print(f"\nPregunta: {args.prompt}", flush=True)
    with model.disable_adapter():
        print(f"\nQwen3 base\n{generate_answer(model, tokenizer, inputs)}", flush=True)
    for adapter, label in (("original", "LoRA original"), ("wide", "LoRA ampliado")):
        model.set_adapter(adapter)
        print(f"\n{label}\n{generate_answer(model, tokenizer, inputs)}", flush=True)


if __name__ == "__main__":
    main()
