"""Entrena un adaptador LoRA de Qwen3 con ejemplos de instrucción y respuesta."""

import argparse
import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference import PROJECT_DIR

MODEL_PATH = PROJECT_DIR / "models/qwen3-0.6b"
TRAIN_PATH = PROJECT_DIR / "data/train.jsonl"
ADAPTER_PATH = PROJECT_DIR / "checkpoints/qwen3-0.6b-lora"
WIDE_ADAPTER_PATH = PROJECT_DIR / "checkpoints/qwen3-0.6b-lora-wide"
WIDE_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
MAX_LENGTH = 128
IGNORE_INDEX = -100


def select_device(force_cpu):
    """Prefiere una GPU reconocida por PyTorch y permite forzar CPU."""
    if not force_cpu:
        # PyTorch también expone las GPU AMD con ROCm mediante torch.cuda.
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.xpu.is_available():
            return torch.device("xpu")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def load_examples(tokenizer):
    """Construye secuencias donde solo los tokens de respuesta aportan pérdida."""
    examples = []
    for line_number, line in enumerate(TRAIN_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            instruction, response = row["instruction"], row["response"]
        except (ValueError, KeyError, TypeError) as exc:
            raise SystemExit(f"{TRAIN_PATH}:{line_number}: ejemplo JSONL inválido.") from exc
        if not all(isinstance(text, str) and text.strip() for text in (instruction, response)):
            raise SystemExit(f"{TRAIN_PATH}:{line_number}: se requieren textos no vacíos.")

        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        # Se añade el fin de turno para que también aprenda cuándo detenerse.
        response_ids = tokenizer.encode(response, add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        input_ids = prompt_ids + response_ids
        if len(input_ids) > MAX_LENGTH:
            raise SystemExit(
                f"{TRAIN_PATH}:{line_number}: {len(input_ids)} tokens, límite {MAX_LENGTH}."
            )

        # El modelo desplaza las etiquetas una posición antes de calcular la
        # pérdida. El último token del prompt predice el primero de respuesta.
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
        examples.append(
            {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
                "labels": torch.tensor([labels]),
            }
        )
    if not examples:
        raise SystemExit(f"No hay ejemplos en {TRAIN_PATH}.")
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Fuerza el entrenamiento en CPU.")
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Aplica LoRA de rango 8 a las proyecciones de atención y de la MLP.",
    )
    args = parser.parse_args()

    if not (MODEL_PATH / "config.json").is_file():
        raise SystemExit("Faltan los pesos originales. Ejecuta python download.py.")

    torch.manual_seed(42)
    device = select_device(args.cpu)
    print(f"Dispositivo de entrenamiento: {device}", flush=True)
    if device.type == "cpu" and not args.cpu:
        print("PyTorch no detectó una GPU compatible. Se usará CPU.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    examples = load_examples(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    model.config.use_cache = False
    rank = 8 if args.wide else 4
    targets = WIDE_TARGET_MODULES if args.wide else ["q_proj", "v_proj"]
    adapter_path = WIDE_ADAPTER_PATH if args.wide else ADAPTER_PATH
    # Conservamos alfa/rango = 2 en ambos perfiles.
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            target_modules=targets,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    ).to(device)
    model.train()
    model.print_trainable_parameters()

    # Una pasada y un ejemplo por paso bastan para la prueba de integración.
    # PEFT congela el modelo base. El optimizador recibe solo los adaptadores.
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4)

    for step, example in enumerate(examples, 1):
        optimizer.zero_grad()
        batch = {name: tensor.to(device) for name, tensor in example.items()}
        loss = model(**batch).loss
        if not torch.isfinite(loss).item():
            raise SystemExit(f"Pérdida no finita en el paso {step}.")
        loss.backward()
        optimizer.step()
        print(f"Paso {step}/{len(examples)}. Pérdida de entrenamiento: {loss.item():.6f}", flush=True)

    model.save_pretrained(adapter_path)
    print(f"Adaptador guardado en {adapter_path}")


if __name__ == "__main__":
    main()
