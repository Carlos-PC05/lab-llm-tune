"""Mide el efecto de sumar 0.1 a un peso de la primera proyección de consulta.

Usa Qwen3-0.6B en CPU y escribe un informe en .cache/. El cambio vive en
memoria durante el experimento y se restaura al terminar.
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference import PROJECT_DIR

# Casos fijos con respuestas de distinta longitud y estructura.
PROMPTS = (
    "¿Cuál es la capital de Francia? Responde en una frase.",
    "Calcula 17 + 25 y explica el cálculo en una frase.",
    "Completa la secuencia: 2, 4, 8, 16, ...",
    "Explica en dos frases qué es una red neuronal.",
)
DELTA = 0.1
MAX_NEW_TOKENS = 48
REPORT_PATH = PROJECT_DIR / ".cache/weight-perturbation.json"


def measure(model, tokenizer, inputs):
    """Captura los logits del siguiente token y una respuesta voraz."""
    # La última posición del prompt predice el primer token de la respuesta.
    logits = model(**inputs, logits_to_keep=1).logits[0, -1].clone()
    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_length = inputs["input_ids"].shape[-1]
    return logits, generated[0, prompt_length:].tolist()


def compare(tokenizer, prompt, before, after):
    """Compara predicciones condicionadas a los mismos tokens del prompt."""
    logits_before, tokens_before = before
    logits_after, tokens_after = after
    probabilities_before = logits_before.double().softmax(dim=-1)
    probabilities_after = logits_after.double().softmax(dim=-1)
    logit_delta = logits_after - logits_before
    probability_delta = probabilities_after - probabilities_before

    # Comparamos los mismos IDs en ambas distribuciones, aunque su ranking cambie.
    candidate_ids = dict.fromkeys(
        logits_before.topk(5).indices.tolist() + logits_after.topk(5).indices.tolist()
    )
    return {
        "prompt": prompt,
        "max_absolute_logit_change": logit_delta.abs().max().item(),
        "mean_absolute_logit_change": logit_delta.abs().mean().item(),
        "max_absolute_probability_change": probability_delta.abs().max().item(),
        "total_variation_distance": 0.5 * probability_delta.abs().sum().item(),
        "top_token_id_before": logits_before.argmax().item(),
        "top_token_id_after": logits_after.argmax().item(),
        "top_token_comparison": [
            {
                "token_id": token_id,
                "token": tokenizer.decode([token_id]),
                "logit_before": logits_before[token_id].item(),
                "logit_after": logits_after[token_id].item(),
                "probability_before": probabilities_before[token_id].item(),
                "probability_after": probabilities_after[token_id].item(),
            }
            for token_id in candidate_ids
        ],
        "generated_tokens_changed": tokens_before != tokens_after,
        "response_before": tokenizer.decode(tokens_before, skip_special_tokens=True),
        "response_after": tokenizer.decode(tokens_after, skip_special_tokens=True),
        "generated_token_ids_before": tokens_before,
        "generated_token_ids_after": tokens_after,
    }


def main():
    model_path = PROJECT_DIR / "models/qwen3-0.6b"
    if not (model_path / "config.json").is_file():
        raise SystemExit("Faltan los pesos originales. Ejecuta primero python download.py.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).eval()
    inputs_by_prompt = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        for prompt in PROMPTS
    ]

    with torch.no_grad():
        weight = model.model.layers[0].self_attn.q_proj.weight
        original = weight[0, 0].item()
        before = [measure(model, tokenizer, inputs) for inputs in inputs_by_prompt]
        try:
            weight[0, 0] += DELTA
            modified = weight[0, 0].item()
            after = [measure(model, tokenizer, inputs) for inputs in inputs_by_prompt]
        finally:
            weight[0, 0] = original
        restored = weight[0, 0].item()

    results = [
        compare(tokenizer, prompt, baseline, perturbed)
        for prompt, baseline, perturbed in zip(PROMPTS, before, after, strict=True)
    ]
    report = {
        "model": "Qwen/Qwen3-0.6B",
        "weight": "model.layers.0.self_attn.q_proj.weight[0, 0]",
        "requested_delta": DELTA,
        "original": original,
        "modified": modified,
        "restored": restored,
        "max_new_tokens": MAX_NEW_TOKENS,
        "results": results,
    }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Peso original {original:.9f} → modificado {modified:.9f} → restaurado {restored:.9f}")
    for result in results:
        print(f"\n{result['prompt']}")
        print(f"  Cambio máximo de logits: {result['max_absolute_logit_change']:.6g}")
        print(f"  Distancia entre probabilidades: {result['total_variation_distance']:.6g}")
        print(f"  Cambiaron los tokens generados: {result['generated_tokens_changed']}")
    print(f"\nInforme completo en {REPORT_PATH}")


if __name__ == "__main__":
    main()
