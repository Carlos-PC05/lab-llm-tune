"""Aprende la relación y = 2x con una capa lineal y SGD en CPU."""

import torch


def main():
    torch.manual_seed(42)
    inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device="cpu")
    targets = 2 * inputs

    # Cada fila contiene una entrada. La capa aprende un peso y un sesgo.
    model = torch.nn.Linear(1, 1, device="cpu")
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    initial_weight = model.weight.item()
    initial_bias = model.bias.item()
    print(f"Peso inicial: {initial_weight:.6f}")
    print(f"Sesgo inicial: {initial_bias:.6f}")

    for epoch in range(1_000):
        # PyTorch acumula los gradientes en .grad. Los borramos antes
        # de calcular los de esta iteración.
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()

        if epoch == 0:
            weight_gradient = model.weight.grad.item()
            bias_gradient = model.bias.grad.item()
            print(f"Pérdida inicial: {loss.item():.6f}")
            print(f"Gradiente del peso: {weight_gradient:.6f}")
            print(f"Gradiente del sesgo: {bias_gradient:.6f}")
            print(f"Peso esperado tras SGD: {initial_weight - learning_rate * weight_gradient:.6f}")

        optimizer.step()

        if epoch == 0:
            print(f"Peso tras el primer paso: {model.weight.item():.6f}")

    # Medimos la pérdida final después de la última actualización y
    # probamos con x = 5, que quedó fuera de los datos de entrenamiento.
    with torch.no_grad():
        final_loss = loss_fn(model(inputs), targets).item()
        prediction_for_five = model(torch.tensor([[5.0]], device="cpu")).item()

    print(f"\nPeso final: {model.weight.item():.6f}")
    print(f"Sesgo final: {model.bias.item():.6f}")
    print(f"Pérdida final: {final_loss:.6f}")
    print(f"Predicción para 5: {prediction_for_five:.6f} (objetivo: 10)")


if __name__ == "__main__":
    main()
