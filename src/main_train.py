import torch
from torch import nn
from dataset import CarPriceDataset
from model import CarPricePredictionModel
from visualize import LossMonitor

import config

def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, loss_monitor):
    """Train a PyTorch model with optional validation and live loss monitoring."""

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # -------------------------
        # Training Step
        # -------------------------
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # -------------------------
        # Validation Step
        # -------------------------        
        should_validate = (
            epoch == 1
            or epoch == config.NUM_EPOCHS
            or epoch % config.VAL_INTERVAL == 0
        )

        val_loss = None
        if should_validate:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    val_loss += loss_fn(outputs, y_batch).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}")
        else:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.8f}")

        # -------------------------
        # Update Training/Validation Loss Graph
        # -------------------------
        loss_monitor.update(epoch, train_loss, val_loss)
            

def test_model(model, dataset, test_loader, device, n_samples=10):
    """Evaluate a trained classification model on a test dataset."""

    model.eval()
    predictions, ground_truths = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            y_pred = dataset.denormalize_targets(outputs)
            y_true = dataset.denormalize_targets(y_batch)

            predictions.append(y_pred)
            ground_truths.append(y_true)

    # Concatenate predictions and labels from all batches into single tensors
    predictions = torch.cat(predictions, dim=0)
    ground_truths = torch.cat(ground_truths, dim=0)

    print("• Sample Predictions:")
    for i in range(min(n_samples, len(predictions))):
        print(f"{i+1}: Predicted={predictions[i].item():.2f}, True={ground_truths[i].item():.2f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"• Selected device: {device}")

    dataset = CarPriceDataset()
    train_loader, val_loader, test_loader = dataset.prepare_data_for_training()

    input_dim = dataset.get_flattened_input_size(train_loader)
    model = CarPricePredictionModel(input_dim=input_dim, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    loss_monitor = LossMonitor()

    train_model(model, train_loader, val_loader, optimizer, loss_fn, device, loss_monitor)
    test_model(model, dataset, test_loader, device)

    model.save()
    dataset.save_statistics()
    dataset.save_feature_transformer()

    loss_monitor.close()

if __name__ == "__main__":
    main()
