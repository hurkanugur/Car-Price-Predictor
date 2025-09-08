import pandas as pd
import torch
from dataset import CarPriceDataset
from model import CarPricePredictionModel

def main():
    # -------------------------
    # Select device
    # -------------------------
    print("-------------------------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"• Selected device: {device}")

    # -------------------------
    # Load dataset normalization params and categorical mappings
    # -------------------------
    dataset = CarPriceDataset()
    dataset.load_statistics()
    dataset.load_feature_transformer()

    # -------------------------
    # Example real-world input
    # -------------------------
    df = pd.DataFrame([
        {
            "Brand": "Honda", "Model": "Civic", "Year": 2004, "Engine_Size": 5.0, "Cylinders": 8,
            "Transmission": "Automatic", "Fuel_Type": "Electric", "Mileage": 56750,
            "Horsepower": 264, "Doors": 4, "Weight": 2282, "Color": "Silver", "Price": 56750
        },
        {
            "Brand": "Ford", "Model": "Fusion", "Year": 1991, "Engine_Size": 5.0, "Cylinders": 10,
            "Transmission": "Manual", "Fuel_Type": "Diesel", "Mileage": 45938,
            "Horsepower": 326, "Doors": 2, "Weight": 1259, "Color": "Blue", "Price": 45938
        },
        {
            "Brand": "Honda", "Model": "Civic", "Year": 2016, "Engine_Size": 4.4, "Cylinders": 4,
            "Transmission": "Manual", "Fuel_Type": "Hybrid", "Mileage": 48707,
            "Horsepower": 239, "Doors": 3, "Weight": 1362, "Color": "White", "Price": 48707
        },
        {
            "Brand": "Toyota", "Model": "Prius", "Year": 1996, "Engine_Size": 1.9, "Cylinders": 12,
            "Transmission": "Manual", "Fuel_Type": "Hybrid", "Mileage": 14659,
            "Horsepower": 63, "Doors": 3, "Weight": 2841, "Color": "Red", "Price": 14659
        }
    ])

    X = dataset.prepare_data_for_inference(df)
    input_dim = X[0].numel()

    # -------------------------
    # Load trained model
    # -------------------------
    model = CarPricePredictionModel(input_dim=input_dim, device=device)
    model.load()

    # -------------------------
    # Model inference
    # -------------------------
    model.eval()
    X = X.to(device)
    with torch.no_grad():
        y_pred = model(X)
        y_pred = dataset.denormalize_targets(y_pred)

    # -------------------------
    # Display predictions
    # -------------------------
    print("Predicted Prices:")
    for idx, row in df.iterrows():
        print(f"{row['Brand']} {row['Model']} ({row['Year']}): Predicted Price = ${y_pred[idx].item():,.2f}")

if __name__ == "__main__":
    main()
