# 🚗 Car Price Prediction with PyTorch

## 📖 Overview
This project predicts **used car prices** using a feedforward neural network regression model built in **PyTorch**.
It covers the complete pipeline from data preprocessing to model training and inference, including:

- 📊 **Neural Network Regression Model** with **Batch Normalization** and **Dropout** for better training stability and generalization 
- 🧩 **Structured data preprocessing** with numeric clipping, categorical encoding, and feature normalization   
- ⚖️ **Mean Squared Error (MSE)** as the loss function  
- 🧠 **Adam optimizer** for training  
- 🔀 **Train/Validation/Test split** for robust evaluation  
- 📈 **Feature & target normalization** using training statistics  
- 💾 **Saving/loading trained model and preprocessing artifacts** for inference  

---

## 🧩 Libraries
- **PyTorch** – model, training, and inference  
- **pandas** – data handling & preprocessing  
- **scikit-learn** – ColumnTransformer, standardization, one-hot encoding  
- **matplotlib** – plotting loss curves  
- **pickle** – saving/loading feature transformer, statistics and trained model

---

## ⚙️ Requirements

- Python **3.13+**  
- Recommended editor: **VS Code**  

---

## 📦 Installation

- Clone the repository
```bash
git clone https://github.com/hurkanugur/Car-Price-Predictor.git
```

- Navigate to the `Car-Price-Predictor` directory
```bash
cd Car_Price_Predictor
```

- Install dependencies
```bash
pip install -r requirements.txt
```

- Navigate to the `Car-Price-Predictor/src` directory
```bash
cd src
```

---

## 🔧 Setup Python Environment in VS Code

1. `View → Command Palette → Python: Create Environment`  
2. Choose **Venv** and your **Python version**  
3. Select **requirements.txt** to install dependencies  
4. Click **OK**

---

## 📂 Project Structure

```bash
data/
└── car_price_dataset.csv        # Raw used car data

model/
├── car_price_model.pth          # Trained PyTorch model
├── feature_transformer.pkl      # Fitted preprocessing transformer
└── statistics.pkl               # Preprocessing stats

src/
├── config.py                    # Paths, hyperparameters, split ratios
├── dataset.py                   # Data loading & preprocessing
├── device_manager.py            # Selects and manages compute device
├── main_train.py                # Training & model saving
├── main_inference.py            # Inference pipeline
├── model.py                     # Neural network definition
├── visualize.py                 # Training/validation plots

requirements.txt                 # Python dependencies
```

---

## 📂 Model Architecture

```bash
Input → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
      → Linear(128) → BatchNorm → ReLU → Dropout(0.2)
      → Linear(64)  → BatchNorm → ReLU → Dropout(0.1)
      → Linear(1)   → Output
```

---

## 📂 Train the Model
```bash
python main_train.py
```
or
```bash
python3 main_train.py
```

---

## 📂 Run Predictions on Real Data
```bash
python main_inference.py
```
or
```bash
python3 main_inference.py
```
