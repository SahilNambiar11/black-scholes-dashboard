import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from model import DeltaNet
from data_generator import generate_delta_dataset as generate_training_data


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():

    # --------------------------------------------------
    # 0️⃣ Reproducibility
    # --------------------------------------------------
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # --------------------------------------------------
    # 1️⃣ Generate Data
    # --------------------------------------------------
    print("\nGenerating synthetic dataset...")
    df = generate_training_data(n_samples=50000)

    X = df[["log_moneyness", "time_to_maturity", "volatility"]].values
    y = df["delta"].values.reshape(-1, 1)

    # --------------------------------------------------
    # 2️⃣ Train / Val / Test Split
    # 60% train, 20% val, 20% test
    # --------------------------------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    # 0.25 * 0.8 = 0.2 total → gives 60/20/20 split

    # --------------------------------------------------
    # 3️⃣ Normalize Features (fit ONLY on train)
    # --------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # --------------------------------------------------
    # 4️⃣ DataLoader for Mini-Batching
    # --------------------------------------------------
    batch_size = 512

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --------------------------------------------------
    # 5️⃣ Initialize Model
    # --------------------------------------------------
    model = DeltaNet().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200

    print("\nStarting training...\n")

    # --------------------------------------------------
    # 6️⃣ Training Loop (Mini-Batch + Validation)
    # --------------------------------------------------
    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            preds = model(batch_X)
            loss = criterion(preds, batch_y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss /= len(val_loader.dataset)

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f}"
            )

    print("\nTraining complete.")

    # --------------------------------------------------
    # 7️⃣ Final Test Evaluation (Used Once)
    # --------------------------------------------------
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_mse = criterion(test_preds, y_test).item()
        max_error = torch.max(torch.abs(test_preds - y_test)).item()

    print("\nFinal Test Results")
    print(f"Test MSE: {test_mse:.10f}")
    print(f"Max Absolute Error: {max_error:.6f}")

    # --------------------------------------------------
    # 8️⃣ Save Model + Scaler
    # --------------------------------------------------
    torch.save(model.state_dict(), "delta_model.pt")

    import joblib
    joblib.dump(scaler, "scaler.pkl")

    print("\nModel saved to delta_model.pt")
    print("Scaler saved to scaler.pkl\n")


if __name__ == "__main__":
    main()