# Libraries preparation:
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import wandb
from pytorch_lightning.loggers import WandbLogger
from uni2ts.model.moirai import MoiraiModule
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Ensure that the necessary libraries are installed:
# !pip install pandas torch transformers pytorch-lightning scikit-learn
# !pip install wandb  # If you plan to use Weights & Biases for logging
# !pip install uni2ts  # If you plan to use Moirai or Chronos models from uni2ts


# Data preparation:
df_train = pd.read_csv("train_combined.csv")
df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
# Compute Vx and Vy (in m/s)
df_train["Vx"] = df_train["x"].diff() / df_train["timestamp"].diff().dt.total_seconds()
df_train["Vy"] = df_train["y"].diff() / df_train["timestamp"].diff().dt.total_seconds()
# Compute Vz using 'altitude' in feet → convert to meters (1 ft = 0.3048 m)
df_train["Vz"] = df_train["altitude"].diff() * 0.3048 / df_train["timestamp"].diff().dt.total_seconds()
# Compute total velocity magnitude
df_train["speed_magnitude"] = (df_train["Vx"] ** 2 + df_train["Vy"] ** 2 + df_train["Vz"] ** 2) ** 0.5
# Feature engineering:
df_train['Vx'] = df_train['Vx'].fillna(0)
df_train['Vy'] = df_train['Vy'].fillna(0)
df_train['Vz'] = df_train['Vz'].fillna(0)
df_train['speed_magnitude'] = df_train['speed_magnitude'].fillna(0)
df_train["altitude"] = df_train["altitude"] * 0.3048  # Convert altitude from feet to meters

# Validation preparation:
df_val = pd.read_csv("validation_combined.csv")
df_val["timestamp"] = pd.to_datetime(df_val["timestamp"])
# Compute Vx and Vy (in m/s)
df_val["Vx"] = df_val["x"].diff() / df_val["timestamp"].diff().dt.total_seconds()
df_val["Vy"] = df_val["y"].diff() / df_val["timestamp"].diff().dt.total_seconds()
# Compute Vz using 'altitude' in feet → convert to meters (1 ft = 0.3048 m)
df_val["Vz"] = df_val["altitude"].diff() * 0.3048 / df_val["timestamp"].diff().dt.total_seconds()
# Compute total velocity magnitude
df_val["speed_magnitude"] = (df_val["Vx"] ** 2 + df_val["Vy"] ** 2 + df_val["Vz"] ** 2) ** 0.5
# Feature engineering:
df_val['Vx'] = df_val['Vx'].fillna(0)
df_val['Vy'] = df_val['Vy'].fillna(0)
df_val['Vz'] = df_val['Vz'].fillna(0)
df_val['speed_magnitude'] = df_val['speed_magnitude'].fillna(0)
df_val["altitude"] = df_val["altitude"] * 0.3048  # Convert altitude from feet to meters

# Test preparation:
df_test = pd.read_csv("test_noise.csv")
df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])
# Compute Vx and Vy (in m/s)
df_test["Vx"] = df_test["x"].diff() / df_test["timestamp"].diff().dt.total_seconds()
df_test["Vy"] = df_test["y"].diff() / df_test["timestamp"].diff().dt.total_seconds()
# Compute Vz using 'altitude' in feet → convert to meters (1 ft = 0.3048 m)
df_test["Vz"] = df_test["altitude"].diff() * 0.3048 / df_test["timestamp"].diff().dt.total_seconds()
# Compute total velocity magnitude
df_test["speed_magnitude"] = (df_test["Vx"] ** 2 + df_test["Vy"] ** 2 + df_test["Vz"] ** 2) ** 0.5
# Feature engineering:
df_test['Vx'] = df_test['Vx'].fillna(0)
df_test['Vy'] = df_test['Vy'].fillna(0)
df_test['Vz'] = df_test['Vz'].fillna(0)
df_test['speed_magnitude'] = df_test['speed_magnitude'].fillna(0)
df_test['altitude'] = df_test['altitude'] * 0.3048

# Features:
numeric_columns = ["longitude", "latitude", "altitude", "Vx", "Vy", "Vz", "speed_magnitude"]
X_train = df_train[["icao24"] + numeric_columns].copy()
scaler = StandardScaler()
X_train.loc[:, numeric_columns] = scaler.fit_transform(X_train[numeric_columns])

numeric_columns = ["longitude", "latitude", "altitude", "Vx", "Vy", "Vz", "speed_magnitude"]
X_val = df_val[["icao24"] + numeric_columns].copy()
scaler = StandardScaler()
X_val.loc[:, numeric_columns] = scaler.fit_transform(X_val[numeric_columns])

numeric_columns = ["longitude", "latitude", "altitude", "Vx", "Vy", "Vz", "speed_magnitude"]
X_test = df_test[["icao24"] + numeric_columns].copy()
scaler = StandardScaler()
X_test.loc[:, numeric_columns] = scaler.fit_transform(X_test[numeric_columns])

# Show the first few rows of the X_train DataFrame:
# print(X_train.head())


# Create dataset class for anomaly detection in flight sequences:
class FlightAnomalyDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, group_by: str = 'icao24', feature_cols=None):
        self.group_by = group_by
        self.feature_cols = feature_cols
        self.sequences, self.ids = self._prepare_sequences(dataframe)

    def _prepare_sequences(self, data: pd.DataFrame):
        sequences = []
        ids = []
        grouped = data.groupby(self.group_by)
        for group_id, group in grouped:
            features = group[self.feature_cols].values
            tensor = torch.tensor(features, dtype=torch.float32)
            sequences.append(tensor)
            ids.append(group_id)
        return sequences, ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]  # [T, F]
        sample_id = self.ids[idx]
        return sequence, sample_id

    @staticmethod
    def collate_fn(batch):
        sequences, sample_ids = zip(*batch)
        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)

        # Pad sequences to match max length
        padded_sequences = pad_sequence(sequences, batch_first=True)  # [B, max_T, F]

        # Optional: create a mask to ignore padded positions
        attention_mask = torch.zeros(padded_sequences.shape[:2], dtype=torch.bool)
        for i, length in enumerate(lengths):
            attention_mask[i, :length] = True

        return {
            'inputs': padded_sequences,  # [B, T, F]
            'attention_mask': attention_mask,  # [B, T]
            'lengths': lengths,
            'sample_ids': sample_ids
        }


train_dataset = FlightAnomalyDataset(X_train, feature_cols=["longitude", "latitude", "altitude", "Vx", "Vy", "Vz", "speed_magnitude"])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=FlightAnomalyDataset.collate_fn)
val_dataset = FlightAnomalyDataset(X_val, feature_cols=["longitude", "latitude", "altitude", "Vx", "Vy", "Vz", "speed_magnitude"])
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=FlightAnomalyDataset.collate_fn)
test_dataset = FlightAnomalyDataset(X_test, feature_cols=["longitude", "latitude", "altitude", "Vx", "Vy", "Vz", "speed_magnitude"])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=FlightAnomalyDataset.collate_fn)

# # Example of iterating through the DataLoader:
# for batch in train_loader:
#     print(batch['inputs'].shape)  # Should be [B, T, F]

# Train Chronos model for anomaly detection to predict masked sequences:
def apply_random_mask(inputs, mask_ratio=0.15):
    # inputs: [B, T, F]
    mask = torch.rand(inputs.shape[:2], device=inputs.device) < mask_ratio  # [B, T]
    masked_inputs = inputs.clone()
    masked_inputs[mask] = 0.0  # mask out by zeroing features (or use other value)
    return masked_inputs, mask


class MoiraiUnsupervisedAnomaly(pl.LightningModule):
    def __init__(self, input_dim=7, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.moirai = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")
        self.input_dim = input_dim
        hidden_size = self.moirai.d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_size))
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim))

        self.loss_fn = nn.MSELoss(reduction="none")
        self.lr = lr

    def forward(self, x):
        x_proj = self.input_proj(x)  # [B, T, hidden_size]
        hidden_states = self.moirai.encoder(x_proj)  # [B, T, hidden_size]
        reconstructed = self.reconstruct_head(hidden_states)  # [B, T, input_dim]
        return reconstructed

    def training_step(self, batch, batch_idx):
        x = batch['inputs']  # [B, T, F]
        masked_x, loss_mask = apply_random_mask(x)
        output = self(masked_x)
        loss = self.loss_fn(output, x)  # [B, T, F]
        # Expand mask dims to match loss shape
        loss_mask = loss_mask.unsqueeze(-1).expand_as(loss)  # [B, T, F]
        masked_loss = loss[loss_mask].mean()
        self.log("train_loss", masked_loss, prog_bar=True, on_step=True, on_epoch=True)
        return masked_loss

    def validation_step(self, batch, batch_idx):
        x = batch['inputs']
        masked_x, loss_mask = apply_random_mask(x)
        output = self(masked_x)
        loss = self.loss_fn(output, x)
        loss_mask = loss_mask.unsqueeze(-1).expand_as(loss)
        masked_loss = loss[loss_mask].mean()
        self.log("val_loss", masked_loss, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def compute_anomaly_score(x, model):
    with torch.no_grad():
        reconstructed = model(x)  # no masking here
        error = torch.mean((x - reconstructed) ** 2, dim=-1)  # [B, T]
        return error


# Run usage:
wandb_logger = WandbLogger(
    project="moirai-anomaly-detection",
    name="moirai-unsupervised-run",
    log_model=True)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",                  # Metric to monitor
    dirpath="checkpoints/",             # Directory to save checkpoints
    filename="moirai-{epoch:02d}-{val_loss:.4f}",  # Checkpoint name format
    save_top_k=3,                       # Keep top 3 checkpoints with lowest val_loss
    mode="min",                         # Lower val_loss is better
    save_weights_only=False,           # Save full model (can set to True if needed)
)

trainer = Trainer(max_epochs=20, accelerator='auto', devices=1, logger=wandb_logger, callbacks=[checkpoint_callback], log_every_n_steps=5)
model = MoiraiUnsupervisedAnomaly()
trainer.fit(model, train_loader, val_loader)
predictions = trainer.predict(model, dataloaders=test_loader)
all_scores = torch.cat(predictions, dim=0)  # [total_samples, seq_len]
flat_scores = all_scores.flatten().cpu().numpy()

df_test["compute_anomaly"] = flat_scores
threshold = df_test["compute_anomaly"].quantile(0.75)
df_test["predicted_anomaly"] = (df_test["compute_anomaly"] > threshold).astype(int)
y_true = df_test["anomaly"]
y_pred = df_test["predicted_anomaly"]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(14, 4))
plt.plot(df_test["compute_anomaly"], label="Anomaly Score")
plt.plot(df_test["anomaly"] * df_test["compute_anomaly"].max(), 'r.', label="Ground Truth Anomaly")
plt.plot(df_test["predicted_anomaly"] * df_test["compute_anomaly"].max(), 'g.', label="Predicted Anomaly", alpha=0.5)
plt.legend()
plt.title("Anomaly Detection Over Time")
plt.xlabel("Timestep")
plt.ylabel("Score")
plt.show()