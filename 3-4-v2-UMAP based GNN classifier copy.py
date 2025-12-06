import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import json
from pathlib import Path
import random
import time
import sys
import contextlib
from collections import defaultdict

## changed Change Dropout to 0.4,LEARNING_RATE=0.0005.epoch =40

import joblib
# --- NEW IMPORTS ---
import matplotlib.pyplot as plt
from torch_geometric.nn import global_add_pool

# Ensure PyTorch Geometric is available for global_add_pool
try:
    from torch_geometric.nn import global_add_pool
except ImportError:
    print("--Error: 'torch_geometric' package not found. - --", file=sys.stderr)
    print("This model requires the PyTorch Geometric library for pooling.", file=sys.stderr)
    print("\nPlease install it by running: pip install torch-geometric\n", file=sys.stderr)
    sys.exit(1)

# --- CONFIGURATION ---
# Corrected input file names to match the output of the previous pipeline
#TRAINING_JSONL = "synthetic_patients_training_cohort_noise_1_200.jsonl"
TRAINING_JSONL = "synthetic_patients_training_cohort_noise_1_200.jsonl"
TESTING_JSONL = "synthetic_patients_internal_test_cohort_noise_1_200.jsonl"
LOG_FILE = "gnn_training_umap_console_output_3_4_v2_training_cohort_noise_1_200.log"
MODEL_OUTPUT_PATH = "gnn_model_and_mappings_umap_1_200_drop0.4_samllerlearningrate.pth"
# --- NEW OUTPUT FILES ---
LOSS_CURVE_PLOT = "gnn_loss_curve.png"
TOPK_ACCURACY_PLOT = "gnn_topk_accuracy.png"

# --- HYPERPARAMETERS ---
UMAP_DIM = 100
HIDDEN_DIM = 64
LEARNING_RATE = 0.0005 #0.001
EPOCHS = 40  #50
BATCH_SIZE = 64
TOPK_LIST = [1, 5, 10, 20]  # Top-K values to track

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ----------------------------------------------------------------------
# --- MODEL DEFINITION (ADAPTED FOR UMAP INPUT) ---
# ----------------------------------------------------------------------

class UmapSetClassifier(nn.Module):
    """
    GNN architecture (Deep Set) using summation aggregation and pre-calculated
    UMAP vectors (100D) as input features.
    """

    def __init__(self, umap_dim, num_diagnoses, hidden_dim):
        super(UmapSetClassifier, self).__init__()

        self.classifier_mlp = nn.Sequential(
            nn.Linear(umap_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.4), #0.3
            nn.Linear(hidden_dim, num_diagnoses),
        )

    def forward(self, symptoms_vector_tensor, batch_index):
        embedded_symptoms = symptoms_vector_tensor
        patient_vector = global_add_pool(embedded_symptoms, batch_index)
        out = self.classifier_mlp(patient_vector)
        return out


# ----------------------------------------------------------------------
# --- DATASET AND DATA LOADING (UNCHANGED) ---
# ----------------------------------------------------------------------

class CaseDataset(Dataset):
    """Dataset for GNN training, yielding lists of 100D UMAP vectors and the target label."""

    def __init__(self, data, label_encoder):
        self.data = data
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        symptom_vectors = record.get('symptoms_vectors_100d', [])
        diagnosis_id = self.label_encoder.transform([record['diagnosis']])[0]
        return symptom_vectors, diagnosis_id


def collate_fn_umap(batch):
    """Custom collate function to handle variable-length symptom vector lists."""
    vectors, targets = zip(*batch)

    all_symptom_vectors = []
    batch_index_list = []
    valid_targets = []

    for i, case_vectors in enumerate(vectors):
        if not case_vectors:
            continue

        all_symptom_vectors.extend(case_vectors)
        # Use the index of the valid_targets list as the new batch index
        batch_index_list.extend([len(valid_targets)] * len(case_vectors))
        valid_targets.append(targets[i])

    if not valid_targets:
        return None

    symptoms_tensor = torch.tensor(all_symptom_vectors, dtype=torch.float)
    batch_index_tensor = torch.tensor(batch_index_list, dtype=torch.long)
    target_tensor = torch.tensor(valid_targets, dtype=torch.long)

    return symptoms_tensor, batch_index_tensor, target_tensor


def load_cohort_data(file_path):
    """Loads and preprocesses the aggregated case data for a single cohort."""
    print(f"Loading data from {file_path}...")
    data = []
    unique_diagnoses = set()

    VECTOR_KEY = 'symptoms_vectors_100d'

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)

                # DEBUG PRINT ADDED HERE
                if not data:
                    print(f"DEBUG: Keys of first record in {file_path.split('/')[-1]}: {list(record.keys())}")
                # ------------------------------

                vectors = record.get(VECTOR_KEY)

                if not isinstance(vectors, list) or not any(isinstance(v, list) for v in vectors):
                    vectors = []

                if not vectors:
                    continue

                data.append({
                    'diagnosis': record['diagnosis'],
                    'symptoms_vectors_100d': vectors,
                })
                unique_diagnoses.add(record['diagnosis'])
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}. Please check your configuration.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: JSON decoding failed for {file_path}. File may be malformed.", file=sys.stderr)
        sys.exit(1)

    print(f"Total valid cases loaded from {file_path.split('/')[-1]}: {len(data)}")

    return data, unique_diagnoses


# ----------------------------------------------------------------------
# --- UTILITY AND LOGGING FUNCTIONS (UNCHANGED) ---
# ----------------------------------------------------------------------

class Tee(object):
    """A class that writes to both the console stream and a file stream."""

    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)

    def flush(self, *args, **kwargs):
        self.stream1.flush()
        self.stream2.flush()

    def fileno(self):
        return self.stream1.fileno()


@contextlib.contextmanager
def redirect_stdout_and_stderr(log_file):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = Tee(original_stdout, log_file)
    tee_stderr = Tee(original_stderr, log_file)
    try:
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


# ----------------------------------------------------------------------
# --- NEW: TOP-K ACCURACY FUNCTION ---
# ----------------------------------------------------------------------

def top_k_accuracy(output, targets, k_list=TOPK_LIST):
    """Computes Top-K accuracy for a list of K values."""
    with torch.no_grad():
        max_k = max(k_list)
        batch_size = targets.size(0)

        # Get the top max_k indices (classes) for each sample
        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t() # Transpose to get [max_k, batch_size]

        # targets needs to be broadcast to [max_k, batch_size] for comparison
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        topk_results = {}
        for k in k_list:
            # Check if target is in the top k predictions
            # CRITICAL FIX: Use .reshape(-1) instead of .view(-1) after slicing
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            topk_results[f'Top-{k}'] = correct_k.mul_(100.0 / batch_size).item() / 100.0

        return topk_results


# ----------------------------------------------------------------------
# --- TRAINING AND EVALUATION FUNCTIONS (MODIFIED) ---
# ----------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []

    for batch in dataloader:
        if batch is None: continue
        symptoms, batch_index, targets = batch
        symptoms, batch_index, targets = symptoms.to(DEVICE), batch_index.to(DEVICE), targets.to(DEVICE)

        if symptoms.numel() == 0: continue

        optimizer.zero_grad()
        output = model(symptoms, batch_index)

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_outputs.append(output)
        all_targets.append(targets)

    # Calculate final metrics
    avg_loss = total_loss / len(dataloader)
    top1_acc = correct / total

    # Calculate Top-K for reporting (optional, but good for diagnostics)
    outputs_tensor = torch.cat(all_outputs)
    targets_tensor = torch.cat(all_targets)
    topk_accs = top_k_accuracy(outputs_tensor, targets_tensor)

    return avg_loss, top1_acc, topk_accs


def evaluate(model, dataloader, criterion):
    """Evaluates the model on the test/validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None: continue
            symptoms, batch_index, targets = batch
            symptoms, batch_index, targets = symptoms.to(DEVICE), batch_index.to(DEVICE), targets.to(DEVICE)

            if symptoms.numel() == 0: continue

            output = model(symptoms, batch_index)
            loss = criterion(output, targets)
            total_loss += loss.item()

            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_outputs.append(output)
            all_targets.append(targets)

    # Calculate final metrics
    avg_loss = total_loss / len(dataloader)
    top1_acc = correct / total

    outputs_tensor = torch.cat(all_outputs)
    targets_tensor = torch.cat(all_targets)
    topk_accs = top_k_accuracy(outputs_tensor, targets_tensor)

    return avg_loss, top1_acc, topk_accs


def save_model_and_mappings(model, label_encoder, id_to_label, path):
    """Saves the model state and all necessary mappings to a single .pth file."""

    artifacts = {
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'id_to_label': id_to_label,
        'config': {'umap_dim': UMAP_DIM, 'hidden_dim': HIDDEN_DIM},
    }

    torch.save(artifacts, path)
    print(f"\nModel and mappings saved successfully to {Path(path).resolve()}")


# ----------------------------------------------------------------------
# --- NEW: PLOTTING FUNCTION ---
# ----------------------------------------------------------------------

def plot_results(train_history, test_history, topk_history, num_epochs):
    """Generates and saves the loss curve and Top-K accuracy plots."""
    epochs = range(1, num_epochs + 1)

    # --- 1. Loss Curve ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_history['loss'], label='Training Loss', marker='o', markersize=4)
    plt.plot(epochs, test_history['loss'], label='Validation Loss', marker='o', markersize=4)
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PLOT)
    print(f"\n✅ Loss curve saved to: {LOSS_CURVE_PLOT}")

    # --- 2. Top-K Accuracy Curve ---
    plt.figure(figsize=(10, 6))

    # topk_history keys are 'Top-1', 'Top-5', etc.
    for k_key in topk_history.keys():
        plt.plot(epochs, topk_history[k_key], label=f'Test {k_key} Accuracy', marker='o', markersize=4)

    plt.title('Test Cohort Top-K Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(TOPK_ACCURACY_PLOT)
    print(f"✅ Top-K accuracy curve saved to: {TOPK_ACCURACY_PLOT}")


def main():
    """Main function to run the UMAP GNN training pipeline."""

    # 1. Load Training Data and Establish Mappings
    train_data_raw, unique_diagnoses_train = load_cohort_data(TRAINING_JSONL)

    if not train_data_raw:
        print("Error: Training data is empty or missing.", file=sys.stderr)
        return

    # 1.a. Establish Mappings (fitted ONLY on training data)
    label_encoder = LabelEncoder()
    label_encoder.fit(list(unique_diagnoses_train))
    id_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}
    num_diagnoses = len(label_encoder.classes_)

    print(f"Model will be initialized using {UMAP_DIM}D UMAP features and {num_diagnoses} diagnosis classes.")

    # 2. Load Testing Data and Filter
    test_data_raw, _ = load_cohort_data(TESTING_JSONL)

    test_diagnoses_valid = set(label_encoder.classes_)
    test_data = [
        record for record in test_data_raw
        if record['diagnosis'] in test_diagnoses_valid
    ]

    # Convert data into DataFrames for easy access (though list of dicts is used for dataset)
    train_data = pd.DataFrame(train_data_raw)
    test_data = pd.DataFrame(test_data)

    print(f"Training set size: {len(train_data)}")
    print(f"Test set size (filtered): {len(test_data)}")

    # 3. Create Datasets and DataLoaders
    train_dataset = CaseDataset(train_data.to_dict('records'), label_encoder)
    test_dataset = CaseDataset(test_data.to_dict('records'), label_encoder)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_umap,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_umap
    )

    # 4. Initialize Model, Optimizer, and Loss
    model = UmapSetClassifier(UMAP_DIM, num_diagnoses, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("\nStarting Training...")

    # --- NEW: HISTORY TRACKING ---
    train_history = defaultdict(list)
    test_history = defaultdict(list)
    # Initialize a dict of lists for each Top-K value (e.g., {'Top-1': [...], 'Top-5': [...]})
    topk_history = {f'Top-{k}': [] for k in TOPK_LIST}

    # 5. Training Loop
    best_test_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc, _ = train_epoch(model, train_loader, optimizer, criterion)

        # Evaluate (Detailed evaluation to get all Top-K metrics)
        test_loss, test_acc, test_topk_accs = evaluate(model, test_loader, criterion)

        # --- NEW: RECORD HISTORY ---
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        test_history['loss'].append(test_loss)
        test_history['acc'].append(test_acc)

        for k_key, acc_val in test_topk_accs.items():
            topk_history[k_key].append(acc_val)

        # Logging
        epoch_duration = time.time() - start_time
        topk_log_str = " | ".join([f"{k_key}: {acc_val:.4f}" for k_key, acc_val in test_topk_accs.items()])

        print(f"Epoch {epoch:02d} | Time: {epoch_duration:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | {topk_log_str}")

        # Save Best Model (based on Top-1 accuracy)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_model_and_mappings(model, label_encoder, id_to_label, MODEL_OUTPUT_PATH)

    print(f"\nTraining complete. Best test accuracy (Top-1): {best_test_acc:.4f}")

    # --- NEW: Generate Plots ---
    plot_results(train_history, test_history, topk_history, EPOCHS)


if __name__ == "__main__":
    print(f"Starting script execution. Console output will be logged to {LOG_FILE} AND displayed here.")
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
            with redirect_stdout_and_stderr(log_file):
                main()
    except Exception:
        import traceback

        traceback.print_exc(file=sys.stderr)
        print(f"\nError: Script finished with an unhandled error. Execution halted.", file=sys.stderr)
        sys.exit(1)

    print(f"\nScript execution finished. Full log is available in {LOG_FILE}.")