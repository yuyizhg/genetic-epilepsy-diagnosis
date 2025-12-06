import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import umap
import hdbscan
import sys
import random
from collections import defaultdict
import string
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import ollama
import warnings
import contextlib
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 💡 io is already imported correctly above

# ======================================================================
# --- CONFIGURATION ---
# ======================================================================

# --- INPUT FILES (Original raw data) ---
INPUT_FILE = "seizure_epilepsy_map.txt"
EMBEDDING_MODEL = "mxbai-embed-large"

# --- OUTPUT FILES (Logging, Clustering, and Synthetic Generation) ---
LOG_FILE = "0-2-unified_pipeline_output.log"
OUTPUT_CLUSTERS_CSV = "clustered_symptom_knowledge_base_1_200.csv"
OUTPUT_CLUSTERS_JSONL = "clustered_symptom_knowledge_base_1_200.jsonl"
OUTPUT_TRAINING_JSONL = "synthetic_patients_training_cohort_noise_1_200.jsonl"
OUTPUT_TRAINING_CSV = "synthetic_patients_training_cohort_noise_1_200.csv"
# CORRECTED: Removed double extensions from internal test cohort files
OUTPUT_INTERNAL_TEST_JSONL = "synthetic_patients_internal_test_cohort_noise_1_200.jsonl"
OUTPUT_INTERNAL_TEST_CSV = "synthetic_patients_internal_test_cohort_noise_1_200.csv"

OUTPUT_PLOT_FILE_2D = "umap_cluster_visualization_2d_optimized_1_200.png"
OUTPUT_PLOT_FILE_3D = "umap_cluster_visualization_3d_1_200.html"
MAX_PLOT_POINTS = 10000  # Use a large, clear number for visualization sampling
UMAP_N_COMPONENTS_2D_VIS = 2  # Dedicated 2D component count for aesthetics

# --- MODEL SAVE PATHS ---
UMAP_MODEL_PATH = "fitted_umap_reducer.joblib"
SCALER_MODEL_PATH = "fitted_scaler.joblib"
KNN_CLASSIFIER_PATH = "fitted_knn_classifier.joblib"
HDBSCAN_MODEL_PATH = "fitted_hdbscan_clusterer.joblib"

# --- HYPERPARAMETERS ---
UMAP_N_COMPONENTS_CLUSTERING = 100
UMAP_N_NEIGHBORS = 30
UMAP_RANDOM_STATE = 42
HDBSCAN_PARAMS = {'min_cluster_size': 2, 'min_samples': 2, 'metric': 'euclidean',
                  'gen_min_span_tree': True, 'prediction_data': True}

# --- PREPROCESSING & NOISE ---
DEL_LIST = ['see also', 'SIBS', 'sibling', 'death', 'die', 'families', 'family', 'prevalence', 'deletion', 'mutation',
            'increased frequency', 'translocation', 'reported', 'report', 'clinical information', 'incidence 1 in',
            'births', 'allelic', 'four major groups', 'consanguineous', 'x-inactivation', 'patient b', '30% of cases',
            'two types']
REP_LIST = ['some patients show', 'patients may present with', 'increased incidence in individuals', 'may occur',
            'some patients']
MIN_SYMPTOM_LENGTH = 3
PUNCTUATION_TO_STRIP = string.punctuation.replace('-', '')

# --- NOISE SYMPTOM DEFINITION (FIXED COMMA) ---
NOISE_SYMPTOMS_STRINGS = [
    "fever", "toothache", "pain", "depression", "claustrophobia", "cough",
    "diarrhea", "headache", "bone fracture", "athlete's foot", "allergy",
    "cold symptoms", "constipation", "sunburn", "carpal tunnel syndrome"
]
NOISE_CLUSTER_IDS = list(range(9000, 9000 + len(NOISE_SYMPTOMS_STRINGS)))


# ======================================================================
# --- UTILITY & LOGGING FUNCTIONS (UNCHANGED) ---
# ======================================================================

class Tee(object):
    """A class that writes to both the console stream and a file stream."""

    def __init__(self, stream1, stream2):
        self.stream1 = stream1  # Console (original sys.stdout or sys.stderr)
        self.stream2 = stream2  # Log file

    def write(self, data):
        self.stream1.write(data)
        self.stream2.write(data)

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

    def fileno(self):
        # Added to make it behave more like a standard stream
        return self.stream1.fileno() if hasattr(self.stream1, 'fileno') else None


@contextlib.contextmanager
def redirect_stdout_and_stderr(log_file_path):
    """
    Context manager to redirect stdout and stderr to a Tee object,
    allowing simultaneous print-to-console and write-to-log.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        tee_stdout = Tee(original_stdout, log_file)
        tee_stderr = Tee(original_stderr, log_file)
        try:
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def load_raw_data(file_path):
    """Loads raw diagnosis and symptom data from JSON array file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}", file=sys.stderr)
        # Note: sys.exit(1) here will exit the program even if wrapped by the context manager
        raise


def preprocess_symptom(symptom_str):
    """Cleans and filters a single symptom string."""
    s_lower = symptom_str.lower()
    if s_lower.startswith('caused by'): return None
    for del_word in DEL_LIST:
        if del_word in s_lower: return None
    s = s_lower
    for k in REP_LIST: s = s.replace(k, ' ')
    s = s.translate(str.maketrans('', '', PUNCTUATION_TO_STRIP))
    s_cleaned = ' '.join(s.split())
    if len(s_cleaned) <= MIN_SYMPTOM_LENGTH: return None
    return s_cleaned


def get_unique_symptoms(raw_knowledge):
    """Extracts unique, cleaned symptoms and their source diagnosis list."""
    unique_cleaned_symptoms = {}
    n_raw_symptoms = 0

    # 1. Collect unique symptoms
    for diagnosis_record in raw_knowledge:
        symptom_list = diagnosis_record.get('clinicalSynopsis', [])
        for raw_symptom in symptom_list:
            n_raw_symptoms += 1
            cleaned_symptom = preprocess_symptom(raw_symptom)
            if cleaned_symptom:
                unique_cleaned_symptoms[cleaned_symptom] = True

    cleaned_list = list(unique_cleaned_symptoms.keys())
    print(f"\nTotal raw symptoms processed: {n_raw_symptoms}")
    print(f"Found {len(cleaned_list)} unique, clean symptoms for embedding.")

    df_unique = pd.DataFrame({'symptom_text': cleaned_list})

    # 2. Prepare list of all symptom occurrences (for final CSV output)
    all_symptom_records = []
    for diag_record in raw_knowledge:
        diagnosis_name = diag_record.get('preferredTitle', 'UNKNOWN DIAGNOSIS')
        for raw_symptom in diag_record.get('clinicalSynopsis', []):
            cleaned_symptom = preprocess_symptom(raw_symptom)
            if cleaned_symptom:
                all_symptom_records.append({
                    'diagnosis': diagnosis_name,
                    'symptom_text': cleaned_symptom
                })
    df_full = pd.DataFrame(all_symptom_records)

    return df_unique, df_full


def apply_umap(X, n_components, n_neighbors, random_state):
    """Applies UMAP dimensionality reduction."""
    umap_reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=0.1,
        metric='cosine', random_state=random_state, verbose=False
    )
    X_reduced = umap_reducer.fit_transform(X)
    return X_reduced, umap_reducer


def apply_hdbscan(X_reduced_scaled, params):
    """Applies HDBSCAN clustering."""
    hdbscan_kwargs = {k: params[k] for k in ['min_cluster_size', 'min_samples', 'metric', 'prediction_data']}
    clusterer = hdbscan.HDBSCAN(**hdbscan_kwargs)
    with warnings.catch_warnings():
        clusterer.fit(X_reduced_scaled)
    return clusterer.labels_, clusterer


def save_cohort_files(cohort_list, jsonl_path, csv_path):
    """Saves a list of patient records to JSONL, vector data to .joblib, and metadata to CSV."""

    # 1. Save JSONL (full data including vectors)
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for patient_record in cohort_list:
            json.dump(patient_record, f)
            f.write('\n')

    df = pd.DataFrame(cohort_list)

    # 2. Extract and Save Vectors to Joblib (as requested)
    vector_data = df['symptoms_vectors_100d'].tolist()
    # Create a corresponding .joblib file name (e.g., synthetic_patients_internal_test_cohort_noise.joblib)
    vector_path = Path(csv_path).with_suffix('.joblib')
    joblib.dump(vector_data, vector_path)

    # 3. Save CSV (metadata only)
    # Exclude large vector list from the CSV for file size/readability
    df_csv = df.drop(columns=['symptoms_vectors_100d'], errors='ignore')

    df_csv.to_csv(csv_path, index=False)

    print(f"  ✅ Saved {len(cohort_list)} records to JSONL: {Path(jsonl_path).name}")
    print(f"  ✅ Saved {len(vector_data)} vector lists to Joblib: {vector_path.name}")
    print(f"  ✅ Saved {len(cohort_list)} records to CSV (metadata only): {Path(csv_path).name}")


def generate_synthetic_patients(diagnosis_data_list, noise_items_classified, umap_dim):
    """
    Generates synthetic patients based on clustered knowledge and adds 0-2 classified noise symptoms.
    """
    all_training_patients = []
    all_test_patients = []
    patient_id_counter_total = 0

    noise_map = {item[0]: item[1] for item in noise_items_classified}
    noise_clusters = list(noise_map.keys())

    def get_noise_symptoms():
        n_noise = random.randint(0, 2)
        if n_noise == 0: return [], []

        # If there are no classified noise symptoms (unlikely), fall back to raw list
        if not noise_clusters:
            # Fallback for safety, though classification should always yield some items
            return random.sample(NOISE_CLUSTER_IDS, n_noise), []

        # Sample from the classified noise (predicted IDs and vectors)
        sampled_noise_ids = random.sample(noise_clusters, n_noise)
        noise_vectors = [noise_map[cid] for cid in sampled_noise_ids]

        return sampled_noise_ids, noise_vectors

    print("\nStarting synthetic patient generation...")

    # 💡 PRINT for Debugging: Check the structure of classified noise items
    print(f"DBG: Noise items ready for injection: {len(noise_items_classified)}")
    if noise_items_classified:
        print(
            f"DBG: Example Noise Item (Cluster ID, 100D Vector first 5 elements): ({noise_items_classified[0][0]}, {noise_items_classified[0][1][:5]})")

    is_first_patient = True  # Flag for printing the first patient details

    for diag_data in tqdm(diagnosis_data_list, desc="Processing Diagnoses"):
        diagnosis = diag_data['diagnosis']
        all_clusters = diag_data['list_of_cluster_label_final']
        all_vectors = np.array(diag_data['list_of_umap_vectors_100d'])
        n_total_symptoms = len(all_clusters)

        if diag_data == diagnosis_data_list[0]:
            print(f"\nDBG: First Diagnosis: {diagnosis}. Total core symptoms: {n_total_symptoms}")
            print(f"DBG: First 3 Core Clusters: {all_clusters[:3]}")
            print(f"DBG: First 3 Core Vectors (first 5 elements): {[v[:5] for v in all_vectors[:3].tolist()]}")

        synthetic_patients = []
        patient_id_counter = 0
        percentages = [0.2, 0.3, 0.4, 0.5, 0.6]
        patients_per_group = 40

        # Generate 200 partial patients
        for p in percentages:
            n_sample = max(1, int(n_total_symptoms * p))

            for _ in range(patients_per_group):
                sampled_indices = random.sample(range(n_total_symptoms), n_sample)

                patient_clusters = [all_clusters[i] for i in sampled_indices]
                patient_vectors = all_vectors[sampled_indices].tolist()

                noise_clusters_add, noise_vectors_add = get_noise_symptoms()

                if p == 0.6 and _ == 0 and diag_data == diagnosis_data_list[0]:
                    print(
                        f"DBG: 60% Patient: Core clusters added: {len(patient_clusters)}. Noise clusters added: {len(noise_clusters_add)}")
                    if noise_clusters_add:
                        print(f"DBG: Noise Cluster IDs added: {noise_clusters_add}")

                patient_clusters.extend(noise_clusters_add)
                patient_vectors.extend(noise_vectors_add)

                patient_record = {
                    'patient_id': f"{diagnosis}_P{patient_id_counter}",
                    'diagnosis': diagnosis,
                    'symptom_completeness_pct': p * 100,
                    'symptoms_clusters': patient_clusters,
                    'symptoms_vectors_100d': patient_vectors,
                }
                synthetic_patients.append(patient_record)
                patient_id_counter += 1
                patient_id_counter_total += 1

                # --- NEW DEBUG PRINT: First generated patient ---
                if is_first_patient:
                    is_first_patient = False
                    print(f"\n--- DEBUG: First Synthetic Patient Details ---")
                    print(f"Patient ID: {patient_record['patient_id']}")
                    print(f"Diagnosis: {patient_record['diagnosis']}")
                    print(f"Completeness: {patient_record['symptom_completeness_pct']}%")
                    print(f"Total Clusters: {len(patient_record['symptoms_clusters'])}")
                    print(f"Total Vectors: {len(patient_record['symptoms_vectors_100d'])}")
                    print(f"Sample Clusters (first 5): {patient_record['symptoms_clusters'][:5]}")
                    print(f"Sample Vector (first 5 elements): {patient_record['symptoms_vectors_100d'][0][:5]}")
                    print(f"---------------------------------------------\n")
                # ---------------------------------------------

        # Generate the 100% complete patient (1 patient)
        noise_clusters_full, noise_vectors_full = get_noise_symptoms()
        full_clusters = all_clusters + noise_clusters_full
        full_vectors = all_vectors.tolist() + noise_vectors_full

        patient_record_full = {
            'patient_id': f"{diagnosis}_P{patient_id_counter}",
            'diagnosis': diagnosis,
            'symptom_completeness_pct': 100,
            'symptoms_clusters': full_clusters,
            'symptoms_vectors_100d': full_vectors,
        }
        synthetic_patients.append(patient_record_full)
        patient_id_counter_total += 1

        # Split the 201 patients (5 * 40 + 1) into cohorts
        random.shuffle(synthetic_patients)
        total_size = len(synthetic_patients)
        train_size = int(total_size * 0.80)
        training_cohort = synthetic_patients[:train_size]
        internal_test_cohort = synthetic_patients[train_size:]

        for patient in training_cohort:
            patient['cohort_type'] = 'training_cohort'
            all_training_patients.append(patient)

        for patient in internal_test_cohort:
            patient['cohort_type'] = 'internal_test_cohort'
            all_test_patients.append(patient)

    print("\n--- SAVING COHORT FILES ---")

    print(f"DBG: Total Training Patients: {len(all_training_patients)}")
    print(f"DBG: Total Internal Test Patients: {len(all_test_patients)}")

    save_cohort_files(all_training_patients, OUTPUT_TRAINING_JSONL, OUTPUT_TRAINING_CSV)
    save_cohort_files(all_test_patients, OUTPUT_INTERNAL_TEST_JSONL, OUTPUT_INTERNAL_TEST_CSV)
    print(f"\nTotal Patients Generated: {patient_id_counter_total}")


def generate_2d_figure(df_unique, output_file, max_points, umap_n_components):
    """
    Plots the first two UMAP components from the DEDICATED 2D run, colored by the final cluster label.
    """
    print("\nGenerating DEDICATED 2D UMAP cluster visualization...")

    x_col, y_col = 'umap_opt_0', 'umap_opt_1'

    if x_col not in df_unique.columns or y_col not in df_unique.columns:
        print(f"Error: Dedicated 2D UMAP columns ({x_col}, {y_col}) not found in DataFrame. Skipping 2D plot.")
        return

    # --- BUG FIX TARGET: Ensure 'cluster_label' is present for this logic ---
    if 'cluster_label' not in df_unique.columns:
        print("Error: Required column 'cluster_label' not found in DataFrame. Skipping 2D plot.")
        return
    # ----------------------------------------------------------------------

    # --- Data Sampling for Visualization ---
    n_rows = len(df_unique)
    df_plot = df_unique.copy()
    if max_points is not None and n_rows > max_points:
        print(f"Sampling {max_points} points from {n_rows} rows for a clearer 2D plot...")
        df_plot = df_unique.groupby('cluster_label_final', group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(max_points / len(df_unique['cluster_label_final'].unique())) + 1),
                               random_state=42)
        )
        if len(df_plot) > max_points:
            df_plot = df_plot.sample(max_points, random_state=42)

    df_plot['cluster_label_final_str'] = df_plot['cluster_label_final'].astype(str)

    max_original_id = df_unique['cluster_label'].max()  # THIS WAS THE LINE THAT FAILED
    is_singleton = df_plot['cluster_label_final'] > max_original_id

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    n_dense_clusters = len(df_plot[~is_singleton]['cluster_label_final'].unique()) if any(~is_singleton) else 0
    n_singleton_clusters = len(df_plot[is_singleton]['cluster_label_final'].unique()) if any(is_singleton) else 0

    if n_dense_clusters <= 20:
        palette = 'tab20'
    else:
        palette = sns.color_palette("hls", n_dense_clusters)

    # 1. Plot non-singleton clusters (Dense Clusters)
    if any(~is_singleton):
        sns.scatterplot(
            x=x_col, y=y_col, data=df_plot[~is_singleton],
            hue='cluster_label_final_str', palette=palette, s=25,
            alpha=0.8, legend=False, ax=ax
        )
        # Proxy plot for Dense Cluster legend
        ax.plot([], [], 'o', color=sns.color_palette(palette)[0], markersize=10, alpha=0.8,
                label=f'Dense Clusters ({n_dense_clusters} groups)')

    # 2. Plot singleton noise clusters
    if any(is_singleton):
        ax.scatter(
            df_plot[is_singleton][x_col], df_plot[is_singleton][y_col],
            color='gray', s=8, alpha=0.4,
            label=f'Singleton Clusters ({n_singleton_clusters} groups)', zorder=1
        )

    # Finalize the plot
    ax.set_title(
        f'Optimized UMAP Clustering for Visualization: {n_dense_clusters} Dense & {n_singleton_clusters} Singleton Groups',
        fontsize=16)

    ax.set_xlabel(f'UMAP Component 1 (Optimized)', fontsize=14)
    ax.set_ylabel(f'UMAP Component 2 (Optimized)', fontsize=14)

    # Single Legend at the Bottom
    handles, labels = ax.get_legend_handles_labels()
    legend_map = {}
    for h, l in zip(handles, labels):
        if l.startswith('Dense Clusters') or l.startswith('Singleton Clusters'):
            legend_map[l] = h

    final_handles = list(legend_map.values())
    final_labels = list(legend_map.keys())

    if final_handles:
        ax.legend(final_handles, final_labels,
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.1),
                  ncol=2, fontsize=10, title='Cluster Categories',
                  markerscale=2)

    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(output_file, dpi=300, facecolor='white', transparent=False)
    plt.close(fig)
    print(f"2D Visualization saved to: {Path(output_file).resolve()}")


def generate_3d_figure(df_unique, output_file, max_points, umap_n_components):
    """Generates an interactive 3D scatter plot using Plotly Express."""
    print("\nGenerating INTERACTIVE 3D Plotly visualization...")

    # We use the first three components of the 100D UMAP output for the 3D plot
    x_col, y_col, z_col = 'umap_scaled_0', 'umap_scaled_1', 'umap_scaled_2'

    if x_col not in df_unique.columns or y_col not in df_unique.columns or z_col not in df_unique.columns:
        print(f"Error: 3D UMAP columns ({x_col}, {y_col}, {z_col}) not found in DataFrame. Skipping 3D plot.")
        return

    # --- Data Sampling for Visualization ---
    n_rows = len(df_unique)
    df_plot = df_unique.copy()
    if max_points is not None and n_rows > max_points:
        print(f"Sampling {max_points} points from {n_rows} rows for 3D visualization clarity...")
        df_plot = df_unique.sample(n=max_points, random_state=42)

    df_plot['cluster_label_final_str'] = 'Cluster ' + df_plot['cluster_label_final'].astype(str)

    fig = px.scatter_3d(
        df_plot,
        x=x_col,
        y=y_col,
        z=z_col,
        color='cluster_label_final_str',
        opacity=0.7,
        title=f'Interactive UMAP Clustering (Clustered in {umap_n_components}D)',
        hover_data=['symptom_text', 'cluster_label']
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=f'UMAP Component 1',
            yaxis_title=f'UMAP Component 2',
            zaxis_title=f'UMAP Component 3',
            aspectmode='data'
        )
    )

    fig.write_html(output_file)
    print(f"3D Interactive Visualization saved to: {Path(output_file).resolve()}")


# ======================================================================
# --- MAIN EXECUTION PIPELINE (CONSOLIDATED) ---
# ======================================================================

def main():
    random.seed(42)
    np.random.seed(42)

    # -----------------------------------------------------------
    # PHASE 1: PREPROCESSING & EMBEDDING (CORE & NOISE SYMPTOMS)
    # -----------------------------------------------------------
    print("PHASE 1: Starting Preprocessing and Embedding...")

    try:
        raw_knowledge = load_raw_data(INPUT_FILE)
    except Exception as e:
        print(f"🚨 CRITICAL FAILURE in Phase 1: Could not load data. {e}", file=sys.stderr)
        sys.exit(1)

    df_unique_core, df_full_core = get_unique_symptoms(raw_knowledge)
    core_symptom_list = df_unique_core['symptom_text'].tolist()

    # Get 1024D embeddings for core symptoms
    print(f"\nGenerating 1024D embeddings for {len(core_symptom_list)} core symptoms...")
    embedded_symptom_map = {}

    # Check model availability first
    try:
        ollama.embeddings(prompt=core_symptom_list[0], model=EMBEDDING_MODEL)
        print(f"DBG: Ollama service connected and model '{EMBEDDING_MODEL}' is responding.")
    except Exception as e:
        print(f"\n🚨 CRITICAL ERROR: Ollama Connection Test Failed. Details: {e}", file=sys.stderr)
        print("ACTION REQUIRED: Ensure 'ollama serve' is running and model 'mxbai-embed-large' is pulled.",
              file=sys.stderr)
        sys.exit(1)

    # Start the main embedding loop
    for symptom in tqdm(core_symptom_list, desc="Embedding Core Symptoms"):
        try:
            response = ollama.embeddings(prompt=symptom, model=EMBEDDING_MODEL)
            embedded_symptom_map[symptom] = response['embedding']
        except Exception as e:
            print(f"\nError during embedding for symptom: '{symptom}'. Details: {e}", file=sys.stderr)
            sys.exit(1)

    df_unique_core['embedding'] = df_unique_core['symptom_text'].map(embedded_symptom_map)
    X_embeddings_core = np.array(df_unique_core['embedding'].tolist())

    print(f"DBG: Core Embeddings Shape (N unique x 1024): {X_embeddings_core.shape}")

    # Get 1024D embeddings for noise symptoms
    print(f"\nGenerating 1024D embeddings for {len(NOISE_SYMPTOMS_STRINGS)} noise symptoms...")
    X_embeddings_noise = []
    for symptom in tqdm(NOISE_SYMPTOMS_STRINGS, desc="Embedding Noise Symptoms"):
        try:
            response = ollama.embeddings(prompt=symptom, model=EMBEDDING_MODEL)
            X_embeddings_noise.append(response['embedding'])
        except Exception as e:
            print(f"\nError during embedding for noise symptom: '{symptom}'. Details: {e}", file=sys.stderr)
            sys.exit(1)
    X_embeddings_noise = np.array(X_embeddings_noise)

    print(f"DBG: Noise Embeddings Shape ({len(NOISE_SYMPTOMS_STRINGS)} x 1024): {X_embeddings_noise.shape}")

    # -----------------------------------------------------------
    # PHASE 2: UMAP & CLUSTERING (Training the Prediction Model)
    # -----------------------------------------------------------
    print("\nPHASE 2: Starting UMAP and HDBSCAN Clustering...")

    # --- NEW: Dedicated 2D UMAP Run for Visualization Aesthetics ---
    print("\n--- Running DEDICATED 2D UMAP for Visualization Aesthetics ---")

    # Run UMAP to 2D
    X_optimized_2d, umap_reducer_2d_opt = apply_umap(
        X_embeddings_core,
        UMAP_N_COMPONENTS_2D_VIS,  # Use the new N=2 component setting
        UMAP_N_NEIGHBORS,
        UMAP_RANDOM_STATE
    )

    # Add the 2D features to the unique symptom DataFrame
    df_unique_core['umap_opt_0'] = X_optimized_2d[:, 0]
    df_unique_core['umap_opt_1'] = X_optimized_2d[:, 1]
    # -----------------------------------------------------------------

    # UMAP to 100 dimensions
    X_reduced_core, umap_reducer = apply_umap(X_embeddings_core, UMAP_N_COMPONENTS_CLUSTERING, UMAP_N_NEIGHBORS,
                                              UMAP_RANDOM_STATE)
    print(f"DBG: UMAP Reduced Core Shape (N unique x 100): {X_reduced_core.shape}")

    # Standard Scaling
    scaler = StandardScaler()
    X_scaled_core = scaler.fit_transform(X_reduced_core)
    print(
        f"DBG: Scaled Core Features Mean/StdDev Check (first feature): Mean={np.mean(X_scaled_core[:, 0]):.4f}, StdDev={np.std(X_scaled_core[:, 0]):.4f}")

    # Apply HDBSCAN Clustering
    cluster_labels_unique, clusterer = apply_hdbscan(X_scaled_core, HDBSCAN_PARAMS)
    df_unique_core['cluster_label'] = cluster_labels_unique  # KEEPING THIS COLUMN IS CRITICAL FOR PHASE 6

    # Noise to Singleton Conversion (N+M approach) for Synthetic Patient Generation (FINAL LABELS)
    is_noise = df_unique_core['cluster_label'] == -1
    max_cluster_id = df_unique_core['cluster_label'].max()
    max_cluster_id = max(0, max_cluster_id)
    num_noise_points = is_noise.sum()
    new_cluster_ids = np.arange(1, num_noise_points + 1,
                                dtype=np.int32) + max_cluster_id
    df_unique_core['cluster_label_final'] = df_unique_core['cluster_label'].copy()
    df_unique_core.loc[is_noise, 'cluster_label_final'] = new_cluster_ids
    print(
        f"HDBSCAN Result: Found {max_cluster_id} dense clusters. Converted {num_noise_points} noise points to singletons.")
    print(f"DBG: Final Cluster ID Range (approx): 1 to {df_unique_core['cluster_label_final'].max()}")

    # -------------------------------------------------------------
    # Train the 1-NN Classifier on the final N+M cluster labels
    # -------------------------------------------------------------

    # Use all points and their final N+M labels for KNN training.
    X_all = X_scaled_core
    y_all = df_unique_core['cluster_label_final'].values.astype(int)

    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(X_all, y_all)
    print(f"1-NN Classifier trained on ALL {len(X_all)} points for {len(np.unique(y_all))} clusters (N+M approach).")

    # Save all trained models
    joblib.dump(umap_reducer, UMAP_MODEL_PATH)
    joblib.dump(scaler, SCALER_MODEL_PATH)
    joblib.dump(knn_classifier, KNN_CLASSIFIER_PATH)
    joblib.dump(clusterer, HDBSCAN_MODEL_PATH)
    print("UMAP, Scaler, and 1-NN Classifier models saved successfully.")

    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # PHASE 3: KNOWLEDGE BASE GENERATION (Output & Intermediate Data)
    # -----------------------------------------------------------
    print("\nPHASE 3: Generating Clustered Knowledge Base...")

    # Add 100D UMAP features to the unique symptom DataFrame
    umap_columns = [f'umap_scaled_{i}' for i in range(UMAP_N_COMPONENTS_CLUSTERING)]
    df_scaled_features = pd.DataFrame(X_scaled_core, columns=umap_columns)

    # --- START OF BUG FIX ---
    # Drop the large 'embedding' column, but KEEP 'cluster_label' for plotting logic in Phase 6.
    df_unique_core = df_unique_core.drop(columns=['embedding'], errors='ignore')

    # Concatenate the scaled features. The original columns ('cluster_label', 'cluster_label_final', 'umap_opt_0', 'umap_opt_1') are preserved.
    df_unique_core = pd.concat([
        df_unique_core.reset_index(drop=True),
        df_scaled_features.reset_index(drop=True)
    ], axis=1)
    # --- END OF BUG FIX ---

    # Merge final cluster labels and 100D vectors back into the full occurrence list (CSV output)
    merge_cols_for_final_csv = ['symptom_text', 'umap_opt_0', 'umap_opt_1', 'cluster_label_final'] + umap_columns

    df_final_csv = pd.merge(
        df_full_core,
        df_unique_core[merge_cols_for_final_csv],  # Merge only the necessary columns
        on='symptom_text',
        how='left'
    )

    # --- NEW DEBUG PRINT: First JSONL record ---
    first_diagnosis = df_final_csv['diagnosis'].iloc[0]
    first_group = df_final_csv[df_final_csv['diagnosis'] == first_diagnosis]

    print(f"\n--- DEBUG: First Diagnosis JSONL Record Generation ({first_diagnosis}) ---")
    print(f"Total symptoms for diagnosis: {len(first_group)}")
    print(f"Sample Cluster IDs (first 5): {first_group['cluster_label_final'].iloc[:5].tolist()}")
    print(f"Sample Vector (first 5 elements): {first_group[umap_columns].iloc[0].tolist()[:5]}")
    print(f"------------------------------------------------------------------------\n")
    # -----------------------------------------

    print(f"DBG: Final CSV Data Structure Head:\n{df_final_csv.head().to_string()}")
    print(f"DBG: Total records in final CSV (full occurrences): {len(df_final_csv)}")

    # Generate output CSV file
    df_final_csv.to_csv(OUTPUT_CLUSTERS_CSV, index=False)
    print(f"✅ Clustered Symptom Knowledge Base (CSV) saved to: {OUTPUT_CLUSTERS_CSV}")

    # Generate JSONL output for synthetic patient input
    df_grouped = df_final_csv.groupby('diagnosis')

    final_jsonl_records = []
    for diagnosis_name, group in df_grouped:
        vectors = group[umap_columns].values.tolist()
        # FIX: Ensure cluster labels are standard Python ints for JSON serialization
        clusters = group['cluster_label_final'].astype(int).values.tolist()

        final_jsonl_records.append({
            'diagnosis': diagnosis_name,
            'list_of_cluster_label_final': clusters,
            'list_of_umap_vectors_100d': vectors
        })

    print(f"DBG: Total diagnoses for synthetic generation: {len(final_jsonl_records)}")

    # Generate output JSONL file (Intermediate Data)
    with open(OUTPUT_CLUSTERS_JSONL, 'w', encoding='utf-8') as f:
        for record in final_jsonl_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✅ Clustered Symptom Knowledge Base (JSONL) saved to: {OUTPUT_CLUSTERS_JSONL}")

    # -----------------------------------------------------------
    # PHASE 4: NOISE SYMPTOM PREPARATION (Classification)
    # -----------------------------------------------------------
    print("\nPHASE 4: Classifying Noise Symptoms using trained models...")

    # 1. Transform 1024D noise embeddings using the fitted UMAP
    X_reduced_noise = umap_reducer.transform(X_embeddings_noise)

    # 2. Scale the 100D features using the fitted Scaler
    X_scaled_noise = scaler.transform(X_reduced_noise)
    print(f"DBG: Scaled Noise Features Shape ({len(NOISE_SYMPTOMS_STRINGS)} x 100): {X_scaled_noise.shape}")

    # 3. Predict the cluster ID using the fitted 1-NN Classifier
    noise_cluster_predictions = knn_classifier.predict(X_scaled_noise)

    noise_items_classified = []
    # Use a high, unique ID for all injected noise symptoms for tracking
    INJECTED_NOISE_ID_START = 9000

    print(f"\n--- DEBUG: Noise Symptom Prediction/Injection ---")
    for i, (text, pred) in enumerate(zip(NOISE_SYMPTOMS_STRINGS, noise_cluster_predictions)):
        # Assign a high, unique ID to each noise symptom to distinguish it in the synthetic patient cohort
        unique_noise_id = INJECTED_NOISE_ID_START + i

        # FIX: Ensure the predicted cluster ID is converted to a standard Python int, but the vector is what is used
        noise_items_classified.append((unique_noise_id, X_scaled_noise[i].tolist()))

        if i < 5:
            print(f"Noise Symptom '{text}' -> Predicted KNN ID: {int(pred)}, Injected as unique ID: {unique_noise_id}")

    print(f"------------------------------------------------\n")

    # -----------------------------------------------------------
    # PHASE 5: SYNTHETIC PATIENT GENERATION
    # -----------------------------------------------------------
    print("\nPHASE 5: Starting Synthetic Patient Generation...")

    generate_synthetic_patients(final_jsonl_records, noise_items_classified, UMAP_N_COMPONENTS_CLUSTERING)

    print("\n✅ Unified Pipeline execution complete.")

    # -----------------------------------------------------------
    # PHASE 6: GENERATE VISUALIZATIONS
    # -----------------------------------------------------------
    print("\nPHASE 6: Generating Visualizations...")

    # The plotting functions rely on df_unique_core having the final labels,
    # the 100D scaled features, AND the 2D optimized features (and 'cluster_label').

    # We use a copy of the master df_unique_core which still contains 'cluster_label'
    # due to the fix in Phase 3.
    df_plot_final = df_unique_core.copy()

    # Note: We use the unique symptom DataFrame (df_plot_final) for visualization,
    # as plotting duplicate symptom occurrences is visually confusing.

    generate_2d_figure(df_plot_final, OUTPUT_PLOT_FILE_2D, MAX_PLOT_POINTS, UMAP_N_COMPONENTS_CLUSTERING)
    generate_3d_figure(df_plot_final, OUTPUT_PLOT_FILE_3D, MAX_PLOT_POINTS, UMAP_N_COMPONENTS_CLUSTERING)

    print("\n✅ Visualization complete.")


# ======================================================================
# --- EXECUTION ENTRY POINT (UNCHANGED) ---
# ======================================================================

if __name__ == "__main__":
    try:
        # Wrap the entire execution in the logging context manager
        with redirect_stdout_and_stderr(LOG_FILE):
            main()
    except Exception as e:
        # Catch any exceptions that escaped main() and log them before crashing
        import traceback

        print(f"\n❌ UNHANDLED FATAL EXCEPTION OUTSIDE MAIN: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)