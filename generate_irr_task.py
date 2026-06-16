import pandas as pd
import numpy as np
import os

def calculate_length_based_weight(text: str) -> float:
    """
    Computes a text length-weighting profile to prioritize information density:
    - 11 to 50 words: Maximum weight of 1.5 (prioritizes descriptive threat reports)
    - Less than 5 words: Minimum weight of 0.1 (suppresses brief chat noise)
    - More than 200 words: Reduced weight of 0.2 (suppresses massive system log dumps)
    - All other lengths: Neutral baseline weight of 1.0
    """
    if pd.isna(text):
        return 0.1
        
    word_count = len(str(text).split())
    
    if 11 <= word_count <= 50:
        return 1.5
    elif word_count < 5:
        return 0.1
    elif word_count > 200:
        return 0.2
    else:
        return 1.0

def generate_weighted_validation_sample(input_csv_path: str = "filtered_dataset.csv", 
                                        total_sample_size: int = 500, 
                                        overlap_ratio: float = 0.10):

    print(f"Ingesting production dataset from: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        print(f"Error: Target file '{input_csv_path}' not found. Verify the file path.")
        return

    df = pd.read_csv(input_csv_path, dtype=str)
    if "text" not in df.columns:
        print("Error: Input CSV must contain a valid text attribute column named 'text'.")
        return
        
    df = df.dropna(subset=["text"]).copy()
    
    print("Computing structural weights based on information density distribution...")
    df["sampling_weight"] = df["text"].apply(calculate_length_based_weight)
    
    print(f"Extracting a statistically representative sample (Size: {total_sample_size})...")
    master_sample = df.sample(
        n=min(total_sample_size, len(df)), 
        weights="sampling_weight", 
        random_state=42  
    ).copy()
    
    master_sample["sample_id"] = [f"SMP_{i:04d}" for i in range(len(master_sample))]
    
    master_sample = master_sample.sample(frac=1, random_state=123).reset_index(drop=True)
    
    total_sampled_rows = len(master_sample)
    overlap_size = int(total_sampled_rows * overlap_ratio)  
    
    print(f"Total Unified Evaluation Pool: {total_sampled_rows} records.")
    print(f"Isolating Cross-Validation Overlap: {overlap_size} rows ({overlap_ratio*100}% ratio).")
    
    overlap_subset = master_sample.iloc[:overlap_size].copy()
    remaining_subset = master_sample.iloc[overlap_size:]
    
    half_split = len(remaining_subset) // 2
    exclusive_rater_a = remaining_subset.iloc[:half_split].copy()
    exclusive_rater_b = remaining_subset.iloc[half_split:].copy()
    
    task_a = pd.concat([exclusive_rater_a, overlap_subset]).sample(frac=1, random_state=9).reset_index(drop=True)
    task_b = pd.concat([exclusive_rater_b, overlap_subset]).sample(frac=1, random_state=99).reset_index(drop=True)
    
    output_columns = ["sample_id", "text"]
    task_a_blind = task_a[output_columns].copy()
    task_b_blind = task_b[output_columns].copy()
    
    task_a_blind["human_label"] = ""
    task_b_blind["human_label"] = ""
    
    task_a_blind.to_csv("annotator_a_tasks.csv", index=False)
    task_b_blind.to_csv("annotator_b_tasks.csv", index=False)
    master_sample.to_csv("secret_pipeline_answer_key.csv", index=False)
    
    print("\n[SUCCESS] IRR Audit infrastructure successfully deployed")
    print("  Generated: 'annotator_a_tasks.csv'")
    print("  Generated: 'annotator_b_tasks.csv'")
    print(f"  Workload Check: Rater A = {len(task_a_blind)} rows | Rater B = {len(task_b_blind)} rows.")

if __name__ == "__main__":
    if not os.path.exists("filtered_dataset.csv"):
        mock_data = pd.DataFrame({
            "text": [
                "short", 
                "Looking for a compromised database. DM immediately.", 
                "System log trace debug drop packet " * 25
            ] * 100
        })
        mock_data.to_csv("filtered_dataset.csv", index=False)

    generate_weighted_validation_sample()