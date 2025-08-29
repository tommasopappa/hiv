import json
import pandas as pd
from collections import Counter
import random

def check_class_balance(data, name="Dataset"):
    """
    Check the balance of classes (0's and 1's) in the dataset.

    Args:
        data: List of dictionaries or DataFrame
        name: Name of the dataset for display

    Returns:
        Dictionary with class counts and percentages
    """
    if isinstance(data, pd.DataFrame):
        targets = data['target'].values
    else:
        # For list of dictionaries in LitGPT format
        targets = [int(item['output']) for item in data]

    class_counts = Counter(targets)
    total = len(targets)

    print(f"\n{name} Class Balance:")
    print(f"Total samples: {total}")
    for class_label in sorted(class_counts.keys()):
        count = class_counts[class_label]
        percentage = (count / total) * 100
        print(f"Class {class_label}: {count} samples ({percentage:.2f}%)")

    return {
        'total': total,
        'class_0': class_counts[0],
        'class_1': class_counts[1],
        'ratio_0': class_counts[0] / total,
        'ratio_1': class_counts[1] / total
    }


def stratified_train_val_split(data, train_ratio=0.8, random_seed=42):
    """
    Create a stratified train/validation split that maintains class balance.

    Args:
        data: List of dictionaries in LitGPT format
        train_ratio: Ratio of training data
        random_seed: Random seed for reproducibility

    Returns:
        train_data, val_data
    """
    random.seed(random_seed)

    # Separate data by class
    class_0_data = [item for item in data if item['output'] == '0']
    class_1_data = [item for item in data if item['output'] == '1']

    # Shuffle each class separately
    random.shuffle(class_0_data)
    random.shuffle(class_1_data)

    # Calculate split indices for each class
    split_idx_0 = int(len(class_0_data) * train_ratio)
    split_idx_1 = int(len(class_1_data) * train_ratio)

    # Split each class
    train_class_0 = class_0_data[:split_idx_0]
    val_class_0 = class_0_data[split_idx_0:]

    train_class_1 = class_1_data[:split_idx_1]
    val_class_1 = class_1_data[split_idx_1:]

    # Combine and shuffle
    train_data = train_class_0 + train_class_1
    val_data = val_class_0 + val_class_1

    random.shuffle(train_data)
    random.shuffle(val_data)

    return train_data, val_data


def convert_to_litgpt_format(df, task_type="classification"):
    """
    Convert the HIV dataset to LitGPT JSONL format with instruction/input/output fields.

    Args:
        df: DataFrame containing the HIV dataset
        task_type: Type of task - "classification", "property_prediction", or "generation"

    Returns:
        List of dictionaries in LitGPT format
    """
    litgpt_data = []

    for idx, row in df.iterrows():
        # Create different instruction templates based on task type
        if task_type == "classification":
            # Binary classification for HIV activity
            instruction = "Classify the following molecule based on its HIV activity. Respond with '1' if the molecule shows HIV activity, or '0' if it does not."
            input_text = f"SMILES: {row['smiles']}"
            output_text = str(int(row['target']))

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        # Create the LitGPT format entry
        litgpt_entry = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }

        litgpt_data.append(litgpt_entry)

    return litgpt_data


def save_to_jsonl(data, filename):
    """
    Save data to JSONL format (one JSON object per line).

    Args:
        data: List of dictionaries
        filename: Output filename
    """
    with open(filename, 'w') as f:
        for item in data:
            json_line = json.dumps(item)
            f.write(json_line + '\n')
    print(f"Saved {len(data)} entries to {filename}")


