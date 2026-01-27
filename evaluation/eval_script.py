# Reference: https://github.com/Alab-NII/SciTabAlign/blob/main/evaluation/eval_claim.py

import json
from pathlib import Path
from typing import List, Dict, Union
from sklearn.metrics import precision_recall_fscore_support


def evaluate_claim_verification(
    predictions: List[str],
    ground_truth: List[str],
    labels: List[str] = ("Supported", "Refuted")
) -> Dict[str, str]:
    """
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0
    )

    return {
        "precision": f"{precision * 100:.1f}",
        "recall": f"{recall * 100:.1f}",
        "macro_f1": f"{f1 * 100:.1f}"
    }


def eval_task_1_individual(
    prediction_file: Union[str, Path],
    gold_file: Union[str, Path]
) -> Dict[str, str]:
    """
    """
    with open(prediction_file, encoding="utf-8") as f:
        pred_listdic = json.load(f)

    # Build a dict: {id: pred_label}
    pred_dic = {item["claim_id"]: item["pred_label"] for item in pred_listdic}

    with open(gold_file, encoding="utf-8") as f:
        gold_listdic = json.load(f)

    predictions, ground_truth = [], []

    for item in gold_listdic:
        cur_id = item["claim_id"]
        if cur_id not in pred_dic:
            print(f"Missing prediction for ID {cur_id}")
            continue
        predictions.append(pred_dic[cur_id])
        ground_truth.append(item["label"])

    return evaluate_claim_verification(predictions, ground_truth)


def eval_task_1_pair(
    prediction_file: Union[str, Path],
    gold_file: Union[str, Path]
) -> Dict[str, str]:
    """
    Evaluate pair accuracy: all samples with the same pair_id must be correct to count as 1.
    """
    with open(prediction_file, encoding="utf-8") as f:
        pred_listdic = json.load(f)

    # Build a dict: {id: pred_label}
    pred_dic = {item["claim_id"]: item["pred_label"] for item in pred_listdic}

    with open(gold_file, encoding="utf-8") as f:
        gold_listdic = json.load(f)

    # Group items by pair_id
    pair_groups = {}
    
    for item in gold_listdic:
        cur_id = item["claim_id"]
        pair_id = item.get("claim_id_pair", "")
        
        # Skip supported_only pairs
        if pair_id == "no pair":
            continue
            
        if pair_id not in pair_groups:
            pair_groups[pair_id] = []
        
        pair_groups[pair_id].append({
            "id": cur_id,
            "label": item["label"],
            "pred_label": pred_dic.get(cur_id, "")
        })
    
    # Evaluate each pair
    correct_pairs = 0
    total_pairs = len(pair_groups)
    
    for pair_id, items in pair_groups.items():
        # Check if all items in this pair are correctly predicted
        all_correct = True
        for item in items:
            if item["pred_label"].lower() != item["label"].lower():
                all_correct = False
                break
        
        if all_correct:
            correct_pairs += 1
    
    # Calculate pair accuracy
    pair_accuracy = (correct_pairs / total_pairs * 100) if total_pairs > 0 else 0.0
    
    return {
        "pair_accuracy": f"{pair_accuracy:.1f}",
        "correct_pairs": f"{correct_pairs}",
        "total_pairs": f"{total_pairs}"
    }


def eval_task_2_accuracy(
    prediction_file: Union[str, Path],
    gold_file: Union[str, Path]
) -> Dict[str, str]:
    """
    Evaluate task 2 accuracy: percentage of correct evidence selection.
    """
    with open(prediction_file, encoding="utf-8") as f:
        pred_listdic = json.load(f)

    # Build a dict: {id: pred_label}
    pred_dic = {item["sample_id"]: item["pred_label"] for item in pred_listdic}

    with open(gold_file, encoding="utf-8") as f:
        gold_listdic = json.load(f)

    predictions, ground_truth = [], []

    for item in gold_listdic:
        cur_id = item["sample_id"]
        if cur_id not in pred_dic:
            print(f"Missing prediction for ID {cur_id}")
            continue
        predictions.append(pred_dic[cur_id])
        ground_truth.append(item["label"])
    
    # Calculate accuracy
    correct = sum(1 for pred, gold in zip(predictions, ground_truth) if pred == gold)
    total = len(predictions)
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    return {
        "accuracy": f"{accuracy:.1f}",
        "correct": f"{correct}",
        "total": f"{total}"
    }

    
