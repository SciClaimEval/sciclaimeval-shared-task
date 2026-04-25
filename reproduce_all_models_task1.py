"""
Run evaluation for claim verification task across multiple models and output results to console and Excel file.
Seperate by use_context: no context, with context, all data
"""

import json
import pandas as pd
from evaluation.eval_script import evaluate_claim_verification, eval_task_1_pair


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_model(model_name, base_folder, prompt_type, gold_file, use_context_filter):
    """Evaluate a single model and return metrics"""
    input_path = f"{base_folder}/{model_name}_{prompt_type}.json"
    
    try:
        data = load_json(input_path)
    except FileNotFoundError:
        return None, None, None, None, 0, 0
    
    # Filter data based on use_context
    predictions = []
    ground_truth = []
    
    for item in data:
        if use_context_filter == "all":
            predictions.append(item["pred_label"])
            ground_truth.append(item["label"])
        elif use_context_filter == "no" and item["use_context"].lower() == "no":
            predictions.append(item["pred_label"])
            ground_truth.append(item["label"])
        elif use_context_filter == "yes" and item["use_context"].lower() != "no":
            predictions.append(item["pred_label"])
            ground_truth.append(item["label"])
    
    if len(predictions) == 0:
        return None, None, None, None, None, 0, 0
    
    # Get Macro-F1, Precision, Recall
    scores = evaluate_claim_verification(predictions, ground_truth)
    precision = float(scores["precision"])
    recall = float(scores["recall"])
    macro_f1 = float(scores["macro_f1"])
    
    # accuracy
    accuracy = float(scores["accuracy"])
    
    # Get Pair Accuracy
    pair_scores = eval_task_1_pair(input_path, gold_file, use_context_filter=use_context_filter)
    pair_accuracy = float(pair_scores["pair_accuracy"])
    num_pairs = int(pair_scores["total_pairs"])
    
    return precision, recall, macro_f1, accuracy, pair_accuracy, len(predictions), num_pairs


def main():
    models = ["llava-mistral-7b", "llava-vicuna-13b", "llama32-vl-11b", "internvl3_5-1b", 
              "internvl3_5-8b", "internvl3_5-14b", "internvl3_5-38b", "qwen3-vl-4b", 
              "qwen3-vl-8b", "qwen3-vl-30b", "o4-mini"]
    
    prompt_type = "zeroshot"
    ## TODO: edit here if you want to evaluate on test set
    data_type = "dev"  
    base_folder = f"outputs_task1/claim_task_{data_type}"
    gold_file = f"data/{data_type}_task1.json"
    
    print(f"\n{'='*100}")
    print(f"EVALUATION RESULTS FOR {data_type.upper()}")
    print(f"{'='*100}\n")
    
    # Collect results for all models
    results = {}
    no_context_samples = 0
    context_samples = 0
    no_context_pairs = 0
    context_pairs = 0
    all_pairs = 0
    
    for model_name in models:
        # No context results
        prec_no, rec_no, f1_no, acc_no, pa_no, n_no, p_no = evaluate_model(model_name, base_folder, prompt_type, gold_file, "no")
        
        # Context results
        prec_ctx, rec_ctx, f1_ctx, acc_ctx, pa_ctx, n_ctx, p_ctx = evaluate_model(model_name, base_folder, prompt_type, gold_file, "yes")
        
        # All data results
        prec_all, rec_all, f1_all, acc_all, pa_all, n_all, p_all = evaluate_model(model_name, base_folder, prompt_type, gold_file, "all")
        
        results[model_name] = {
            "no_context": (prec_no, rec_no, f1_no, acc_no, pa_no),
            "context": (prec_ctx, rec_ctx, f1_ctx, acc_ctx, pa_ctx),
            "all": (prec_all, rec_all, f1_all, acc_all, pa_all)
        }
        
        if n_no > 0:
            no_context_samples = n_no
            no_context_pairs = p_no
        if n_ctx > 0:
            context_samples = n_ctx
            context_pairs = p_ctx
        if n_all > 0:
            all_pairs = p_all
    
    # Print number of samples and pairs
    print(f"Number of samples - No Context: {no_context_samples}, Context: {context_samples}, All: {no_context_samples + context_samples}")
    print(f"Number of pairs - No Context: {no_context_pairs}, Context: {context_pairs}, All: {all_pairs}\n")
    
    # Print header
    print(f"{'Model':<25} {'No Context':^50} {'Context':^50} {'All':^50}")
    print(f"{'':<25} {'Prec':<10} {'Rec':<10} {'F1':<10} {'Acc':<10} {'P-Acc':<10} {'Prec':<10} {'Rec':<10} {'F1':<10} {'Acc':<10} {'P-Acc':<10} {'Prec':<10} {'Rec':<10} {'F1':<10} {'Acc':<10} {'P-Acc':<10}")
    print("-" * 180)
    
    # Print results for each model
    for model_name in models:
        res = results[model_name]
        prec_no, rec_no, f1_no, acc_no, pa_no = res["no_context"]
        prec_ctx, rec_ctx, f1_ctx, acc_ctx, pa_ctx = res["context"]
        prec_all, rec_all, f1_all, acc_all, pa_all = res["all"]
        
        prec_no_str = f"{prec_no:.1f}" if prec_no is not None else "N/A"
        rec_no_str = f"{rec_no:.1f}" if rec_no is not None else "N/A"
        f1_no_str = f"{f1_no:.1f}" if f1_no is not None else "N/A"
        acc_no_str = f"{acc_no:.1f}" if acc_no is not None else "N/A"
        pa_no_str = f"{pa_no:.1f}" if pa_no is not None else "N/A"
        
        prec_ctx_str = f"{prec_ctx:.1f}" if prec_ctx is not None else "N/A"
        rec_ctx_str = f"{rec_ctx:.1f}" if rec_ctx is not None else "N/A"
        f1_ctx_str = f"{f1_ctx:.1f}" if f1_ctx is not None else "N/A"
        acc_ctx_str = f"{acc_ctx:.1f}" if acc_ctx is not None else "N/A"
        pa_ctx_str = f"{pa_ctx:.1f}" if pa_ctx is not None else "N/A"
        
        prec_all_str = f"{prec_all:.1f}" if prec_all is not None else "N/A"
        rec_all_str = f"{rec_all:.1f}" if rec_all is not None else "N/A"
        f1_all_str = f"{f1_all:.1f}" if f1_all is not None else "N/A"
        acc_all_str = f"{acc_all:.1f}" if acc_all is not None else "N/A"
        pa_all_str = f"{pa_all:.1f}" if pa_all is not None else "N/A"
        
        print(f"{model_name:<25} {prec_no_str:<10} {rec_no_str:<10} {f1_no_str:<10} {acc_no_str:<10} {pa_no_str:<10} {prec_ctx_str:<10} {rec_ctx_str:<10} {f1_ctx_str:<10} {acc_ctx_str:<10} {pa_ctx_str:<10} {prec_all_str:<10} {rec_all_str:<10} {f1_all_str:<10} {acc_all_str:<10} {pa_all_str:<10}")
    
    print(f"\n{'='*180}\n")
    
    # Export to Excel
    excel_data = []
    for model_name in models:
        res = results[model_name]
        prec_no, rec_no, f1_no, acc_no, pa_no = res["no_context"]
        prec_ctx, rec_ctx, f1_ctx, acc_ctx, pa_ctx = res["context"]
        prec_all, rec_all, f1_all, acc_all, pa_all = res["all"]
        
        excel_data.append({
            "Model": model_name,
            "No_Context_Precision": f"{prec_no:.1f}" if prec_no is not None else "",
            "No_Context_Recall": f"{rec_no:.1f}" if rec_no is not None else "",
            "No_Context_Macro-F1": f"{f1_no:.1f}" if f1_no is not None else "",
            "No_Context_Acc": f"{acc_no:.1f}" if acc_no is not None else "",
            "No_Context_Pair-Acc": f"{pa_no:.1f}" if pa_no is not None else "",
            "Context_Precision": f"{prec_ctx:.1f}" if prec_ctx is not None else "",
            "Context_Recall": f"{rec_ctx:.1f}" if rec_ctx is not None else "",
            "Context_Macro-F1": f"{f1_ctx:.1f}" if f1_ctx is not None else "",
            "Context_Acc": f"{acc_ctx:.1f}" if acc_ctx is not None else "",
            "Context_Pair-Acc": f"{pa_ctx:.1f}" if pa_ctx is not None else "",
            "All_Precision": f"{prec_all:.1f}" if prec_all is not None else "",
            "All_Recall": f"{rec_all:.1f}" if rec_all is not None else "",
            "All_Macro-F1": f"{f1_all:.1f}" if f1_all is not None else "",
            "All_Acc": f"{acc_all:.1f}" if acc_all is not None else "",
            "All_Pair-Acc": f"{pa_all:.1f}" if pa_all is not None else ""
        })
    
    df = pd.DataFrame(excel_data)
    # TODO: edit output file name if needed
    output_excel = f"evaluation_results_{data_type}.xlsx"
    df.to_excel(output_excel, index=False, sheet_name=data_type.upper())
    print(f"Results saved to {output_excel}")


if __name__ == "__main__":
    main()