import os
import json
from pathlib import Path
import pandas as pd

def main():
    # Configuration
    predictions_dir = "outputs_task2/evidence_selection_dev"
    output_excel = "results_excel/task2_evaluation_results.xlsx"
    
    # Get all JSON files in the predictions directory
    pred_files = list(Path(predictions_dir).glob("*.json"))
    
    if not pred_files:
        print(f"No JSON files found in {predictions_dir}")
        return
    
    # Store results
    results = []
    
    # Evaluate each prediction file
    for pred_file in sorted(pred_files):
        print(f"Evaluating: {pred_file.name}")
        
        try:
            # Get model name from filename
            model_name = pred_file.stem
            
            # Load predictions file (contains both label and pred_label)
            with open(pred_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Calculate accuracy by comparing label vs pred_label
            correct = 0
            empty_preds = 0
            total = len(data)
            
            for item in data:
                gold_label = item.get("label", "").lower()
                pred_label = item.get("pred_label", "")
                
                # Count empty predictions
                if not pred_label or str(pred_label).strip() == "":
                    empty_preds += 1
                
                if gold_label == pred_label.lower():
                    correct += 1
            
            accuracy = (correct / total * 100) if total > 0 else 0.0
            
            results.append({
                "Model": model_name,
                "Accuracy (%)": f"{accuracy:.1f}",
                "Correct": correct,
                "Total": total,
                "Empty": empty_preds
            })
            
            print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total}), Empty: {empty_preds}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results.append({
                "Model": pred_file.stem,
                "Accuracy (%)": "Error",
                "Correct": "Error",
                "Total": "Error",
                "Empty": "Error"
            })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(results)
    # Convert accuracy to float for proper sorting, then back to string with 1f format
    df["Accuracy (%)"] = pd.to_numeric(df["Accuracy (%)"], errors='coerce')
    df = df.sort_values("Accuracy (%)", ascending=False)
    df["Accuracy (%)"] = df["Accuracy (%)"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "Error")
    
    # Ensure output directory exists
    Path(output_excel).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_excel, index=False)
    
    print(f"\nResults saved to {output_excel}")
    print("\nSummary:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()