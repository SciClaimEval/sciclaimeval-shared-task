import argparse
from evaluation.eval_script import eval_task_1_individual, eval_task_1_pair, eval_task_2_accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate SciClaimEval Shared Task submissions")
    parser.add_argument(
        "--task",
        type=str,
        choices=["task1", "task2", "both"],
        default="both",
        help="Choose which task to evaluate: task1, task2, or both (default: both)"
    )
    parser.add_argument(
        "--ground_truth_task1",
        type=str,
        default="examples/task1_ground_truth.json",
        help="Path to Task 1 ground truth file"
    )
    parser.add_argument(
        "--pred_task1",
        type=str,
        default="examples/task1_pred_format.json",
        help="Path to Task 1 prediction file"
    )
    parser.add_argument(
        "--ground_truth_task2",
        type=str,
        default="examples/task2_ground_truth.json",
        help="Path to Task 2 ground truth file"
    )
    parser.add_argument(
        "--pred_task2",
        type=str,
        default="examples/task2_pred_format.json",
        help="Path to Task 2 prediction file"
    )
    
    args = parser.parse_args()
    
    ## Task 1
    if args.task in ["task1", "both"]:
        print("\n" + "="*50)
        print("Evaluating Task 1")
        print("="*50)
        
        scores = eval_task_1_individual(args.pred_task1, args.ground_truth_task1)
        print("\nIndividual scores:", scores)
        
        pair_scores = eval_task_1_pair(args.pred_task1, args.ground_truth_task1)
        print("Pair scores:", pair_scores)
    
    ## Task 2
    if args.task in ["task2", "both"]:
        print("\n" + "="*50)
        print("Evaluating Task 2")
        print("="*50)
        
        task2_scores = eval_task_2_accuracy(args.pred_task2, args.ground_truth_task2)
        print("\nTask 2 scores:", task2_scores)


if __name__ == "__main__":
    main()