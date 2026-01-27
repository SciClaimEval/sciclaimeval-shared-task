from evaluation.eval_script import eval_task_1_individual, eval_task_1_pair, eval_task_2_accuracy

## Task 1

ground_truth_path = "examples/task1_ground_truth.json"
pred_path = "examples/task1_pred_format.json"

scores = eval_task_1_individual(pred_path, ground_truth_path)
print("Individual scores:", scores)

pair_scores = eval_task_1_pair(pred_path, ground_truth_path)
print("Pair scores:", pair_scores)


## Task 2

ground_truth_path_2 = "examples/task2_ground_truth.json"
pred_path_2 = "examples/task2_pred_format.json"

task2_scores = eval_task_2_accuracy(pred_path_2, ground_truth_path_2)
print("Task 2 scores:", task2_scores)