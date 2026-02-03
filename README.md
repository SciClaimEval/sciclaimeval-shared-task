# SciClaimEval Shared Task: Dataset Information

## Subtask 1: Claim Label Prediction Task
Each sample includes the following information:

- paper_id: the ID of the paper; it can be an arXiv ID or a PeerJ ID
- claim_id: the ID of the claim
- claim: the claim for which the label needs to be predicted
- label: there are two labels in our dataset: Supported and Refuted
- caption: the caption of the evidence file
- context: the preceding sentences from the same paragraph, provided as a short contextual field for each claim sentence
- domain: three domains, ML, NLP, and PeerJ (medical domain)
- use_context: No (the claim is understandable without context), Yes (short context is needed; information is taken from the context field), or Other sources (the full paper is needed to understand the claim)
- operation: how the evidence is modified to obtain the modified evidence that pairs with the same claim to create a refuted sample
- paper_path: the path to the paper
- detail_others: if the operation is Other, a description is provided here
- claim_id_pair: one claim is paired with two pieces of evidence, creating two labels: Supported and Refuted

Please refer to the file [here](https://github.com/SciClaimEval/sciclaimeval-shared-task/blob/main/examples/task1_ground_truth.json) for an example.


Please prepare your prediction file following the format in [this file](https://github.com/SciClaimEval/sciclaimeval-shared-task/blob/main/examples/task1_pred_format.json).


### Information about the Test Set:

- You will receive the input for the test set, but the gold labels are not available.
Please refer to the file [here](https://github.com/SciClaimEval/sciclaimeval-shared-task/blob/main/examples/task1_test_input.json) for an example; the following keys are missing: label, operation, detail_others, and claim_id_pair.

- [Here](https://github.com/SciClaimEval/sciclaimeval-shared-task/blob/main/examples/task2_test_input.json) is the test input example for the second task.


## Subtask 2: Claim Evidence Prediction Task

- Please refer to the file [here](https://github.com/SciClaimEval/sciclaimeval-shared-task/blob/main/examples/task2_ground_truth.json) for an example.


- Please prepare your prediction file following the format in [this file](https://github.com/SciClaimEval/sciclaimeval-shared-task/blob/main/examples/task2_pred_format.json).


## Evaluation
### Run both tasks (default)
```bash
python3 run_eval.py
```

### Run only Task 1
```bash
python3 run_eval.py --task task1
```

### Run only Task 2
```bash
python3 run_eval.py --task task2
```

### Use custom file paths
```bash
python3 run_eval.py --task task1 --ground_truth_task1 path/to/gt.json --pred_task1 path/to/pred.json
```

## Note on License Information
The dataset is licensed under CC BY 4.0; however, individual samples may have their own licenses.

