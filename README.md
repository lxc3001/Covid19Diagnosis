# COVID-19 Multimodal Learning Framework

## Dependencies
Python version 3.11.3

Other dependencies in requirements.txt

## Project Description

This project implements contrastive learning methods to learn joint representations of chest X-ray images and corresponding clinical notes for COVID-19-related tasks. It includes the original ConVIRT model, a temporal extension (Sequential ConVIRT), and a hierarchical contrastive learning model. The trained models are evaluated on two downstream tasks: COVID-19 classification and image-text, text-image retrieval.

## To train ConVIRT, Sequential ConVIRT and Hierarchical Model

Run the following scripts respectively:

- `convirt.py` for ConVIRT architecture
- `sequentialConvirt.py` for sequential temporal model
- `contrastiveHierLoss.py` for hierarchical model with strong/weak positive pairs

## To do COVID-19 classification test

Run `baseline.py`

- Default is using image encoder only for classification.
- If you want to use both image and text encoders, change the `use_text` parameter in the `train_and_evaluate` function to `True`.
  - Remember to change the result file and output log directory to new ones to prevent overwriting previous results.

### Output:

- Result will be stocked in `classification_img_results.csv`
- Terminal output log file in `experiment_img_log.txt`

## To do Retrieval Test

Run `retrieval.py`

- This test evaluates retrieval between images and texts using learned embeddings.
- Includes the same three baselines as classification (Random Init., ImageNet Init., Ours).

### Output:

- Result will be stocked in `retrieval_results.csv`
- Terminal output log file in `experiment_retrieval_log.txt`

## For dataset

- Train dataset used for our model is in:
  - `./data/longitudinal_training_data.csv`

- Test dataset for classification and retrieval tasks is in:
  - `./data/test_data.csv`

- Original dataset with metadata is in:
  - `./data/metadata.csv`

- Medical images are stocked in:
  - `./data/new_images` (training images)
  - `./data/remaining_images` (testing images)

### Notes
Baselines reasults are also included in classification and retrieval task:

- **Random Init.**: ResNet-50 and BERT initialized randomly.
- **ImageNet Init.**: ResNet-50 pretrained on ImageNet and Clinical-BERT

Both baselines will be evaluated together and saved in the result files.

### Preliminairy results
Trained model results too big cannot upload to github

Classification Results:
- `classification_img_results.csv` for image-only classification
- `classification_imgtxt_results.csv` for image + text classification

Retrieval Results:
`retrieval_results.csv`
