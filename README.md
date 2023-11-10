# Project Setup Instructions

This README provides a step-by-step guide to setting up your Python project. Follow these instructions to initialize a virtual environment, install necessary dependencies, and run preprocessing scripts.

## Initialising a Virtual Environment

To isolate the project dependencies, it's recommended to use a virtual environment. Follow these steps to set it up:

1. **Create a Virtual Environment:** Run the following command in your terminal:

   ```bash
   python -m venv venv

2. **Activate the Virtual Environment:** Run the following command in your terminal:

   ```bash
   source venv/bin/activate

3. **Install Dependencies:** Run the following command in your terminal:

   ```bash
   pip install -r requirements.txt

## Running Preprocessing Scripts

To run the preprocessing scripts, follow these steps:


1. **Run the Script:** Run the following command in your terminal:

   ```bash
   python Baseline/tweets_emotions/preprocessing.py
   python Baseline/emo2019/preprocessing.py

## Repository Contents

This section describes the contents of the repository, divided into two main directories: `tweets_emotions` and `emo2019`. Each directory contains notebooks associated with various models.

To view the results of the baseline models, please refer to the `model_comparison` notebooks in each directory. For BERT + CNN models, please refer to the `bert_cnn_cf.ipynb` and `bert_cnn_emo.ipynb` notebooks in the `tweets_emotions` and `emo2019` directories respectively.


### tweets_emotions Directory

| Notebook             | Description                                  |
|----------------------|----------------------------------------------|
| [CNN](Baseline/tweets_emotions/cnn_te.ipynb)                  | Notebook for the Convolutional Neural Network model. |
| [BiLSTM](Baseline/tweets_emotions/bi_lstm_te.ipynb)               | Notebook for the Bidirectional Long Short-Term Memory model. |
| [AttBiLSTM](Baseline/tweets_emotions/att_bi_lstm_te.ipynb)            | Notebook for the Attention-based Bidirectional LSTM model. |
| [Model Comparison]Baseline/(tweets_emotions/model_comparison.ipynb)     | Notebook for comparing the performance of the different models. |

### emo2019 Directory

| Notebook             | Description                                  |
|----------------------|----------------------------------------------|
| [CNN](Baseline/emo2019/cnn_emo.ipynb)                  | Notebook for the Convolutional Neural Network model. |
| [BiLSTM](Baseline/emo2019/bi_lstm_emo.ipynb)               | Notebook for the Bidirectional Long Short-Term Memory model. |
| [AttBiLSTM](Baseline/emo2019/att_bi_lstm_emo.ipynb)            | Notebook for the Attention-based Bidirectional LSTM model. |
| [Model Comparison](Baseline/emo2019/model_comparison.ipynb)     | Notebook for comparing the performance of the different models. |



## Running BERT-CNN files
   use the following command in terminal to install extra library required for the files, this libray will rewrite some files use by keras libray which makes some files in previous steps to become unexecutable. (Create new virtual environment if necessary)
   ```bash
   pip install tf-models-official
   
### BERT-CNN Directory
| Notebook             | Description                                  |
|----------------------|----------------------------------------------|
| [BERT + CNN](BERT-CNN/bert_cnn_cf.ipynb)           | Notebook for the combination of BERT and CNN models. |
| [BERT + CNN](BERT-CNN/bert_cnn_emo.ipynb)           | Notebook for the combination of BERT and CNN models. |



