# BERT Sentiment Analysis with Fine-Tuning

This repository contains a pre-trained BERT model fine-tuned for sentiment analysis. The model is trained on a dataset with 'review' and 'sentiment' fields. It is then used to predict the sentiment of reviews in a test set.

## Setup

1. Clone this repository to your local machine.
2. Install the necessary Python packages. This project requires `numpy`, `pandas`, `torch`, `scikit-learn`, and `transformers`. You can install these with pip:

```sh
pip install numpy pandas torch scikit-learn transformers
```

## Data

The data used for this project should be in CSV format with 'review' and 'sentiment' columns. The 'review' column contains the text of the review, and the 'sentiment' column contains the sentiment of the review. The sentiment should be a binary value, with 0 representing negative sentiment and 1 representing positive sentiment.

The data is split into a training set and a validation set. The training set is used to fine-tune the model, and the validation set is used to evaluate the model's performance during training.

## Training

The model used for this project is the 'bert-base-uncased' model from Hugging Face's transformers library. It is a BERT model that has been pre-trained on a large corpus of uncased (lowercased) English text.

The model is fine-tuned for 3 epochs. During training, the model's performance is evaluated on the validation set every 500 steps, and the best-performing model is saved for later use. Training can be stopped early if the model's performance on the validation set does not improve for 3 consecutive evaluations.

## Prediction

After training, the fine-tuned model is used to predict the sentiment of reviews in a test set. The test set should be in the same format as the training and validation sets. The model's raw predictions are converted to binary sentiment predictions, with the higher value representing the predicted sentiment.

## Usage

To use this model, run the included Python script. The script includes code for reading the data, fine-tuning the model, and predicting sentiments of the test set. You can adjust this code as necessary for your specific task.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
