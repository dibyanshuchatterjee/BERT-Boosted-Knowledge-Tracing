"""
This framework assumes that the KT dataset provided is pre-processed and balanced
"""
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
import numpy as np
import time
import user_input as U
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


file_path, target_col = U.get_input()

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Specify device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model.to(device)

print(f"The device being used is {device}")


# Define a custom dataset
class LearnerDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


def preprocess_data(data, labels_column=None):
    columns_except_target = [col for col in data.columns.tolist() if col.lower() != target_col.lower()]
    # Combine the columns into a single string

    data['text'] = data[columns_except_target].astype(str).apply(' '.join, axis=1)

    # Tokenize the text
    encodings = tokenizer(data['text'].tolist(), truncation=True, padding=True)

    # Add the labels
    # encodings['labels'] = data['answer'].tolist()
    # Add the labels if labels_column is not None
    if labels_column is not None:
        encodings['labels'] = data[labels_column].tolist()

    return encodings


def evaluate(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []
    for batch in data_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.logits.argmax(-1).tolist())
            true_labels.extend(labels.tolist())
    return predictions, true_labels


def main():
    data = pd.read_csv(file_path)
    unique_labels = data['correct'].nunique()
    # Print the counts
    print("Number of unique labels in target:", unique_labels)

    unique_values = data['correct'].unique()

    print("Unique values:", unique_values)

    # Create mapping for target column
    label_mapping = {value: i for i, value in enumerate(unique_values)}
    print(label_mapping)
    data[target_col] = data[target_col].replace(label_mapping)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    test_size = input("Enter size of the test dataset ")
    while not test_size:
        print("This field is required. Please enter a value.")
        test_size = input("Enter size of the test dataset ")

    # Make train and test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=int(test_size), random_state=42)

    # Create DataFrames for training and validation
    train_data = pd.concat([X_train, y_train], axis=1)
    validation_data = pd.concat([X_val, y_val], axis=1)

    # Encode the train data
    training_encodings = preprocess_data(train_data, target_col)
    testing_encodings = preprocess_data(validation_data, target_col)

    # Create DataLoader
    training_data_loader = DataLoader(LearnerDataset(training_encodings), batch_size=32)
    testing_data_loader = DataLoader(LearnerDataset(testing_encodings), batch_size=32)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    epochs = input("Enter number of epochs the model should be trained for: ")
    while not epochs:
        print("This field is required. Please enter a value.")
        epochs = input("Enter number of epochs the model should be trained for: ")

    # Train the model using training_data_loader
    num_epochs = int(epochs)
    model.train()

    training_start_time = time.time()
    for epoch in range(num_epochs):
        for batch in training_data_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    training_stop_time = time.time()
    overall_time = (training_stop_time - training_start_time) / 60
    print(f"The training time taken for this model was : {overall_time:.2f} minutes")

    # Evaluate model
    predictions, true_labels = evaluate(model, testing_data_loader)

    # Creating checks to ensure logic switch between multiclass and binary classes
    if unique_labels > 2:
        predictions_array = np.array(predictions)
        onehot_encoder = OneHotEncoder(sparse_output=False)
        predictions_onehot = onehot_encoder.fit_transform(predictions_array.reshape(-1, 1))

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(true_labels, predictions, squared=False))
        print(f'RMSE: {rmse}')

        # Compute ACC
        acc = accuracy_score(true_labels, predictions)
        print(f'ACC: {acc}')

        auc = roc_auc_score(true_labels, predictions_onehot, multi_class='ovo')

        # Compute AUC
        print(f'AUC: {auc}')
    else:
        auc = roc_auc_score(true_labels, predictions)
        print(f'AUC: {auc}')

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(true_labels, predictions, squared=False))
        print(f'RMSE: {rmse}')

        # Compute ACC
        acc = accuracy_score(true_labels, predictions)
        print(f'ACC: {acc}')


if __name__ == '__main__':
    main()


