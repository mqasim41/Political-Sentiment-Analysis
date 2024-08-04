from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os
from sentimentAnalysis.components.dataset.koo_dataset import KooDataset
from sentimentAnalysis.components.training.trainers.model_trainer import ModelTrainer


class BERTTraining(ModelTrainer):
    def __init__(self, config):
        self.tokenizer = None
        self.model = None
        self.config = config
        self.encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self):
        config = self.config
        train_data_path = Path(str(config.data_path) + '/train_data.json')
        df = pd.read_json(train_data_path, lines=True)
        texts = df['title'].values
        labels = df[config.type].values
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Encode labels to integers
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        num_labels = len(set(encoded_labels))
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.model.to(self.device)

        encodings = self.tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')
        inputs = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        targets = torch.tensor(encoded_labels)

        dataset = KooDataset(inputs, targets, attention_mask)

        training_args = TrainingArguments(
            per_device_train_batch_size=config.params_batch_size,
            num_train_epochs=config.params_epochs,
            logging_dir='./logs',
            output_dir=config.results_dir
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        trainer.train()
        self.save(config)

    def predict(self, data):
        self.model.to(self.device)
        encodings = self.tokenizer(data, truncation=True, padding=True, max_length=512, return_tensors='pt')
        inputs = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        return predictions.cpu().numpy()

    def save(self, config):
        if self.model is not None:
            model_dir = config.model_save_dir
            os.makedirs(model_dir, exist_ok=True)
            model_path = Path(str(model_dir) + '/bert_model')
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
        else:
            raise ValueError("Model not trained. Cannot save.")
