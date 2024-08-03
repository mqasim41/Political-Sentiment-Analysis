import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
from pathlib import Path
import pandas as pd
from sentimentAnalysis.components.dataset.koo_dataset import KooDataset
from sentimentAnalysis.utils.common import save_json


class Evaluation:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def load_model(path: Path) -> BertForSequenceClassification:
        return BertForSequenceClassification.from_pretrained(path)

    def _valid_generator(self):
        test_data_path = Path(str(self.config.data_path)+'/test_data.json')
        df = pd.read_json(test_data_path, lines=True)
        texts = df['title'].values
        labels = df[self.config.type].values
        encodings = self.tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors='pt')
        inputs = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        dataset = KooDataset(inputs, labels, attention_mask)
        self.valid_loader = DataLoader(dataset, batch_size=self.config.params_batch_size, shuffle=False)

    def evaluate(self):
        if self.config.type == "SVM":
            return
        self.model = self.load_model(self.config.path_of_model)
        self.model.to(self.device)
        self.model.eval()
        self._valid_generator()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in self.valid_loader:
                inputs = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=inputs, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        self.accuracy = accuracy_score(all_labels, all_preds)
        self.f1 = f1_score(all_labels, all_preds, average='weighted')
        self.precision = precision_score(all_labels, all_preds, average='weighted')
        self.recall = recall_score(all_labels, all_preds, average='weighted')
        self.save_score()

    def save_score(self):
        if self.config.type == "SVM":
            return
        scores = {
            "accuracy": self.accuracy,
            "f1_score": self.f1,
            "precision": self.precision,
            "recall": self.recall
        }
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        if self.config.type == "SVM":
            return
        with mlflow.start_run():
            mlflow.log_params(vars(self.config))
            mlflow.log_metrics({
                "accuracy": self.accuracy,
                "f1_score": self.f1,
                "precision": self.precision,
                "recall": self.recall
            })
            mlflow.pytorch.log_model(self.model, "model", registered_model_name="BERTModel")
            
