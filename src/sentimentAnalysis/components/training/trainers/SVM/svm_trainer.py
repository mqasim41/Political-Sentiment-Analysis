from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sentimentAnalysis.components.training.trainers.model_trainer import ModelTrainer
from sentimentAnalysis import logger

class SVMTraining(ModelTrainer):
    def __init__(self, config):
        self.model = None
        self.config = config
        self.encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer()

    def train(self):
        config = self.config
        train_data_path = Path(config.data_path) / "train_data.json"
        df = pd.read_json(train_data_path, lines=True)
        X = df['title'].values
        y = df[config.type].values
        X_transformed = self.vectorizer.fit_transform(X)
        y_encoded = self.encoder.fit_transform(y)
        self.model = SVC(kernel='rbf')
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5)
        grid_search.fit(X_transformed, y_encoded)
        self.model = grid_search.best_estimator_
        self.save()
        self.test()
        

    def predict(self, data):
        vectorizer = joblib.load(Path(str(self.config.model_save_dir) + '/vectorizer.joblib'))
        model = joblib.load(Path(str(self.config.model_save_dir) + '/svm_model.joblib'))
        X = vectorizer.transform(data)
        return model.predict(X)

    def save(self):
        if self.model is not None:
            model_save_dir = Path(self.config.model_save_dir)
            model_save_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_save_dir / 'svm_model.joblib'
            joblib.dump(self.model, model_path)
            vectorizer_path = model_save_dir / 'vectorizer.joblib'
            encoder_path = model_save_dir / 'encoder.joblib'
            joblib.dump(self.vectorizer, vectorizer_path)
            joblib.dump(self.encoder, encoder_path)
        else:
            raise ValueError("Model not trained. Cannot save.")

    def test(self):
        test_data_path = Path(str(self.config.data_path) + '/test_data.json')
        test_df = pd.read_json(test_data_path, lines=True)
        X_test = test_df['title'].values
        y_test = test_df[self.config.type].values
        vectorizer = joblib.load(Path(str(self.config.model_save_dir) + '/vectorizer.joblib'))
        encoder = joblib.load(Path(str(self.config.model_save_dir) + '/encoder.joblib'))
        X_test_transformed = vectorizer.transform(X_test)
        y_test_encoded = encoder.transform(y_test)
        y_pred = self.model.predict(X_test_transformed)
        report = classification_report(y_test_encoded, y_pred, target_names=encoder.classes_)
        print(report)
