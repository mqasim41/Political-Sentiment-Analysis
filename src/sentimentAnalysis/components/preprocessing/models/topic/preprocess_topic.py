from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sentimentAnalysis.components.preprocessing.models.preprocess_model import PreprocessModel

class TopicPreprocessor(PreprocessModel):

    def __init__(self, config):
        self.config = config

    def preprocess(self):
        config = self.config
        num_classes = len(config.topics)
        df = pd.read_json(config.data_path)

        def reassign_topic(topic):
            for t in config.topics:
                if t in str(topic):
                    return t
            return 'N/A'

        df['topic'] = df['topic'].apply(reassign_topic)
        topic_mask = df['topic'] != 'N/A'
        df = df[topic_mask]
        unique_classes = df['topic'].unique()
        assert len(unique_classes) == num_classes, f"Expected {num_classes} classes, but found {len(unique_classes)}."
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[self.config.type])
        train_save_path = Path(str(config.save_path) + '/train_data.json')
        test_save_path = Path(str(config.save_path) + '/test_data.json')
        train_df.to_json(train_save_path, orient='records', lines=True)
        test_df.to_json(test_save_path, orient='records', lines=True)

        
