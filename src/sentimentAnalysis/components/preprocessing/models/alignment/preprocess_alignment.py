from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sentimentAnalysis.components.preprocessing.models.preprocess_model import PreprocessModel
from sentimentAnalysis import logger

class AlignmentPreprocessor(PreprocessModel):

    def __init__(self, config):
        self.config = config
   
    def preprocess(self):
        config = self.config
        num_classes = 3
        df = pd.read_json(config.data_path)
        df['political_alignment'] = df['political_alignment'].fillna('')
        df.loc[df['political_alignment'].str.contains('Left-Wing Politics'), 'political_alignment'] = 'Left-Wing'
        df.loc[df['political_alignment'].str.contains('Right-Wing Politics'), 'political_alignment'] = 'Right-Wing'
        df.loc[df['political_alignment'].str.contains('Centrist Politics'), 'political_alignment'] = 'Centrist'
        df.loc[~df['political_alignment'].isin(['Left-Wing', 'Right-Wing', 'Centrist']), 'political_alignment'] = 'N/A'
        alignment_mask = df['political_alignment'] != 'N/A'
        df = df[alignment_mask]
        unique_classes = df['political_alignment'].unique()
        assert len(unique_classes) == num_classes, f"Expected {num_classes} classes, but found {len(unique_classes)}."
        logger.info(f"Size of Data After Preprocessing: {len(df)}")
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[self.config.type])
        train_save_path = Path(str(config.save_path) + '/train_data.json')
        test_save_path = Path(str(config.save_path) + '/test_data.json')
        train_df.to_json(train_save_path, orient='records', lines=True)
        test_df.to_json(test_save_path, orient='records', lines=True)