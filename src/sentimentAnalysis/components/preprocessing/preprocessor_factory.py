from sentimentAnalysis.components.preprocessing.models.alignment.preprocess_alignment import AlignmentPreprocessor
from sentimentAnalysis.components.preprocessing.models.political.preprocess_political_A import PoliticalPreprocessor
from sentimentAnalysis.components.preprocessing.models.topic.preprocess_topic import TopicPreprocessor


class PreprocessorFactory:
    @staticmethod
    def get_preprocessor(config):
        if config.type == 'classification':
            return PoliticalPreprocessor(config)
        elif config.type == 'topic':
            return TopicPreprocessor(config)
        elif config.type == 'political_alignment':
            return AlignmentPreprocessor(config)
        else:
            raise ValueError("Unknown preprocess type")
        
