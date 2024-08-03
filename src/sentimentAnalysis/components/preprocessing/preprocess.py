from sentimentAnalysis.components.preprocessing.preprocessor_factory import PreprocessorFactory
from sentimentAnalysis.config.configuration import PreprocessDataConfig

class PreprocessTopicData:
    def __init__(self, config: PreprocessDataConfig):
        self.config = config
        self.preprocessor = PreprocessorFactory.get_preprocessor(self.config)
    
    def preprocess_data_topic(self):
        self.preprocessor.preprocess()
            




    
    

