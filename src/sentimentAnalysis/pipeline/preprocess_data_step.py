from sentimentAnalysis.config.configuration import ConfigurationManager
from sentimentAnalysis.components.preprocessing.preprocess import PreprocessTopicData
from sentimentAnalysis import logger



STAGE_NAME = "Preprocess Data"


class PreprocessDataPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        preprocess_data_config = config.get_preprocess_data_config()
        preprocess_data_model = PreprocessTopicData(config=preprocess_data_config)
        preprocess_data_model.preprocess_data_topic()



if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PreprocessDataPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e