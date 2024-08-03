import os
from sentimentAnalysis.constants import *
from sentimentAnalysis.entity.config_entity import DataIngestionConfig, PreprocessDataConfig
from sentimentAnalysis.entity.config_entity import TrainingConfig,EvaluationConfig
from sentimentAnalysis.utils.common import read_yaml, create_directories


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    def get_preprocess_data_config(self) -> PreprocessDataConfig:
        params = self.params
        config = self.config.preprocess_data
        data_ingestion_config = self.config.data_ingestion
        create_directories([Path(config.save_path+"/"+params.TYPE)])

        preprocess_topic_data_config = PreprocessDataConfig(
            data_path=Path(data_ingestion_config.unzip_dir+"/labelled_data_new.json"),
            type = params.TYPE,
            save_path=Path(config.save_path+"/"+params.TYPE),
            topics=self.params.TOPICS
        )

        return preprocess_topic_data_config
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        params = self.params
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            data_path=Path(training.data_path+"/"+params.TYPE),
            type=params.TYPE,
            results_dir=Path(training.results_dir),
            model_save_dir=Path(training.model_save_dir),
            model_type=training.model_type,
            C=params.C,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
        )

        return training_config
    def get_evaluation_config(self) -> EvaluationConfig:
        trainer = self.config.training
        eval_config = EvaluationConfig(
            data_path=Path(self.config.evaluation.data_path),
            path_of_model=Path(trainer.model_save_dir+'/bert_model'),
            all_params=self.params,
            type=self.params.TYPE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config
      