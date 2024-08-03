from sentimentAnalysis.components.training.trainers.BERT.bert_trainer import BERTTraining
from sentimentAnalysis.components.training.trainers.SVM.svm_trainer import SVMTraining


class TrainerFactory:
    @staticmethod
    def get_training_model(config):
        if config.model_type == 'SVM':
            return SVMTraining(config)
        elif config.model_type == 'BERT':
            return BERTTraining(config)
        else:
            raise ValueError("Unknown model type")
