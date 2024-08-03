import os
import urllib.request as request
from zipfile import ZipFile
import time
from sentimentAnalysis.components.training.trainer_factory import TrainerFactory
from sentimentAnalysis.entity.config_entity import TrainingConfig
from pathlib import Path



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trainer = TrainerFactory.get_training_model(self.config)
    
    def train(self):
        self.trainer.train()




    
   
