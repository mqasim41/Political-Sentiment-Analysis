from abc import ABC, abstractmethod

class PreprocessModel(ABC):
    @abstractmethod
    def preprocess(self, data):
        pass
