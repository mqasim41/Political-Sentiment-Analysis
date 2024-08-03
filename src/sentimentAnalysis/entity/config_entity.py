from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PreprocessDataConfig:
    data_path: Path
    type: str
    save_path:Path
    topics: list



@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    data_path: Path
    results_dir: Path
    model_save_dir:Path
    type:str
    model_type:str
    C:float
    params_epochs: int
    params_batch_size: int


@dataclass(frozen=True)
class EvaluationConfig:
    data_path: Path
    path_of_model: Path
    all_params: dict
    type: str
    params_batch_size: int