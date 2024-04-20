import os
from dataclasses import dataclass
from enum import Enum

class ModelPrefix(Enum):
    Transformer = 1
    CNN = 2
    Sem_MSCNN = 3
    CNN_LSTM = 4
    Hybrid = 5

    def to_string(self):
        match self:
            case ModelPrefix.Transformer:
                return "Transformer"
            case ModelPrefix .CNN:
                return "cnn"
            case ModelPrefix.Sem_MSCNN:
                return "sem-mscnn"
            case ModelPrefix.CNN_LSTM:
                return "cnn-lstm"
            case ModelPrefix.Hybrid:
                return "hybrid"


@dataclass
class TrainConfig:
    data_path: str #": f"{model_env.data_root}/nch_30x64.npz",
    model_path: str | None # f"{model_env.model_path}" if model_env.model_path is not None else None,
    model_dir: str # f"{model_env.model_dir}",
    # "model_name": "sem-mscnn_" + chstr,  # Must be one of: "Transformer", "cnn", "sem-mscnn", "cnn-lstm", "hybrid"
    model_name: str # "model_name": "Transformer_" + chstr,
    # Must be one of: Transformer: "cnn", "sem-mscnn", "cnn-lstm", "hybrid"
    epochs: int # model_env.n_epochs,  # best 200
    channels: list[float] #"channels": chs,

    regression: bool = False
    transformer_layers: int = 5
    transformer_layers: int= 5  # best 5
    drop_out_rate: float = 0.25  # best 0.25
    num_patches: int = 30  # best 30 TBD
    transformer_units: int = 32  # best 32
    regularization_weight: float = 0.001  # best 0.001
    num_heads: int = 4


@dataclass
class TrainEnv:
    data_root: str
    model_path: str | None
    model_dir: str
    n_epochs: int
    force_retrain: bool

    def to_train_config(
        self,
        # Must be one of: Transformer: "cnn", "sem-mscnn", "cnn-lstm", "hybrid"
        model_prefix: ModelPrefix,
        chan_str: str,
        channels: list[float],
    ) -> TrainConfig:
        return TrainConfig(
            data_path=f"{self.data_root}/nch_30x64.npz",
            model_path=self.model_path,
            model_dir=self.model_dir,
            model_name=f"{model_prefix.to_string()}_{chan_str}",
            epochs=self.n_epochs,
            channels=channels,
        )

    


def parse_train_env() -> TrainEnv:
    data_root = os.getenv("DLHPROJ_DATA_ROOT", "/mnt/e/data")
    # i.e. "./weights/semscnn_ecgspo2/f"
    model_name = os.getenv("DLHPROJ_MODEL_PATH", None)
    model_dir = os.getenv("DLHPROJ_MODEL_DIR", "./weights")
    # the paper reports the best is 200
    n_epochs = int(os.getenv("DLHPROJ_NUM_EPOCHS", "100"))
    force_retrain = bool(os.getenv("DLHPROJ_FORCE_RETRAIN", "False"))
    return TrainEnv(
        data_root,
        model_name,
        model_dir,
        n_epochs,
        force_retrain
    )
