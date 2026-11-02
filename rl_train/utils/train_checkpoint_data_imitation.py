from dataclasses import dataclass
from rl_train.utils.train_checkpoint_data import TrainCheckpointData

@dataclass
class ImitationTrainCheckpointData(TrainCheckpointData):
    # reward_accumulate and reward_weights are now inherited from TrainCheckpointData
    pass