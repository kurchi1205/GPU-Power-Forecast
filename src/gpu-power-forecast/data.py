import torch
import json
import numpy as np
from torch.utils.data import Dataset


def make_power_lag_features(power, lags):
    """
    power: np.ndarray, shape (T,)
    lags: list[int], e.g. [1, 2]

    Returns:
        lagged_power: np.ndarray, shape (T, len(lags))
        lagged_power[t, i] = power[t - lags[i]]
    """
    T = len(power)
    lagged = np.full((T, len(lags)), np.nan, dtype=np.float32)

    for i, lag in enumerate(lags):
        lagged[lag:, i] = power[:-lag]

    return lagged



class PowerDataset(Dataset):
    def __init__(self, json_path, seq_len=60, train_ratio=0.8, add_power_as_lag=False):
        self.seq_len = seq_len
        
        # load records
        records = []
        with open(json_path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except:
                    pass
        
        # convert to arrays
        self.power = np.array([r["gpu_power_watts"] for r in records], dtype=np.float32)

        # choose features you want
        self.features = np.array([
            [
                r["fps"],
                r["gs_fps"],
                r["num_pixels"],
                r["num_splats"],
                r["gpu_sm_util_percent"],
                r["gpu_mem_util_percent"],
                r["gpu_memory_used_MB"]
            ]
            for r in records
        ], dtype=np.float32)

        if add_power_as_lag:
            POWER_LAGS = [1, 2]
            power_lag_feats = make_power_lag_features(self.power, POWER_LAGS)
            self.features = np.concatenate(
                [self.features, power_lag_feats],
                axis=1
            )

            max_lag = max(POWER_LAGS)
            self.features = self.features[max_lag:]
            self.power = self.power[max_lag:]

        # normalize features
        train_size = int(len(self.features) * train_ratio)
        self.feature_mean = self.features[:train_size].mean(axis=0)
        self.feature_std = self.features[:train_size].std(axis=0) + 1e-9
        self.power_mean = self.power[:train_size].mean()
        self.power_std = self.power[:train_size].std() + 1e-9


        self.features = (self.features - self.feature_mean) / self.feature_std
        self.power = (self.power - self.power_mean) / self.power_std

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        X = self.features[idx : idx + self.seq_len]
        y = self.power[idx + self.seq_len]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

if __name__=="__main__":
    data = PowerDataset(json_path='../metrics_log/merged_log.json')
    print("Length of data: ", len(data))
    print(data[0])
    print("======================")
    print(data[1])