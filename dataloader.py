import numpy as np
import torch
import random
import torch.utils.data as utils
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def init_dataloader(dataset_config, dataseed):
    ts_data = np.load(dataset_config["time_seires"], allow_pickle=True)
    pearson_data = np.load(dataset_config["node_feature"], allow_pickle=True)
    label_df = np.load(dataset_config["label"], allow_pickle=True)
    final_pearson, final_ts, final_label = pearson_data, ts_data, label_df

    _, _, timeseries = final_ts.shape
    _, node_size, node_feature_size = final_pearson.shape
    out_size = np.max(final_label) + 1

    #对数据集时间序列进行归一化
    scaler = StandardScaler(mean=np.mean(
        final_ts), std=np.std(final_ts))
    final_ts = scaler.transform(final_ts)
    labels = final_label
    length = final_ts.shape[2]

    #读取数据划分占比
    train_set = dataset_config["train_set"]
    val_set = dataset_config["val_set"]
    n_splits = int(1.0 / val_set)

    #对数据集进行k-fold划分
    con_data = np.concatenate([final_ts, final_pearson], axis=2)
    kf = StratifiedKFold(n_splits=n_splits, random_state=dataseed, shuffle=True)
    zip_list = list(zip(con_data, labels))
    random.Random(dataseed).shuffle(zip_list)
    con_data, labels = zip(*zip_list)
    data_loaders = []
    con_data = np.array(con_data)
    labels = np.array(labels)
    for kk, (train_index, test_index) in enumerate(kf.split(con_data, labels)):
        train_con_data, val_con_data = con_data[train_index], con_data[test_index]
        train_ts, train_pearson = (train_con_data[:,:,0:length], train_con_data[:,:, length: ])
        val_ts, val_pearson = (val_con_data[:,:,0:length], val_con_data[:,:, length: ])
        train_label, val_label = labels[train_index], labels[test_index]

        train_ts, train_pearson, val_ts, val_pearson = [torch.from_numpy(
            data).float() for data in (train_ts, train_pearson, val_ts, val_pearson)]

        train_label, val_label = [torch.from_numpy(
            data).to(torch.int64) for data in (train_label, val_label)]

        train_label = F.one_hot(train_label)
        val_label = F.one_hot(val_label)

        train_dataset = utils.TensorDataset(
            train_ts,
            train_pearson,
            train_label
        )
        val_dataset = utils.TensorDataset(
            val_ts,
            val_pearson,
            val_label
        )
        train_dataloader = utils.DataLoader(
            train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

        val_dataloader = utils.DataLoader(
            val_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

        data_loaders.append((train_dataloader,val_dataloader))

    return data_loaders, node_size, node_feature_size, timeseries, out_size
