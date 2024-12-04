import torch
from util import Logger, accuracy, Metrics
import numpy as np
import torch.nn.functional as F
from util.loss import mixup_criterion, mixup_data, mixup_cluster_loss, multi_mixup_cluster_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLCTrain:
    def __init__(self, config, model, optimizer, scheduler, dataloaders, log_folder):
        train_config = config['train']
        self.logger = Logger()
        self.model = model.to(device)
        self.train_dataloader, self.val_dataloader = dataloaders
        self.epochs = train_config['epochs']
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.group_loss = train_config['group_loss']
        self.group_loss_weight = train_config['group_loss_weight']
        self.sparsity_loss = train_config['sparsity_loss']
        self.sparsity_loss_weight = train_config['sparsity_loss_weight']
        self.save_path = log_folder

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.train_accuracy, self.val_accuracy = [Metrics() for _ in range(5)]

    def reset_meters(self):
        for meter in [self.train_loss, self.val_loss, self.train_accuracy, self.val_accuracy]:
            meter.reset()

    def train_per_epoch(self, optimizer, scheduler):
        self.model.train()
        for data_in, pearson, label in self.train_dataloader:
            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)
            inputs, nodes, targets_a, targets_b, lam = mixup_data(
                data_in, pearson, label, 1, device)
            output, learnable_matrixs = self.model(inputs, nodes)
            loss = mixup_criterion(self.loss_fn, output, targets_a, targets_b, lam)
            if self.group_loss:
                group_loss = 0
                for learnable_matrix in learnable_matrixs:
                    group_loss += self.group_loss_weight * multi_mixup_cluster_loss(learnable_matrix,
                                            targets_a, targets_b, lam)
                group_loss = group_loss / len(learnable_matrixs)
                loss += group_loss
            if self.sparsity_loss:
                sp_loss = 0
                for learnable_matrix in learnable_matrixs:
                    sparsity_loss = self.sparsity_loss_weight * torch.norm(learnable_matrix, p=1)
                    sp_loss += sparsity_loss
                sp_loss = sp_loss / len(learnable_matrixs)
                loss += sp_loss
            self.train_loss.update_with_weight(loss.item(), label.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []
        self.model.eval()
        with torch.no_grad():
            for data_in, pearson, label in dataloader:
                label = label.long()
                data_in, pearson, label = data_in.to(
                    device), pearson.to(device), label.to(device)
                output, _ = self.model(data_in, pearson)
                label = torch.argmax(label, dim=1)
                loss = self.loss_fn(output, label)
                loss_meter.update_with_weight(
                    loss.item(), label.shape[0])
                top1 = accuracy(output, label)[0]
                acc_meter.update_with_weight(top1, label.shape[0])
                result += F.softmax(output, dim=1)[:, 1].tolist()
                labels += label.tolist()

    def save_result(self, results, best_model_dict, best_epoch):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "training_process.npy",
                results, allow_pickle=True)
        np.save(self.save_path / "best_epoch.npy",
                best_epoch, allow_pickle=True)
        torch.save(best_model_dict, self.save_path / "model.pt")

    def train(self):
        best_epoch = 0
        best_acc = 0
        best_model_dict = self.model.state_dict()
        training_process = []

        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizer, self.scheduler)
            self.test_per_epoch(self.val_dataloader, self.val_loss, self.val_accuracy)
            if self.val_accuracy.avg > best_acc:
                best_epoch = epoch
                best_acc = self.val_accuracy.avg
                best_model_dict = self.model.state_dict()
            if epoch % 10 == 0:
                self.logger.info(" | ".join([
                    f'Epoch[{epoch}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .4f}',
                    f'Val Loss:{self.val_loss.avg: .4f}',
                    f'Val Accuracy:{self.val_accuracy.avg: .4f}%',
                ]))
            training_process.append([self.train_loss.avg, self.val_loss.avg, self.val_accuracy.avg])
        print('best epcoh:{}'.format(best_epoch))
        best_info = training_process[best_epoch]
        self.logger.info(" | ".join([
            f'Epoch[{best_epoch}/{self.epochs}]',
            f'Train Loss:{best_info[0]: .4f}',
            f'Val Loss:{best_info[1]: .4f}',
            f'Val Accuracy:{best_info[2]: .4f}',
        ]))
        self.save_result(training_process, best_model_dict, best_epoch)


class MutiMLCTrain:
    def __init__(self, config, model, optimizer, scheduler, dataloaders, log_folder):
        train_config = config['train']
        self.logger = Logger()
        self.model = model.to(device)
        self.train_dataloader, self.val_dataloader = dataloaders
        self.epochs = train_config['epochs']
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.group_loss = train_config['group_loss']
        self.group_loss_weight = train_config['group_loss_weight']
        self.sparsity_loss = train_config['sparsity_loss']
        self.sparsity_loss_weight = train_config['sparsity_loss_weight']
        self.save_path = log_folder
        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.train_accuracy, self.val_accuracy, self.edges_num = [Metrics() for _ in
                                                                                                  range(5)]
    def reset_meters(self):
        for meter in [self.train_loss, self.val_loss, self.train_accuracy, self.val_accuracy, self.edges_num]:
            meter.reset()

    def train_per_epoch(self, optimizer, scheduler):
        self.model.train()
        for data_in, pearson, label in self.train_dataloader:
            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)
            inputs, nodes, targets_a, targets_b, lam = mixup_data(
                data_in, pearson, label, 1, device)
            output, learnable_matrixs = self.model(inputs, nodes)
            loss = mixup_criterion(self.loss_fn, output, targets_a, targets_b, lam)
            if self.group_loss:
                group_loss = 0
                for learnable_matrix in learnable_matrixs:
                    group_loss += self.group_loss_weight * multi_mixup_cluster_loss(learnable_matrix,
                                                                                    targets_a, targets_b, lam)
                group_loss = group_loss / len(learnable_matrixs)
                loss += group_loss
            if self.sparsity_loss:
                sp_loss = 0
                for learnable_matrix in learnable_matrixs:
                    sparsity_loss = self.sparsity_loss_weight * torch.norm(learnable_matrix, p=1)
                    sp_loss += sparsity_loss
                sp_loss = sp_loss / len(learnable_matrixs)
                loss += sp_loss
            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []
        self.model.eval()
        with torch.no_grad():
            for data_in, pearson, label in dataloader:
                label = label.long()
                data_in, pearson, label = data_in.to(
                    device), pearson.to(device), label.to(device)
                output, _ = self.model(data_in, pearson)
                label = torch.argmax(label, dim=1)
                loss = self.loss_fn(output, label)
                loss_meter.update_with_weight(
                    loss.item(), label.shape[0])
                top1 = accuracy(output, label)[0]
                acc_meter.update_with_weight(top1, label.shape[0])
                result += F.softmax(output, dim=1).tolist()
                labels += label.tolist()

    def save_result(self, results, best_model_dict, best_epoch):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "training_process.npy",
                results, allow_pickle=True)
        np.save(self.save_path / "best_epoch.npy",
                best_epoch, allow_pickle=True)
        torch.save(best_model_dict, self.save_path / "model.pt")

    def train(self):
        best_epoch = 0
        best_acc = 0
        best_model_dict = self.model.state_dict()
        training_process = []
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizer, self.scheduler)
            self.test_per_epoch(self.val_dataloader, self.val_loss, self.val_accuracy)

            # 更新最佳模型epoch，性能指标优先级Acc>AUC
            if self.val_accuracy.avg > best_acc:
                best_epoch = epoch
                best_acc = self.val_accuracy.avg

            # 打印输出该次迭代信息
            if epoch % 10 == 0:
                self.logger.info(" | ".join([
                    f'Epoch[{epoch}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .4f}',
                    f'Val Loss:{self.val_loss.avg: .4f}',
                    f'Val Accuracy:{self.val_accuracy.avg: .4f}%'
                ]))
            training_process.append([self.train_loss.avg, self.val_loss.avg, self.val_accuracy.avg])

        # 完成训练输出最佳模型epoch以及对应性能指标
        print('best epcoh:{}'.format(best_epoch))
        best_info = training_process[best_epoch]
        self.logger.info(" | ".join([
            f'Epoch[{best_epoch}/{self.epochs}]',
            f'Train Loss:{best_info[0]: .4f}',
            f'Val Loss:{best_info[1]: .4f}',
            f'Val Accuracy:{best_info[2]: .4f}'
        ]))
        self.save_result(training_process, best_model_dict, best_epoch)