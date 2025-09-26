import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pickle
import random
from torch.utils.data import DataLoader, TensorDataset


# 自定义归一化函数，假设 norm 是已定义的
def norm(sig_u):
    if len(sig_u.shape) == 3:
        pwr = np.sqrt(np.mean(np.sum(sig_u ** 2, axis=-1), axis=-1))
        sig_u = sig_u / pwr[:, None, None]
    if len(sig_u.shape) == 2:
        pwr = np.sqrt(np.mean(np.sum(sig_u ** 2, axis=-1), axis=-1))
        sig_u = sig_u / pwr
    # print(sig_u.shape)
    return sig_u


# 准备数据
def load_pkl(path):
    file = open(path, 'rb')
    many_sig = pickle.load(file)
    file.close()
    iq_data = many_sig['data']
    tx_list = many_sig['node_list']
    return tx_list, iq_data


# 数据集预处理
def preprocess_data_1(file_name):
    with open(file_name, 'rb') as f:
        f_data = pickle.load(f)
    data = f_data['data']  # 数据样本列表
    label = f_data['labels']  # 标签列表，保持不变

    X = np.array(data, dtype=np.float32)

    # 计算 Q1, Q3 和 IQR
    q1 = np.percentile(X.flatten(), 25)
    q3 = np.percentile(X.flatten(), 75)
    iqr = q3 - q1

    # 定义上下限阈值
    upper_threshold = q3 + 1.5 * iqr
    lower_threshold = q1 - 1.5 * iqr

    outlier_indices = np.any((X > upper_threshold) | (X < lower_threshold), axis=(1, 2))
    # 删除包含极端值的样本及其对应的标签
    X = X[~outlier_indices]

    y = np.array(label) - 1
    y = np.array(y)[~outlier_indices]

    for k in range(len(X)):
        X[k] = norm(X[k])

    # 重塑数据为 (-1, 320, 2)，适配模型
    X = np.reshape(X, (-1, 320, 2))

    return torch.Tensor(X), torch.LongTensor(y)


def load_pkl_2(path):
    file = open(path, 'rb')
    many_sig = pickle.load(file)
    file.close()
    iq_data = many_sig['data']
    tx_list = many_sig['node_list']
    return tx_list, iq_data


def preprocess_data_2(file_name, indices):
    label, data = load_pkl_2(file_name)

    new_label = []
    for idx, label_val in enumerate(label):
        new_label.append(idx)  # 重新分配标签
    label = new_label

    x = []
    y = []
    count = 0
    for temp_tx_index in indices:
        temp_data = data[temp_tx_index]

        # 随机采样100个数据点
        indices_1 = random.sample(range(200), 200)
        temp_data = temp_data[indices_1]

        temp_tx_label = count
        if len(temp_data) > 0:
            x.extend(temp_data)
            y.extend(np.tile(temp_tx_label, len(temp_data)))
        count += 1

    X = np.array(x, dtype=np.float32)

    # 计算 Q1, Q3 和 IQR
    q1 = np.percentile(X.flatten(), 25)
    q3 = np.percentile(X.flatten(), 75)
    iqr = q3 - q1

    # 定义上下限阈值
    upper_threshold = q3 + 1.5 * iqr
    lower_threshold = q1 - 1.5 * iqr

    outlier_indices = np.any((X > upper_threshold) | (X < lower_threshold), axis=(1, 2))

    # 删除包含极端值的样本及其对应的标签
    X = X[~outlier_indices]
    y = np.array(y)[~outlier_indices]

    # 功率归一化
    for k in range(len(X)):
        X[k] = norm(X[k])

    # 重塑数据为 (-1, 320, 2)，适配模型
    X = np.reshape(X, (-1, 320, 2))

    return torch.Tensor(X), torch.LongTensor(y)


# 定义TCN模型
class TCNModel(nn.Module):
    def __init__(self, input_size, sequence_length, num_filters, filter_size, output_size):
        super(TCNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=filter_size, stride=1,
                               padding=(filter_size - 1) // 2)
        self.BatchNorm1d1 = nn.BatchNorm1d(num_filters)
        self.relu1 = nn.ReLU()
        # self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 4, kernel_size=filter_size, stride=1,
                               padding=(filter_size - 1) // 2)
        self.BatchNorm1d2 = nn.BatchNorm1d(num_filters * 4)
        self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=num_filters * 4, out_channels=num_filters, kernel_size=filter_size, stride=1,
                               padding=(filter_size - 1) // 2)
        self.BatchNorm1d3 = nn.BatchNorm1d(num_filters)
        self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters // 2, kernel_size=filter_size,
                               stride=1, padding=(filter_size - 1) // 2)
        self.BatchNorm1d4 = nn.BatchNorm1d(num_filters // 2)
        self.relu4 = nn.ReLU()
        # self.pool4 = nn.MaxPool1d(kernel_size=2)
        self.Dropout = nn.Dropout(0.5)

        conv_output_size = sequence_length
        # for _ in range(4):
        #     conv_output_size = (conv_output_size - filter_size + 1) // 2
        # self.fc = nn.Linear(num_filters * 8 * conv_output_size, output_size)

        self.fc = nn.Linear((num_filters // 2) * conv_output_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.BatchNorm1d1(x)
        x = self.relu1(x)
        # x = self.pool(x)
        x = self.conv2(x)
        x = self.BatchNorm1d2(x)
        x = self.relu2(x)
        # x = self.pool2(x)
        x = self.conv3(x)
        x = self.BatchNorm1d3(x)
        x = self.relu3(x)
        # x = self.pool3(x)
        x = self.conv4(x)
        x = self.BatchNorm1d4(x)
        x = self.relu4(x)
        x = self.Dropout(x)
        # x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 定义辅助模型GA
class AuxiliaryModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AuxiliaryModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 源域数据预训练GS
def pretrain_gs(gs, xS, target_S, epochs=100):
    optimizer_gs = optim.Adam(gs.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        gs.train()
        optimizer_gs.zero_grad()
        output = gs(xS)
        loss = criterion(output, target_S)
        loss.backward()
        optimizer_gs.step()
        if epoch % 10 == 0:
            print(f"Pretraining GS - Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    return gs


# 训练模型时源域数据不可得
def train_without_source(gs, gt, ga, train_loader, test_loader, epochs):
    optimizer_gt = optim.Adam(gt.parameters(), lr=1e-3)
    optimizer_ga = optim.Adam(ga.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_gt, T_max=epochs)

    classification_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    best_accuracy = 0.0

    for epoch in range(epochs):
        gt.train()
        for x_batch, y_batch in train_loader:
            optimizer_gt.zero_grad()
            output_gt = gt(x_batch)
            loss_gt = classification_loss_fn(output_gt, y_batch)

            output_gs_t = gs(x_batch).detach()
            output_ga = ga(output_gs_t)
            loss_ga = mse_loss_fn(output_ga, output_gt.detach())

            optimizer_ga.zero_grad()
            loss_ga.backward()
            optimizer_ga.step()

            loss_gt.backward()
            optimizer_gt.step()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch + 1}/{epochs}], lr:{current_lr}')

        # 测试模型
        gt.eval()
        with torch.no_grad():
            total_loss, total_acc, total_f1 = 0, 0, 0
            for x_test_batch, y_test_batch in test_loader:
                output_gt_test = gt(x_test_batch)
                test_loss = classification_loss_fn(output_gt_test, y_test_batch).item()
                pred_test = torch.argmax(output_gt_test, dim=1)
                acc_test = accuracy_score(y_test_batch.cpu(), pred_test.cpu())
                f1_test = f1_score(y_test_batch.cpu(), pred_test.cpu(), average='weighted')
                total_loss += test_loss
                total_acc += acc_test
                total_f1 += f1_test
            avg_loss = total_loss / len(test_loader)
            avg_acc = total_acc / len(test_loader)
            avg_f1 = total_f1 / len(test_loader)
            if avg_acc > best_accuracy:
                best_accuracy = avg_acc
            print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}, Test F1 Score: {avg_f1:.4f}")

    print(f"{file_name3} Best accuracy: {best_accuracy:.4f}")


# 数据加载和处理
file_name1 = '/home/wjj/my_project/dataset/cleaned_combined_sltf_eq_dataset_25tx_2_1000_21070.pkl'
file_name2 = '/home/wjj/my_project/dataset/combined_pkt_dataset_89tx_10rx_30.pkl'
file_name3 = '/home/wjj/my_project/dataset/cleaned_processed_combined_320sltf_eq_pkt_dataset_2024_01_27_10tx_4node.pkl'
file_name5 = '/home/wjj/my_project/dataset/combined_320sltf_pkt_dataset_2024_01_27_10tx_4node.pkl'
file_name4 = '/home/wjj/my_project/dataset/combined_320sltf_eq_pkt_dataset_89tx_10rx_30.pkl'
indices1 = list(range(0, 10))
indices2 = list(range(0, 30))
xT, target_T = preprocess_data_1(file_name3)
# xT, target_T = preprocess_data_2(file_name5, indices1)
xS, target_S = preprocess_data_2(file_name2, indices2)

# 数据集划分为训练集和测试集
xT_train, xT_test, target_T_train, target_T_test = train_test_split(xT, target_T, test_size=0.2, random_state=42)

# 定义批大小
batch_size = 128
train_loader = DataLoader(TensorDataset(xT_train, target_T_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(xT_test, target_T_test), batch_size=batch_size, shuffle=False)

# 初始化模型
input_size = 2
sequence_length = 320
num_filters = 16
filter_size = 7
output_size1 = 10
output_size2 = 30
hidden_size = 128

gs = TCNModel(input_size, sequence_length, num_filters, filter_size, output_size2)
gs = pretrain_gs(gs, xS, target_S)  # 预训练模型 GS
gt = TCNModel(input_size, sequence_length, num_filters, filter_size, output_size1)  # 目标模型 GT
ga = AuxiliaryModel(output_size2, hidden_size, output_size1)  # 辅助模型 GA

# 训练目标模型，不使用源域数据
train_without_source(gs, gt, ga, train_loader, test_loader, epochs=100)