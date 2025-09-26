import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pickle
import random
from torch.utils.data import DataLoader, TensorDataset


# 自定义归一化函数
def norm(sig_u):
    sig_u = np.nan_to_num(sig_u, nan=0.0, posinf=1.0, neginf=-1.0)
    epsilon = 1e-10  # 避免零除
    if len(sig_u.shape) == 3:
        pwr = np.sqrt(np.mean(np.sum(sig_u ** 2, axis=-1), axis=-1))
        pwr = np.where(pwr < epsilon, epsilon, pwr)  # 确保 pwr 不为零或极小
        sig_u = sig_u / pwr[:, None, None]
    elif len(sig_u.shape) == 2:
        pwr = np.sqrt(np.mean(np.sum(sig_u ** 2, axis=-1), axis=-1))
        pwr = np.where(pwr < epsilon, epsilon, pwr)
        sig_u = sig_u / pwr
    return sig_u


# 数据集预处理
def preprocess_data_1(file_name):
    with open(file_name, 'rb') as f:
        f_data = pickle.load(f)
    data = f_data['data']
    label = f_data['labels']

    X = np.array(data, dtype=np.float32)
    for k in range(len(X)):
        X[k] = norm(X[k])

    # 重塑数据为 (-1, 320, 2)，适配模型
    X = np.reshape(X, (-1, 320, 2))
    y = np.array(label) - 1  # 将标签调整为 [0, 24]

    return torch.Tensor(X), torch.LongTensor(y)


def preprocess_data_2(file_name, indices):
    with open(file_name, 'rb') as f:
        f_data = pickle.load(f)
    label = f_data['node_list']
    data = f_data['data']

    x, y = [], []
    for count, temp_tx_index in enumerate(indices):
        temp_data = data[temp_tx_index]
        indices_1 = random.sample(range(300), 200)
        temp_data = temp_data[indices_1]
        if len(temp_data) > 0:
            x.extend(temp_data)
            y.extend(np.tile(count, len(temp_data)))

    X = np.array(x, dtype=np.float32)
    for k in range(len(X)):
        X[k] = norm(X[k])

    X = np.reshape(X, (-1, 320, 2))
    y = np.array(y)

    return torch.Tensor(X), torch.LongTensor(y)


# 定义TCN模型
class TCNModel(nn.Module):
    def __init__(self, input_size, sequence_length, num_filters, filter_size, output_size):
        super(TCNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=filter_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=filter_size)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=num_filters * 2, out_channels=num_filters * 4, kernel_size=filter_size)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(in_channels=num_filters * 4, out_channels=num_filters * 8, kernel_size=filter_size)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        conv_output_size = sequence_length
        for _ in range(4):
            conv_output_size = (conv_output_size - filter_size + 1) // 2
        self.fc = nn.Linear(num_filters * 8 * conv_output_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 定义辅助模型
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
def pretrain_gs(gs, train_loader, epochs=100):
    optimizer_gs = optim.Adam(gs.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        gs.train()
        for x_batch, y_batch in train_loader:
            optimizer_gs.zero_grad()
            output = gs(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer_gs.step()
        if epoch % 10 == 0:
            print(f"Pretraining GS - Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    return gs


# 训练目标模型GT
def train_model(gs, gt, ga_s, ga_t, train_loader, test_loader, source_loader, source_available, epochs):
    optimizer_gt = optim.Adam(gt.parameters(), lr=1e-3)
    optimizer_ga_s = optim.Adam(ga_s.parameters(), lr=1e-3) if ga_s else None
    optimizer_ga_t = optim.Adam(ga_t.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_gt, T_max=epochs)

    classification_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        gt.train()
        total_loss_gt, total_loss_ga_s, total_loss_ga_t = 0, 0, 0  # 用于累加每个批次的损失

        for x_batch, y_batch in train_loader:
            optimizer_gt.zero_grad()
            output_gt = gt(x_batch)
            loss_gt = classification_loss_fn(output_gt, y_batch)

            if source_available:
                for x_s_batch, y_s_batch in source_loader:
                    output_gs_s = gs(x_s_batch).detach()
                    output_gs_t = gs(x_batch).detach()
                    output_ga_s = ga_s(output_gs_s)
                    output_ga_t = ga_t(output_gs_t)

                    loss_ga_s = mse_loss_fn(output_ga_s, output_gs_s)
                    loss_ga_t = mse_loss_fn(output_ga_t, output_gt.detach())

                    optimizer_ga_s.zero_grad()
                    optimizer_ga_t.zero_grad()
                    loss_ga_s.backward()
                    loss_ga_t.backward()
                    optimizer_ga_s.step()
                    optimizer_ga_t.step()

                    # 累加每个批次的损失
                    total_loss_gt += loss_gt.item()
                    total_loss_ga_s += loss_ga_s.item()
                    total_loss_ga_t += loss_ga_t.item()
            else:
                output_gs_t = gs(x_batch).detach()
                output_ga_t = ga_t(output_gs_t)

                loss_ga_t = mse_loss_fn(output_ga_t, output_gt.detach())
                optimizer_ga_t.zero_grad()
                loss_ga_t.backward()
                optimizer_ga_t.step()

                # 累加每个批次的损失
                total_loss_gt += loss_gt.item()
                total_loss_ga_t += loss_ga_t.item()

            loss_gt.backward()
            optimizer_gt.step()

        # 打印每个 epoch 的平均损失
        avg_loss_gt = total_loss_gt / len(train_loader)
        avg_loss_ga_s = total_loss_ga_s / len(source_loader) if source_available else 0
        avg_loss_ga_t = total_loss_ga_t / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Avg Loss_GT: {avg_loss_gt:.4f}, Avg Loss_GA_S: {avg_loss_ga_s:.4f}, Avg Loss_GA_T: {avg_loss_ga_t:.4f}")

        scheduler.step()
        print(f'Epoch [{epoch + 1}/{epochs}], lr: {scheduler.get_last_lr()[0]}')

        # Evaluation on test data
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
            print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}, Test F1 Score: {avg_f1:.4f}")


# 数据集加载和预处理
file_name1 = ''
file_name2 = ''
indices1 = list(range(0, 50))
indices2 = list(range(0, 85))
xT, target_T = preprocess_data_2(file_name1, indices1)
xS, target_S = preprocess_data_2(file_name2, indices2)

# 数据集划分为训练集和测试集
xT_train, xT_test, target_T_train, target_T_test = train_test_split(xT, target_T, test_size=0.2, random_state=42)

# 定义批大小
batch_size = 64
train_loader = DataLoader(TensorDataset(xT_train, target_T_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(xT_test, target_T_test), batch_size=batch_size, shuffle=False)
source_loader = DataLoader(TensorDataset(xS, target_S), batch_size=batch_size, shuffle=True)

# 初始化模型
input_size = 2
sequence_length = 320
num_filters = 16
filter_size = 7
output_size1 = 50
output_size2 = 85
hidden_size = 128

gs = TCNModel(input_size, sequence_length, num_filters, filter_size, output_size2)
gt = TCNModel(input_size, sequence_length, num_filters, filter_size, output_size1)
ga_s = AuxiliaryModel(output_size2, hidden_size, output_size2)  # 辅助模型 G_A_S
ga_t = AuxiliaryModel(output_size2, hidden_size, output_size1)  # 辅助模型 G_A_T 或 G_A

# 源域预训练
gs = pretrain_gs(gs, source_loader)

# 训练目标模型，假设源域数据可得
train_model(gs, gt, ga_s, ga_t, train_loader, test_loader, source_loader=source_loader, source_available=True,
            epochs=100)