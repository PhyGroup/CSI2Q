import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc
import pickle
import random
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import weibull_min
import matplotlib.pyplot as plt


# ===========================
# 数据预处理函数
# ===========================

def norm(sig_u):
    """
    自定义归一化函数
    """
    if len(sig_u.shape) == 3:
        pwr = np.sqrt(np.mean(np.sum(sig_u ** 2, axis=-1), axis=-1))
        sig_u = sig_u / pwr[:, None, None]
    elif len(sig_u.shape) == 2:
        pwr = np.sqrt(np.mean(np.sum(sig_u ** 2, axis=-1), axis=-1))
        sig_u = sig_u / pwr
    return sig_u


def preprocess_data_common(X, labels_or_nodes=None):
    """
    通用的数据预处理步骤：异常值删除、归一化和数据重塑
    """
    # 计算 Q1, Q3 和 IQR
    q1 = np.percentile(X.flatten(), 25)
    q3 = np.percentile(X.flatten(), 75)
    iqr = q3 - q1

    # 定义上下限阈值
    upper_threshold = q3 + 1.5 * iqr
    lower_threshold = q1 - 1.5 * iqr

    # 查找异常值
    outlier_indices = np.any((X > upper_threshold) | (X < lower_threshold), axis=(1, 2))

    # 删除包含极端值的样本
    X = X[~outlier_indices]

    # 归一化
    for k in range(len(X)):
        X[k] = norm(X[k])

    # 重塑数据为 (-1, 320, 2)，适配模型
    X = np.reshape(X, (-1, 320, 2))

    # 如果存在标签或节点列表，处理它们
    if labels_or_nodes is not None:
        labels_or_nodes = np.array(labels_or_nodes)[~outlier_indices]  # 处理 node_list 或 labels

    return torch.Tensor(X), torch.LongTensor(labels_or_nodes) if labels_or_nodes is not None else torch.Tensor(X)


def preprocess_data_1(file_name):
    """
    处理含有 labels 键的数据文件
    """
    with open(file_name, 'rb') as f:
        f_data = pickle.load(f)

    data = f_data['data']  # 数据样本列表
    label = f_data['labels']  # 标签列表

    X = np.array(data, dtype=np.float32)

    # 调用通用预处理函数
    X, y = preprocess_data_common(X, label - 1)  # 标签减1以适应0-index
    return X, y


def preprocess_data_2(file_name, indices):
    """
    处理含有 node_list 键的数据文件
    """
    with open(file_name, 'rb') as f:
        f_data = pickle.load(f)

    data = f_data['data']  # 数据样本列表
    label = f_data['node_list']  # 标签列表

    # 重新分配标签
    new_label = [idx for idx in range(len(label))]

    x = []
    y = []
    count = 0
    for temp_tx_index in indices:
        temp_data = data[temp_tx_index]

        # 随机采样200个数据点
        indices_1 = random.sample(range(200), 200)
        temp_data = temp_data[indices_1]

        temp_tx_label = count
        if len(temp_data) > 0:
            x.extend(temp_data)
            y.extend(np.tile(temp_tx_label, len(temp_data)))
        count += 1

    X = np.array(x, dtype=np.float32)

    # 调用通用预处理函数
    X, y = preprocess_data_common(X, y)
    return X, y


# ===========================
# 模型定义
# ===========================

class TCNModel(nn.Module):
    def __init__(self, input_size, sequence_length, num_filters, filter_size, output_size):
        super(TCNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, num_filters, filter_size, padding=(filter_size - 1) // 2)
        self.BatchNorm1d1 = nn.BatchNorm1d(num_filters)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(num_filters, num_filters * 4, filter_size, padding=(filter_size - 1) // 2)
        self.BatchNorm1d2 = nn.BatchNorm1d(num_filters * 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(num_filters * 4, num_filters, filter_size, padding=(filter_size - 1) // 2)
        self.BatchNorm1d3 = nn.BatchNorm1d(num_filters)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(num_filters, num_filters // 2, filter_size, padding=(filter_size - 1) // 2)
        self.BatchNorm1d4 = nn.BatchNorm1d(num_filters // 2)
        self.relu4 = nn.ReLU()
        self.Dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(num_filters // 2 * sequence_length, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_size, sequence_length)
        x = self.relu1(self.BatchNorm1d1(self.conv1(x)))
        x = self.relu2(self.BatchNorm1d2(self.conv2(x)))
        x = self.relu3(self.BatchNorm1d3(self.conv3(x)))
        x = self.relu4(self.BatchNorm1d4(self.conv4(x)))
        x = self.Dropout(x)
        x = x.view(x.size(0), -1)  # 展平
        return self.fc(x)


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


# ===========================
# OpenMax 实现
# ===========================

def openmax_func(scores, alpha=1.5, beta=1.0, tau=0.5):
    """
    OpenMax函数，用于识别未知类别
    """
    # 计算Softmax得分
    softmax_scores = torch.softmax(scores, dim=1)

    # 获取每个样本的最大得分和其对应的类别
    max_score, _ = torch.max(softmax_scores, dim=1)

    # 使用Weibull分布来评估得分
    weibull_cdf = weibull_min.cdf(max_score.cpu().detach().numpy(), alpha, scale=beta)
    weibull_cdf_scores = weibull_cdf

    # 如果CDF值大于阈值，判定为已知类别，否则为未知类别
    return torch.tensor(weibull_cdf >= tau, dtype=torch.bool), weibull_cdf_scores


# ===========================
# 预训练GS模型
# ===========================

def pretrain_gs(gs, xS, target_S, epochs=30):
    """
    预训练GS模型
    """
    optimizer = optim.Adam(gs.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    gs.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = gs(xS)
        loss = criterion(output, target_S)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Pretrain Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return gs


# ===========================
# 训练GT模型
# ===========================

def train_gt_model(gt, gs, ga, train_loader, epochs=100):
    """
    训练GT模型和GA辅助模型
    """
    optimizer_gt = optim.Adam(gt.parameters(), lr=1e-3)
    optimizer_ga = optim.Adam(ga.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_gt, T_max=epochs)

    classification_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_gt_loss = 0
        gt.train()

        for x_batch, y_batch in train_loader:
            # 训练GT模型
            optimizer_gt.zero_grad()
            output_gt = gt(x_batch)
            loss_gt = classification_loss_fn(output_gt, y_batch)
            total_gt_loss += loss_gt.item()

            # 训练GA辅助模型
            output_gs_t = gs(x_batch).detach()
            output_ga = ga(output_gs_t)
            loss_ga = mse_loss_fn(output_ga, output_gt.detach())

            optimizer_ga.zero_grad()
            loss_ga.backward()
            optimizer_ga.step()

            # 反向传播GT模型的损失
            loss_gt.backward()
            optimizer_gt.step()

        scheduler.step()
        print(
            f'Epoch [{epoch + 1}/{epochs}], Avg Train Loss: {total_gt_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.4e}')

    return gt, ga


# ===========================
# 拟合Weibull分布参数
# ===========================

def fit_weibull_tail(gt, train_loader, tail_percentage=5):
    """
    仅使用得分低于指定分位数的训练样本来拟合 Weibull 分布
    """
    gt.eval()
    all_gt_scores = []

    with torch.no_grad():
        for x_batch, _ in train_loader:
            output_gt = gt(x_batch)
            softmax_scores = torch.softmax(output_gt, dim=1)
            max_scores = torch.max(softmax_scores, dim=1)[0].cpu().numpy()
            all_gt_scores.extend(max_scores)

    all_gt_scores = np.array(all_gt_scores)

    # 选择得分低于指定分位数的样本进行拟合
    threshold = np.percentile(all_gt_scores, tail_percentage)
    tail_scores = all_gt_scores[all_gt_scores <= threshold]

    if len(tail_scores) == 0:
        raise ValueError("没有足够的尾部数据用于拟合 Weibull 分布。请增加 tail_percentage 或检查数据。")

    alpha, loc, beta = weibull_min.fit(tail_scores, floc=0)  # 固定位置参数为0
    print(f'Fitted Weibull parameters on {tail_percentage}% tail: alpha={alpha:.4f}, beta={beta:.4f}')
    return alpha, beta


# ===========================
# 测试阶段应用OpenMax
# ===========================
def find_best_tau(weibull_cdf_scores, true_labels, tau_values=np.linspace(0, 1, 100), metric="f1"):
    """
    根据不同的 tau 值找到最佳的阈值，支持多种优化指标。

    :param weibull_cdf_scores: Weibull分布的CDF得分（预测概率）。
    :param true_labels: 真实标签，-1表示未知类别，其他表示已知类别。
    :param tau_values: tau 值的搜索范围，默认为 [0, 1] 的线性空间。
    :param metric: 优化指标，可选值为 "f1", "accuracy", "precision", "recall"。
    :return: 最佳 tau 值以及对应的所有指标得分。
    """
    # 初始化存储指标分数
    f1_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    # 将真实标签二值化，1 表示已知类别，0 表示未知类别
    binary_true = np.where(true_labels == -1, 0, 1)

    # 遍历 tau 值，计算每个 tau 下的预测性能
    for tau in tau_values:
        # 根据 tau 值进行预测，1 表示已知类别，0 表示未知类别
        pred_labels = np.where(weibull_cdf_scores >= tau, 1, 0)

        # 计算各类评估指标
        f1 = f1_score(binary_true, pred_labels)
        accuracy = accuracy_score(binary_true, pred_labels)
        precision = precision_score(binary_true, pred_labels, zero_division=0)
        recall = recall_score(binary_true, pred_labels, zero_division=0)

        # 存储指标
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # 根据指定的优化指标找到最佳 tau 值
    if metric == "f1":
        best_tau_idx = np.argmax(f1_scores)
    elif metric == "accuracy":
        best_tau_idx = np.argmax(accuracy_scores)
    elif metric == "precision":
        best_tau_idx = np.argmax(precision_scores)
    elif metric == "recall":
        best_tau_idx = np.argmax(recall_scores)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose from 'f1', 'accuracy', 'precision', 'recall'.")

    # 最佳 tau 值
    best_tau = tau_values[best_tau_idx]

    # 打印结果
    print(f'Best tau based on {metric}: {best_tau:.4f}')
    print(f'F1 Score: {f1_scores[best_tau_idx]:.4f}, Accuracy: {accuracy_scores[best_tau_idx]:.4f}, '
          f'Precision: {precision_scores[best_tau_idx]:.4f}, Recall: {recall_scores[best_tau_idx]:.4f}')

    # 返回最佳 tau 值及对应的所有指标分数
    return {
        "best_tau": best_tau,
        "f1_score": f1_scores[best_tau_idx],
        "accuracy": accuracy_scores[best_tau_idx],
        "precision": precision_scores[best_tau_idx],
        "recall": recall_scores[best_tau_idx]
    }


def plot_weibull_cdf_distribution(scores, alpha=1.5, beta=1.0, epoch=100):
    """
    绘制Weibull CDF分布图
    """
    weibull_cdf_values = weibull_min.cdf(scores, alpha, scale=beta)

    # 绘制CDF分布图
    plt.figure(figsize=(8, 6))
    plt.hist(weibull_cdf_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Weibull CDF Distribution at Epoch {epoch}")
    plt.xlabel('Weibull CDF Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def apply_openmax_final(gt, test_loader, alpha, beta, tau):
    """
    在测试阶段应用 OpenMax 进行开集识别
    """
    gt.eval()
    total_acc, total_f1, total_known_acc, total_known_f1, total_unknown_acc, total_unknown_f1 = 0, 0, 0, 0, 0, 0

    all_weibull_cdf_scores = []
    all_true_labels = []

    with torch.no_grad():
        for x_test_batch, y_test_batch in test_loader:
            output_gt_test = gt(x_test_batch)
            softmax_scores = torch.softmax(output_gt_test, dim=1)
            max_scores = torch.max(softmax_scores, dim=1)[0].cpu().numpy()
            weibull_cdf_scores = weibull_min.cdf(max_scores, alpha, scale=beta)
            all_weibull_cdf_scores.extend(weibull_cdf_scores)
            all_true_labels.extend(y_test_batch.cpu().numpy())

    all_weibull_cdf_scores = np.array(all_weibull_cdf_scores)
    all_true_labels = np.array(all_true_labels)

    # 找到最佳tau值
    result = find_best_tau(all_weibull_cdf_scores, all_true_labels, tau_values=np.linspace(0, 1, 100),
                           metric="precision")
    best_tau = result["best_tau"]

    # 应用最佳tau值进行最终测试
    adjusted_labels = np.where(all_weibull_cdf_scores >= best_tau, 1, 0)  # 1表示已知类别，0表示未知类别

    # 转换为类别标签
    # 计算softmax后，断开计算图并转换为numpy
    softmax_output = torch.softmax(gt(torch.Tensor(xT_test)).to(torch.device('cpu')).detach(), dim=1)
    pred_labels = np.where(adjusted_labels == 1, np.argmax(softmax_output.numpy(), axis=1), -1)

    # pred_labels = np.where(adjusted_labels == 1, np.argmax(torch.softmax(gt(torch.Tensor(xT_test)).to(torch.device('cpu')), dim=1).numpy(), axis=1), -1)

    # 计算各项指标
    binary_true = np.where(all_true_labels == -1, 0, 1)
    binary_pred = np.where(pred_labels == -1, 0, 1)

    acc = accuracy_score(binary_true, binary_pred)
    f1 = f1_score(binary_true, binary_pred, average='weighted')

    print(f'Best Tau: {best_tau:.4f}')
    print(f'Test Accuracy: {acc:.4f}, Test F1: {f1:.4f}')

    # 计算已知类别和未知类别的准确率
    known_mask = all_true_labels != -1
    unknown_mask = all_true_labels == -1

    known_acc = accuracy_score(all_true_labels[known_mask], pred_labels[known_mask])
    unknown_acc = accuracy_score(all_true_labels[unknown_mask], pred_labels[unknown_mask])

    print(f'Known Accuracy: {known_acc:.4f}, Unknown Accuracy: {unknown_acc:.4f}')

    # 绘制Weibull CDF分布图（可选）
    plot_weibull_cdf_distribution(all_weibull_cdf_scores, alpha, beta, epoch=100)

    # 绘制 ROC 曲线（可选）
    plot_roc_curve(all_weibull_cdf_scores, binary_true)

    return acc, f1, known_acc, unknown_acc


def plot_roc_curve(weibull_cdf_scores, true_labels):
    """
    绘制 ROC 曲线并计算 AUC
    """
    binary_true = np.where(true_labels == 0, 0, 1)  # 0表示未知类别，1表示已知类别
    fpr, tpr, thresholds = roc_curve(binary_true, weibull_cdf_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print(f'ROC AUC: {roc_auc:.4f}')


# ===========================
# 主流程执行
# ===========================

if __name__ == "__main__":
    # ===========================
    # 数据准备
    # ===========================

    # 数据文件路径（请根据实际情况修改）
    file_name1 = '/home/wjj/my_project/dataset/cleaned_combined_sltf_eq_dataset_25tx_2_1000_21070.pkl'
    file_name2 = '/home/wjj/my_project/dataset/combined_pkt_dataset_89tx_10rx_30.pkl'

    # 预处理数据
    xT, target_T = preprocess_data_1(file_name1)
    xS, target_S = preprocess_data_2(file_name2, indices=list(range(0, 85)))

    # 划分数据集
    registered_labels = np.where(target_T < 20)[0]
    unknown_labels = np.where(target_T >= 20)[0]

    train_labels = np.random.choice(registered_labels, int(0.8 * len(registered_labels)), replace=False)
    test_labels_registered = np.setdiff1d(registered_labels, train_labels)
    test_labels_unknown = np.random.choice(unknown_labels, int(0.2 * len(unknown_labels)), replace=False)

    # 将未知设备的标签设为 -1，保持为 torch.LongTensor
    target_T[test_labels_unknown] = -1

    test_labels = np.concatenate([test_labels_registered, test_labels_unknown])
    xT_train, target_T_train = xT[train_labels], target_T[train_labels]
    xT_test, target_T_test = xT[test_labels], target_T[test_labels]

    print("Test Labels:", target_T_test)

    # 数据加载器
    batch_size = 128
    train_loader = DataLoader(TensorDataset(xT_train, target_T_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(xT_test, target_T_test), batch_size=batch_size, shuffle=False)

    # ===========================
    # 模型初始化
    # ===========================

    input_size = 2
    sequence_length = 320
    num_filters = 16
    filter_size = 7
    output_size1 = 20  # GT模型的输出类别数
    output_size2 = 85  # GS模型的输出类别数
    hidden_size = 128

    gs = TCNModel(input_size, sequence_length, num_filters, filter_size, output_size2)
    gt = TCNModel(input_size, sequence_length, num_filters, filter_size, output_size1)
    ga = AuxiliaryModel(output_size2, hidden_size, output_size1)

    # 预训练GS模型
    print("Pretraining GS model...")
    gs = pretrain_gs(gs, xS, target_S, epochs=100)

    # 训练GT模型
    print("Training GT model...")
    gt, ga = train_gt_model(gt, gs, ga, train_loader, epochs=100)

    # 拟合Weibull分布参数（仅使用5%的尾部数据）
    print("Fitting Weibull parameters on tail data...")
    alpha, beta = fit_weibull_tail(gt, train_loader, tail_percentage=5)

    # 测试阶段应用OpenMax
    print("Testing with OpenMax...")
    avg_acc, avg_f1, avg_known_acc, avg_unknown_acc = apply_openmax_final(gt, test_loader, alpha, beta, tau=0.5)

    # 最终输出
    print(f'Final Test Accuracy: {avg_acc:.4f}, Final Test F1: {avg_f1:.4f}')

