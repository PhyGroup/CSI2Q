# mat_to_pkl_with_numeric_nodes.py

# mat_to_pkl_with_numeric_nodes.py

import os
import scipy.io
import numpy as np
import pickle


def convert_folder_to_pkl(mat_folder, out_pkl_path, sample_num=1000):
    """
    将 mat_folder 下每个 .mat 文件里的 packet_log 或 csi_all（cell array）
    转换成 real/imag 格式的 ndarray，形状 (N_samples, 320, 2)，并存为一个 .pkl：
        {
          'data':   [arr1, arr2, ...],    # 每个 arr.shape = (<=sample_num, 320, 2)
          'node_list': [0, 1, 2, ...]      # 对应每个 .mat 文件的数值标签
        }
    并在控制台打印 字符串名称 -> 数值标签 的映射。"""
    data = []
    raw_names = []

    for fn in sorted(os.listdir(mat_folder)):
        if not fn.endswith('.mat'):
            continue
        fullpath = os.path.join(mat_folder, fn)
        mat = scipy.io.loadmat(fullpath, squeeze_me=False, struct_as_record=False)

        # 获取 csi 数据字段，避免使用 or 引发数组歧义
        arr = mat.get('packet_log', None)
        if arr is None:
            arr = mat.get('csi_all', None)
        if arr is None:
            print(f"[WARN] 跳过 {fn}（无 'packet_log'/'csi_all'）")
            continue

        # 将 MATLAB cell array 展平成 Python list
        try:
            cells = arr.flatten().tolist()
        except Exception:
            print(f"[WARN] 无法展开数组: {fn}")
            continue

        N = len(cells)
        if N > sample_num:
            idx = np.round(np.linspace(0, N-1, sample_num)).astype(int)
            cells = [cells[i] for i in idx]

        squeezed = [np.squeeze(c) for c in cells]
        complex_arr = np.stack(squeezed, axis=0)           # (N,320)
        real_imag   = np.stack([complex_arr.real,
                                complex_arr.imag], axis=2) # (N,320,2)

        data.append(real_imag)
        raw_names.append(os.path.splitext(fn)[0])
        print(f"  • {fn}: cells {len(cells)} → array {real_imag.shape}")

    # 创建 名称 -> 数值标签 映射
    name_to_label = {name: idx for idx, name in enumerate(raw_names)}
    numeric_labels = [name_to_label[name] for name in raw_names]

    print("\nName -> Numeric Label Mapping:")
    for name, label in name_to_label.items():
        print(f"  {name} -> {label}")

    pkl_dict = {
        'data': data,
        'node_list': numeric_labels
    }
    with open(out_pkl_path, 'wb') as f:
        pickle.dump(pkl_dict, f)

    print(f"\n✅ 保存 {len(data)} 条数据到 {out_pkl_path}")
    return pkl_dict


def inspect_pkl_dict(pkl_dict, n_preview=3):
    print("\n=== Inspect PKL ===")
    if not isinstance(pkl_dict, dict):
        print("✖️ 不是 dict，而是", type(pkl_dict))
        return

    print("Keys:", list(pkl_dict.keys()))
    data = pkl_dict.get('data', [])
    nl   = pkl_dict.get('node_list', [])
    print(f"'data' 条目数 = {len(data)}")
    for i, arr in enumerate(data[:n_preview]):
        print(f"  data[{i}].shape = {arr.shape}, dtype={arr.dtype}")
    print(f"'node_list' (数值标签) 长度 = {len(nl)}, 示例 = {nl[:n_preview]}")
    print("=== End ===\n")


if __name__ == '__main__':
    mat_folder   = r"D:\deeplearning\CSI2Q\combined_wisig_IQ_4days_300"
    out_pkl_path = r"D:\deeplearning\CSI2Q\combined_wisig_IQ_4days_300.pkl"

    pkl = convert_folder_to_pkl(mat_folder, out_pkl_path, sample_num=1000)
    inspect_pkl_dict(pkl)
    # 验证加载
    with open(out_pkl_path,'rb') as f:
        loaded = pickle.load(f)
    inspect_pkl_dict(loaded)
