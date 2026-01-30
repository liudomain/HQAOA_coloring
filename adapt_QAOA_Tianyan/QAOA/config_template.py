"""
天衍平台配置文件模板
使用说明：
1. 复制此文件为 config.py
2. 填写实际的 LOGIN_KEY 和 LAB_ID
3. config.py 已被 .gitignore 忽略，不会上传到 GitHub
"""

# 天衍平台配置
TIANYAN_CONFIG = {
    "login_key": "YOUR_LOGIN_KEY_HERE",  # 请替换为实际的登录密钥
    "lab_id": None,  # 可选：如果已有实验室ID，请填写；否则留None自动创建
    "machine": "tianyan_sa",  # 使用的量子计算机：tianyan_sa (200 qubits)
}

# 数据集配置
DATA_CONFIG = {
    "data_dir": None,  # 数据集目录，会自动计算
    "large_datasets": [
        "cora.col",
        "citeseer.col"
        # "pubmed.col",
    ],
}

# 算法参数
ALGORITHM_PARAMS = {
    "max_k": 20,  # 最大颜色数
    "p": 1,  # QAOA 层数
    "num_shots": 1000,  # 量子测量次数
    "max_iter": 20,  # 最大迭代次数
    "early_stop_threshold": 5,  # 早停阈值
    "Q": 20,  # 贪心着色参数
}

# 训练参数
TRAINING_CONFIG = {
    "train_params": True,  # 是否训练参数
    "train_max_iter": 50,  # 参数训练最大迭代次数
    "train_lr": 0.01,  # 学习率
}

# 可视化配置
VISUALIZATION_CONFIG = {
    "save_plots": True,  # 是否保存图片
    "show_plots": False,  # 是否显示图片
    "plot_format": "png",  # 图片格式：png, svg, pdf
    "dpi": 300,  # 图片分辨率
}
