# GitHub 安全配置说明

本文档说明项目的安全配置和敏感信息处理方式，确保代码安全地上传到 GitHub。

## 概述

本项目包含多个模块：
- `adapt_QAOA_Tianyan` - 天衍平台 QAOA 实现（需要登录密钥）
- `classical_algorithms` - 经典图着色算法库
- `standard_and_adapt_QAOA` - MindSpore QAOA 实现

**平台信息**:
- **天衍平台**: 中电信量子计算平台，官网: https://qc.zdxlz.com/home?lang=zh
- **MindSpore**: 华为昇思量子框架，官网: https://www.mindspore.cn/

## 敏感信息处理

### 1. 天衍平台登录密钥

**状态**: ✅ 已处理

所有天衍平台相关的代码已移除硬编码的登录密钥，改为从配置文件或环境变量读取。

**配置方式**:

#### 方式 1: 配置文件

```bash
cd HAdaQAOA/HadaQAOA/adapt_QAOA_Tianyan/QAOA
cp config_template.py config.py
```

编辑 `config.py`:
```python
TIANYAN_CONFIG = {
    "login_key": "your_login_key_here",  # 填入实际的登录密钥
    "lab_id": None,
    "machine": "tianyan_sa",
}
```

#### 方式 2: 环境变量

```bash
# Linux/Mac
export TIANYAN_LOGIN_KEY="your_login_key_here"

# Windows PowerShell
$env:TIANYAN_LOGIN_KEY="your_login_key_here"
```

### 2. Git 忽略规则

项目使用 `.gitignore` 文件忽略以下内容：

#### 敏感配置文件
- `*.key`, `*.pem`, `credentials.json`, `.env`
- `config.py` (天衍平台配置文件)

#### 数据文件
- 输出数据文件: `*.csv`, `*.log`, `*.pkl`
- 实验结果: `coloring_results/`, `csvs/`, `logs/`
- 可视化输出: `*.png`, `*.jpg`, `*.pdf`
- 可视化目录: `*visualizations*/`

**注意**: `Data/` 目录下的图数据集（.col, .pkl）可以上传，仅忽略运行生成的结果文件。

#### 可视化输出
- `*.png`, `*.jpg`, `*.pdf`
- `*visualizations*/`
- `coloring_results/`
- `csvs/`
- `logs/`

#### Python 缓存和 IDE
- `__pycache__/`, `*.pyc`
- `.vscode/`, `.idea/`
- `.ipynb_checkpoints/`

#### 操作系统文件
- `.DS_Store`, `Thumbs.db`

## 各模块安全状态

### adapt_QAOA_Tianyan

**状态**: ✅ 安全

- 登录密钥已移除硬编码
- 配置文件被 `.gitignore` 忽略
- 所有输出文件被忽略

**需要用户操作**:
1. 创建 `config.py` 文件（复制自 `config_template.py`）
2. 填写实际的登录密钥
3. `config.py` 已被 `.gitignore` 忽略，不会上传

### classical_algorithms

**状态**: ✅ 安全

- 无敏感信息
- 无需配置密钥
- 所有输出文件被忽略

### standard_and_adapt_QAOA

**状态**: ✅ 安全

- 无敏感信息
- 无需配置密钥
- 所有输出文件被忽略
- Jupyter Notebook 已被忽略（含本地路径）

## 上传前检查清单

### ✅ 必须做

1. 确认 `config.py` 已被 `.gitignore` 忽略
2. 检查是否有其他包含敏感信息的文件
3. 验证输出文件和可视化目录被忽略

### ❌ 不要上传

- ❌ `config.py`（包含敏感密钥）
- ❌ 运行生成的 `*.csv`, `*.log`, `*.pkl` 等结果文件
- ❌ `*visualizations*/` 等图片目录
- ❌ `coloring_results/`, `csvs/`, `logs/` 等结果目录
- ❌ `__pycache__/` 等缓存目录
- ❌ `.env` 等环境变量文件
- ✅ `Data/` 目录下的图数据集（可以上传）

## 验证配置

### 检查将要上传的文件

```bash
# 查看将被 git 跟踪的文件
git add .
git status

# 确认敏感文件不在跟踪列表中
```

### 验证配置文件

```bash
cd HAdaQAOA/HadaQAOA/adapt_QAOA_Tianyan/QAOA
python -c "from config import TIANYAN_CONFIG; print('配置加载成功')"
```

## 可视化配置

### 禁用可视化输出

如需禁用可视化输出，修改 `VISUALIZATION_CONFIG`:

```python
VISUALIZATION_CONFIG = {
    "save_plots": False,  # 不保存图片
    "show_plots": False,  # 不显示图片
    "plot_format": "pdf",  # 图片格式：png, svg, pdf
    "dpi": 300,  # 图片分辨率
}
```

### 输出目录说明

代码中生成的可视化图片保存在以下目录：
- `large_graph_visualizations/` (adapt_QAOA_Tianyan)
- `large_subgraph_visualizations/` (adapt_QAOA_Tianyan)
- `coloring_results/` (classical_algorithms)
- `graph_visualizations/` (standard_and_adapt_QAOA)
- `subgraph_visualizations/` (standard_and_adapt_QAOA)
- `experiment_visualizations/` (standard_and_adapt_QAOA)

这些目录已被 `.gitignore` 忽略，不会上传。

## 其他安全建议

### 1. 启用 GitHub Secret Scanning

GitHub 提供自动密钥扫描功能，建议在仓库设置中启用。

### 2. 定期审计依赖

使用工具检查依赖包的安全漏洞：
```bash
pip install safety
safety check
```

### 3. 使用预提交钩子

使用 pre-commit 框架防止意外提交敏感文件：
```bash
pip install pre-commit
pre-commit install
```

## 总结

通过以上配置，项目可以安全地上传到 GitHub：

- ✅ 敏感密钥已移除硬编码
- ✅ 配置文件被 `.gitignore` 忽略
- ✅ 可视化输出目录被忽略
- ✅ 提供了清晰的配置模板

**注意**: 上传后，其他用户需要：
1. 创建自己的 `config.py` 文件才能运行 adapt_QAOA_Tianyan 代码
2. Data/ 目录已包含部分图数据集（可直接使用），也可添加自己的数据集
3. 安装所有依赖包（MindSpore 或 cqlib）
