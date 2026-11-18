# RA-predict

根据关节超声图像、ROI掩码及临床指标执行疗效预测的skill。可放进agent skills目录，仅包含预测需要的代码、配置、输入模板和安装/启动入口。模型从[百度网盘](https://pan.baidu.com/s/5fw4WoWDQetc2hS4LvNa7zQ?#list/path=%2FRA-treatment-response-prediction%2Fpytorch_deployment)下载，直接使用网盘中已命名的fold_1至fold_5文件；下载后校验SHA256，无需重命名。

## 安装

支持Linux x86_64、Python3.9。安装支持CUDA 12.1的PyTorch，推理默认优先使用NVIDIA GPU；CUDA不可用时自动使用CPU，无GPU也可运行。安装器会安装CUDA用户态依赖，不修改系统驱动。

以Claude Code个人skill为例（私有仓库需访问权限）：

```bash
git clone git@github.com:sunyan1024/RA-treatment-response-prediction.git ~/.claude/skills/RA-predict
cd ~/.claude/skills/RA-predict
bash scripts/install.sh --python /path/to/python3.9
```

如果没有Python3.9，可先用已有conda创建 `conda create -n RA-predict python=3.9 pip -y` 并激活，再执行 `bash scripts/install.sh --python "$(command -v python)"`。安装器只在当前skill创建.venv和安装依赖；不编译/打包模型，不创建分发运行时。首次安装需联网，环境和模型就绪后推理可离线运行。

其他支持SKILL.md的本地agent：将整个仓库放在其skills搜索目录，根目录SKILL.md为入口。agent必须能访问这些文件并执行本地命令。

## 模型

将以下5个文件放入目录 `pth/`：

```text
fold_1.pth
fold_2.pth
fold_3.pth
fold_4.pth
fold_5.pth
```

下载后执行 `python scripts/verify_weights.py`。启动预测前也会自动检查5个模型的SHA256；没有旧名称兼容或自动改名逻辑。

## 使用

复制并填写患者模板，影像及掩码路径相对于该JSON；全部临床值必须填写，不能保留null：

```bash
cp references/patient.example.json /path/to/patient.json
bash scripts/run.sh --patient /path/to/patient.json --out /path/to/report.md --json-out /path/to/result.json
```

默认 `--device auto`：CUDA可用时使用GPU，否则使用CPU。可加 `--device cpu` 强制CPU，或 `--device cuda` 强制GPU（不可用时报错）。设备选择会输出到终端；运行中显存不足等错误会直接报错。

启动本地agent后可输入：`/RA-predict 使用 /absolute/patient.json 预测并生成报告和JSON结果`。无需激活环境或手动配置库路径，run.sh使用本skill的.venv和本地模型。迁移skill时需要重新安装.venv；模型文件可直接复制并重新校验。

临床连续值使用原始测量值，单位和性别编码必须与训练约定一致；定性抗体支持P/N/O或1/0/-1。超声图像需为8位单通道灰度PNG，ROI为同尺寸二值掩码。每患者1–50张，关节类型及图像/掩码按原始文件名字典序同步排列。模型输出和阈值提示不是治疗建议或校准置信度。

本仓库仅保留使用者必需文件，不包含训练、导出、构建、基准对比、历史包或开发环境。第三方说明见THIRD_PARTY.md。
