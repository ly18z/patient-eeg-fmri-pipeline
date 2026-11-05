# EEG / MRI Analysis Pipelines

两条独立的研究分析流程，各自填写一份JSON，再用一条命令执行。支持人工复核暂停、逐步日志和跳过已完成步骤。

| 流程 | 包含的计算 | 入门指南 |
|---|---|---|
| EEG | 连续预处理 → FastICA → 自动/人工成分处理 → 配对PSD/ERD → Morlet时频与头皮拓扑 | [EEG/README.md](EEG/README.md) |
| MRI | DICOM转换、BIDS整理、fMRIPrep、任务GLM；可选脑区标注、静息态连接、DTI/CST和病灶候选 | [fMRI/README.md](fMRI/README.md) |

## 1. 安装Python环境

建议使用Python 3.12。进入下载或克隆后的仓库根目录，创建独立环境：

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

然后安装需要的流程依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -r EEG/requirements.txt
# 使用MRI时另装主机端依赖
python -m pip install -r fMRI/requirements.txt
```

只使用一条流程时，不需要安装另一条流程的依赖。MRI的计算容器、转换器及可选模型准备见MRI指南。病灶推理使用独立环境，避免与EEG的NumPy版本冲突。

## 2. 创建本次配置

在仓库内或外部建立病例目录，例如：

```text
cases/
  eeg_case/
    pipeline_config.json   ← 复制EEG/pipeline_config.example.json
    data/record.bdf
  mri_case/
    pipeline_config.json   ← 复制fMRI/pipeline_config.example.json
    dicom/
    license.txt
```

填写复制后的配置。JSON路径统一可用`/`；相对路径以配置文件位置为基准，MRI输入文件等另相对`root`。默认使用启动命令所用的Python；只有需要指定另一个解释器时才添加`python`字段。

## 3. 预览并启动

从仓库根目录执行：

```bash
# EEG
python EEG/run_pipeline.py --config cases/eeg_case/pipeline_config.json --plan
python EEG/run_pipeline.py --config cases/eeg_case/pipeline_config.json

# MRI：先完成MRI指南中的Docker、模板及模块资源准备
python fMRI/run_pipeline.py --config cases/mri_case/pipeline_config.json --plan
python fMRI/run_pipeline.py --config cases/mri_case/pipeline_config.json
```

`--plan`只列出步骤，不读取影像、不启动计算、不下载资源。

## 4. 人工复核和继续

在普通交互终端中，流程会在人工节点列出查看位置，检查后按回车继续；输入`q`保存进度并退出。重新运行原命令，会跳过已完成步骤。

在没有交互输入的环境中，流程停在人工节点并退出。完成当前复核后，原命令加`--continue`确认该节点；后面的人工节点仍会暂停。EEG人工模式还要求提供本次ICA分解导出的审核JSON。

`run_directory`中保存：

- `pipeline_state.json`：配置、已完成步骤和当前节点。
- `logs/`：每步的命令及输出。

修改输入、分析参数或ICA决定后，应使用新的运行版本与输出目录，不能沿用旧完成记录。报错步骤不会标为完成；若留下部分输出，按日志修复并清理该步骤的部分目录后重试。手动删除已完成输出，不会使运行记录自动失效。

## 仓库结构

```text
EEG/                  EEG脚本、配置示例与方法说明
fMRI/                 MRI入口及按阶段组织的计算脚本
runner_common.py      日志、暂停和继续的共享实现
.gitignore            本地环境、数据、权重及结果的排除规则
```

运行时生成的`tools/`、病例目录和实际配置不属于代码仓库。发布时仅提交代码、模板、依赖清单和示例配置；原始记录、影像、审核名单、结果、许可证及模型权重留在本地。第三方模型和图谱按原作者来源获取，见各流程指南。

本项目提供研究分析和可视化，自动伪迹判断、图谱标签、候选纤维束或病灶候选均需结合数据质控和人工复核。
