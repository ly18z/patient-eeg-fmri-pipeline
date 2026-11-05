# MRI pipeline

任务fMRI主线为DICOM → NIfTI/JSON → BIDS → fMRIPrep → 质控 → GLM → 原生T1叠加及BOLD曲线。脑区标注、静息态、DTI/CST和病灶候选是可选模块。

以下命令都从仓库根目录执行。使用Python 3.12主机环境和固定的Linux/amd64 fMRIPrep容器，不依赖某个用户的安装路径。

## 1. 安装基础依赖

先按[仓库说明](../README.md#1-安装python环境)创建并激活环境：

```bash
python -m pip install -r fMRI/requirements.txt
```

另行安装：

1. **Docker**：Linux安装Docker Engine；Windows/macOS安装Docker Desktop并使用Linux容器。`docker version`必须能连接Server。
2. **dcm2niix**：从[官方发布页](https://github.com/rordenlab/dcm2niix/releases)安装适合平台的版本，放入PATH，使`dcm2niix --version`可执行。已有NIfTI时可跳过。
3. **FreeSurfer许可证**：按[官方注册页面](https://surfer.nmr.mgh.harvard.edu/registration.html)申请`license.txt`，保存在本地，不提交仓库。

提前下载固定容器：

```bash
docker pull --platform linux/amd64 nipreps/fmriprep:25.2.7
```

启动入口不自动拉取镜像。默认使用8 CPU、13 GB容器内存，Docker资源分配需满足；资源限制分别在`preprocess.py`的`PARAMETERS`及`run.py`的容器参数处调整。ARM主机需具备amd64仿真，本流程未验证其性能。

fMRIPrep的安装和输入约定见[25.2.7官方文档](https://fmriprep.org/en/25.2.7/installation.html)。主机无需另外安装FSL/ANTs；相关计算在容器内执行。DWI模块的额外eddy和DIPY资源在第4步准备。

## 2. 准备本次目录和配置

复制`fMRI/pipeline_config.example.json`到`cases/mri_case/pipeline_config.json`：

```text
cases/mri_case/
  pipeline_config.json
  dicom/                     原始DICOM，保持保留
  license.txt
  converted/                 转换产生NIfTI、JSON和DWI梯度
  bids/                      后续整理产生
  derivatives/               处理及模型结果
  pipeline_run01/            运行进度和逐步日志
```

`root: "."`表示配置文件所在目录。填写下表：

| 字段 | 内容 |
|---|---|
| `root` | 本次MRI工程目录，相对配置文件位置 |
| `dicom` | 原始DICOM目录，相对配置文件位置 |
| `run_directory` | 本次进度/日志目录，相对配置文件位置 |
| `subject` | BIDS匿名编号，例如`sub-01` |
| `modules` | 要执行的模块，见第3步 |
| `inputs` | T1、FLAIR、DWI、静息态及每个任务run的序列编号和NIfTI路径 |
| `task` | 任务名称、条件名、说明、时序来源、起点和持续时间 |
| `preprocessing.templateflow` | 模板缓存，相对`root` |
| `preprocessing.license` | 许可证文件，相对`root` |
| `preprocessing.download_templates` | 首次运行允许下载缺少的模板；缓存完整后可设为`false`离线运行 |
| `preprocessing.ignore` | 本次明确省略的步骤；默认空列表，不统一忽略切片时间/场图 |
| `manual_qc` | 是否在质控节点暂停，默认`true` |
| `tools_directory` | 可选，公共资源目录；省略时使用仓库根目录的`tools/` |
| `font_file` | 可选，中文字体文件，相对配置文件位置；映射进容器供绘图使用 |
| `lesion_python` | 可选，病灶专用环境解释器路径，见第4步 |
| `docker`、`dcm2niix` | 可选，程序的完整路径；省略时从PATH调用 |
| `analysis_image` | 可选，分析容器；默认固定fMRIPrep 25.2.7，改动后需验证依赖兼容 |
| `python` | 可选，主机端其他解释器；通常使用启动流程的当前Python |

`inputs.*.file`及任务事件表相对`root`。dcm2niix默认命名为`converted/series-序列号.nii.gz`，必须按真实扫描修改示例编号，不按示例猜序列类型。DWI需同名`.bval`、`.bvec`、`.json`；其他NIfTI需同名JSON。未采集的输入条目可以删除。

TR与体积数从影像读取，不填写固定帧数。默认任务为一个显式条件相对隐式静息，`task.condition`需与事件表条件名一致。可以使用：

- `task.onsets_seconds`和`duration_seconds`给所有run提供已确认区块时序；单个`inputs.task_runs`条目可覆盖它们。
- 在run条目填`"events": "events/run-01.tsv"`，读取含`onset`、`duration`、`trial_type`的TSV。

时间单位秒，以第一个保存体积为零点。填写真实来源，不根据激活强弱移动起点，也不把示例时序当采集事实。

## 3. 选择模块

| 模块 | 执行内容与条件 |
|---|---|
| `convert` | dcm2niix转换；已有NIfTI可省略 |
| `bids` | 根据`inputs`整理BIDS并写任务事件；已有BIDS可省略 |
| `preprocess` | fMRIPrep → 数值质控 → 摘要 → 可选人工复核；已有对应输出可省略 |
| `glm` | 逐run GLM、固定效应合并、统计图和报告 |
| `native` | 安排GLM，再映射回患者T1并生成查看页 |
| `curves` | 安排GLM，再恢复模型BOLD曲线并插入报告 |
| `regions` | 安排GLM、模板配准、HO及AAL3标注；需要图谱资源 |
| `rest` | 模板配准、六个运动/感觉ROI、联合去噪及连接；需要独立静息BOLD和图谱 |
| `dti` | 图谱/ROI、梯度核查、MP-PCA、eddy、DTI指标和候选CST；需要DWI、原始DICOM及额外资源 |
| `lesion` | 图谱、FLAIR准备、两套T1模型与TrUE-Net候选；需要T1、FLAIR、模型及专用环境 |

共享依赖只执行一次。`rest`与`dti`同时启用时还生成合并报告；单独启用时输出各自图表、影像与摘要。`native`/`curves`/`regions`会安排GLM计算，复用已完成分析应使用本次运行记录或单步入口。

## 4. 准备可选资源

只运行基本GLM、原生叠加和曲线时，不需要图谱或病灶模型。先预览所选模块需要的资源：

```bash
python fMRI/prepare_resources.py --config cases/mri_case/pipeline_config.json
```

确认后显式启动下载：

```bash
python fMRI/prepare_resources.py --config cases/mri_case/pipeline_config.json --download
```

此命令根据`modules`选择资源；也可加`--groups atlases`、`--groups dti`或`--groups models`单独准备。模型包可能较大，下载是一次性准备步骤；分析本身不触发这些下载。脚本尊重标准网络/代理环境变量，不写入代理地址。

准备内容：

| 资源 | 位置 |
|---|---|
| HO皮层下、JHU及MNI6模板 | `root/work/lesion_atlas/` |
| HO皮层图谱 | `root/analysis_rest_dti/work/atlas/` |
| AAL3及Colin27模板（选择`regions`时） | `root/analysis_glm_regions/atlas/AAL3_source/` |
| 固定容器的Linux CPython 3.12 DIPY及附加包 | `tools_directory/python/dti_linux_packages/` |
| FSL eddy_cpu | `tools_directory/fsl_eddy/bin/eddy_cpu` |
| SynthStroke源码、两套权重 | `tools_directory/lesion_models/synthstroke*` |
| TrUE-Net源码和三平面权重 | `tools_directory/lesion_models/truenet/` |

图谱使用指定来源与标签编码，不能随意以同名但编号不同的图谱替换。资源准备不会覆盖已有文件；切换资源版本使用新的资源目录。资源来源与版本见[THIRD_PARTY.md](../THIRD_PARTY.md)。

### 病灶专用环境

仅在选择`lesion`时创建。不要安装到EEG环境：

```bash
python -m venv .venv-lesion
# Windows PowerShell
.venv-lesion/Scripts/python.exe -m pip install -r fMRI/requirements-lesion.txt
# Linux / macOS
.venv-lesion/bin/python -m pip install -r fMRI/requirements-lesion.txt
```

运行该环境的`python -c "import sys; print(sys.executable)"`获取解释器完整路径，填入`lesion_python`。模型推理在主机端运行；T1/FLAIR配准、图谱和报告仍走容器。

### 中文字体

中文图表建议准备Noto Sans CJK SC等支持中文的字体文件，并设置`font_file`。例如将字体放在仓库的`tools/`，配置位于`cases/mri_case/`时可以填`"font_file": "../../tools/NotoSansCJKsc-Regular.otf"`。字体可从[Noto CJK官方仓库](https://github.com/notofonts/noto-cjk)取得。不指定时使用可用字体，部分容器可能只能显示英文字符。

## 5. 一行启动与人工复核

```bash
python fMRI/run_pipeline.py --config cases/mri_case/pipeline_config.json --plan
python fMRI/run_pipeline.py --config cases/mri_case/pipeline_config.json
```

`download_templates=true`时，fMRIPrep容器可联网并写入模板缓存；原始BIDS仍只读。设为`false`时，模板缓存只读且容器不联网，缺少模板会停止。其他分析容器均不联网。镜像需提前拉取。

人工节点会列出fMRIPrep报告或质控目录。检查脑提取、覆盖、头动及局部配准后按回车继续；DTI另检查候选束的走行和目标约束；病灶模块复核T1/FLAIR上的模型候选。`q`退出、重新运行和`--continue`的行为见[运行约定](../README.md#4-人工复核和继续)。

## 6. 查找输出

| 位置 | 内容 |
|---|---|
| `root/converted`、`root/bids` | 转换和BIDS输入 |
| `root/derivatives/fmriprep`、`root/qc/preprocessing` | 预处理、confounds及质控 |
| `root/derivatives/glm`、`root/GLM分析报告.html` | GLM图、设计矩阵、事件、统计和报告 |
| `root/GLM_患者原生影像.html` | 原生T1叠加查看 |
| `root/derivatives/glm_curves` | 区域模型曲线及数据 |
| `root/analysis_glm_regions` | 脑区峰值/整簇标注和交互报告 |
| `root/analysis_rest_dti/rest` | 静息态曲线、保留索引、矩阵和种子图 |
| `root/analysis_rest_dti/dti` | FA/MD/AD/RD、配准、区域指标及候选束 |
| `root/derivatives/lesion` | 候选概率、掩膜和定位统计 |
| `root/病灶候选定位报告.html` | 候选报告及交互影像入口 |
| `run_directory/logs` | 逐步运行日志 |

本仓库不包含旧病例的ASL/SNAP、设备束图截图或多序列拼接专用页面。通用影像查看可单独运行：

```bash
python fMRI/view_images.py --background cases/mri_case/T1w.nii.gz --space "patient native T1w" --output cases/mri_case/viewer
```

可追加`--overlay`或`--lesion`，但必须已与背景在同一网格/空间；查看器不自动配准，也不执行病灶分割。

## 方法摘要

- fMRIPrep输出T1w及MNI152NLin2009cAsym 2 mm，不运行FreeSurfer表面重建；切片时间和场图处理由实际输入及`ignore`决定。
- GLM：5 mm平滑，SPM HRF＋时间导数，24项头动、5项combined aCompCor、已有余弦漂移、FD>0.5 mm或stdDVARS>1.5异常帧回归，AR(1)，百分比尺度；对比仅标准HRF系数。run间逆方差固定效应；双侧体素FDR q<0.05，另保存Bonferroni结果。连通簇只是分组，不称簇水平显著。
- 静息态：默认0.01–0.1 Hz；混杂、带外DCT和异常帧联合回归；排除帧不进入连接估计。保存原始时间和保留索引，四维结果中的零帧不是观测。输出Pearson r、Fisher变换及GSR/滞后描述性对照，不标体素显著性。
- DTI：MP-PCA → eddy → 旋转梯度 → WLS张量。当前没有topup/磁敏感畸变校正。候选CST采用单张量方向、解剖约束、0.5 mm步长，FA停止0.15与0.10敏感性；40–250 mm长度规则只是当前实现，不能直接当儿童等数据的通用标准。
- 病灶：固定SynthStroke baseline/synth与TrUE-Net MWSC权重，输出候选。白质高信号不自动等于卒中；图谱标签、FA、流线数和模型概率不单独构成临床诊断。

## 常见问题

- **Docker无法连接**：启动Docker服务或Desktop，确认使用Linux容器，先让`docker version`成功。
- **找不到dcm2niix**：加入PATH或填`dcm2niix`完整路径；已有转换数据则省略`convert`。
- **缺少模板**：首次使用设`download_templates=true`并确保容器联网；离线前完整准备缓存。
- **找不到DIPY/eddy/图谱/模型**：运行第4步；若资源在外部，填写`tools_directory`。不要把Windows Python包放进Linux容器包目录。
- **资源下载遇到网络或TLS错误**：按日志中的URL和目标位置下载完整文件，再重跑准备命令。AAL3原站在部分网络不可直连，可从官方页面下载同一发布包，放到`tools_directory/downloads/AAL3v2.tar.gz`；脚本会复用缓存。不要关闭TLS验证或把其他版本改名代替。
- **源DICOM没有标准扩散标签**：当前梯度核查依赖`DiffusionBValue`、`DiffusionGradientOrientation`；厂商私有标签数据需先适配该检查，不能假装已验证。
- **配置改变或已有GLM结果拒绝覆盖**：使用新工程/输出版本，或按日志处理部分运行结果；不要删除原始输入。
