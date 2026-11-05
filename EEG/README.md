# EEG pipeline

从带事件注释的BDF、EDF或FIF记录，生成ICA清洗结果、配对PSD/ERD、Morlet时频图和头皮拓扑图。分类和ERP不包含在当前流程中。

## 1. 安装

按[仓库环境说明](../README.md#1-安装python环境)创建Python 3.12环境，在仓库根目录执行：

```bash
python -m pip install -r EEG/requirements.txt
```

依赖清单固定已验证的MNE、NumPy、SciPy等版本。安装方式可参考[MNE官方说明](https://mne.tools/stable/install/manual_install.html)。若升级依赖，应另存环境版本并重新检查结果。

## 2. 准备输入

- 文件应包含可识别的头皮EEG通道名称及事件注释。数据单位由MNE读取为伏。
- ICA眼电代理使用`Fp1`、`Fp2`，当前流程需要这两个通道；PSD默认显示C3/Cz/C4中实际存在的通道。
- `montage`填MNE模板名称或电极坐标文件。标准模板只是位置参考，不代表患者实测电极位置。
- 默认事件：`miSuccess`、`miFailed`均映射为任务尝试，`rest`、`initialRest`映射为静息。同状态标记间隔≤3秒合并为一簇。
- 先确认本次标记语义确实符合该映射；软件标签不等于真实运动起点或执行成功。事件名称不同，需要修改`psd.py`顶部的`EVENT_CONFIG`，未映射注释会报错。

原始记录不覆盖。处理保持连续时间轴，坏段通过注释和质量筛选排除，不删除后拼接。

## 3. 填写配置

复制`EEG/pipeline_config.example.json`到`cases/eeg_case/pipeline_config.json`。示例路径对应：

```text
cases/eeg_case/
  pipeline_config.json
  data/record.bdf
```

| 字段 | 如何填写 |
|---|---|
| `input` | BDF/EDF/FIF输入路径，相对配置文件所在目录 |
| `run_directory` | 本次输出与进度目录；新患者或新分析版本使用新目录 |
| `montage` | 例如`standard_1020`，或相对配置位置的坐标文件 |
| `manual_ica` | `true`进行可选形貌复核；`false`按固定阈值直接清洗 |
| `tfr_channels` | 时频图展示通道；省略则使用存在的C3/Cz/C4 |
| `open_review_pages` | 在交互终端中是否自动打开审核页 |
| `review_file` | 可选，自定义人工审核JSON位置；默认在`03_ica_review/ica_review.json` |
| `python` | 可选，其他Python解释器的完整路径；通常省略 |

滤波、ICA和功率计算的方法参数在脚本中固定；路径与人工复核选择集中在配置中。

## 4. 一行运行

```bash
python EEG/run_pipeline.py --config cases/eeg_case/pipeline_config.json
```

执行顺序：

```text
pre_ica → fit_ica → [show_ica / 人工暂停] → apply_ica
        → psd → erd → plot_tfr → plot_topography
```

人工模式生成`run_directory/03_ica_review/index.html`。页面已按自动规则预选，优先复核候选，必要时展开其他IC；不要求逐个重新判断全部成分。修改后点击“导出审核JSON”，保存为同目录`ica_review.json`，再回终端按回车。审核JSON只对应本次ICA分解。

预览、退出及继续方式见[共享运行约定](../README.md#4-人工复核和继续)。不需要额外的PSD配置JSON或`reviewed`开关。

## 5. 输出

| 目录 | 主要内容 |
|---|---|
| `01_preica` | 预处理FIF、通道选择和实际秩 |
| `02_ica` | FastICA模型、成分评分和源特征 |
| `03_ica_review` | 人工模式的诊断图、审核页面及导出的JSON |
| `04_clean` | 清洗FIF、实际排除名单和局部源污染标记 |
| `05_psd` | PSD/配对ERD图、全部有效通道数组、片段/试次接受表、事件及统计 |
| `06_morlet` | 逐试次Morlet功率、基线、频带值和接受表 |
| `07_tfr` | 时频图和展示数组 |
| `08_topography` | 频带ERD头皮拓扑及通道效应 |

PSD和Morlet各自筛选样本，有效试次数可能不同。只更改图的样式时可独立调用绘图脚本，不必重算小波。

## 方法参数

| 环节 | 当前实现 |
|---|---|
| 通道 | 按头信息选EEG，排除辅助/参考、已标坏道和整段精确平线；不固定导联数 |
| 工频 | 50 Hz FIR零相位陷波；过渡带3 Hz，长度auto，Hamming/firwin |
| 带通 | 连续数据1–45 Hz，四阶Butterworth原型，SOS前后向零相位 |
| 重采样/参考 | 限带后FFT至250 Hz；实际应用有效EEG平均参考并测秩 |
| ICA | FastICA，成分数取实测秩；seed=123，max_iter=2000，tol=1e-4，拟合decim=2 |
| 眼电排除 | 1–10 Hz源/额极代理最大绝对相关>0.4、额部/中央后部载荷RMS比>3、1–8/1–40 Hz功率比>0.45三项同时满足 |
| 肌电排除 | MNE频谱/空间联合评分S>0.512，且最高10%块贡献的20–40 Hz能量比例Q≤0.5；S>0.15仅标候选 |
| 局部源质量 | 保留IC逐0.5秒块的1–8和20–40 Hz log10均方功率；尺度max(1.4826×MAD,0.25)，任一abs(z)>6标记污染 |
| 通道质量 | 分析窗任一有效EEG通道峰峰值≥150 μV拒绝 |
| 任务/基线 | 任务为MI+[1,5)秒；基线为紧邻前序rest+[6,10)秒 |
| PSD | 0.5/1/2秒不重叠DPSS；主分析2秒；NW=2、3 taper、low_bias=True、adaptive=False、normalization=full，去均值、不补零 |
| Morlet | 6–35 Hz，1 Hz步进，5周期，zero_mean=True，FFT，全250 Hz计算 |
| 展示/频带 | MI相对[-5,5)秒；显示抽点20 ms；μ8–13、β13–30、低β13–20、高β20–30 Hz均含端点；色标±3 dB |

PSD先在每个配对试次内平均合格片段的线性功率，再求任务/静息比并转dB，最后对试次dB等权平均。频带效应先平均频带内线性PSD，再求比值；不直接平均频点dB作为频带效应。

Morlet要求完整显示窗、任务窗及独立静息基线通过质量筛选；卷积两侧使用真实数据支持。时频图采用`10log10(mean_trial(功率/本试次基线))`；拓扑采用逐试次频带dB的平均，两者口径明确区分。

PSD主分析使用100,000次、seed=123的双侧sign-flip；BH-FDR分别用于实际中央区通道的2–40 Hz频点和μ/β频带。少于两对时不计算p/q。这是记录内探索性统计，不能当作患者群体推断。小波图及拓扑为描述性展示。

## 单独运行某一步

```bash
python EEG/psd.py --clean cases/eeg_case/results/eeg_run01/04_clean --output cases/eeg_case/psd_new
python EEG/plot_tfr.py --morlet cases/eeg_case/results/eeg_run01/06_morlet --output cases/eeg_case/tfr_new
```

## 常见问题

- **提示未映射事件**：核对原始注释含义，再修改`EVENT_CONFIG`；不要仅为通过报错随意忽略标签。
- **缺少Fp1/Fp2或电极位置**：当前ICA评分假定有额极代理和可用montage，需要适配数据后再运行。
- **ICA不收敛**：脚本不保存接受模型。检查通道和数据质量后另开版本处理。
- **无可用配对或Morlet试次**：先看接受表中的具体拒绝原因，检查任务/静息时窗和数据质量；不以更强ERD为目的放宽规则。
- **输出目录已存在**：使用新的运行版本；中断留下的部分目录按日志清理后重试。
