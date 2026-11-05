# 第三方软件与资源

代码仓库不打包外部软件、模型、图谱、字体或患者数据。使用时从原作者来源取得，并遵守各自条款。

| 用途 | 来源及当前版本 |
|---|---|
| EEG | [MNE-Python](https://mne.tools/stable/index.html)，精确Python依赖见EEG/requirements.txt |
| 影像转换 | [dcm2niix](https://github.com/rordenlab/dcm2niix) |
| 结构/BOLD预处理及基础分析环境 | [fMRIPrep 25.2.7](https://fmriprep.org/en/25.2.7/installation.html) |
| 扩散拟合/追踪 | [DIPY 1.11.0](https://docs.dipy.org/1.11.0/) |
| 扩散校正 | [FSL eddy](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/eddy/index.html)，固定Linux包2401.2-h500b71b_4 |
| 图谱 | [FSL图谱说明](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html)；HO副本取自[FreeSurfer tutorial](https://www.freesurfer.net/pub/dist/freesurfer/tutorial_packages_centos6/OSX/fsl_501/data/atlases/) |
| 模板 | [TemplateFlow](https://www.templateflow.org/)：MNI152NLin6Asym、MNIColin27；fMRIPrep另使用其所需模板 |
| AAL3 | [GIN官方页面](https://www.gin.cnrs.fr/en/tools/aal/)，AAL3v2_for_SPM12包，实际1 mm标签文件仍名AAL3v1 |
| T1病灶模型 | [SynthStroke](https://github.com/liamchalcroft/SynthStroke)，源码提交f627b57b8f9291bfa870cb54cb21ddbeb4941d39 |
| baseline权重 | [synthstroke-baseline](https://huggingface.co/liamchalcroft/synthstroke-baseline)，revision b693a650026359705688fbce409219c4dbb5d6be |
| synth权重 | [synthstroke-synth](https://huggingface.co/liamchalcroft/synthstroke-synth)，revision ced72e683ed68d3d110b675714e1a07bac584948 |
| WMH候选 | [TrUE-Net](https://github.com/v-sundaresan/truenet)，源码d9e3118029e8be93c9930574751efd9e78533cf0；[FSL模型包](https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/noarch/fsl-truenet_data-2406.0-1.tar.bz2)中MWSC T1+FLAIR三平面权重 |
| 中文字体 | [Noto CJK](https://github.com/notofonts/noto-cjk)，可选 |

资源准备入口是`fMRI/prepare_resources.py`，只有显式加`--download`才会开始下载，不会上传患者文件。DIPY额外轮子固定针对当前容器的Linux/amd64、CPython 3.12；变更基础镜像需要重新验证二进制兼容性。
