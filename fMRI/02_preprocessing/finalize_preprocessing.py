"""Write a preprocessing summary from the existing QC results."""
from case_config import CASE
import json
root = CASE.root
summary=json.loads((root/'qc/preprocessing/qc_summary.json').read_text(encoding='utf-8'))
rows=[]
for run in summary['runs']:
    rows.append(f"| {run['run']} | {run['source_volumes']} | {run['mean_FD_mm']:.4f} | {run['max_FD_mm']:.4f} | {run['frames_FD_gt_0_5']} |")
report="""# 预处理结果说明

| Run | 体积数 | 平均FD（mm） | 最大FD（mm） | FD>0.5mm帧数 |
|---|---:|---:|---:|---:|
"""+'\n'.join(rows)+"""

处理后影像与混杂表位于derivatives/fmriprep，质控图和完整指标位于qc/preprocessing。
请结合fMRIPrep报告检查脑提取、组织分割、空间覆盖及配准，再进行下游分析。
"""
(root/'预处理结果说明.md').write_text(report,encoding='utf-8')
print(root/'预处理结果说明.md')
