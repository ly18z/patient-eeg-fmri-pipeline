"""Insert the saved BOLD curve section into the GLM report."""
from case_config import CASE
import re
root = CASE.root
report=root/'GLM分析报告.html';source=report.read_text(encoding='utf-8')
section=(root/'derivatives/glm_curves/report_section.html').read_text(encoding='utf-8')
if '<section id="bold-curves">' in source:
    source=re.sub(r'<section id="bold-curves">.*?</section>',lambda _:section,source,flags=re.S)
else:
    source=source.replace('<section><h2>合并结果</h2>',section+'\n<section><h2>合并结果</h2>')
report.write_text(source,encoding='utf-8')
print(report)
