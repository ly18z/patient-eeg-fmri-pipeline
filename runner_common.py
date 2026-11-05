"""Small execution helper: logs, manual checkpoints and completed-step resume."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import webbrowser


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def manual_review(step, allow_continue, open_pages):
    print('\n人工处理：'+step['message'], flush=True)
    for page in step['pages']:
        print('  '+page, flush=True)
    if not allow_continue:
        if not sys.stdin.isatty():
            return False
        if open_pages:
            for page in step['pages']:
                if Path(page).is_file() and Path(page).suffix == '.html':
                    webbrowser.open(Path(page).as_uri())
                    break
        answer = input('完成后按回车继续，输入 q 保存进度并退出：').strip().lower()
        if answer == 'q':
            return False
    if step.get('required'):
        file = Path(step['required'])
        if not file.is_file():
            print('尚未找到审核文件，请导出到上述位置后继续。', flush=True)
            return False
        review = json.loads(file.read_text(encoding='utf-8-sig'))
        if Path(review['fit']).resolve() != Path(step['fit']).resolve():
            raise ValueError('ICA审核JSON不属于本次拟合，不能复用其他分解的IC编号。')
    return True


def run_command(command, log_file):
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    with log_file.open('w', encoding='utf-8') as log:
        log.write(subprocess.list2cmdline(command)+'\n')
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', env=env) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def launch(make_plan, description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--plan', action='store_true', help='Only list ordered steps and manual checkpoints')
    p.add_argument('--continue', dest='continue_review', action='store_true', help='Confirm the previously paused checkpoint')
    a = p.parse_args()
    config = json.loads(a.config.read_text(encoding='utf-8-sig'))
    run, steps = make_plan(config, a.config.resolve())
    if not steps:
        p.error('没有启用的处理步骤。')
    if a.plan:
        for i, step in enumerate(steps, 1):
            print(f'{i:02d}. {step["name"]}'+(' [人工暂停]' if 'message' in step else ''))
        return
    run.mkdir(parents=True, exist_ok=True)
    state_file = run/'pipeline_state.json'
    state = json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else dict(configuration=config, completed=[])
    if state['configuration'] != config:
        p.error('配置已变更。请使用新的 run_directory 开始新版本，避免混用已完成结果。')
    confirmed = state.get('paused') if a.continue_review else None
    if a.continue_review and not confirmed:
        p.error('当前没有等待确认的人工节点。')
    logs = run/'logs'
    logs.mkdir(exist_ok=True)
    for i, step in enumerate(steps, 1):
        name = step['name']
        if name in state['completed']:
            print('跳过已完成：'+name, flush=True)
            continue
        print(f'\n[{i}/{len(steps)}] {name}', flush=True)
        if 'message' in step:
            state.update(paused=name, running=None)
            write_json(state_file, state)
            if not manual_review(step, name == confirmed, config.get('open_review_pages', True)):
                print('进度已保存。完成处理后重跑同一命令；非交互运行时加 --continue 确认当前人工节点。', flush=True)
                return
        else:
            state.update(running=name, paused=None)
            write_json(state_file, state)
            run_command(step['command'], logs/(name+'.log'))
        state['completed'].append(name)
        state.update(running=None, paused=None)
        write_json(state_file, state)
    print('\n所选流程已完成。运行记录与日志：'+str(run), flush=True)


def entry(make_plan, description):
    try:
        launch(make_plan, description)
    except KeyboardInterrupt:
        print("\n已停止，进度保存在运行目录中。", file=sys.stderr)
        sys.exit(130)
    except (subprocess.CalledProcessError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
