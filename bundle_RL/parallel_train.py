import subprocess
import os
import sys
import threading
from pathlib import Path
from queue import Queue, Empty

def get_project_root():
    return Path(__file__).parent.absolute()

def enqueue_output(out, queue):
    for line in iter(out.readline, ''):
        queue.put(line)
    out.close()

def run_experiment(experiment_name, ent_coef, K, steps_per_config_per_round, rounds, 
                   learning_rate, clip_range, clip_range_decay):
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    code = f'''import sys
sys.path.insert(0, "{get_project_root()}")
from bundle_RL.main_train import main
main(
    experiment_name="{experiment_name}",
    ent_coef={ent_coef},
    K={K},
    steps_per_config_per_round={steps_per_config_per_round},
    rounds={rounds},
    learning_rate={learning_rate},
    clip_range={clip_range},
    clip_range_decay={clip_range_decay}
)'''
    
    cmd = [sys.executable, "-c", code]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=get_project_root(),
        env=env
    )
    
    return process

def run_parallel_experiments(experiments):
    processes = []
    output_queues = []
    
    print(f"开始并行启动 {len(experiments)} 个实验...")
    print("="*80)
    
    for exp in experiments:
        print(f"启动实验: {exp['experiment_name']}")
        print(f"  参数: ent_coef={exp['ent_coef']}, lr={exp['learning_rate']}, clip={exp['clip_range']}")
        
        process = run_experiment(
            experiment_name=exp['experiment_name'],
            ent_coef=exp['ent_coef'],
            K=exp['K'],
            steps_per_config_per_round=exp['steps_per_config_per_round'],
            rounds=exp['rounds'],
            learning_rate=exp['learning_rate'],
            clip_range=exp['clip_range'],
            clip_range_decay=exp['clip_range_decay']
        )
        processes.append((exp['experiment_name'], process))
        
        stdout_queue = Queue()
        stderr_queue = Queue()
        stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, stdout_queue))
        stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, stderr_queue))
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        output_queues.append((stdout_queue, stderr_queue, stdout_thread, stderr_thread))
    
    print("="*80)
    print("所有实验已启动！")
    print("等待所有实验完成...")
    print("="*80)
    
    results = []
    while processes:
        for i, (exp_name, process) in enumerate(processes):
            retcode = process.poll()
            if retcode is not None:
                stdout_queue, stderr_queue, _, _ = output_queues[i]
                
                stdout_lines = []
                while True:
                    try:
                        line = stdout_queue.get_nowait()
                        stdout_lines.append(line)
                    except Empty:
                        break
                
                stderr_lines = []
                while True:
                    try:
                        line = stderr_queue.get_nowait()
                        stderr_lines.append(line)
                    except Empty:
                        break
                
                stdout_text = ''.join(stdout_lines)
                stderr_text = ''.join(stderr_lines)
                
                results.append({
                    'experiment_name': exp_name,
                    'return_code': retcode,
                    'stdout': stdout_text,
                    'stderr': stderr_text
                })
                
                if retcode == 0:
                    print(f"[完成] 实验 {exp_name}")
                else:
                    print(f"[失败] 实验 {exp_name}，错误码: {retcode}")
                    if stderr_text:
                        print(f"错误信息:\n{stderr_text[:500]}...")
                
                processes.pop(i)
                output_queues.pop(i)
                break
        else:
            import time
            time.sleep(1)
    
    print("="*80)
    print("所有实验执行完毕！")
    return results

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    experiments = [
        # {
        #     'experiment_name': 'exp07',
        #     'ent_coef': 0.01,
        #     'K': 10,
        #     'steps_per_config_per_round': 2000,
        #     'rounds': 10,
        #     'learning_rate': 1e-4,
        #     'clip_range': 0.2,
        #     'clip_range_decay': True
        # },
        # {
        #     'experiment_name': 'exp08',
        #     'ent_coef': 0.01,
        #     'K': 10,
        #     'steps_per_config_per_round': 2000,
        #     'rounds': 10,
        #     'learning_rate': 5e-5,
        #     'clip_range': 0.2,
        #     'clip_range_decay': True
        # },
        # {
        #     'experiment_name': 'exp09',
        #     'ent_coef': 0.02,
        #     'K': 10,
        #     'steps_per_config_per_round': 2000,
        #     'rounds': 10,
        #     'learning_rate': 1e-4,
        #     'clip_range': 0.2,
        #     'clip_range_decay': True
        # },
        # {
        #     'experiment_name': 'exp10',
        #     'ent_coef': 0.005,
        #     'K': 10,
        #     'steps_per_config_per_round': 2000,
        #     'rounds': 10,
        #     'learning_rate': 1e-4,
        #     'clip_range': 0.2,
        #     'clip_range_decay': True
        # },
        # {
        #     'experiment_name': 'exp11',
        #     'ent_coef': 0.01,
        #     'K': 10,
        #     'steps_per_config_per_round': 2000,
        #     'rounds': 10,
        #     'learning_rate': 1e-4,
        #     'clip_range': 0.15,
        #     'clip_range_decay': True
        # },
        {
            'experiment_name': 'exp12',
            'ent_coef': 0.01,
            'K': 10,
            'steps_per_config_per_round': 2000,
            'rounds': 10,
            'learning_rate': 1e-4,
            'clip_range': 0.2,
            'clip_range_decay': False
        },
        {
            'experiment_name': 'exp13',
            'ent_coef': 0.015,
            'K': 10,
            'steps_per_config_per_round': 2000,
            'rounds': 10,
            'learning_rate': 8e-5,
            'clip_range': 0.15,
            'clip_range_decay': True
        },
        {
            'experiment_name': 'exp14',
            'ent_coef': 0.02,
            'K': 10,
            'steps_per_config_per_round': 2000,
            'rounds': 10,
            'learning_rate': 5e-5,
            'clip_range': 0.15,
            'clip_range_decay': True
        },
        {
            'experiment_name': 'exp15',
            'ent_coef': 0.01,
            'K': 10,
            'steps_per_config_per_round': 1000,
            'rounds': 20,
            'learning_rate': 1e-4,
            'clip_range': 0.2,
            'clip_range_decay': True
        }
    ]

    run_parallel_experiments(experiments)
