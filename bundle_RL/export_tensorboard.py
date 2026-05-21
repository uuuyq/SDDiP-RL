import os
import json
from pathlib import Path
import pandas as pd

def get_experiment_dirs(base_dir):
    """获取所有实验目录"""
    tensorboard_dir = os.path.join(base_dir, 'train_result', 'ppo_tensorboard')
    if not os.path.exists(tensorboard_dir):
        print(f"TensorBoard目录不存在: {tensorboard_dir}")
        return {}
    
    experiments = {}
    for exp_name in os.listdir(tensorboard_dir):
        exp_path = os.path.join(tensorboard_dir, exp_name)
        if os.path.isdir(exp_path):
            events_file = None
            for f in os.listdir(exp_path):
                if f.startswith('events.out.tfevents'):
                    events_file = os.path.join(exp_path, f)
                    break
            if events_file:
                experiments[exp_name] = events_file
    return experiments

def extract_tensorboard_data(events_file, output_json):
    """从tensorboard事件文件中提取数据"""
    try:
        import tensorflow as tf
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(events_file)
        ea.Reload()
        
        data = {}
        scalar_keys = ea.scalars.Keys()
        
        for key in scalar_keys:
            events = ea.scalars.Items(key)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[key] = {'steps': steps, 'values': values}
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return True, list(scalar_keys)
    except Exception as e:
        print(f"提取数据失败: {e}")
        return False, []

def export_to_csv(data, output_csv):
    """将数据导出为CSV格式"""
    records = []
    for metric_name, metric_data in data.items():
        steps = metric_data['steps']
        values = metric_data['values']
        for step, value in zip(steps, values):
            records.append({
                'step': step,
                'metric': metric_name,
                'value': value
            })
    
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    return df

def analyze_experiments(base_dir, output_dir=None):
    """分析所有实验的数据"""
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'train_result', 'analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    
    experiments = get_experiment_dirs(base_dir)
    
    if not experiments:
        print("未找到任何实验数据！")
        return
    
    print(f"找到 {len(experiments)} 个实验:")
    for exp_name in experiments:
        print(f"  - {exp_name}")
    
    summary_data = []
    
    for exp_name, events_file in experiments.items():
        print(f"\n处理实验: {exp_name}")
        
        exp_output_dir = os.path.join(output_dir, exp_name)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        json_file = os.path.join(exp_output_dir, 'tensorboard_data.json')
        csv_file = os.path.join(exp_output_dir, 'tensorboard_data.csv')
        
        success, scalar_keys = extract_tensorboard_data(events_file, json_file)
        
        if success:
            print(f"  提取到 {len(scalar_keys)} 个指标")
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            export_to_csv(data, csv_file)
            
            summary = analyze_metrics(data, exp_name)
            summary_data.append(summary)
            
            print(f"  数据已保存到: {exp_output_dir}")
    
    summary_file = os.path.join(output_dir, 'experiment_summary.csv')
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    print(f"\n实验对比摘要已保存到: {summary_file}")
    
    return summary_df

def analyze_metrics(data, exp_name):
    """分析单个实验的关键指标"""
    summary = {'experiment': exp_name}
    
    key_metrics = [
        'rollout/ep_rew_mean',
        'rollout/ep_len_mean',
        'train/policy_loss',
        'train/value_loss',
        'train/entropy_loss',
        'train/approx_kl',
        'train/clip_fraction'
    ]
    
    for metric in key_metrics:
        if metric in data:
            values = data[metric]['values']
            steps = data[metric]['steps']
            
            summary[f'{metric}_final'] = values[-1] if values else None
            summary[f'{metric}_mean'] = sum(values) / len(values) if values else None
            summary[f'{metric}_min'] = min(values) if values else None
            summary[f'{metric}_max'] = max(values) if values else None
            
            last_10k_idx = max(0, len(values) - 10)
            summary[f'{metric}_last_10k_mean'] = sum(values[last_10k_idx:]) / len(values[last_10k_idx:]) if values else None
        else:
            summary[f'{metric}_final'] = None
            summary[f'{metric}_mean'] = None
            summary[f'{metric}_min'] = None
            summary[f'{metric}_max'] = None
            summary[f'{metric}_last_10k_mean'] = None
    
    return summary

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("="*60)
    print("TensorBoard 数据导出与分析工具")
    print("="*60)
    print(f"项目目录: {base_dir}")
    
    df = analyze_experiments(base_dir)
    
    if df is not None and not df.empty:
        print("\n" + "="*60)
        print("实验对比摘要:")
        print("="*60)
        
        key_cols = ['experiment', 'rollout/ep_rew_mean_final', 'rollout/ep_rew_mean_last_10k_mean',
                   'train/approx_kl_final', 'train/approx_kl_last_10k_mean',
                   'train/clip_fraction_final', 'train/clip_fraction_last_10k_mean']
        
        available_cols = [c for c in key_cols if c in df.columns]
        print(df[available_cols].to_string(index=False))
