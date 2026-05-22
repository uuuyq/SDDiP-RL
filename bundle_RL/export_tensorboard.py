import os
import json
import pandas as pd

def get_experiment_dirs(tensorboard_dir):
    """获取所有实验目录"""
    print(f"查找TensorBoard目录: {tensorboard_dir}")
    
    if not os.path.exists(tensorboard_dir):
        print(f"ERROR: TensorBoard目录不存在!")
        print(f"请确保目录存在: {tensorboard_dir}")
        return {}
    
    experiments = {}
    for exp_name in os.listdir(tensorboard_dir):
        exp_path = os.path.join(tensorboard_dir, exp_name)
        if os.path.isdir(exp_path):
            # 递归查找 events 文件
            events_file = None
            for root, dirs, files in os.walk(exp_path):
                for f in files:
                    if f.startswith('events.out.tfevents'):
                        events_file = os.path.join(root, f)
                        break
                if events_file:
                    break
            
            if events_file:
                experiments[exp_name] = events_file
                print(f"  找到实验: {exp_name}")
            else:
                print(f"  警告: {exp_name}目录中未找到events文件")
    
    return experiments

def extract_tensorboard_data(events_file, output_json):
    """从tensorboard事件文件中提取数据"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        print(f"  提取数据: {os.path.basename(events_file)}")
        
        ea = event_accumulator.EventAccumulator(events_file)
        ea.Reload()
        
        data = {}
        scalar_keys = ea.scalars.Keys()
        
        for key in scalar_keys:
            events = ea.scalars.Items(key)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[key] = {'steps': steps, 'values': values}
        
        output_dir = os.path.dirname(output_json)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return True, list(scalar_keys)
    except ImportError:
        print("  ERROR: 需要安装tensorflow库")
        return False, []
    except Exception as e:
        print(f"  ERROR: 提取数据失败: {e}")
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
    
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    return df

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
            
            summary[f'{metric}_final'] = values[-1] if values else None
            summary[f'{metric}_mean'] = sum(values) / len(values) if values else None
            summary[f'{metric}_min'] = min(values) if values else None
            summary[f'{metric}_max'] = max(values) if values else None
            
            last_idx = max(0, len(values) - 10)
            summary[f'{metric}_last_mean'] = sum(values[last_idx:]) / len(values[last_idx:]) if values else None
        else:
            summary[f'{metric}_final'] = None
            summary[f'{metric}_mean'] = None
            summary[f'{metric}_min'] = None
            summary[f'{metric}_max'] = None
            summary[f'{metric}_last_mean'] = None
    
    return summary

def analyze_experiments(tensorboard_dir, output_dir):
    """分析所有实验的数据"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    print("-" * 60)
    
    experiments = get_experiment_dirs(tensorboard_dir)
    
    if not experiments:
        print("\n未找到任何实验数据！")
        return None
    
    print(f"\n找到 {len(experiments)} 个实验")
    print("-" * 60)
    
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
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            export_to_csv(data, csv_file)
            
            summary = analyze_metrics(data, exp_name)
            summary_data.append(summary)
            
            print(f"  数据已保存到: {exp_output_dir}")
    
    summary_file = os.path.join(output_dir, 'experiment_summary.csv')
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        print(f"\n实验对比摘要已保存到: {summary_file}")
        
        print("\n" + "=" * 60)
        print("实验对比摘要:")
        print("=" * 60)
        
        key_cols = ['experiment', 
                   'rollout/ep_rew_mean_final', 'rollout/ep_rew_mean_last_mean',
                   'train/approx_kl_final', 'train/approx_kl_last_mean',
                   'train/clip_fraction_final', 'train/clip_fraction_last_mean']
        
        available_cols = [c for c in key_cols if c in summary_df.columns]
        if available_cols:
            print(summary_df[available_cols].to_string(index=False))
        else:
            print("没有可用的指标数据")
        
        return summary_df
    else:
        print("\n没有成功提取任何实验数据")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("TensorBoard 数据导出与分析工具")
    print("=" * 60)
    
    # ===============================
    # 固定目录配置（修改这里！）
    # ===============================
    
    # TensorBoard数据目录（events.out.tfevents文件所在的目录）
    TENSORBOARD_DIR = r"D:\tools\workspace_pycharm\SDDiP-RL\bundle_RL\train_result\ppo_tensorboard"
    
    # 输出目录（分析结果保存位置）
    OUTPUT_DIR = r"D:\tools\workspace_pycharm\SDDiP-RL\bundle_RL\train_result\analysis"
    
    # ===============================
    
    print(f"TensorBoard目录: {TENSORBOARD_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    # 检查目录是否存在
    if os.path.exists(TENSORBOARD_DIR):
        print("TensorBoard目录存在")
        print("包含的实验:")
        for item in os.listdir(TENSORBOARD_DIR):
            if os.path.isdir(os.path.join(TENSORBOARD_DIR, item)):
                print(f"  - {item}")
    else:
        print(f"WARNING: TensorBoard目录不存在: {TENSORBOARD_DIR}")
    
    print("\n" + "-" * 60)
    
    analyze_experiments(TENSORBOARD_DIR, OUTPUT_DIR)
