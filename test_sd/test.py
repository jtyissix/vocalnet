import sys
import os
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/home/jiangtianyuan/resource/voice/vocalnet/')
sys.path.append('/home/jiangtianyuan/resource/voice/vocalnet/')

# 导入投机解码模型
from vocalnet_speculative_decoding import VocalNetSpeculativeDecoding, DRAFT_MODEL_PATH, TARGET_MODEL_PATH

def test_speculative_decoding_batch(
    audio_files: list,
    k_values: list = [3, 5, 7, 15],
    save_dir: str = "./audio_acceptance_results"
):
    """
    批量测试不同k值下的audio token接受概率
    
    Args:
        audio_files: 音频文件路径列表
        k_values: 要测试的k值列表 (每次draft生成的text token数)
        save_dir: 结果保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    
    for k in k_values:
        print(f"\n{'='*70}")
        print(f"Testing with k={k} (text tokens)")
        print(f"{'='*70}")
        
        # 初始化模型
        vocalnet = VocalNetSpeculativeDecoding(
            draft_model_path=DRAFT_MODEL_PATH,
            target_model_path=TARGET_MODEL_PATH,
            k=k,
            s2s=True
        )
        
        vocalnet.__initialize__()
        audio_save_dir = os.path.join(save_dir, f"k_{k}")
        os.makedirs(audio_save_dir, exist_ok=True)
        vocalnet.set_audio_dir(audio_save_dir)
        
        for audio_file in audio_files:
            if not os.path.exists(audio_file):
                print(f"Warning: {audio_file} not found, skipping...")
                continue
            
            print(f"\nProcessing: {audio_file}")
            audio_messages = [{"role": "user", "content": "<speech>", "path": audio_file}]
            
            try:
                response = vocalnet(audio_messages)
                stats = vocalnet.stats.get_summary()
                
                # 收集每个样本的结果
                result = {
                    'audio_file': os.path.basename(audio_file),
                    'k': k,
                    'acceptance_rate': stats['avg_acceptance_rate'],
                    'expected_accepted': stats['avg_expected_accepted'],
                    'num_audio_tokens': stats['avg_num_audio_tokens'],
                    'draft_time_ms': stats['avg_draft_time'] * 1000,
                    'verify_time_ms': stats['avg_verify_time'] * 1000,
                }
                
                all_results.append(result)
                
                print(f"  Audio Tokens: {result['num_audio_tokens']:.0f}")
                print(f"  Expected Accepted: {result['expected_accepted']:.2f}")
                print(f"  Acceptance Rate: {result['acceptance_rate']*100:.2f}%")
                
                # 重置统计信息
                vocalnet.stats = type(vocalnet.stats)()
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # 保存结果
    if all_results:
        df = pd.DataFrame(all_results)
        results_csv = os.path.join(save_dir, "audio_acceptance_results.csv")
        df.to_csv(results_csv, index=False)
        print(f"\nResults saved to: {results_csv}")
        
        # 生成统计报告
        generate_analysis_report(df, save_dir)
        
        return df
    else:
        print("No results to save!")
        return None


def generate_analysis_report(df: pd.DataFrame, save_dir: str):
    """生成详细的分析报告"""
    
    print("\n" + "="*70)
    print("Audio Token Acceptance Test Results")
    print("="*70)
    
    # 按k值分组统计
    summary = df.groupby('k').agg({
        'acceptance_rate': ['mean', 'std', 'min', 'max'],
        'expected_accepted': ['mean', 'std', 'min', 'max'],
        'num_audio_tokens': ['mean', 'std'],
        'draft_time_ms': 'mean',
        'verify_time_ms': 'mean'
    }).round(4)
    
    print("\n=== Summary by k (重点指标) ===")
    print(summary)
    
    # 保存统计表格
    summary.to_csv(os.path.join(save_dir, "summary_by_k.csv"))
    
    # 打印重点统计
    print("\n" + "="*70)
    print("KEY METRICS BY k")
    print("="*70)
    for k in sorted(df['k'].unique()):
        k_data = df[df['k'] == k]
        print(f"\nk = {k}:")
        print(f"  平均接受率: {k_data['acceptance_rate'].mean()*100:.2f}% (±{k_data['acceptance_rate'].std()*100:.2f}%)")
        print(f"  平均期望接受数: {k_data['expected_accepted'].mean():.2f} (±{k_data['expected_accepted'].std():.2f})")
        print(f"  平均audio token数: {k_data['num_audio_tokens'].mean():.1f}")
        print(f"  效率: {(k_data['expected_accepted'].mean()/k_data['num_audio_tokens'].mean())*100:.1f}%")
    
    # 生成可视化
    generate_visualizations(df, save_dir)
    
    # 生成文本报告
    generate_text_report(df, summary, save_dir)


def generate_visualizations(df: pd.DataFrame, save_dir: str):
    """生成可视化图表 - 重点展示接受率和期望长度"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    k_values = sorted(df['k'].unique())
    
    # 1. 接受率 vs k值 (左上)
    ax1 = fig.add_subplot(gs[0, 0])
    acceptance_rates = [df[df['k']==k]['acceptance_rate'].mean() for k in k_values]
    acceptance_stds = [df[df['k']==k]['acceptance_rate'].std() for k in k_values]
    
    ax1.errorbar(k_values, acceptance_rates, yerr=acceptance_stds, marker='o', 
                 capsize=5, capthick=2, linewidth=2, markersize=10, color='#2E86AB')
    ax1.set_xlabel('k (Text Tokens)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Acceptance Rate', fontsize=13, fontweight='bold')
    ax1.set_title('平均接受率 vs k', fontsize=15, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0, 1])
    for i, (k, rate) in enumerate(zip(k_values, acceptance_rates)):
        ax1.text(k, rate + 0.03, f'{rate*100:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    # 2. 期望接受数 vs k值 (右上)
    ax2 = fig.add_subplot(gs[0, 1])
    expected_accepted = [df[df['k']==k]['expected_accepted'].mean() for k in k_values]
    expected_stds = [df[df['k']==k]['expected_accepted'].std() for k in k_values]
    num_audio_tokens = [df[df['k']==k]['num_audio_tokens'].mean() for k in k_values]
    
    ax2.errorbar(k_values, expected_accepted, yerr=expected_stds, marker='s', 
                 capsize=5, capthick=2, linewidth=2, markersize=10, color='#06A77D', 
                 label='Expected Accepted')
    ax2.plot(k_values, num_audio_tokens, 'r--', marker='^', linewidth=2, markersize=10, 
             alpha=0.7, label='Total Audio Tokens')
    ax2.set_xlabel('k (Text Tokens)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Number of Audio Tokens', fontsize=13, fontweight='bold')
    ax2.set_title('期望接受数 vs 总token数', fontsize=15, fontweight='bold', pad=10)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    for i, (k, exp) in enumerate(zip(k_values, expected_accepted)):
        ax2.text(k, exp + 2, f'{exp:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    # 3. 接受率分布箱线图 (左中)
    ax3 = fig.add_subplot(gs[1, 0])
    data_for_box = [df[df['k']==k]['acceptance_rate'].values for k in k_values]
    bp = ax3.boxplot(data_for_box, labels=k_values, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    ax3.set_xlabel('k (Text Tokens)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Acceptance Rate', fontsize=13, fontweight='bold')
    ax3.set_title('接受率分布', fontsize=15, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_ylim([0, 1])
    
    # 4. 期望接受数分布箱线图 (右中)
    ax4 = fig.add_subplot(gs[1, 1])
    data_for_box = [df[df['k']==k]['expected_accepted'].values for k in k_values]
    bp = ax4.boxplot(data_for_box, labels=k_values, patch_artist=True,
                     boxprops=dict(facecolor='lightgreen', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    ax4.set_xlabel('k (Text Tokens)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Expected Accepted Tokens', fontsize=13, fontweight='bold')
    ax4.set_title('期望接受数分布', fontsize=15, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 5. 效率对比表格 (底部跨两列)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')
    
    # 构建表格数据
    table_data = []
    table_data.append(['k', 'Acceptance Rate', 'Expected Accepted', 'Audio Tokens', 'Efficiency'])
    for k in k_values:
        k_data = df[df['k'] == k]
        acc_rate = k_data['acceptance_rate'].mean()
        exp_acc = k_data['expected_accepted'].mean()
        num_tokens = k_data['num_audio_tokens'].mean()
        efficiency = exp_acc / num_tokens if num_tokens > 0 else 0
        table_data.append([
            f'{k}',
            f'{acc_rate*100:.2f}%',
            f'{exp_acc:.2f}',
            f'{num_tokens:.1f}',
            f'{efficiency*100:.1f}%'
        ])
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.25, 0.25, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置行交替颜色
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.suptitle('Audio Token 投机解码接受概率分析', fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(save_dir, 'audio_acceptance_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {os.path.join(save_dir, 'audio_acceptance_analysis.png')}")
    plt.close()


def generate_text_report(df: pd.DataFrame, summary: pd.DataFrame, save_dir: str):
    """生成文本报告"""
    
    report_path = os.path.join(save_dir, "analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Audio Token 投机解码接受概率测试报告\n")
        f.write("Draft Model: 1B | Target Model: 8B\n")
        f.write("="*70 + "\n\n")
        
        f.write("测试说明:\n")
        f.write("-" * 70 + "\n")
        f.write("本测试评估draft model生成的audio tokens被target model接受的概率\n")
        f.write("使用标准投机解码验证逻辑: accept_prob = min(1, p(x)/q(x))\n")
        f.write("k = draft model生成的text token数量\n")
        f.write("期望接受数 = 考虑前缀性质的累积接受概率\n\n")
        
        f.write("整体统计:\n")
        f.write("-" * 70 + "\n")
        f.write(f"测试样本数: {len(df)}\n")
        f.write(f"测试的k值: {sorted(df['k'].unique())}\n")
        f.write(f"总体平均接受率: {df['acceptance_rate'].mean()*100:.2f}%\n")
        f.write(f"总体平均期望接受数: {df['expected_accepted'].mean():.2f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("各k值详细统计\n")
        f.write("="*70 + "\n\n")
        
        for k in sorted(df['k'].unique()):
            k_data = df[df['k'] == k]
            f.write(f"k = {k} (text tokens):\n")
            f.write("-" * 70 + "\n")
            f.write(f"  样本数量: {len(k_data)}\n")
            f.write(f"  \n")
            f.write(f"  【接受率】\n")
            f.write(f"    平均值: {k_data['acceptance_rate'].mean()*100:.2f}%\n")
            f.write(f"    标准差: {k_data['acceptance_rate'].std()*100:.2f}%\n")
            f.write(f"    最小值: {k_data['acceptance_rate'].min()*100:.2f}%\n")
            f.write(f"    最大值: {k_data['acceptance_rate'].max()*100:.2f}%\n")
            f.write(f"  \n")
            f.write(f"  【期望接受数】\n")
            f.write(f"    平均值: {k_data['expected_accepted'].mean():.2f} tokens\n")
            f.write(f"    标准差: {k_data['expected_accepted'].std():.2f}\n")
            f.write(f"    最小值: {k_data['expected_accepted'].min():.2f}\n")
            f.write(f"    最大值: {k_data['expected_accepted'].max():.2f}\n")
            f.write(f"  \n")
            f.write(f"  【Audio Token数量】\n")
            f.write(f"    平均值: {k_data['num_audio_tokens'].mean():.1f} tokens\n")
            f.write(f"    标准差: {k_data['num_audio_tokens'].std():.1f}\n")
            f.write(f"  \n")
            f.write(f"  【效率】\n")
            efficiency = k_data['expected_accepted'].mean() / k_data['num_audio_tokens'].mean() if k_data['num_audio_tokens'].mean() > 0 else 0
            f.write(f"    接受效率: {efficiency*100:.1f}%\n")
            f.write(f"    (期望接受数 / 总audio tokens)\n")
            f.write(f"  \n")
            f.write(f"  【时间开销】\n")
            f.write(f"    Draft平均时间: {k_data['draft_time_ms'].mean():.2f}ms\n")
            f.write(f"    Verify平均时间: {k_data['verify_time_ms'].mean():.2f}ms\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("结论与建议\n")
        f.write("="*70 + "\n")
        
        best_k_rate = df.groupby('k')['acceptance_rate'].mean().idxmax()
        best_rate = df.groupby('k')['acceptance_rate'].mean().max()
        best_k_exp = df.groupby('k')['expected_accepted'].mean().idxmax()
        best_exp = df.groupby('k')['expected_accepted'].mean().max()
        
        f.write(f"1. 最高接受率: k={best_k_rate}, 接受率={best_rate*100:.2f}%\n")
        f.write(f"2. 最高期望接受数: k={best_k_exp}, 期望接受={best_exp:.2f} tokens\n")
        f.write(f"\n")
        
        if best_rate > 0.8:
            f.write("3. 评估: 接受率很高，投机解码效果优秀\n")
        elif best_rate > 0.6:
            f.write("3. 评估: 接受率良好，投机解码有明显效果\n")
        else:
            f.write("3. 评估: 接受率较低，可能需要优化draft model或调整k值\n")
    
    print(f"Text report saved to: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Audio Token Acceptance Test')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--audio_list', type=str, default=None,
                        help='Text file containing list of audio paths')
    parser.add_argument('--k_values', type=int, nargs='+', default=[3, 5, 7, 15],
                        help='List of k values (text tokens) to test')
    parser.add_argument('--save_dir', type=str, default='./audio_acceptance_results',
                        help='Directory to save results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to test')
    
    args = parser.parse_args()
    
    # 收集音频文件
    if args.audio_list:
        with open(args.audio_list, 'r') as f:
            audio_files = [line.strip() for line in f if line.strip()]
    else:
        audio_dir = Path(args.audio_dir)
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(list(audio_dir.glob(ext)))
        audio_files = [str(f) for f in audio_files]
    
    if args.max_samples:
        audio_files = audio_files[:args.max_samples]
    
    print(f"Found {len(audio_files)} audio files to test")
    print(f"Testing k values (text tokens): {args.k_values}")
    
    if len(audio_files) == 0:
        print("No audio files found!")
        exit(1)
    
    # 运行测试
    df = test_speculative_decoding_batch(
        audio_files=audio_files,
        k_values=args.k_values,
        save_dir=args.save_dir
    )
    
    print("\n" + "="*70)
    print("Testing completed!")
    print(f"Results saved to: {args.save_dir}")
    print("="*70)