import sys
import os
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '/home/jiangtianyuan/resource/voice/vocalnet/')
sys.path.append('/home/jiangtianyuan/resource/voice/vocalnet/')

# 导入投机解码模型
from vocalnet_speculative_decoding import VocalNetSpeculativeDecoding, DRAFT_MODEL_PATH, TARGET_MODEL_PATH

def test_speculative_decoding_batch(
    audio_files: list,
    k_values: list = [3, 5, 7, 10],
    save_dir: str = "./speculative_results",
    max_new_tokens: int = 512
):
    """
    批量测试不同k值下的投机解码性能
    
    Args:
        audio_files: 音频文件路径列表
        k_values: 要测试的k值列表 (每次draft生成的token数)
        save_dir: 结果保存目录
        max_new_tokens: 最大生成token数
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Testing with k={k}")
        print(f"{'='*60}")
        
        # 初始化模型
        vocalnet = VocalNetSpeculativeDecoding(
            draft_model_path=DRAFT_MODEL_PATH,
            target_model_path=TARGET_MODEL_PATH,
            k=k,
            s2s=True,
            max_new_tokens=max_new_tokens
        )
        
        vocalnet.__initialize__()
        audio_save_dir = os.path.join(save_dir, f"k_{k}")
        os.makedirs(audio_save_dir, exist_ok=True)
        vocalnet.set_audio_dir(audio_save_dir)
        breakpoint()
        for audio_file in audio_files:
            if not os.path.exists(audio_file):
                print(f"Warning: {audio_file} not found, skipping...")
                continue
            
            print(f"\nProcessing: {audio_file}")
            audio_messages = [{"role": "user", "content": "<speech>", "path": audio_file}]
            
            try:
                response = vocalnet(audio_messages)
                stats = vocalnet.stats.get_summary()
                
                result = {
                    'audio_file': os.path.basename(audio_file),
                    'k': k,
                    'acceptance_rate': stats['acceptance_rate'],
                    'total_draft_tokens': stats['total_draft_tokens'],
                    'total_accepted_tokens': stats['total_accepted_tokens'],
                    'num_iterations': stats['num_iterations'],
                    'avg_accepted_per_iteration': stats['avg_accepted_per_iteration'],
                    'avg_draft_time_ms': stats['avg_draft_time'] * 1000,
                    'avg_verify_time_ms': stats['avg_verify_time'] * 1000,
                    'generated_text': response['text']
                }
                
                all_results.append(result)
                
                print(f"  Acceptance Rate: {stats['acceptance_rate']*100:.2f}%")
                print(f"  Iterations: {stats['num_iterations']}")
                print(f"  Avg Accepted/Iter: {stats['avg_accepted_per_iteration']:.2f}")
                
                # 重置统计信息
                vocalnet.stats = type(vocalnet.stats)()
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                continue
    
    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(save_dir, "speculative_decoding_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")
    
    # 生成统计报告
    generate_analysis_report(results_df, save_dir)
    
    return results_df


def generate_analysis_report(df: pd.DataFrame, save_dir: str):
    """生成详细的分析报告"""
    
    # 1. 按k值分组统计
    print("\n" + "="*60)
    print("Summary Statistics by k")
    print("="*60)
    
    summary_by_k = df.groupby('k').agg({
        'acceptance_rate': ['mean', 'std', 'min', 'max'],
        'avg_accepted_per_iteration': ['mean', 'std'],
        'num_iterations': ['mean', 'std'],
        'avg_draft_time_ms': 'mean',
        'avg_verify_time_ms': 'mean'
    }).round(4)
    
    print(summary_by_k)
    
    # 保存统计表格
    summary_by_k.to_csv(os.path.join(save_dir, "summary_by_k.csv"))
    
    # 2. 生成可视化图表
    generate_visualizations(df, save_dir)
    
    # 3. 生成文本报告
    generate_text_report(df, summary_by_k, save_dir)


def generate_visualizations(df: pd.DataFrame, save_dir: str):
    """生成可视化图表"""
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 接受率 vs k值
    ax1 = axes[0, 0]
    k_values = sorted(df['k'].unique())
    acceptance_rates = [df[df['k']==k]['acceptance_rate'].mean() for k in k_values]
    acceptance_stds = [df[df['k']==k]['acceptance_rate'].std() for k in k_values]
    
    ax1.errorbar(k_values, acceptance_rates, yerr=acceptance_stds, marker='o', 
                 capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.set_xlabel('k (Draft Tokens per Iteration)', fontsize=12)
    ax1.set_ylabel('Acceptance Rate', fontsize=12)
    ax1.set_title('Acceptance Rate vs k', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 2. 每次迭代接受的平均token数 vs k值
    ax2 = axes[0, 1]
    avg_accepted = [df[df['k']==k]['avg_accepted_per_iteration'].mean() for k in k_values]
    avg_accepted_stds = [df[df['k']==k]['avg_accepted_per_iteration'].std() for k in k_values]
    
    ax2.errorbar(k_values, avg_accepted, yerr=avg_accepted_stds, marker='s', 
                 capsize=5, capthick=2, linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('k (Draft Tokens per Iteration)', fontsize=12)
    ax2.set_ylabel('Avg Accepted Tokens per Iteration', fontsize=12)
    ax2.set_title('Efficiency vs k', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 时间开销对比
    ax3 = axes[1, 0]
    draft_times = [df[df['k']==k]['avg_draft_time_ms'].mean() for k in k_values]
    verify_times = [df[df['k']==k]['avg_verify_time_ms'].mean() for k in k_values]
    
    x = range(len(k_values))
    width = 0.35
    ax3.bar([i - width/2 for i in x], draft_times, width, label='Draft Time', alpha=0.8)
    ax3.bar([i + width/2 for i in x], verify_times, width, label='Verify Time', alpha=0.8)
    ax3.set_xlabel('k (Draft Tokens per Iteration)', fontsize=12)
    ax3.set_ylabel('Time (ms)', fontsize=12)
    ax3.set_title('Average Time Cost per Iteration', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(k_values)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 接受率分布箱线图
    ax4 = axes[1, 1]
    data_for_box = [df[df['k']==k]['acceptance_rate'].values for k in k_values]
    bp = ax4.boxplot(data_for_box, labels=k_values, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax4.set_xlabel('k (Draft Tokens per Iteration)', fontsize=12)
    ax4.set_ylabel('Acceptance Rate', fontsize=12)
    ax4.set_title('Acceptance Rate Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'speculative_decoding_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {os.path.join(save_dir, 'speculative_decoding_analysis.png')}")
    plt.close()


def generate_text_report(df: pd.DataFrame, summary: pd.DataFrame, save_dir: str):
    """生成文本报告"""
    
    report_path = os.path.join(save_dir, "analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Speculative Decoding Analysis Report\n")
        f.write("Draft Model: 1B | Target Model: 8B\n")
        f.write("="*70 + "\n\n")
        
        # 整体统计
        f.write("Overall Statistics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total samples tested: {len(df)}\n")
        f.write(f"k values tested: {sorted(df['k'].unique())}\n")
        f.write(f"Overall average acceptance rate: {df['acceptance_rate'].mean()*100:.2f}%\n")
        f.write(f"Best k value (by acceptance rate): k={df.groupby('k')['acceptance_rate'].mean().idxmax()}\n")
        f.write(f"Best k value (by efficiency): k={df.groupby('k')['avg_accepted_per_iteration'].mean().idxmax()}\n")
        f.write("\n")
        
        # 按k值详细统计
        f.write("Detailed Statistics by k:\n")
        f.write("-" * 70 + "\n")
        
        for k in sorted(df['k'].unique()):
            k_data = df[df['k'] == k]
            f.write(f"\nk = {k}:\n")
            f.write(f"  Acceptance Rate: {k_data['acceptance_rate'].mean()*100:.2f}% ")
            f.write(f"(±{k_data['acceptance_rate'].std()*100:.2f}%)\n")
            f.write(f"  Avg Accepted per Iteration: {k_data['avg_accepted_per_iteration'].mean():.2f} ")
            f.write(f"(±{k_data['avg_accepted_per_iteration'].std():.2f})\n")
            f.write(f"  Avg Iterations: {k_data['num_iterations'].mean():.1f}\n")
            f.write(f"  Avg Draft Time: {k_data['avg_draft_time_ms'].mean():.2f}ms\n")
            f.write(f"  Avg Verify Time: {k_data['avg_verify_time_ms'].mean():.2f}ms\n")
            f.write(f"  Speedup potential: {k_data['avg_accepted_per_iteration'].mean():.2f}x per iteration\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Recommendations:\n")
        f.write("-" * 70 + "\n")
        
        best_k = df.groupby('k')['acceptance_rate'].mean().idxmax()
        best_rate = df.groupby('k')['acceptance_rate'].mean().max()
        f.write(f"1. Optimal k value: k={best_k} with {best_rate*100:.2f}% acceptance rate\n")
        
        efficiency_k = df.groupby('k')['avg_accepted_per_iteration'].mean().idxmax()
        efficiency_val = df.groupby('k')['avg_accepted_per_iteration'].mean().max()
        f.write(f"2. Best efficiency: k={efficiency_k} with {efficiency_val:.2f} tokens accepted per iteration\n")
        
        if best_rate > 0.7:
            f.write("3. High acceptance rate achieved! Speculative decoding is effective.\n")
        elif best_rate > 0.5:
            f.write("3. Moderate acceptance rate. Consider tuning or using smaller k values.\n")
        else:
            f.write("3. Low acceptance rate. Draft model may be too different from target model.\n")
    
    print(f"Text report saved to: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch test speculative decoding')
    parser.add_argument('--audio_dir', type=str, required=True, 
                        help='Directory containing audio files to test')
    parser.add_argument('--audio_list', type=str, default=None,
                        help='Text file containing list of audio paths (one per line)')
    parser.add_argument('--k_values', type=int, nargs='+', default=[3, 5, 7, 10],
                        help='List of k values to test')
    parser.add_argument('--save_dir', type=str, default='./speculative_results',
                        help='Directory to save results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to test')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of tokens to generate')
    
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
    print(f"Testing k values: {args.k_values}")
    
    if len(audio_files) == 0:
        print("No audio files found!")
        exit(1)
    
    # 运行批量测试
    results_df = test_speculative_decoding_batch(
        audio_files=audio_files,
        k_values=args.k_values,
        save_dir=args.save_dir,
        max_new_tokens=args.max_new_tokens
    )
    
    print("\n" + "="*70)
    print("Batch testing completed!")
    print(f"Results saved to: {args.save_dir}")
    print("="*70)