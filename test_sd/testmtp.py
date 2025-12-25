import sys
import os
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/home/jiangtianyuan/resource/voice/vocalnet/')
sys.path.append('/home/jiangtianyuan/resource/voice/vocalnet/')

from vocalnet_speculative_decoding_mtp import VocalNetSpeculativeDecodingMTP, DRAFT_MODEL_PATH, TARGET_MODEL_PATH

def test_speculative_decoding_mtp_batch(
    audio_files: list,
    k_values: list = [3, 5, 7, 15],
    mtp_values: list = [1, 2, 3],
    save_dir: str = "./audio_acceptance_results_mtp"
):
    """
    批量测试不同k值和MTP层数下的audio token接受概率
    
    Args:
        audio_files: 音频文件路径列表
        k_values: 要测试的k值列表 (每次draft生成的text token数)
        mtp_values: 要测试的MTP层数列表
        save_dir: 结果保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    
    for mtp_num in mtp_values:
        for k in k_values:
            print(f"\n{'='*70}")
            print(f"Testing with k={k} (text tokens), MTP layers={mtp_num}")
            print(f"{'='*70}")
            
            # 初始化模型
            vocalnet = VocalNetSpeculativeDecodingMTP(
                draft_model_path=DRAFT_MODEL_PATH,
                target_model_path=TARGET_MODEL_PATH,
                k=k,
                mtp_num=mtp_num,
                s2s=True
            )
            
            vocalnet.__initialize__()
            audio_save_dir = os.path.join(save_dir, f"mtp_{mtp_num}_k_{k}")
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
                        'mtp_num': mtp_num,
                        'acceptance_rate': stats['avg_acceptance_rate'],
                        'expected_accepted': stats['avg_expected_accepted'],
                        'num_audio_tokens': stats['avg_num_audio_tokens'],
                        'draft_time_ms': stats['avg_draft_time'] * 1000,
                        'verify_time_ms': stats['avg_verify_time'] * 1000
                    }
                    all_results.append(result)
                    
                    print(f"  Acceptance Rate: {result['acceptance_rate']*100:.2f}%")
                    print(f"  Expected Accepted: {result['expected_accepted']:.2f}")
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {str(e)}")
                    continue
    
    # 保存结果
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(save_dir, "results_mtp.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # 生成可视化
    generate_visualizations(df, save_dir)
    
    # 生成报告
    generate_text_report(df, save_dir)
    
    return df


def generate_visualizations(df: pd.DataFrame, save_dir: str):
    """生成可视化图表"""
    
    # 1. 接受率 vs k值 (不同MTP层数)
    plt.figure(figsize=(12, 6))
    for mtp_num in sorted(df['mtp_num'].unique()):
        mtp_data = df[df['mtp_num'] == mtp_num]
        avg_by_k = mtp_data.groupby('k')['acceptance_rate'].mean()
        plt.plot(avg_by_k.index, avg_by_k.values * 100, marker='o', label=f'MTP={mtp_num}')
    
    plt.xlabel('k (text tokens)', fontsize=12)
    plt.ylabel('Average Acceptance Rate (%)', fontsize=12)
    plt.title('Acceptance Rate vs k (Different MTP Layers)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'acceptance_rate_vs_k_mtp.png'), dpi=300)
    plt.close()
    
    # 2. 期望接受数 vs k值 (不同MTP层数)
    plt.figure(figsize=(12, 6))
    for mtp_num in sorted(df['mtp_num'].unique()):
        mtp_data = df[df['mtp_num'] == mtp_num]
        avg_by_k = mtp_data.groupby('k')['expected_accepted'].mean()
        plt.plot(avg_by_k.index, avg_by_k.values, marker='o', label=f'MTP={mtp_num}')
    
    plt.xlabel('k (text tokens)', fontsize=12)
    plt.ylabel('Average Expected Accepted Tokens', fontsize=12)
    plt.title('Expected Accepted Tokens vs k (Different MTP Layers)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'expected_accepted_vs_k_mtp.png'), dpi=300)
    plt.close()
    
    # 3. 接受率热力图
    pivot_table = df.groupby(['k', 'mtp_num'])['acceptance_rate'].mean().unstack()
    plt.figure(figsize=(10, 8))
    im = plt.imshow(pivot_table.values * 100, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Acceptance Rate (%)')
    plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
    plt.yticks(range(len(pivot_table.index)), pivot_table.index)
    plt.xlabel('MTP Layers', fontsize=12)
    plt.ylabel('k (text tokens)', fontsize=12)
    plt.title('Acceptance Rate Heatmap', fontsize=14)
    
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            text = plt.text(j, i, f'{pivot_table.values[i, j]*100:.1f}%',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'acceptance_rate_heatmap_mtp.png'), dpi=300)
    plt.close()
    
    # 4. Draft时间 vs Verify时间
    plt.figure(figsize=(12, 6))
    for mtp_num in sorted(df['mtp_num'].unique()):
        mtp_data = df[df['mtp_num'] == mtp_num]
        avg_draft = mtp_data.groupby('k')['draft_time_ms'].mean()
        avg_verify = mtp_data.groupby('k')['verify_time_ms'].mean()
        
        x = np.arange(len(avg_draft.index))
        width = 0.15
        offset = (mtp_num - 2) * width
        
        plt.bar(x + offset, avg_draft.values, width, label=f'Draft MTP={mtp_num}', alpha=0.7)
        plt.bar(x + offset, avg_verify.values, width, bottom=avg_draft.values, 
                label=f'Verify MTP={mtp_num}', alpha=0.7)
    
    plt.xlabel('k (text tokens)', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('Draft and Verify Time vs k (Different MTP Layers)', fontsize=14)
    plt.xticks(x, avg_draft.index)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_comparison_mtp.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to: {save_dir}")


def generate_text_report(df: pd.DataFrame, save_dir: str):
    """生成文本报告"""
    
    report_path = os.path.join(save_dir, "analysis_report_mtp.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Audio Token 投机解码接受概率测试报告 (MTP版本)\n")
        f.write("Draft Model: 1B | Target Model: 8B\n")
        f.write("="*70 + "\n\n")
        
        f.write("测试说明:\n")
        f.write("-" * 70 + "\n")
        f.write("本测试评估使用MTP头时draft model生成的audio tokens被target model接受的概率\n")
        f.write("使用标准投机解码验证逻辑: accept_prob = min(1, p(x)/q(x))\n")
        f.write("k = draft model生成的text token数量\n")
        f.write("mtp_num = 使用的MTP层数\n")
        f.write("期望接受数 = 考虑前缀性质的累积接受概率\n\n")
        
        f.write("整体统计:\n")
        f.write("-" * 70 + "\n")
        f.write(f"测试样本数: {len(df)}\n")
        f.write(f"测试的k值: {sorted(df['k'].unique())}\n")
        f.write(f"测试的MTP层数: {sorted(df['mtp_num'].unique())}\n")
        f.write(f"总体平均接受率: {df['acceptance_rate'].mean()*100:.2f}%\n")
        f.write(f"总体平均期望接受数: {df['expected_accepted'].mean():.2f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("各组合详细统计\n")
        f.write("="*70 + "\n\n")
        
        for mtp_num in sorted(df['mtp_num'].unique()):
            f.write(f"\n{'='*70}\n")
            f.write(f"MTP Layers = {mtp_num}\n")
            f.write(f"{'='*70}\n\n")
            
            mtp_data = df[df['mtp_num'] == mtp_num]
            
            for k in sorted(mtp_data['k'].unique()):
                k_data = mtp_data[mtp_data['k'] == k]
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
        
        # 找最佳组合
        best_idx = df['acceptance_rate'].idxmax()
        best_row = df.loc[best_idx]
        f.write(f"最高接受率组合: k={best_row['k']}, MTP={best_row['mtp_num']}, ")
        f.write(f"接受率={best_row['acceptance_rate']*100:.2f}%\n\n")
        
        best_exp_idx = df['expected_accepted'].idxmax()
        best_exp_row = df.loc[best_exp_idx]
        f.write(f"最高期望接受数组合: k={best_exp_row['k']}, MTP={best_exp_row['mtp_num']}, ")
        f.write(f"期望接受={best_exp_row['expected_accepted']:.2f}\n\n")
        
        # MTP层数影响分析
        f.write("MTP层数影响:\n")
        for mtp_num in sorted(df['mtp_num'].unique()):
            mtp_avg = df[df['mtp_num'] == mtp_num]['acceptance_rate'].mean()
            f.write(f"  MTP={mtp_num}: 平均接受率 {mtp_avg*100:.2f}%\n")
        
        f.write("\n")
        if df['acceptance_rate'].max() > 0.8:
            f.write("评估: 使用MTP头后接受率很高,投机解码效果优秀\n")
        elif df['acceptance_rate'].max() > 0.6:
            f.write("评估: 使用MTP头后接受率良好,投机解码有明显效果\n")
        else:
            f.write("评估: 接受率较低,可能需要优化MTP层数或k值\n")
    
    print(f"Text report saved to: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Audio Token Acceptance Test with MTP')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--audio_list', type=str, default=None,
                        help='Text file containing list of audio paths')
    parser.add_argument('--k_values', type=int, nargs='+', default=[3, 5, 7, 15],
                        help='List of k values (text tokens) to test')
    parser.add_argument('--mtp_values', type=int, nargs='+', default=[1, 2, 3],
                        help='List of MTP layer numbers to test')
    parser.add_argument('--save_dir', type=str, default='./audio_acceptance_results_mtp',
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
    print(f"Testing MTP layers: {args.mtp_values}")
    
    if len(audio_files) == 0:
        print("No audio files found!")
        exit(1)
    
    # 运行测试
    df = test_speculative_decoding_mtp_batch(
        audio_files=audio_files,
        k_values=args.k_values,
        mtp_values=args.mtp_values,
        save_dir=args.save_dir
    )
    
    print("\n" + "="*70)
    print("Testing completed!")
    print(f"Results saved to: {args.save_dir}")
    print("="*70)