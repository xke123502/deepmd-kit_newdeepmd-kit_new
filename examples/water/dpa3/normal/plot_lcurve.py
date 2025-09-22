#!/usr/bin/env python3
"""
分析和可视化DeepMD标准训练学习曲线脚本
读取lcurve.out文件并生成多子图可视化（无MAD正则化）
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def read_lcurve_data(filename):
    """读取lcurve.out文件数据"""
    # 读取文件，跳过第二行注释，使用第一行作为列名
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 提取列名（第一行，去掉开头的#和多余空格）
    header_line = lines[0].strip().lstrip('#').strip()
    column_names = header_line.split()
    
    # 读取数据（跳过前两行）
    data = pd.read_csv(filename, sep='\s+', skiprows=2, names=column_names)
    return data

def plot_training_curves(data, save_path=None):
    """绘制标准训练曲线（无MAD正则化）"""
    # 设置字体和图像样式
    plt.rcParams['font.size'] = 10
    
    # 创建4行1列子图
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle('DeepMD Standard Training Curves Analysis', fontsize=16, fontweight='bold')
    
    # 1. 总RMSE
    ax1 = axes[0]
    ax1.plot(data['step'], data['rmse_val'], 'b-', label='Validation', linewidth=1.5)
    ax1.plot(data['step'], data['rmse_trn'], 'r-', label='Training', linewidth=1.5)
    ax1.set_ylabel('Total RMSE')
    ax1.set_title('Total RMSE')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.tick_params(labelbottom=False)  # 隐藏x轴标签
    
    # 2. 能量RMSE
    ax2 = axes[1]
    ax2.plot(data['step'], data['rmse_e_val'], 'b-', label='Validation', linewidth=1.5)
    ax2.plot(data['step'], data['rmse_e_trn'], 'r-', label='Training', linewidth=1.5)
    ax2.set_ylabel('Energy RMSE')
    ax2.set_title('Energy RMSE')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    ax2.tick_params(labelbottom=False)  # 隐藏x轴标签
    
    # 3. 力RMSE
    ax3 = axes[2]
    ax3.plot(data['step'], data['rmse_f_val'], 'b-', label='Validation', linewidth=1.5)
    ax3.plot(data['step'], data['rmse_f_trn'], 'r-', label='Training', linewidth=1.5)
    ax3.set_ylabel('Force RMSE')
    ax3.set_title('Force RMSE')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_yscale('log')
    ax3.tick_params(labelbottom=False)  # 隐藏x轴标签
    
    # 4. 学习率
    ax4 = axes[3]
    ax4.plot(data['step'], data['lr'], 'g-', label='Learning Rate', linewidth=1.5)
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_yscale('log')
    
    # 调整子图间距
    plt.tight_layout(rect=[0, 0.01, 1, 0.97], h_pad=2.0)
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    
    plt.show()
    
    return fig

def print_data_summary(data):
    """打印数据摘要信息"""
    print("=" * 60)
    print("数据摘要信息（标准训练）")
    print("=" * 60)
    print(f"训练步数范围: {data['step'].min()} - {data['step'].max()}")
    print(f"总数据点数: {len(data)}")
    print()
    
    print("最终收敛值:")
    final_row = data.iloc[-1]
    print(f"  Total RMSE (val/trn): {final_row['rmse_val']:.6f} / {final_row['rmse_trn']:.6f}")
    print(f"  Energy RMSE (val/trn): {final_row['rmse_e_val']:.6f} / {final_row['rmse_e_trn']:.6f}")
    print(f"  Force RMSE (val/trn): {final_row['rmse_f_val']:.6f} / {final_row['rmse_f_trn']:.6f}")
    print(f"  Learning Rate: {final_row['lr']:.6e}")
    print()
    
    # 检查是否有应力数据
    if 'rmse_v_val' in data.columns and not pd.isna(data['rmse_v_val']).all():
        print(f"  Stress RMSE (val/trn): {final_row['rmse_v_val']:.6f} / {final_row['rmse_v_trn']:.6f}")
    else:
        print("  Stress RMSE: 无数据 (nan)")
    
    print()
    print("训练性能分析:")
    
    # 分析收敛情况
    final_10_percent = len(data) // 10
    if final_10_percent > 0:
        final_portion = data.tail(final_10_percent)
        rmse_val_std = final_portion['rmse_val'].std()
        rmse_trn_std = final_portion['rmse_trn'].std()
        print(f"  后10%步数RMSE标准差 (val/trn): {rmse_val_std:.6f} / {rmse_trn_std:.6f}")
        
        if rmse_val_std < 0.1:
            print("  收敛状态: 良好 ✓")
        else:
            print("  收敛状态: 可能需要更多训练步数")
    
    # 过拟合检查
    overfitting_ratio = final_row['rmse_val'] / final_row['rmse_trn']
    print(f"  验证/训练RMSE比率: {overfitting_ratio:.3f}")
    if overfitting_ratio > 1.5:
        print("  过拟合警告: 验证误差明显高于训练误差 ⚠️")
    elif overfitting_ratio < 1.2:
        print("  拟合状态: 良好 ✓")
    else:
        print("  拟合状态: 轻微过拟合")
    
    print("=" * 60)

def compare_metrics(data):
    """比较不同指标的收敛表现"""
    print("\n指标改善分析:")
    print("-" * 40)
    
    initial = data.iloc[0]
    final = data.iloc[-1]
    
    metrics = [
        ('Total RMSE (val)', 'rmse_val'),
        ('Total RMSE (trn)', 'rmse_trn'),
        ('Energy RMSE (val)', 'rmse_e_val'),
        ('Energy RMSE (trn)', 'rmse_e_trn'),
        ('Force RMSE (val)', 'rmse_f_val'),
        ('Force RMSE (trn)', 'rmse_f_trn'),
    ]
    
    for name, col in metrics:
        if col in data.columns:
            improvement = (initial[col] - final[col]) / initial[col] * 100
            print(f"  {name:20s}: {improvement:6.1f}% 改善")

def main():
    """主函数"""
    # 定义输入和输出文件路径
    lcurve_file = "lcurve.out"
    output_image = "training_curves_analysis.png"
    
    # 检查文件是否存在
    if not os.path.exists(lcurve_file):
        print(f"错误: 找不到文件 {lcurve_file}")
        print("请确保在包含lcurve.out文件的目录中运行此脚本")
        return
    
    try:
        # 读取数据
        print("正在读取学习曲线数据...")
        data = read_lcurve_data(lcurve_file)
        
        # 打印数据摘要
        print_data_summary(data)
        
        # 指标改善分析
        compare_metrics(data)
        
        # 绘制图表
        print("\n正在生成可视化图表...")
        fig = plot_training_curves(data, output_image)
        
        print("分析完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请检查数据文件格式是否正确")

if __name__ == "__main__":
    main()
