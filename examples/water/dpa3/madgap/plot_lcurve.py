#!/usr/bin/env python3
"""
分析和可视化DeepMD训练学习曲线脚本
读取lcurve.out文件并生成多子图可视化
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
    """绘制训练曲线"""
    # 设置中文字体和图像样式
    plt.rcParams['font.size'] = 10
    
    # 创建6行1列子图
    fig, axes = plt.subplots(6, 1, figsize=(12, 18))
    fig.suptitle('DeepMD Training Curves Analysis', fontsize=16, fontweight='bold')
    
    # 1. MAD Gap值
    ax1 = axes[0]
    ax1.plot(data['step'], data['mad_gap_val'], 'b-', label='Validation', linewidth=1.5)
    ax1.plot(data['step'], data['mad_gap_trn'], 'r-', label='Training', linewidth=1.5)
    ax1.set_ylabel('MAD Gap Value')
    ax1.set_title('MAD Gap Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('symlog')  # 使用对称对数坐标，因为可能有负值
    ax1.tick_params(labelbottom=False)  # 隐藏x轴标签
    
    # 2. MAD正则化损失
    ax2 = axes[1]
    ax2.plot(data['step'], data['mad_reg_loss_val'], 'b-', label='Validation', linewidth=1.5)
    ax2.plot(data['step'], data['mad_reg_loss_trn'], 'r-', label='Training', linewidth=1.5)
    ax2.set_ylabel('MAD Regularization Loss')
    ax2.set_title('MAD Regularization Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_yscale('log')
    ax2.tick_params(labelbottom=False)  # 隐藏x轴标签
    
    # 3. 总RMSE
    ax3 = axes[2]
    ax3.plot(data['step'], data['rmse_val'], 'b-', label='Validation', linewidth=1.5)
    ax3.plot(data['step'], data['rmse_trn'], 'r-', label='Training', linewidth=1.5)
    ax3.set_ylabel('Total RMSE')
    ax3.set_title('Total RMSE')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_yscale('log')
    ax3.tick_params(labelbottom=False)  # 隐藏x轴标签
    
    # 4. 能量RMSE
    ax4 = axes[3]
    ax4.plot(data['step'], data['rmse_e_val'], 'b-', label='Validation', linewidth=1.5)
    ax4.plot(data['step'], data['rmse_e_trn'], 'r-', label='Training', linewidth=1.5)
    ax4.set_ylabel('Energy RMSE')
    ax4.set_title('Energy RMSE')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_yscale('log')
    ax4.tick_params(labelbottom=False)  # 隐藏x轴标签
    
    # 5. 力RMSE
    ax5 = axes[4]
    ax5.plot(data['step'], data['rmse_f_val'], 'b-', label='Validation', linewidth=1.5)
    ax5.plot(data['step'], data['rmse_f_trn'], 'r-', label='Training', linewidth=1.5)
    ax5.set_ylabel('Force RMSE')
    ax5.set_title('Force RMSE')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_yscale('log')
    ax5.tick_params(labelbottom=False)  # 隐藏x轴标签
    
    # 6. 学习率
    ax6 = axes[5]
    ax6.plot(data['step'], data['lr'], 'g-', label='Learning Rate', linewidth=1.5)
    ax6.set_xlabel('Training Steps')
    ax6.set_ylabel('Learning Rate')
    ax6.set_title('Learning Rate Schedule')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    ax6.set_yscale('log')
    
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
    print("数据摘要信息")
    print("=" * 60)
    print(f"训练步数范围: {data['step'].min()} - {data['step'].max()}")
    print(f"总数据点数: {len(data)}")
    print()
    
    print("最终收敛值:")
    final_row = data.iloc[-1]
    print(f"  MAD Gap (val/trn): {final_row['mad_gap_val']:.6f} / {final_row['mad_gap_trn']:.6f}")
    print(f"  MAD Reg Loss (val/trn): {final_row['mad_reg_loss_val']:.6f} / {final_row['mad_reg_loss_trn']:.6f}")
    print(f"  Total RMSE (val/trn): {final_row['rmse_val']:.6f} / {final_row['rmse_trn']:.6f}")
    print(f"  Energy RMSE (val/trn): {final_row['rmse_e_val']:.6f} / {final_row['rmse_e_trn']:.6f}")
    print(f"  Force RMSE (val/trn): {final_row['rmse_f_val']:.6f} / {final_row['rmse_f_trn']:.6f}")
    print(f"  Learning Rate: {final_row['lr']:.6e}")
    print()
    
    # 检查是否有应力数据
    if not pd.isna(data['rmse_v_val']).all():
        print(f"  Stress RMSE (val/trn): {final_row['rmse_v_val']:.6f} / {final_row['rmse_v_trn']:.6f}")
    else:
        print("  Stress RMSE: 无数据 (nan)")
    print("=" * 60)

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
        
        # 绘制图表
        print("正在生成可视化图表...")
        fig = plot_training_curves(data, output_image)
        
        print("分析完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请检查数据文件格式是否正确")

if __name__ == "__main__":
    main()
