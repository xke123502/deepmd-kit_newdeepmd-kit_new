# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.pt.utils.preprocess import (
    compute_exp_sw,     # 指数开关函数
    compute_smooth_weight,  # 多项式平滑权重函数
)


def _make_env_mat(
    nlist,              # 邻居列表 [bsz, natoms, nnei]
    coord,              # 坐标信息 [bsz, nall*3] 
    rcut: float,        # 截断半径
    ruct_smth: float,   # 平滑截断起始半径
    radial_only: bool = False,  # 是否只返回径向信息(1/r)
    protection: float = 0.0,    # 数值保护参数，防止除零
    use_exp_switch: bool = False,  # 是否使用指数开关函数
):
    """构建平滑环境矩阵
    
    环境矩阵是DPA模型中编码原子局部几何环境的核心数据结构。
    包含径向信息(1/r)和角度信息(xyz/r²)，用于后续的GRRG对称化操作。
    """
    # =============================================================================
    # 1. 基础数据准备
    # =============================================================================
    bsz, natoms, nnei = nlist.shape  # 批次大小, 局部原子数, 邻居数
    coord = coord.view(bsz, -1, 3)   # 重塑坐标: [bsz, nall, 3]
    nall = coord.shape[1]            # 扩展区域总原子数
    
    # 生成邻居掩码: True表示真实邻居，False表示填充邻居
    mask = nlist >= 0
    
    # 将无效邻居索引(-1)替换为边界外索引(nall)，避免索引错误
    # 注释掉的方法会导致Hessian计算中的NaN问题
    nlist = torch.where(mask, nlist, nall)
    
    # =============================================================================
    # 2. 坐标提取和相对位移计算  
    # =============================================================================
    # 中心原子坐标: [bsz, natoms, 1, 3]
    coord_l = coord[:, :natoms].view(bsz, -1, 1, 3)
    
    # 准备邻居原子索引: [bsz, natoms*nnei, 3]
    index = nlist.view(bsz, -1).unsqueeze(-1).expand(-1, -1, 3)
    
    # 坐标填充: 添加虚拟原子坐标(距离rcut之外)，防止无效索引
    coord_pad = torch.concat([coord, coord[:, -1:, :] + rcut], dim=1)
    
    # 根据邻居列表提取邻居原子坐标
    coord_r = torch.gather(coord_pad, 1, index)
    coord_r = coord_r.view(bsz, natoms, nnei, 3)  # [bsz, natoms, nnei, 3]
    
    # 计算相对位移向量: 邻居 - 中心
    diff = coord_r - coord_l # [bsz, natoms, nnei, 3] 相对位移向量
    
    # 计算原子间距离
    length = torch.linalg.norm(diff, dim=-1, keepdim=True) # [bsz, natoms, nnei, 1]
    
    # 对无效邻居的距离加1，避免影响后续计算
    length = length + ~mask.unsqueeze(-1)
    
    # =============================================================================
    # 3. 环境矩阵的核心组件计算
    # =============================================================================
    # t0: 径向特征 = 1/r (物理意义: 库仑势、范德华力等都与1/r^n相关)
    # 为什么用1/r而不是r？
    # 1. 物理合理性: 大多数原子间相互作用∝1/r^n (库仑力1/r², 范德华力1/r^6等)
    # 2. 数值特性: 1/r在近距离时数值大，远距离时趋向0，符合相互作用强度衰减
    # 3. 传统经验: DeePMD系列模型验证了1/r的有效性
    t0 = 1 / (length + protection)  # [bsz, natoms, nnei, 1] 径向项
    
    # t1: 角度特征 = xyz/r² (归一化方向向量，额外除以r提供距离权重)
    # 物理意义: 方向信息 + 距离权重，用于GRRG操作中的几何对称化
    t1 = diff / (length + protection) ** 2  # [bsz, natoms, nnei, 3] 方向项
    # =============================================================================
    # 4. 平滑截断权重计算
    # =============================================================================
    # 计算平滑权重函数: 在rcut_smth到rcut之间平滑衰减到0
    # 两种选择: 多项式平滑 或 指数平滑
    weight = (
        compute_smooth_weight(length, ruct_smth, rcut)  # 多项式平滑截断
        if not use_exp_switch
        else compute_exp_sw(length, ruct_smth, rcut)    # 指数平滑截断
    ) # [bsz, natoms, nnei, 1] 平滑权重
    
    # 应用邻居掩码: 将填充邻居的权重设为0
    weight = weight * mask.unsqueeze(-1) # [bsz, natoms, nnei, 1]
    
    # =============================================================================
    # 5. 环境矩阵组装
    # =============================================================================
    if radial_only:
        # 仅径向模式: 只返回1/r信息，用于某些简化场景
        env_mat = t0 * weight  # [bsz, natoms, nnei, 1]
    else:
        # 完整模式: 返回[1/r, x/r², y/r², z/r²]
        # 这是DPA3的标准环境矩阵格式
        env_mat = torch.cat([t0, t1], dim=-1) * weight  # [bsz, natoms, nnei, 4]
    
    # 返回值说明:
    # env_mat: 环境矩阵 [bsz, natoms, nnei, 4或1] 
    # diff: 相对位移向量(应用掩码) [bsz, natoms, nnei, 3]
    # weight: 平滑权重 [bsz, natoms, nnei, 1]
    return env_mat, diff * mask.unsqueeze(-1), weight


def prod_env_mat(
    extended_coord,     # 扩展坐标 [nframes, nall*3]
    nlist,             # 邻居列表 [nframes, nloc, nnei]  
    atype,             # 原子类型 [nframes, nloc]
    mean,              # 统计均值 [ntypes, nnei, 4]
    stddev,            # 统计标准差 [ntypes, nnei, 4] 
    rcut: float,       # 截断半径
    rcut_smth: float,  # 平滑截断起始半径
    radial_only: bool = False,   # 是否仅径向模式
    protection: float = 0.0,     # 数值保护参数
    use_exp_switch: bool = False, # 是否使用指数开关函数
):
    """生成标准化的环境矩阵
    
    这是环境矩阵生成的高级接口，在_make_env_mat基础上添加了统计标准化。
    标准化是为了让不同原子类型的特征具有相似的数值范围，提高训练稳定性。
    
    Args:
    - extended_coord: 扩展区域原子坐标 [nframes, nall*3]
    - nlist: 邻居列表 [nframes, nloc, nnei]  
    - atype: 局部原子类型索引 [nframes, nloc]
    - mean: 各原子类型环境矩阵的统计均值 [ntypes, nnei, 4]
    - stddev: 各原子类型环境矩阵的统计标准差 [ntypes, nnei, 4]
    - rcut: 截断半径
    - rcut_smth: 平滑截断起始半径  
    - radial_only: 是否只返回径向信息(1/r)
    - protection: 数值保护参数，防止除零错误
    - use_exp_switch: 是否使用指数开关函数

    Returns:
    -------
    - env_mat_se_a: 标准化后的环境矩阵 [nframes, nloc, nnei, 4]
    - diff: 相对位移向量 [nframes, nloc, nnei, 3] (用于h2，GRRG操作的几何输入)
    - switch: 平滑权重函数 [nframes, nloc, nnei, 1] (用于sw，开关函数)
    
    关于r vs 1/r的选择:
    - 这里固定使用1/r，这是DPA系列模型的设计选择
    - 如果要使用r，需要在repflows.py中设置edge_init_use_dist=True
    - 那样会在后续流程中用torch.linalg.norm(diff)替换1/r作为边输入
    """
    # 调用底层环境矩阵构建函数
    _env_mat_se_a, diff, switch = _make_env_mat(
        nlist,
        extended_coord,
        rcut,
        rcut_smth,
        radial_only,
        protection=protection,
        use_exp_switch=use_exp_switch,
    )  # _env_mat_se_a: [nframes, nloc, nnei, 4], diff: [nframes, nloc, nnei, 3]
    
    # =============================================================================
    # 统计标准化: 基于原子类型的均值和标准差进行归一化
    # =============================================================================
    # 根据原子类型索引提取对应的统计量
    t_avg = mean[atype]    # [nframes, nloc, nnei, 4] 原子类型对应的均值
    t_std = stddev[atype]  # [nframes, nloc, nnei, 4] 原子类型对应的标准差
    
    # Z-score标准化: (x - μ) / σ
    # 这确保了不同原子类型的环境特征具有相似的数值分布
    env_mat_se_a = (_env_mat_se_a - t_avg) / t_std
    
    # 返回值详解:
    # env_mat_se_a: 标准化环境矩阵，用于后续的边嵌入计算
    # diff: 原始相对位移向量，用作h2(旋转等变特征)  
    # switch: 平滑权重，用作sw(开关函数)
    return env_mat_se_a, diff, switch
