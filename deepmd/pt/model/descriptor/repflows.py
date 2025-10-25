# SPDX-License-Identifier: LGPL-3.0-or-later
"""
RepFlow描述符块实现模块

本模块实现了DescrptBlockRepflows类，是DPA3描述符的核心组件，负责：
1. 管理多个RepFlow层的堆叠
2. 处理环境矩阵的构建和统计
3. 协调节点、边、角度的嵌入计算
4. 实现动态邻居选择和图索引管理

RepFlow描述符块是DPA3模型的主要描述符实现，通过多层RepFlow层
实现复杂的图神经网络消息传递机制。

修改记录（Modified in 2025-10-14）:
- 实现EGNN风格的坐标更新功能
- 每层RepFlow后更新extended_coord并重新计算几何量(diff,h2,sw,a_sw)（905-960行）
- 拓扑量(nlist,nlist_mask,a_nlist,a_nlist_mask,edge_index,angle_index)保持不变
- 仅支持非并行模式的坐标更新，并行模式暂不支持
"""
from errno import ESTALE
from typing import (
    Callable,
    Optional,
    Union,
)

import torch

from deepmd.dpmodel.utils.seed import (
    child_seed,  # 子种子生成器
)
from deepmd.pt.model.descriptor.descriptor import (
    DescriptorBlock,  # 描述符块基类
)
from deepmd.pt.model.descriptor.env_mat import (
    prod_env_mat,  # 环境矩阵生成函数
)
from deepmd.pt.model.network.mlp import (
    MLPLayer,  # MLP层
)
from deepmd.pt.model.network.utils import (
    get_graph_index,  # 图索引生成函数
)
from deepmd.pt.utils import (
    env,  # 环境配置
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,  # 精度字典
)
from deepmd.pt.utils.env_mat_stat import (
    EnvMatStatSe,  # 环境矩阵统计
)
from deepmd.pt.utils.exclude_mask import (
    PairExcludeMask,  # 原子对排除掩码
)
from deepmd.pt.utils.spin import (
    concat_switch_virtual,  # 虚拟原子拼接
)
from deepmd.pt.utils.utils import (
    ActivationFn,  # 激活函数
)
from deepmd.utils.env_mat_stat import (
    StatItem,  # 统计项
)
from deepmd.utils.path import (
    DPPath,  # 路径处理
)

from .repflow_layer import (
    RepFlowLayer,  # RepFlow层
)

if not hasattr(torch.ops.deepmd, "border_op"):

    def border_op(
        argument0,
        argument1,
        argument2,
        argument3,
        argument4,
        argument5,
        argument6,
        argument7,
        argument8,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "border_op is not available since customized PyTorch OP library is not built when freezing the model. "
            "See documentation for DPA3 for details."
        )

    # Note: this hack cannot actually save a model that can be run using LAMMPS.
    torch.ops.deepmd.border_op = border_op


import math

class BesselBasisDPA3(torch.nn.Module):
    """
    Bessel基函数，与SevenNet完全一致
    f : (*, 1) -> (*, bessel_basis_num)
    """
    def __init__(
        self,
        cutoff_length: float,
        bessel_basis_num: int = 8,
        trainable_coeff: bool = True,
    ):
        super().__init__()
        self.num_basis = bessel_basis_num
        self.prefactor = 2.0 / cutoff_length
        self.coeffs = torch.FloatTensor([
            n * math.pi / cutoff_length for n in range(1, bessel_basis_num + 1)
        ])
        if trainable_coeff:
            self.coeffs = torch.nn.Parameter(self.coeffs)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # r: [nb, nloc, nnei, 1]
        #ur = r.unsqueeze(-1)  # [nb, nloc, nnei, 1, 1] --- 这行是sevenent原本的我们注释掉了。
        #coeffs = self.coeffs.view(1, 1, 1, 1, -1)  # [1, 1, 1, 1, bessel_num]
        #coeffs = self.coeffs.view(1, 1, 1, -1)  # [1, 1, 1, 1, bessel_num]
        #print("self.coeffs", self.coeffs.shape) # self.coeffs torch.Size([8])
        #print(self.coeffs)
        # 计算 sin(n*π*r/r_cut) / r
        #bessel_vals = self.prefactor * torch.sin(coeffs * ur) / (ur + 1e-8)
        #print("bessel_vals.squenze(-2)", bessel_vals.squeeze(-2)) # bessel_vals torch.Size([1, 1, 1, 1, 8])
        #return bessel_vals.squeeze(-2)  # [nb, nloc, nnei, bessel_num]
        #print("result", self.prefactor * torch.sin(self.coeffs * r) / (r + 1e-8))
        return self.prefactor * torch.sin(self.coeffs * r) / (r + 1e-8)

class PolynomialCutoffDPA3(torch.nn.Module):
    """
    多项式截断函数，与SevenNet一致  
    f : (*, 1) -> (*, 1)
    """
    def __init__(self, cutoff_length: float, poly_cut_p_value: int = 6):
        super().__init__()
        p = poly_cut_p_value
        self.cutoff_length = cutoff_length
        self.p = poly_cut_p_value
        self.coeff_p0 = (p + 1.0) * (p + 2.0) / 2.0
        self.coeff_p1 = p * (p + 2.0)  
        self.coeff_p2 = p * (p + 1.0) / 2.0

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r_norm = r / self.cutoff_length
        return (
            1
            - self.coeff_p0 * torch.pow(r_norm, self.p)
            + self.coeff_p1 * torch.pow(r_norm, self.p + 1.0) 
            - self.coeff_p2 * torch.pow(r_norm, self.p + 2.0)
        )

@DescriptorBlock.register("se_repflow")
class DescrptBlockRepflows(DescriptorBlock):
    """RepFlow描述符块
    
    这是DPA3模型的核心描述符块，实现了基于RepFlow的图神经网络描述符。
    通过堆叠多个RepFlow层，实现复杂的消息传递机制，用于建模原子间的相互作用。

    Parameters
    ----------
    n_dim : int, optional
        节点表征的维度，默认为128
    e_dim : int, optional
        边表征的维度，默认为64
    a_dim : int, optional
        角度表征的维度，默认为64
    nlayers : int, optional
        RepFlow层的数量，默认为6
    e_rcut : float, optional
        边的截断半径
    e_rcut_smth : float, optional
        边平滑截断的起始位置，例如1/r项从rcut到rcut_smth平滑
    e_sel : int, optional
        边邻居的最大选择数量
    a_rcut : float, optional
        角度的截断半径
    a_rcut_smth : float, optional
        角度平滑截断的起始位置
    a_sel : int, optional
        角度邻居的最大选择数量
    a_compress_rate : int, optional
        角度消息的压缩率，默认为0表示无压缩。
        如果提供非零整数c，节点和边维度将在角度消息中分别压缩到a_dim/c和a_dim/2c
    a_compress_e_rate : int, optional
        角度消息压缩中边的额外压缩率，默认为1。
        当使用角度消息压缩时，边维度将压缩到(c_e * a_dim / 2c)
    a_compress_use_split : bool, optional
        在角度消息压缩期间是否分割第一个子向量而不是线性映射
        The default value is False.
    n_multi_edge_message : int, optional
        The head number of multiple edge messages to update node feature.
        Default is 1, indicating one head edge message.
    axis_neuron : int, optional
        The number of dimension of submatrix in the symmetrization ops.
    update_angle : bool, optional
        Where to update the angle rep. If not, only node and edge rep will be used.
    update_style : str, optional
        Style to update a representation.
        Supported options are:
        -'res_avg': Updates a rep `u` with: u = 1/\\sqrt{n+1} (u + u_1 + u_2 + ... + u_n)
        -'res_incr': Updates a rep `u` with: u = u + 1/\\sqrt{n} (u_1 + u_2 + ... + u_n)
        -'res_residual': Updates a rep `u` with: u = u + (r1*u_1 + r2*u_2 + ... + r3*u_n)
        where `r1`, `r2` ... `r3` are residual weights defined by `update_residual`
        and `update_residual_init`.
    update_residual : float, optional
        When update using residual mode, the initial std of residual vector weights.
    update_residual_init : str, optional
        When update using residual mode, the initialization mode of residual vector weights.
    fix_stat_std : float, optional
        If non-zero (default is 0.3), use this constant as the normalization standard deviation
        instead of computing it from data statistics.
    smooth_edge_update : bool, optional
        Whether to make edge update smooth.
        If True, the edge update from angle message will not use self as padding.
    edge_init_use_dist : bool, optional
        Whether to use direct distance r to initialize the edge features instead of 1/r.
        Note that when using this option, the activation function will not be used when initializing edge features.
    use_exp_switch : bool, optional
        Whether to use an exponential switch function instead of a polynomial one in the neighbor update.
        The exponential switch function ensures neighbor contributions smoothly diminish as the interatomic distance
        `r` approaches the cutoff radius `rcut`. Specifically, the function is defined as:
        s(r) = \\exp(-\\exp(20 * (r - rcut_smth) / rcut_smth)) for 0 < r \\leq rcut, and s(r) = 0 for r > rcut.
        Here, `rcut_smth` is an adjustable smoothing factor and `rcut_smth` should be chosen carefully
        according to `rcut`, ensuring s(r) approaches zero smoothly at the cutoff.
        Typical recommended values are `rcut_smth` = 5.3 for `rcut` = 6.0, and 3.5 for `rcut` = 4.0.
    use_dynamic_sel : bool, optional
        Whether to dynamically select neighbors within the cutoff radius.
        If True, the exact number of neighbors within the cutoff radius is used
        without padding to a fixed selection numbers.
        When enabled, users can safely set larger values for `e_sel` or `a_sel` (e.g., 1200 or 300, respectively)
        to guarantee capturing all neighbors within the cutoff radius.
        Note that when using dynamic selection, the `smooth_edge_update` must be True.
    sel_reduce_factor : float, optional
        Reduction factor applied to neighbor-scale normalization when `use_dynamic_sel` is True.
        In the dynamic selection case, neighbor-scale normalization will use `e_sel / sel_reduce_factor`
        or `a_sel / sel_reduce_factor` instead of the raw `e_sel` or `a_sel` values,
        accommodating larger selection numbers.
    use_loc_mapping : bool, Optional
        Whether to use local atom index mapping in training or non-parallel inference.
        When True, local indexing and mapping are applied to neighbor lists and embeddings during descriptor computation.
    optim_update : bool, optional
        Whether to enable the optimized update method.
        Uses a more efficient process when enabled. Defaults to True
    ntypes : int
        Number of element types
    activation_function : str, optional
        The activation function in the embedding net.
    set_davg_zero : bool, optional
        Set the normalization average to zero.
    precision : str, optional
        The precision of the embedding net parameters.
    exclude_types : list[list[int]], optional
        The excluded pairs of types which have no interaction with each other.
        For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    env_protection : float, optional
        Protection parameter to prevent division by zero errors during environment matrix calculations.
        For example, when using paddings, there may be zero distances of neighbors, which may make division by zero error during environment matrix calculations without protection.
    seed : int, optional
        Random seed for parameter initialization.
    """

    def __init__(
        self,
        e_rcut,
        e_rcut_smth,
        e_sel: int,
        a_rcut,
        a_rcut_smth,
        a_sel: int,
        ntypes: int,
        nlayers: int = 6,
        n_dim: int = 128,
        e_dim: int = 64,
        a_dim: int = 64,
        a_compress_rate: int = 0,
        a_compress_e_rate: int = 1,
        a_compress_use_split: bool = False,
        n_multi_edge_message: int = 1,
        axis_neuron: int = 4,
        update_angle: bool = True,
        activation_function: str = "silu",
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        set_davg_zero: bool = True,
        exclude_types: list[tuple[int, int]] = [],
        env_protection: float = 0.0,
        precision: str = "float64",
        fix_stat_std: float = 0.3,
        smooth_edge_update: bool = False,
        edge_init_use_dist: bool = False,
        use_exp_switch: bool = False,
        use_dynamic_sel: bool = False,
        sel_reduce_factor: float = 10.0,
        use_loc_mapping: bool = True,
        optim_update: bool = True,
        seed: Optional[Union[int, list[int]]] = None,
        init: str = "default",  # new added in 2025 0923 - MLP initialization method
        edge_use_bessel: bool = False,
        bessel_basis_num: int = 8, 
        bessel_trainable: bool = True,
        update_coord: bool = False,  # new added in 2025 1012 - 是否更新坐标(类似EGNN)
        normalize_coord: bool = False,  # new added in 2025 1012 - 是否归一化坐标差(类似EGNN)
        coords_agg: str = "mean",  # new added in 2025 1012 - 坐标聚合方式
    ) -> None:
        super().__init__()
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.e_sel = e_sel
        self.a_rcut = float(a_rcut)
        self.a_rcut_smth = float(a_rcut_smth)
        self.a_sel = a_sel
        self.ntypes = ntypes
        self.nlayers = nlayers
        # for other common desciptor method
        sel = [e_sel] if isinstance(e_sel, int) else e_sel
        self.nnei = sum(sel)
        self.ndescrpt = self.nnei * 4  # use full descriptor.
        assert len(sel) == 1
        self.sel = sel
        self.rcut = e_rcut
        self.rcut_smth = e_rcut_smth
        self.sec = self.sel
        self.split_sel = self.sel
        self.a_compress_rate = a_compress_rate
        self.a_compress_e_rate = a_compress_e_rate
        self.n_multi_edge_message = n_multi_edge_message
        self.axis_neuron = axis_neuron
        self.set_davg_zero = set_davg_zero
        self.fix_stat_std = fix_stat_std
        self.set_stddev_constant = fix_stat_std != 0.0
        self.a_compress_use_split = a_compress_use_split
        self.use_loc_mapping = use_loc_mapping
        self.optim_update = optim_update
        self.smooth_edge_update = smooth_edge_update
        self.edge_init_use_dist = edge_init_use_dist
        self.use_exp_switch = use_exp_switch
        self.use_dynamic_sel = use_dynamic_sel
        self.sel_reduce_factor = sel_reduce_factor
        if self.use_dynamic_sel and not self.smooth_edge_update:
            raise NotImplementedError(
                "smooth_edge_update must be True when use_dynamic_sel is True!"
            )
        if self.sel_reduce_factor <= 0:
            raise ValueError(
                f"`sel_reduce_factor` must be > 0, got {self.sel_reduce_factor}"
            )

        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.update_angle = update_angle

        self.activation_function = activation_function
        self.update_style = update_style
        self.update_residual = update_residual
        self.update_residual_init = update_residual_init
        self.act = ActivationFn(activation_function)
        self.prec = PRECISION_DICT[precision]

        # order matters, placed after the assignment of self.ntypes
        self.reinit_exclude(exclude_types)
        self.env_protection = env_protection
        self.precision = precision
        self.epsilon = 1e-4
        self.seed = seed
        self.edge_use_bessel = edge_use_bessel
        self.bessel_basis_num = bessel_basis_num
        self.bessel_trainable = bessel_trainable
        # new added in 2025 1012 - 保存坐标更新相关参数
        self.update_coord = update_coord
        self.normalize_coord = normalize_coord
        self.coords_agg = coords_agg

        if edge_use_bessel:
            # 启用Bessel展开时，强制使用距离r而不是1/r
            self.edge_init_use_dist = True
            
            # 初始化Bessel基函数和截断函数
            self.bessel_basis = BesselBasisDPA3(
                cutoff_length=self.e_rcut,
                bessel_basis_num=bessel_basis_num,
                trainable_coeff=bessel_trainable
            )
            self.poly_cutoff = PolynomialCutoffDPA3(self.e_rcut)
            
            # 修改edge_embd的输入维度：bessel_num → e_dim
            self.edge_embd = MLPLayer(
                bessel_basis_num,
                self.e_dim, 
                precision=precision,
                seed=child_seed(seed, 3),
            )
        else:
            # added in 2025 10 01, edge basis - JIT兼容性：即使不使用也要初始化为None
            self.bessel_basis = None
            self.poly_cutoff = None
            self.edge_embd = MLPLayer(
                1, self.e_dim, precision=precision, init=init, seed=child_seed(seed, 0)  # new added in 2025 0923 - Pass init method
            ) # 创建边嵌入, 从一个标量到e_dim维度的嵌入

        # 创建角度嵌入---这里还没有展开angle, 不要缩进了。
        self.angle_embd = MLPLayer(
                1, self.a_dim, precision=precision, bias=False, init=init, seed=child_seed(seed, 1)  # new added in 2025 0923 - Pass init method
            ) # 创建角度嵌入, 从一个标量到a_dim维度的嵌入
        layers = []
        for ii in range(nlayers):
            layers.append(
                RepFlowLayer(
                    e_rcut=self.e_rcut,
                    e_rcut_smth=self.e_rcut_smth,
                    e_sel=self.sel,
                    a_rcut=self.a_rcut,
                    a_rcut_smth=self.a_rcut_smth,
                    a_sel=self.a_sel,
                    ntypes=self.ntypes,
                    n_dim=self.n_dim,
                    e_dim=self.e_dim,
                    a_dim=self.a_dim,
                    a_compress_rate=self.a_compress_rate,
                    a_compress_use_split=self.a_compress_use_split,
                    a_compress_e_rate=self.a_compress_e_rate,
                    n_multi_edge_message=self.n_multi_edge_message,
                    axis_neuron=self.axis_neuron,
                    update_angle=self.update_angle,
                    activation_function=self.activation_function,
                    update_style=self.update_style,
                    update_residual=self.update_residual,
                    update_residual_init=self.update_residual_init,
                    precision=precision,
                    optim_update=self.optim_update,
                    use_dynamic_sel=self.use_dynamic_sel,
                    sel_reduce_factor=self.sel_reduce_factor,
                    smooth_edge_update=self.smooth_edge_update,
                    init=init,  # new added in 2025 0923 - Pass MLP initialization method to RepFlowLayer
                    update_coord=self.update_coord,  # new added in 2025 1012 - Pass coordinate update flag
                    normalize_coord=self.normalize_coord,  # new added in 2025 1012 - Pass normalization flag
                    coords_agg=self.coords_agg,  # new added in 2025 1012 - Pass aggregation method
                    seed=child_seed(child_seed(seed, 1), ii),
                )
            ) # 创建RepFlow层
        self.layers = torch.nn.ModuleList(layers)

        wanted_shape = (self.ntypes, self.nnei, 4)
        mean = torch.zeros(wanted_shape, dtype=self.prec, device=env.DEVICE)
        stddev = torch.ones(wanted_shape, dtype=self.prec, device=env.DEVICE)
        if self.set_stddev_constant:
            stddev = stddev * self.fix_stat_std
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)
        self.stats = None

    def get_rcut(self) -> float:
        """Returns the cut-off radius."""
        return self.e_rcut

    def get_rcut_smth(self) -> float:
        """Returns the radius where the neighbor information starts to smoothly decay to 0."""
        return self.e_rcut_smth

    def get_nsel(self) -> int:
        """Returns the number of selected atoms in the cut-off radius."""
        return sum(self.sel)

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.sel

    def get_ntypes(self) -> int:
        """Returns the number of element types."""
        return self.ntypes

    def get_dim_out(self) -> int:
        """Returns the output dimension."""
        return self.dim_out

    def get_dim_in(self) -> int:
        """Returns the input dimension."""
        return self.dim_in

    def get_dim_emb(self) -> int:
        """Returns the embedding dimension e_dim."""
        return self.e_dim

    def __setitem__(self, key, value) -> None:
        if key in ("avg", "data_avg", "davg"):
            self.mean = value
        elif key in ("std", "data_std", "dstd"):
            self.stddev = value
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key in ("avg", "data_avg", "davg"):
            return self.mean
        elif key in ("std", "data_std", "dstd"):
            return self.stddev
        else:
            raise KeyError(key)

    def mixed_types(self) -> bool:
        """If true, the descriptor
        1. assumes total number of atoms aligned across frames;
        2. requires a neighbor list that does not distinguish different atomic types.

        If false, the descriptor
        1. assumes total number of atoms of each atom type aligned across frames;
        2. requires a neighbor list that distinguishes different atomic types.

        """
        return True

    def get_env_protection(self) -> float:
        """Returns the protection of building environment matrix."""
        return self.env_protection

    @property
    def dim_out(self):
        """Returns the output dimension of this descriptor."""
        return self.n_dim

    @property
    def dim_in(self):
        """Returns the atomic input dimension of this descriptor."""
        return self.n_dim

    @property
    def dim_emb(self):
        """Returns the embedding dimension e_dim."""
        return self.get_dim_emb()

    def reinit_exclude(
        self,
        exclude_types: list[tuple[int, int]] = [],
    ) -> None:
        self.exclude_types = exclude_types
        self.emask = PairExcludeMask(self.ntypes, exclude_types=exclude_types)

    def forward(
        self,
        nlist: torch.Tensor,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        extended_atype_embd: Optional[torch.Tensor] = None,
        mapping: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ):
        """RepFlow描述符块的前向传播函数
        
        这是RepFlow描述符块的核心函数，负责：
        1. 构建环境矩阵和邻居信息
        2. 计算初始的边和角度嵌入
        3. 通过多层RepFlow层进行消息传递
        4. 生成最终的节点、边、角度表征

        Parameters
        ----------
        nlist : torch.Tensor
            邻居列表，形状为 nf x nloc x nnei
        extended_coord : torch.Tensor
            扩展坐标，形状为 nf x (nall*3)
        extended_atype : torch.Tensor
            扩展原子类型，形状为 nf x nall
        extended_atype_embd : Optional[torch.Tensor]
            扩展原子类型嵌入，形状为 nf x nall x n_dim
        mapping : Optional[torch.Tensor]
            索引映射，将扩展区域索引映射到局部区域
        comm_dict : Optional[dict[str, torch.Tensor]]
            并行推理所需的通信数据

        Returns
        -------
        node_ebd : torch.Tensor
            节点嵌入，形状为 nf x nloc x n_dim
        edge_ebd : torch.Tensor
            边嵌入，形状为 nf x nloc x nnei x e_dim
        h2 : torch.Tensor
            旋转等变表征，形状为 nf x nloc x nnei x 3
        rot_mat : torch.Tensor
            旋转矩阵，形状为 nf x nloc x e_dim x 3
        sw : torch.Tensor
            开关函数，形状为 nf x nloc x nnei
        """
        # =============================================================================
        # 1. 输入预处理和模式检测
        # =============================================================================
        parallel_mode = comm_dict is not None
        if not parallel_mode:
            assert mapping is not None
        nframes, nloc, nnei = nlist.shape
        #print("extended_coord", extended_coord.shape) # extended_coord torch.Size([1, 5184, 3])
        #print("extended_coord", extended_coord[0]) # extended_coord torch.Size([1, 5184, 3])
        nall = extended_coord.view(nframes, -1).shape[1] // 3
        atype = extended_atype[:, :nloc]
        #print("nall", nall) # nall 5184
        #print("extend_ coord", extended_coord.shape) # extend_ coord torch.Size([1, 5184, 3])
        #print("extend_ coord", extended_coord[0]) # extend_ coord torch.Size([1, 5184, 3])
        # =============================================================================
        # 2. 处理排除的原子对
        # =============================================================================
        # 应用排除掩码：将排除的原子对设为-1
        exclude_mask = self.emask(nlist, extended_atype)
        nlist = torch.where(exclude_mask != 0, nlist, -1)
        #print("nlist", nlist.shape) # nlist torch.Size([1, 192, 120])
        #print("nlist", nlist[0]) # nlist torch.Size([1, 192, 120])
        #print("atype", atype.shape) # atype torch.Size([1, 192])
        #print("atype", atype[0]) # atype torch.Size([1, 192])
        #print("exclude_mask", exclude_mask.shape) # exclude_mask torch.Size([1, 192, 120])
        #print("exclude_mask", exclude_mask[0]) # exclude_mask torch.Size([1, 192, 120])

        # =============================================================================
        # 3. 构建环境矩阵
        # =============================================================================
        # 生成环境矩阵：包含距离矩阵、方向向量、开关函数
        # 形状: nb x nloc x nnei x 4, nb x nloc x nnei x 3, nb x nloc x nnei x 1
        dmatrix, diff, sw = prod_env_mat(
            extended_coord,
            nlist,
            atype,
            self.mean,
            self.stddev,
            self.e_rcut,
            self.e_rcut_smth,
            protection=self.env_protection,
            use_exp_switch=self.use_exp_switch,
        ) # 这里有一个diff--> 方向向量
        #print("extended_coord", extended_coord.shape) # extended_coord torch.Size([1, 5184, 3])
        #print("diff_pre", diff.shape)
        #print("diff", diff)
        # 处理邻居列表掩码和开关函数
        nlist_mask = nlist != -1 # nlist_mask 是邻居列表掩码，真实邻居为1，否则为0， -1是填充的邻居
        sw = torch.squeeze(sw, -1)
        # 在截断半径之外，开关函数应该为0.0
        sw = sw.masked_fill(~nlist_mask, 0.0)

        # =============================================================================
        # 4. 构建角度环境矩阵
        # =============================================================================
        # 获取角度邻居列表（可能比边邻居列表小）
        a_dist_mask = (torch.linalg.norm(diff, dim=-1) < self.a_rcut)[
            :, :, : self.a_sel
        ]
        a_nlist = nlist[:, :, : self.a_sel]
        a_nlist = torch.where(a_dist_mask, a_nlist, -1)
        
        # 为角度计算生成环境矩阵：包含距离矩阵、方向向量、开关函数
        _, a_diff, a_sw = prod_env_mat(
            extended_coord,
            a_nlist,
            atype,
            self.mean[:, : self.a_sel],
            self.stddev[:, : self.a_sel],
            self.a_rcut,
            self.a_rcut_smth,
            protection=self.env_protection,
            use_exp_switch=self.use_exp_switch,
        )
        
        # 处理角度邻居列表掩码和开关函数
        a_nlist_mask = a_nlist != -1
        a_sw = torch.squeeze(a_sw, -1)
        # 在截断半径之外，开关函数应该为0.0
        a_sw = a_sw.masked_fill(~a_nlist_mask, 0.0)
        
        # 将所有填充位置设为索引0
        # 邻居是否真实由nlist_mask指示
        nlist[nlist == -1] = 0
        a_nlist[a_nlist == -1] = 0

        # =============================================================================
        # 5. 获取节点嵌入
        # =============================================================================
        # 从扩展原子类型嵌入中提取局部原子嵌入
        # 形状: [nframes, nloc, tebd_dim]
        assert extended_atype_embd is not None
        atype_embd = extended_atype_embd[:, :nloc, :]
        assert list(atype_embd.shape) == [nframes, nloc, self.n_dim]
        assert isinstance(atype_embd, torch.Tensor)  # 用于jit编译
        node_ebd = self.act(atype_embd)  # 应用激活函数
        n_dim = node_ebd.shape[-1]

        # =============================================================================
        # 6. 获取边和角度嵌入输入
        # =============================================================================
        # 从环境矩阵中分离边输入和旋转等变表征
        # 形状: nb x nloc x nnei x 1,  nb x nloc x nnei x 3
        edge_input, h2 = torch.split(dmatrix, [1, 3], dim=-1)
        # print("edge_input", edge_input.shape) # edge_input torch.Size([1, 192, 120, 1])
            # 如果使用直接距离初始化边特征， nb = nframe
        if self.edge_init_use_dist:
            # 形状: nb x nloc x nnei x 1
            edge_input = torch.linalg.norm(diff, dim=-1, keepdim=True)
        # 如果使用Bessel基函数展开，使用欧式距离
        if self.edge_use_bessel:
            edge_input = torch.linalg.norm(diff, dim=-1, keepdim=True)
        # 计算角度输入：归一化的方向向量
        # 形状: nf x nloc x a_nnei x 3
        normalized_diff_i = a_diff / (
            torch.linalg.norm(a_diff, dim=-1, keepdim=True) + 1e-6
        )
        # 形状: nf x nloc x 3 x a_nnei
        normalized_diff_j = torch.transpose(normalized_diff_i, 2, 3)
        # 计算角度输入：原子对之间的余弦值
        # 形状: nf x nloc x a_nnei x a_nnei
        # 1 - 1e-6 用于torch.acos的数值稳定性
        cosine_ij = torch.matmul(normalized_diff_i, normalized_diff_j) * (1 - 1e-6)
        angle_input = cosine_ij.unsqueeze(-1) / (torch.pi**0.5)

        # =============================================================================
        # 7. 处理索引映射和动态选择
        # =============================================================================
        if not parallel_mode and self.use_loc_mapping:
            assert mapping is not None
            # 将邻居列表从nall索引转换为nloc索引
            nlist = torch.gather(
                mapping,
                1,
                index=nlist.reshape(nframes, -1),
            ).reshape(nlist.shape)
            
        if self.use_dynamic_sel:
            # 获取图索引：用于动态邻居选择
            edge_index, angle_index = get_graph_index(
                nlist,
                nlist_mask,
                a_nlist_mask,
                nall,
                use_loc_mapping=self.use_loc_mapping,
            )
            
            # 扁平化所有张量以适应动态选择
            # 形状: n_edge x 1
            edge_input = edge_input[nlist_mask] #
            # 形状: n_edge x 3
            h2 = h2[nlist_mask]
            # 形状: n_edge x 1
            sw = sw[nlist_mask]
            
            # 计算角度掩码：两个邻居都有效
            # 形状: nb x nloc x a_nnei x a_nnei
            a_nlist_mask = a_nlist_mask[:, :, :, None] & a_nlist_mask[:, :, None, :]
            # 形状: n_angle x 1
            angle_input = angle_input[a_nlist_mask]
            # 形状: n_angle x 1
            a_sw = (a_sw[:, :, :, None] * a_sw[:, :, None, :])[a_nlist_mask]
        else:
            # 避免jit断言错误
            edge_index = angle_index = torch.zeros(
                [1, 3], device=nlist.device, dtype=nlist.dtype
            )
            
        # =============================================================================
        # 8. 计算边和角度嵌入
        # =============================================================================
        # 计算边嵌入
        # 形状: nb x nloc x nnei x e_dim [OR] n_edge x e_dim
        if self.edge_use_bessel:
            # 使用Bessel基函数展开（类似SevenNet）
            
            # JIT兼容性：确保bessel_basis和poly_cutoff不为None
            assert self.bessel_basis is not None, "bessel_basis should not be None when edge_use_bessel=True"
            assert self.poly_cutoff is not None, "poly_cutoff should not be None when edge_use_bessel=True"
            
            # 强制使用欧式距离r
            #edge_length = torch.linalg.norm(diff, dim=-1, keepdim=True)  # [nb, nloc, nnei, 1]
            edge_length = edge_input
            #print(edge_length.shape)  # torch.Size([1, 192, 120, 1])
            # Bessel基函数展开 
            bessel_features = self.bessel_basis(edge_length)    # [nb, nloc, nnei, bessel_num] torch.Size([1, 192, 120, 1, 8])
            #print(bessel_features.shape)
            # 多项式截断函数
            cutoff_weight = self.poly_cutoff(edge_length)       # [nb, nloc, nnei, 1] torch.Size([1, 192, 120, 1])
            #print(cutoff_weight.shape)
            # 组合特征：Bessel × 截断函数
            edge_features = bessel_features * cutoff_weight #.unsqueeze(-1)    # [nb, nloc, nnei, bessel_num]
            #print(edge_features.shape)
            # MLP处理：bessel_num → e_dim
            edge_ebd = self.edge_embd(edge_features)            # [nb, nloc, nnei, e_dim]

        else:
            if not self.edge_init_use_dist:
                edge_ebd = self.act(self.edge_embd(edge_input))  # 应用激活函数
            else:
                edge_ebd = self.edge_embd(edge_input)  # 直接使用距离，不应用激活函数
        
        # 【调试】检查edge_ebd是否有NaN
        #print("=== edge_ebd 初始化后检查 ===")
        #print("edge_input has nan:", torch.isnan(edge_input).any().item())
        #print("edge_input sample:", edge_input.shape)
        #print("edge_ebd has nan:", torch.isnan(edge_ebd).any().item())
        #print("edge_ebd sample:", edge_ebd.shape)
            
        # 计算角度嵌入
        # 形状: nf x nloc x a_nnei x a_nnei x a_dim [OR] n_angle x a_dim
        angle_ebd = self.angle_embd(angle_input)

        # =============================================================================
        # 9. 通过多层RepFlow层进行消息传递
        # =============================================================================
        # 准备映射张量（非并行模式）
        # 初始化mapping_orig，确保所有分支都有定义（TorchScript要求）
        mapping_orig: torch.Tensor = torch.zeros([1, 1], dtype=torch.int64, device=nlist.device)
        if not parallel_mode:
            assert mapping is not None
            # Modified in 2025-10-14 - 准备mapping_coord用于坐标扩展
            mapping_orig = mapping.view(nframes, nall)
            mapping = mapping_orig.unsqueeze(-1).expand(-1, -1, self.n_dim)
            #mapping_coord = mapping_orig.unsqueeze(-1).expand(-1, -1, 3)
        # extended_coord_new = extended_coord.detach().clone()
        
        extended_coord_new = extended_coord.clone()

        # 遍历所有RepFlow层进行消息传递 --- 循环
        for idx, ll in enumerate(self.layers):
            # n_prev, e_prev, a_prev = node_ebd, edge_ebd, angle_ebd
            # 准备扩展节点嵌入
            # node_ebd:     nb x nloc x n_dim
            # node_ebd_ext: nb x nall x n_dim [OR] nb x nloc x n_dim when not parallel_mode
            #print("parallel_mode", parallel_mode) # parallel_mode False
            #print("use_loc_mapping", self.use_loc_mapping) # use_loc_mapping True
            if not parallel_mode:
                assert mapping is not None
                node_ebd_ext = (
                    torch.gather(node_ebd, 1, mapping)
                    if not self.use_loc_mapping
                    else node_ebd
                ) # node_ebd → 派生 node_ebd_ext → RepFlowLayer → 得到新 node_ebd → 再派生下一层的 node_ebd_ext → …
                #print("node_ebd_ext", node_ebd_ext.shape)       # node_ebd_ext torch.Size([1, 192, 128]) 
            else:
                #print("parallel_mode --- True")
                #print("node_ebd", node_ebd.shape)       # node_ebd torch.Size([1, 192, 128]) 
                # 并行模式：处理通信和自旋
                assert comm_dict is not None
                has_spin = "has_spin" in comm_dict
                if not has_spin:
                    # 无自旋：简单填充
                    n_padding = nall - nloc
                    node_ebd = torch.nn.functional.pad(
                        node_ebd.squeeze(0), (0, 0, 0, n_padding), value=0.0
                    )
                    real_nloc = nloc
                    real_nall = nall
                else:
                    # 有自旋：处理实部和虚部
                    real_nloc = nloc // 2
                    real_nall = nall // 2
                    real_n_padding = real_nall - real_nloc
                    node_ebd_real, node_ebd_virtual = torch.split(
                        node_ebd, [real_nloc, real_nloc], dim=1
                    )
                    # 混合节点嵌入：拼接实部和虚部
                    # 形状: nb x real_nloc x (n_dim * 2)
                    mix_node_ebd = torch.cat([node_ebd_real, node_ebd_virtual], dim=2)
                    # 形状: nb x real_nall x (n_dim * 2)
                    node_ebd = torch.nn.functional.pad(
                        mix_node_ebd.squeeze(0), (0, 0, 0, real_n_padding), value=0.0
                    )

                # 检查并行通信所需的字典键
                assert "send_list" in comm_dict
                assert "send_proc" in comm_dict
                assert "recv_proc" in comm_dict
                assert "send_num" in comm_dict
                assert "recv_num" in comm_dict
                assert "communicator" in comm_dict
                
                # 执行并行通信操作 - node_ebd
                ret = torch.ops.deepmd.border_op(
                    comm_dict["send_list"],
                    comm_dict["send_proc"],
                    comm_dict["recv_proc"],
                    comm_dict["send_num"],
                    comm_dict["recv_num"],
                    node_ebd,
                    comm_dict["communicator"],
                    torch.tensor(
                        real_nloc,
                        dtype=torch.int32,
                        device=torch.device("cpu"),
                    ),  # 应该是c++的int，放在cpu上
                    torch.tensor(
                        real_nall - real_nloc,
                        dtype=torch.int32,
                        device=torch.device("cpu"),
                    ),  # 应该是c++的int，放在cpu上
                )
                node_ebd_ext = ret[0].unsqueeze(0) # 并行模式的node_ebd_ext重新生成
                
                # 如果有自旋，分离实部和虚部
                if has_spin:
                    node_ebd_real_ext, node_ebd_virtual_ext = torch.split(
                        node_ebd_ext, [n_dim, n_dim], dim=2
                    )
                    node_ebd_ext = concat_switch_virtual(
                        node_ebd_real_ext, node_ebd_virtual_ext, real_nloc
                    )
            # print("node_ebd_ext", node_ebd_ext.shape)       # node_ebd_ext torch.Size([1, 192, 128]) 
            # 调用RepFlow层的前向传播函数 --- 这里输出了最终的node_ebd, edge_ebd, angle_ebd --- from repflow_layer.py
            # node_ebd, edge_ebd, angle_ebd = ll.forward(
            # new added in 2025 1012 - 添加diff参数和coord_update返回值
            node_ebd, edge_ebd, angle_ebd, coord_update = ll.forward(
                node_ebd_ext, # node 嵌入
                edge_ebd, # edge 嵌入，距离embd
                h2, # 旋转等变表征, dmatrix的后面三维
                angle_ebd, # 角度嵌入
                nlist, # 邻居列表 --- 不需要改吧
                nlist_mask, # 邻居列表掩码
                sw, # 开关函数
                a_nlist, # 角度邻居列表
                a_nlist_mask, # 角度邻居列表掩码
                a_sw, # 角度开关函数
                edge_index=edge_index, # 边索引
                angle_index=angle_index, # 角度索引
                diff=diff,  # new added in 2025 1012 - 传入未归一化的坐标差, 更新坐标需要diff
            ) # 返回：node_ebd, edge_ebd, angle_ebd, coord_update
            #if coord_update is not None:
                # print("coord_update", coord_update.shape)      
                # 计算每个原子的移动距离
                #distances = torch.linalg.norm(coord_update, dim=-1)  # [nf, nloc]
                #mean_distance = distances.mean()
                # print(f"  mean dist: {mean_distance:.6f} Å")
                #max_update = coord_update.abs().max()
                # print(f"  max: {max_update:.6f} Å")
                # print("coord_diff", diff.shape)
                # print("diff", diff)

            # =============================================================================
            # Modified in 2025-10-14 - 坐标更新和环境矩阵重算
            # =============================================================================
            # 在 repflows.py 中，坐标更新前添加
            '''
            def diagnose_prod_env_mat_gradient(coord, nlist, atype, mean, stddev, rcut, rcut_smth):
                """诊断prod_env_mat的梯度问题"""
                print("\n=== Diagnosing prod_env_mat gradients ===")
                
                # 确保coord有梯度
                coord_test = coord.detach().clone().requires_grad_(True)
                
                # 调用prod_env_mat
                dmatrix, diff, sw = prod_env_mat(
                    coord_test, nlist, atype, mean, stddev,
                    rcut, rcut_smth,
                    protection=1e-6,
                    use_exp_switch=False
                )
                
                # 测试各个输出的梯度
                outputs = {
                    'dmatrix': dmatrix,
                    'diff': diff,
                    'sw': sw
                }
                
                for name, output in outputs.items():
                    if output.requires_grad:
                        try:
                            # 计算梯度
                            grad = torch.autograd.grad(
                                output.sum(),
                                coord_test,
                                retain_graph=True
                            )[0]
                            
                            grad_norm = grad.norm().item()
                            grad_max = grad.abs().max().item()
                            has_nan = torch.isnan(grad).any().item()
                            has_inf = torch.isinf(grad).any().item()
                            
                            print(f"\n{name}:")
                            print(f"  ∂{name}/∂coord norm: {grad_norm:.2e}")
                            print(f"  ∂{name}/∂coord max: {grad_max:.2e}")
                            print(f"  has NaN: {has_nan}")
                            print(f"  has Inf: {has_inf}")
                            
                            if has_nan or has_inf or grad_norm > 1e6:
                                print(f"  ❌ PROBLEM DETECTED in {name}!")
                                
                        except RuntimeError as e:
                            print(f"  ❌ Error computing gradient for {name}: {e}")
            '''
            if self.update_coord and coord_update is not None:
                # =====================================================================
                # 坐标更新逻辑（完全仿照node_ebd/node_ebd_ext的处理模式）
                # =====================================================================
                # 对应关系：
                # node_ebd:        [nf, nloc, n_dim]  ←→  coord:          [nf, nloc, 3]
                # node_ebd_ext:    [nf, nall, n_dim]  ←→  extended_coord: [nf, nall, 3]
                # 
                # 处理流程：
                # 1. 从node_ebd_ext提取局部node_ebd（实际上node_ebd就是局部的）
                # 2. RepFlowLayer更新node_ebd
                # 3. 根据mapping生成新的node_ebd_ext
                
                # 步骤1：将局部位移增量广播到所有extended原子，并对extended坐标做增量更新
                # 数学依据：X_ext'[k] = X_ext[k] + ΔX_local[mapping[k]]
                if not parallel_mode:
                    # mapping_orig: [nf, nall]，扩展到坐标维
                    mapping_coord_ext = mapping_orig.unsqueeze(-1).expand(-1, -1, 3)
                    # 将局部位移按映射广播到extended区
                    delta_ext = torch.gather(coord_update, 1, mapping_coord_ext)  # [nf, nall, 3]
                    # 增量更新extended坐标（保持ghost的PBC平移不变）
                    extended_coord_new = extended_coord_new + delta_ext #* 0.01
                else:
                    # 并行模式暂不支持坐标更新
                    raise NotImplementedError("update_coord in parallel_mode is not supported yet")
                # =====================================================================
                # diagnose_prod_env_mat_gradient(
                #    extended_coord_new,  # 更新后的坐标
                #    nlist, atype,
                #    self.mean, self.stddev,
                #    self.e_rcut, self.e_rcut_smth
                # )
                # 2. 重新计算边环境矩阵（基于更新后的extended_coord）
                # =====================================================================
                dmatrix, diff, sw = prod_env_mat(
                    extended_coord_new.detach(),  # 使用更新后的坐标
                    nlist,           # 拓扑不变，复用原始nlist
                    atype,
                    self.mean,
                    self.stddev,
                    self.e_rcut,
                    self.e_rcut_smth,
                    protection=self.env_protection,
                    use_exp_switch=self.use_exp_switch,
                )
                #print("diff", diff.shape)
                #print("diff", diff)
                #r = torch.linalg.norm(diff, dim=-1)
                #print(f"[layer {idx}] env_prot={self.env_protection:.1e} r_min={(r[r>0]).min().item():.3e}")
                # 更新边几何量（复用原始nlist_mask，拓扑不变）
                sw = torch.squeeze(sw, -1).masked_fill(~nlist_mask, 0.0)
                _, h2 = torch.split(dmatrix, [1, 3], dim=-1)  # h2: [nf, nloc, nnei, 3]
                #epsilon = 1e-6  # 比默认的1e-6大得多
                #diff_norm = torch.linalg.norm(diff.detach().clone(), dim=-1, keepdim=True)
                #diff_norm = torch.linalg.norm(diff, dim=-1, keepdim=True)

                #h2 = diff / (diff_norm + epsilon)  # 避免除以很小的数
                #h2 = diff
                #h2 = h2_new.detach().clone()
                if self.use_dynamic_sel:
                    # 扁平化以匹配动态选择的数据流
                    h2 = h2[nlist_mask]   # [n_edge, 3]
                    sw = sw[nlist_mask]   # [n_edge]

                # 角度权重a_sw重算：保持与更新后的坐标一致
            '''
            if self.use_inter_layer_res:
                node_ebd = n_prev + self.alpha_n[idx] * (node_ebd - n_prev)
                edge_ebd = e_prev + self.alpha_e[idx] * (edge_ebd - e_prev)
                if self.update_angle:
                    angle_ebd = a_prev + self.alpha_a[idx] * (angle_ebd - a_prev)
            '''
        # =============================================================================
        # 10. 计算最终的旋转矩阵
        # =============================================================================
        # 计算转置旋转矩阵：用于生成旋转等变的几何信息
        # 形状: nb x nloc x 3 x e_dim
        h2g2 = (
            RepFlowLayer._cal_hg(edge_ebd, h2, nlist_mask, sw)
            if not self.use_dynamic_sel
            else RepFlowLayer._cal_hg_dynamic(
                edge_ebd,
                h2,
                sw,
                owner=edge_index[:, 0],
                num_owner=nframes * nloc,
                nb=nframes,
                nloc=nloc,
                scale_factor=(self.nnei / self.sel_reduce_factor) ** (-0.5),
            )
        )
        
        # 转置旋转矩阵：从 (nb x nloc) x 3 x e_dim 到 (nb x nloc) x e_dim x 3
        rot_mat = torch.permute(h2g2, (0, 1, 3, 2))

        # =============================================================================
        # 11. 返回最终结果
        # =============================================================================
        #print("isnan node_ebd", torch.isnan(node_ebd).sum()) # nan nan nan nan 
        #print("isnan edge_ebd", torch.isnan(edge_ebd).sum()) # nan nan nan nan 
        #print("isnan h2", torch.isnan(h2).sum()) # nan nan nan nan 
        #print("isnan rot_mat", torch.isnan(rot_mat).sum()) # nan nan nan nan 
        #print("isnan sw", torch.isnan(sw).sum()) # nan nan nan nan 
        return node_ebd, edge_ebd, h2, rot_mat.view(nframes, nloc, self.dim_emb, 3), sw
 
    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        path: Optional[DPPath] = None,
    ) -> None:
        """
        Compute the input statistics (e.g. mean and stddev) for the descriptors from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        path : Optional[DPPath]
            The path to the stat file.

        """
        if self.set_stddev_constant and self.set_davg_zero:
            return
        env_mat_stat = EnvMatStatSe(self)
        if path is not None:
            path = path / env_mat_stat.get_hash()
        if path is None or not path.is_dir():
            if callable(merged):
                # only get data for once
                sampled = merged()
            else:
                sampled = merged
        else:
            sampled = []
        env_mat_stat.load_or_compute_stats(sampled, path)
        self.stats = env_mat_stat.stats
        mean, stddev = env_mat_stat()
        if not self.set_davg_zero:
            self.mean.copy_(
                torch.tensor(mean, device=env.DEVICE, dtype=self.mean.dtype)
            )
        if not self.set_stddev_constant:
            self.stddev.copy_(
                torch.tensor(stddev, device=env.DEVICE, dtype=self.stddev.dtype)
            )

    def get_stats(self) -> dict[str, StatItem]:
        """Get the statistics of the descriptor."""
        if self.stats is None:
            raise RuntimeError(
                "The statistics of the descriptor has not been computed."
            )
        return self.stats

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor block has message passing."""
        return True

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the descriptor block needs sorted nlist when using `forward_lower`."""
        return True
