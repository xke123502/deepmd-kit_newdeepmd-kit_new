# SPDX-License-Identifier: LGPL-3.0-or-later  # 许可证声明
import logging  # 日志模块
from typing import (
    Optional,  # 可选类型（参数可为 None）
    Union,     # 联合类型（允许多种类型）
)

import torch  # PyTorch 张量与模块

from deepmd.dpmodel import (
    FittingOutputDef,   # 拟合网络输出定义集合
    OutputVariableDef,  # 单个输出变量的定义（形状、是否可归约等）
)
from deepmd.pt.model.task.ener import (
    InvarFitting,  # 继承自 GeneralFitting，封装“可归约且可求导”的不变量读出
)
from deepmd.pt.model.task.fitting import (
    Fitting,  # 插件注册基类（根据 type 选择具体 FittingNet 子类）
)
from deepmd.pt.utils import (
    env,  # 环境配置（精度、设备等）
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,  # 缺省数值精度（如 float32/float64）
)
from deepmd.utils.version import (
    check_version_compatibility,  # 序列化/反序列化的版本兼容性检查
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION  # 全局浮点精度（用于张量）
device = env.DEVICE                    # 当前设备（CPU/GPU）

log = logging.getLogger(__name__)  # 模块级日志记录器


@Fitting.register("property")  # 在插件系统中注册名为 "property" 的读出头
class PropertyFittingNet(InvarFitting):
    """Fitting the rotationally invariant properties of `task_dim` of the system.
    # 说明：用于拟合“旋转不变”的每原子性质（如标量/向量的每原子分量），
    # 通过 InvarFitting 实现：每原子输出（dim_out=task_dim），是否做系统级归约由 intensive 与 reducible 控制。

    Parameters
    ----------
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    task_dim : int
        The dimension of outputs of fitting net.
    property_name:
        The name of fitting property, which should be consistent with the property name in the dataset.
        If the data file is named `humo.npy`, this parameter should be "humo".
    neuron : list[int]
        Number of neurons in each hidden layers of the fitting net.
    bias_atom_p : torch.Tensor, optional
        Average property per atom for each element.
    intensive : bool, optional
        Whether the fitting property is intensive.
    resnet_dt : bool
        Using time-step in the ResNet construction.
    numb_fparam : int
        Number of frame parameters.
    numb_aparam : int
        Number of atomic parameters.
    dim_case_embd : int
        Dimension of case specific embedding.
    activation_function : str
        Activation function.
    precision : str
        Numerical precision.
    mixed_types : bool
        If true, use a uniform fitting net for all atom types, otherwise use
        different fitting nets for different atom types.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        property_name: str,      # 输出变量名称（训练数据中的键名）
        task_dim: int = 1,       # 每原子输出维度（标量=1，向量=3）
        neuron: list[int] = [128, 128, 128],
        bias_atom_p: Optional[torch.Tensor] = None,  # 每元素的性质偏置（与能量 bias_atom_e 同构）
        intensive: bool = False,       # 是否为强度量（True：平均/独立于系统大小；False：可按原子归约）
        resnet_dt: bool = True,        # 残差网络时间步 dt（ResNet 构造）
        numb_fparam: int = 0,          # 帧参数维度（追加到描述符）
        numb_aparam: int = 0,          # 原子参数维度（可作为掩码或追加特征）
        dim_case_embd: int = 0,        # case embedding 维度（任务索引等场景）
        activation_function: str = "tanh",  # 激活函数
        precision: str = DEFAULT_PRECISION,  # 数值精度
        mixed_types: bool = True,      # 是否跨元素共享同一个读出网络
        trainable: Union[bool, list[bool]] = True,  # 可训练性（可布尔或逐层列表）
        seed: Optional[int] = None,    # 随机种子
        **kwargs,
    ) -> None:
        self.task_dim = task_dim       # 记录输出维度
        self.intensive = intensive     # 记录强度量标志
        super().__init__(
            var_name=property_name,    # 输出变量名（用于结果与标签对齐）
            ntypes=ntypes,             # 元素类型个数
            dim_descrpt=dim_descrpt,   # 每原子描述符维度（来自 DPA3 描述符）
            dim_out=task_dim,          # 每原子输出维度
            neuron=neuron,             # 读出 MLP 隐层结构
            bias_atom_e=bias_atom_p,   # 偏置张量（沿用 InvarFitting 接口名）
            resnet_dt=resnet_dt,
            numb_fparam=numb_fparam,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            trainable=trainable,
            seed=seed,
            **kwargs,
        )

    def output_def(self) -> FittingOutputDef:
        # 定义本读出头的输出规格：
        # - 变量名为 self.var_name（如 "magnetic_moment"）
        # - 形状为 [task_dim]
        # - reducible=True：框架会同时生成原子级与归约后的系统级键；
        #   若只需每原子值，训练/评估时使用未归约键即可（例如 magnetic_moment）。
        # - r/c_differentiable=False：默认不对该变量做关于坐标/应变的导数（与能量不同）。
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                    intensive=self.intensive,
                ),
            ]
        )

    def get_intensive(self) -> bool:
        """Whether the fitting property is intensive."""
        return self.intensive

    @classmethod
    def deserialize(cls, data: dict) -> "PropertyFittingNet":
        # 从字典反序列化：版本校验、字段重映射（var_name ← property_name）
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 4, 1)
        data.pop("dim_out")
        data["property_name"] = data.pop("var_name")
        obj = super().deserialize(data)

        return obj

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        dd = {
            **InvarFitting.serialize(self),
            "type": "property",
            "task_dim": self.task_dim,
            "intensive": self.intensive,
        }
        dd["@version"] = 4

        return dd

    # make jit happy with torch 2.0.0
    exclude_types: list[int]  # 仅为 TorchScript 类型推断提供注解
