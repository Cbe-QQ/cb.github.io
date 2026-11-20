---
title: PTOAS 编译器内部机制全解
date: 2026-11-20
categories: Ascend
tags:
  - PTOAS
  - MLIR
  - Ascend
  - Compiler
  - PTODSL
layout: docs
group: ascend
order: 2
description: 从 PTO dialect、binding、PTODSL 到 emitc / vpto 双后端:PTOAS 编译链路全解
---

# PTOAS 编译器内部机制全解

> 基于 `/home/chenben/llvm-workspace/PTOAS` 源码、`/home/chenben/code/tilelang/docs/compiler_internals/tilelang_to_ptodsl.md` 与 `/home/chenben/code/pypto`（第七部分）整理。所有行号引用均可直接在仓库中跳转验证。

---

## 第一部分 · 全局观

### 1.1 组成部分

| 组成 | 定义 |
|---|---|
| **PTO dialect** | 一个 MLIR 方言（dialect），定义昇腾 NPU 虚拟指令集的 op 与类型，覆盖 Cube / Vector / SIMT 三个计算单元与 MTE 数据搬移 |
| **Python binding** | 将 PTO dialect 与编译器能力暴露给 Python 的接口层（`ptoas.mlir.`* + `ptoas._core`） |
| **PTODSL** | 基于 Python binding 实现的高层 Python 前端，通过 `@pto.jit` 与 tracing 构造 PTO IR |
| **PTOAS** | 编译器：解析并验证 PTO IR，执行优化 pass，经 emitc / vpto 两条后端生成目标代码 |

四者的关系：**PTO dialect 定义指令集语义，Python binding 把 dialect 与编译器能力暴露给 Python，PTODSL 构建于 binding 之上提供编写方式，PTOAS 完成 lowering、优化与代码生成。**

### 1.2 全景图

```text
来源 A (PTODSL 手写):   @pto.jit;  pto.tile.* / pto.vmi.* / pto.vadd ...
                        └ 构建于 Python binding 之上（子命名空间适配 + tracing）
来源 B (裸 binding):    build(): Context → pto.TLoadOp(...) → print(module)
                        └ 直接调用 Python binding 逐个建 op
来源 C (TileLang 生成): tilelang.lower(target="pto") → 打印 PTODSL 文本
                        └ 不经 binding，产物即来源 A 的文本，殊途同归
                      │  A/B/C 经 Python binding（Python 构造 MLIR 的唯一通道）汇聚
                      ▼
            PTO dialect 的 MLIR 模块（.pto 文本 / PTOBC 二进制）
                      ▲                     │
        路线B(预留):  │                     ▼
        PyPTO IR      │           ┌─ ptoas 编译器 ────────────────────┐
        → PTO MLIR    │           │ 前处理 → 共享 mainline（优化）      │
        (发射器未实现) │           │  ┌ emitc: → 调用 pto-isa 的 C++ ─┼──┐
                      │           │  └ vpto : → 物理微指令 → hivm     │  │
                      │           └─────────────────┬────────────────┘  │
                      │                             │ (ptoas 内部 bisheng)│
来源 D (PyPTO):       │                             │                   │
@pl.jit → parser      │                             │                   │
→ PyPTO IR ──────路线A(现网，不经 binding/dialect)──┼───────────────────┤
（自带独立 IR：        │                             │                   │
 block.*/system.*）   │                             │                   │
                      │                             ▼                   ▼
                      │                    fatobj(ELF)         emitc 产物：
                      │                    + launch.o          调用 pto-isa 的 C++
                      ▼                                               │
        路线A 产物：BackendCCE 产码                                    │
        pto-isa C++ kernel.cpp（与 emitc 产物同构）◄───────────────────┘
                      │
                      └──────────────┬──────────────┘
                                     │ 指令语义唯一来源：pto-isa C++ 头文件库
                                     ▼
                  bisheng 设备编译器（--enable-pto-tile-fusion：
                  TLOAD / TMATMUL 等虚拟指令 → 硬件指令）
                                     ▼
                  fatobj (ELF) + launch.o ──链接──► .so
                                     ▼
                  CANN runtime（真机）/ camodel（仿真），ctypes 启动
```

图中体现的组件关系（后文各部分的索引）：

- **A/B/C → binding → dialect**：PTODSL 构建于 Python binding 之上（3.2），裸 binding 是直接使用（3.3），TileLang 的产物是 PTODSL 文本、回到来源 A（第六部分）；三者殊途同归于同一份 PTO MLIR。MLIR 到目标代码的完整过程——三层指令如何经两条后端走到实现——是第四部分的主线。
- **D（PyPTO）与 PTOAS 平行**：PyPTO 自带独立 IR 与编译链，不经过 binding / dialect——路线 A 自带 "IR → pto-isa C++" 发射器，与 ptoas 的 emitc 后端**同构**（产物都是调用 pto-isa 的 C++，再交 bisheng）；路线 B 是预留的汇聚口，发射器未实现（第七部分）。
- **共享底座**：pto-isa C++ 库是指令语义的唯一来源（emitc 与路线 A 共同依赖），bisheng 是唯一的设备编译器（emitc / 路线 A 在外部调用，vpto 在 ptoas 内部调用）——所以四条来源最终都落到同一种产物与运行方式（第五部分）。

---

## 第二部分 · 基础概念

#### MLIR 基础

**IR（中间表示）**：编译器内部表示程序的数据结构。MLIR 中程序是一棵 op 嵌套树（模块 ⊃ 函数 ⊕ 基本块 ⊃ op），值以 SSA 形式命名（`%0`、`%1`，每个值只有一个定义点）。前端与编译器之间传递的只有 IR——本文所有"构造 MLIR"都是构造这棵树。

**dialect（方言）**：MLIR 的扩展单位：一组自定义 op / 类型 / 属性共用一个命名空间前缀（如 `pto.tadd` 属于 pto dialect）。同一个模块可以混用多个 dialect，本文涉及 `pto`（虚拟指令集）、`arith` / `scf` / `memref`（MLIR 内置）、`emitc`（C++ 发射）与 `llvm.hivm`（昇腾设备指令层）。"PTO 是一个 dialect"意味着它复用 MLIR 的全部基础设施（解析、验证、pass 框架、bytecode）。

**TableGen / .td**：LLVM/MLIR 的声明式代码生成机制：op 在 `.td` 文件中统一定义（名字、操作数、约束、描述），由 `mlir-tblgen` 生成 C++类与 Python binding（3.1 线 1）——增删一个 op 只需改 `.td` 后重新构建，C++ 与 Python 两侧同步更新。

**pass**：MLIR 编译器对 IR 执行处理的基本单元，实现为 `runOnOperation()`。按作用分为变换（transform，改写 IR）、分析（analysis，只读并产出信息）、校验（verify，发现非法 IR 则报错）。pass 由 PassManager 按 pipeline 编排顺序执行；pipeline 中的 `validate` 系列 pass 用于在关键节点检查 IR 合法性。

**lowering（下降）**：把高层抽象 op 改写为更低层、更接近硬件的 op 的过程，如 tile → vmi → mi → `llvm.hivm`。与"优化"相对：优化在同一抽象层内改写 IR，lowering 跨抽象层。

**memref**：MLIR 内置的内存引用类型 / dialect，表示一块带形状与步长的缓冲区视图。`pto-plan-memory` 在 UB 上规划 tile 缓冲（4.2）；PyPTO IR 侧的对应概念见 7.2。

#### 昇腾硬件

**计算 / 搬移单元**：昇腾 NPU 核内有三类计算单元——Cube（矩阵乘，核号体系中称 AIC）、Vector（SIMD 向量，称 AIV）、SIMT（标量组），外加 MTE（数据搬移引擎，负责各存储层级间的 DMA）。Cube 与 Vector 是两条异步执行的硬件流水线，一个 kernel 同时用两者（CV 混合）时必须显式同步（`set_flag` / `wait_flag`，见 4.2 同步插入与第六部分的 flag 用例）。

**存储层级（GM / L1 / L0 / UB）**：GM 为全局内存（核外、容量最大）；核内片上存储分 L1、L0A/L0B/L0C（Cube 专用）与 UB（Vector 专用的统一缓冲），容量逐级递减、带宽逐级递增。kernel 的核心工作就是在这几级之间搬数并计算——`pto.mte_gm_ub`、`pto.mte_l1_l0a` 等搬移指令的名字即"源层级_目的层级"（第六部分）。

**tile**：驻留在近核存储（L1 / L0 / UB）上的一块多维数据及其形状 / 位置描述，是 PTO "算子级"抽象的基本操作单位：tile 指令（`TLOAD` / `TADD` / `TMATMUL` 等）以整块 tile 为操作数，一条指令对应硬件上的一批流水操作，屏蔽了寄存器、mask、步长等细节。

#### 执行模型

**SIMD（单指令多数据）**：无线程概念；**一条向量指令**对一组数据通道（lane）同时执行同一操作，并行度由指令宽度表达。PTO 中对应 vmi / mi 两层：vmi 是逻辑 SIMD（`pto.vmi.vadd`，`!pto.vmi.vreg<L×T>`，只指定 lane 数 L，物理寄存器由编译器分配），mi 是物理 SIMD（`pto.vadd`，直接操作 256B 物理寄存器）。TileLang 前端的 `T.SimdVF` 也走此模型（6.4）。

**SIMT（单指令多线程）**：**N 个 workitem** 各自执行一份（多为标量）代码，每个 workitem 有自己的线程编号，并行度由线程数表达。PTO 中对应 `with pto.simt(tx, ty, tz):` 块内的 `pto.load` / `pto.store`（6.4）；线程编号经 `pto.get_tid_x()` 等获取。注意区分：`pto.simt(128)` 是 128 个 workitem 各执行一遍，`pto.vmi.*(size=128)` 是一条指令 128 个 lane——`threads=128` ≠ `lanes=128`。

**lane / VL / mask**：lane 是向量指令的一条数据通道；VL（向量长度）是一条指令的 lane 数（vmi 逻辑 lane 数 ∈ {1,2,4,8,64,128,256}，物理寄存器数由 `K = ⌈L·bitwidth/2048⌉` 决定，见 4.4）；mask 是谓词，控制一条指令中哪些 lane 的计算 / 写回生效——load 不带 mask 读满全部 lane、计算与 store 用动态 mask 是处理"不能整除尾巴"的惯用手段（6.4 带尾例）。

#### 寄存器

PTO dialect 中的"寄存器值"共 5 种类型，分两层，本质是三类硬件寄存器：

| 层 | 类型 | 对应的硬件寄存器 |
|---|---|---|
| mi（物理，`VPTOTypeDefs.td`） | `!pto.vreg<L×T>` | 向量数据寄存器：固定 256B 一条 |
| | `!pto.mask<b8/b16/b32>` | 谓词 / mask 寄存器：带物理粒度视图 |
| | `!pto.align` | 对齐寄存器（vector_align）：`vlds` / `vsts` 非对齐访存用 |
| vmi（逻辑，`VMITypeDefs.td`） | `!pto.vmi.vreg<L×T[, layout]>` | 逻辑向量寄存器：只有 lane 数与元素类型，物理布局由 `vmi-layout-assignment` 分配（4.4） |
| | `!pto.vmi.mask<L, granularity[, layout]>` | 逻辑 mask：比物理版多 `pred` 粒度（未物理化的纯谓词） |

两层是 lowering 关系：`vmi-to-vpto` 把逻辑 vreg / mask 翻译为物理 vreg（K 条 256B 寄存器）与确定粒度的物理 mask。两类不算寄存器：标量与指针（`pto.load` / `pto.store`、地址计算）使用 MLIR 内置类型，标量寄存器分配交给下游 bisheng；tile 层的 `!pto.tile_buf` 是近核存储的描述符，属于内存概念而非寄存器。

#### 工具链与运行

**pto-isa**：以 C++头文件库形式提供的虚拟指令语义实现（`TLOAD` / `TADD` / `TMATMUL` 等模板函数），`#include <pto/pto-inst.hpp>` 即可在 C++ 中调用；虚拟指令到硬件指令的 lowering 不在库内，而由 bisheng 的 tile-fusion 后端完成。ptoas 的 emitc 后端与 PyPTO 路线 A（第七部分）共同依赖它。

**bisheng**：昇腾 CCE 工具链编译器（clang/LLVM 风格命令行），同时编译 host 与设备代码并产出 fatobj；对 PTO 而言，它承担"虚拟指令 → 硬件指令"的最后一段 lowering（`--enable-pto-tile-fusion`），emitc / vpto / 路线 A 三条路径最终都经它（5.2）。

**hivm**：bisheng/LLVM 内部面向昇腾设备的指令层（`llvm.hivm.`* intrinsic）。vpto 后端把物理微指令翻译到这里，交由 LLVM 完成寄存器分配与指令发射（4.5）。

**fatobj（fat object）**：CCE 工具链的目标文件格式，在单个 ELF 中同时包含 host 符号与设备二进制。host 部分为空的 `__global__` 函数符号（用于链接与启动寻址）；设备二进制（aibinary）以数据段形式嵌入（`-fcce-include-aibinary`）。运行时由 CCE runtime 提取 aibinary 并加载到 NPU 执行。

**PTOBC（PTO bytecode）**：PTO IR 的 MLIR bytecode 二进制编码，与 `.pto` 文本互为序列化形式。它是**输入格式**而非执行格式——ptoas 解码后进入与文本输入完全相同的 pipeline（5.2）。

**launch.o / launcher**：host 侧启动器：`launch.cpp` 用 CCE 启动语法 `kernel<<<grid, nullptr, stream>>>(...)` 包装设备函数，编译为 `launch.o` 后与 kernel fatobj 链接成 `.so`，Python 侧经 ctypes 调用（5.2 / 5.3）。

**CANN runtime**：昇腾异构计算架构（CANN）的运行时软件栈：stream、内存管理、kernel 加载与下发。kernel 最终在 CANN runtime（真机）上执行。

**camodel**：NPU 的 CPU 功能仿真器：不插卡即可运行 kernel（链接 `libruntime_camodel` 并设 `MSPROF_SIMULATOR_MODE=1`），用于功能验证与核内流水报告（5.3）。

#### 构造机制

**Python binding（语言绑定）**：将一种语言（C++）实现的库暴露给另一种语言（Python）调用的接口层，负责函数映射、对象生命周期与类型转换。本身不引入新功能：删除 Python 侧，C++ 编译器功能不变；删除 C++ 侧，Python 无法构造任何 op。

**tracing**：PTODSL 的 IR 构造机制：**执行**（而非静态解析）用户的 Python kernel 函数，函数体里每个 `pto.`* 调用在执行时立即向当前 MLIR 模块插入一条 op，函数执行完毕即得到完整 IR（3.2）。

**回调（callback）**：将函数注册给被调用方，由被调用方在特定时机反向调用的机制。本文涉及的场景在 4.7：C++ 编译器在 `pto-expand-tile-op` 执行期间调用 Python 侧的 TileLib/SoftLib 模板。

---

## 第三部分 · Python 构造 MLIR：Python binding 与两条构造路径

本章是全文核心之一。结论：**Python 构造 MLIR 的通道只有一条——Python binding；PTODSL 与 pto-func 是这条通道的两种使用方式。**（第三条来源 TileLang 不经过 binding，直接打印 PTODSL 文本，见第六部分。）

### 3.1 Python binding 在构建期生成

Python binding 不是由任何 Python 代码生成的，而是编译 PTOAS 源码时的构建产物，包含三条生成线：

```text
线 1：PTOOps.td ⊃ VMIOps.td + VPTOOps.td
      ──mlir-tblgen -gen-python-op-bindings──► _pto_ops_gen.py（708 个 op 类）
线 2：PTOModule.cpp + NativeModule.cpp ──C++ 编译──► ptoas/_core.so（类型/属性/枚举 + 编译器入口 main）
线 3：LLVM 上游 MLIR Python API ──打包──► ptoas/mlir/ir/...（Context / Module 等）
```

线 1 说明：**一个 dialect 对应一份 TableGen 生成文件**，因此 mi 的 `VaddOp`（`:14885`）、tileop 的 `TAddOp`（`:19913`）、vmi 的 `VMIVaddOp`（`:30383`）位于同一个 `_pto_ops_gen.py`——"三层同属一个 dialect"在构建产物上即为此事实。每个 op 提供两种入口形式：op 类（如 `VaddOp`，位置参数，构造即插入）与便捷函数（如 `tadd(...)`、`vmi_vadd(...)`，关键字参数）。三层调用方选择不同：mi 发射函数使用 op 类，tileop 发射函数使用便捷函数，vmi 通过 `getattr(_pto, f"vmi_{op_name}")` 按名查找（查找失败时可给出该 VMI 形式不可用的明确错误）。

PTODSL 由 70+ 个纯 Python 文件组成，构建时原样打包，不产生 binding 的任何部分；运行期通过 `_ops_imports.py`（`from ptoas.mlir.dialects import pto as _pto`）依赖 binding。依赖关系是单向的：binding 不依赖 PTODSL（pto-func 路径可不经过 PTODSL 独立使用 binding）；PTODSL 完全依赖 binding（binding 缺失时 `import ptodsl` 即失败）；两者构造的 IR 最终由 C++ 编译器消费。

### 3.2 路径 A：PTODSL 的 tracing 流程

PTODSL 将 Python 函数转换为 MLIR 的过程分为五步，tileop / vmi / mi 三层共用同一流程：

```text
① 装饰      @pto.jit 记录 target/backend/mode 等编译配置，不产生 IR
② AST 改写  for/if 改写为 pto.for_/pto.if_（纯 Python 到 Python 的变换）
③ 构造容器  创建 Context、module 骨架与 func.func 入口（含 pto.backend 等属性）
④ tracing   执行改写后的函数体，每个 pto.* 调用立即插入一条 MLIR op（三层的差异在此）
⑤ 输出      str(module) 得到 .pto 文本，交由 ptoas 编译
```

**第③步"构造容器"详解**：MLIR 是层层嵌套的容器结构（Module 装 func.func，func.func 装 Block，Block 装 op），op 创建时必须落在已有块里的插入点上，所以 tracing 之前要先把空骨架搭好。这一步由两处代码完成：`_context.py:21-27` 的 `make_context()` 创建 Context 并显式注册 pto dialect；`_tracing/module_builder.py` 的 `create_kernel_module` 按 `KernelModuleSpec`（即 `@pto.jit` 装饰器记录的 target/backend/mode 等配置）建 Module 与 func.func 入口。第③步结束时打印 module，就是一个 op 都没有的空架子：

```mlir
module attributes {pto.target_arch = "a5", pto.mode = "auto"} {
  module attributes {pto.backend = "vpto", pto.target_arch = "a5"} {
    func.func @main(%arg0: !pto.ptr<f32, #pto.addr_space<gm>>, ...)
        attributes {pto.entry} {
      // ← 空的，等第④步 tracing 往这里逐条插 op
    }
  }
}
```

**第④步三层的调用入口不同：mi 直接调用发射函数，tileop / vmi 先经过子命名空间模块。**

存在这一差异的原因是 Python 对象结构：`pto.vadd` 可以在 `pto.py` 中直接再导出，而 `pto.tile.add` / `pto.vmi.vadd` 要求 `pto.tile` / `pto.vmi` 是带方法的独立子对象。`_tile_namespace.py` / `_vmi_namespace.py` 即这两个子命名空间的实现（在 `pto.py:87-88` 导入挂载）。除命名空间组织外，它们还承担参数适配与分派：

- `_tile_namespace.py`：多数成员为直通转发（`add = staticmethod(_ops.tadd)`），少数方法提供参数补全——例如 `tile.store` 自动构造 partition view，行归约类 op 在未提供临时 buffer 时按源 tile 元数据自动分配（`_tile_namespace.py:33-51`）；
- `_vmi_namespace.py`：按操作数类型在指令族内分派——`vmi.vadd` 依据 rhs 为向量或标量分别发射 `pto.vmi.vadd` 或 `pto.vmi.vadds`（`_vmi_namespace.py:703-720`）。

第④步完整调用链：

```text
mi:      pto.vadd      → pto.py 再导出 → _ops_vmath.py:vadd()（校验/解包/推导 result_type）→ binding VaddOp
tileop:  pto.tile.add  → _tile_namespace.py（转发与参数补全）→ _ops_tile.py:tadd() ───────→ binding tadd
vmi:     pto.vmi.vadd  → _vmi_namespace.py（vadd/vadds 分派、mask 适配）────────────────→ binding vmi_vadd
                                        │
                                        ▼ 三条路径在此汇合，之后完全相同
              【binding】 _pto_ops_gen.py 的 op 类/便捷函数（OPERATION_NAME = "pto.vadd" 等）
              →【上游 MLIR Python】 ir.Operation.create
              →【MLIR C API】 mlirOperationCreate(...)
              →【C++】 mlir::Operation 创建并插入当前 block
              →【PTODSL】 wrap_surface_value(返回值)，封装为 PTODSL 表面对象
```

结论：**PTODSL（含子命名空间模块与 `_ops_*` 发射模块）不直接创建任何 MLIR 对象**，所有创建动作均经 binding 转发至 C++ 完成。职责划分：子命名空间模块负责面向用户的参数适配，`_ops_*.py` 保持发射函数与 MLIR op 一一对应。

### 3.3 路径 B：pto-func 的直接调用方式（裸 binding）

pto-func 用例（`test/pto-func/<arch>/<类别>/<OpName>/<Case>.py`）不经过 `@pto.jit`，手动完成 PTODSL 自动处理的每个环节。以 `TImg2colOp_conv_p1_...py` 为例，共六步：

**① 创建 Context**：`Context()` + `pto.register_dialect(ctx, load=True)`——pto dialect 不自动加载，必须显式注册。

**② 构造类型与属性——两种并行方式**：

```python
# 方式 a：binding 构造器（经 CAPI 由 C++ 创建）
left_ty = pto.TileBufType.get([16,32], f16, left_space, [6,32], left_cfg, ctx)
# 方式 b：字符串解析（MLIR AsmParser 调用 dialect 的 parser）
fmap_tv_ty = Type.parse("!pto.tensor_view<1x1x3x4x16xf16, #pto.conv_layout<nc1hwc0>>", ctx)
```

**③ 构造容器**（手动完成路径 A 第③步自动完成的工作）：`func.FuncOp` + 设置 `pto.entry` 属性 + `add_entry_block()`。

**④ 逐 op 插入**——与路径 A 使用完全相同的 binding → CAPI → C++ 通道：

```python
with InsertionPoint(entry):
    c0 = arith.ConstantOp(idx, 0).result                      # 标量常量使用 arith
    fmap_tv = pto.MakeTensorViewOp(fmap_tv_ty, fmap_ptr, [...], [...]).result
    x_tile  = pto.AllocTileOp(x_tile_ty, config=cfg_fmap).result
    pto.TLoadOp(None, fmap_pv, x_tile)                        # DPS 形式，无结果时首参传 None
    pto.TImg2colOp(left, x_tile, posM=c0, posK=c0, fmatrixMode=fm)
    pto.TMatmulOp(None, left, right, acc)
    func.ReturnOp([])
```

**⑤ 手动校验** `m.operation.verify()`（路径 A 中由 tracing runtime 自动执行）。

**⑥ 输出与编译**：`__main__` 中 `print(build())`——`str(module)` 即 .pto 文本；运行该 `.py` 得到 MLIR，交由 `ptoas` 编译（批量入口 `tools/ptoastest/script/run.sh`）。

### 3.4 两条路径对照

| | 路径 A（PTODSL） | 路径 B（pto-func 裸 binding） |
|---|---|---|
| Context / 容器 / InsertionPoint | tracing runtime 自动创建 | 手动创建 |
| 控制流 | AST 改写 + `pto.for_` / `pto.if_` | 直接构造 `scf.ForOp` 等 |
| 标量常量 | `pto.const` → `pto.constant` | `arith.ConstantOp` |
| 子命名空间 | tileop / vmi 有参数适配层 | 无（直接使用 binding op 类） |
| verify | 自动 | 手动 |
| **op 插入通道** | **相同：binding op 类 → CAPI → C++** | **相同** |
| **产物** | **相同：PTO dialect MLIR 文本** | **相同** |
| 测试定位 | 三层作者面的端到端行为及 DSL 封装本身 | dialect op 的原始契约（类型约束、属性组合） |

**两条路径在 binding 之后完全相同；差异全部位于 binding 之前的自动化封装。**

### 3.5 Python 与编译器之间的三条交互路径

| 路径 | 方向 | 时机 | 章节 |
|---|---|---|---|
| 构建路径 | C++ 源码 + TableGen → Python binding | 构建期一次 | 3.1 |
| 使用路径 | PTODSL / pto-func → binding → C++（构造 IR、启动编译） | 运行期 | 3.2 / 3.3 |
| 回调路径 | C++ pass → Python TileLib / SoftLib 模板 | 编译期 `pto-expand-tile-op` 等触发 | 见 4.7 |

---

## 第四部分 · 三层作者面到最终实现：tileop / vmi / mi × emitc / vpto

本部分是全文主线：**三层指令各自如何一路走到设备上可执行的实现**。pass 与编译选项是实现细节，作为子内容归并在各层路径与 4.8 速查中。

### 4.1 三层作者面

三层不是三个独立的包，而是同一个 `pto` dialect 内的三层抽象（op 分别定义于 `PTOOps.td` / `VMIOps.td` / `VPTOOps.td`）：

| 层 | 抽象级别 | 语义 | Python 入口 | MLIR 类型 |
|---|---|---|---|---|
| **tileop** | 整块 2D tile | tile 级数据搬移与计算，不涉及寄存器 | `pto.tile.add`、`pto.tload`，`mode="auto"` | `!pto.tile_buf<...>` |
| **vmi** | 逻辑 SIMD 向量（Virtual Micro-instruction Set） | 逻辑向量逐 lane 操作，指定 lane 与谓词，不指定物理寄存器布局 | `pto.vmi.vadd` 等 | `!pto.vmi.vreg<L×T>`、`!pto.vmi.mask` |
| **mi** | 物理微指令（VPTO） | 直接操作 256B 物理向量寄存器、物理 mask 粒度与分布模式 | 顶层 `pto.vadd` / `pto.vlds` / `pto.mad` / `pto.mte_gm_ub`（不存在 `pto.mi` 命名空间） | `!pto.vreg`、`!pto.mask<b32>` |

vmi 逻辑 lane 数 ∈ {1,2,4,8,64,128,256}，物理寄存器数由 `K = ⌈L·bitwidth/2048⌉` 决定，寄存器布局由编译器分配。注意区分：`@pto.tileop` / `with pto.tileop():` 定义可复用的单核 helper（outline 为 `func.func` + `func.call`），与 tile 指令面是两个概念。

三层 IR 对照（同一加法）：

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<...>) outs(%c : ...)                    // tileop
%vc = pto.vmi.vadd %va, %vb, %m : !pto.vmi.vreg<64xf32>, ... -> ...          // vmi
%vc = pto.vadd %va, %vb, %m : !pto.vreg<64xf32>, !pto.vreg<64xf32>, ...      // mi
```

测试流程中"一次只走一层"的原因是三层的验证契约不同（tile 层验证 tile 指令语义、vmi 层验证逻辑到物理的 lowering、mi 层验证物理指令本身），并非机制上的限制。

### 4.2 进入编译器：前处理与共享 mainline

无论哪一层、走哪条后端，IR 进入 ptoas 后先经过两段**与层无关的公共处理**：

```text
.pto / PTOBC ──► 前处理 ──► 共享 mainline（--emit-pto-ir 在此观测）──► 后端分叉（emitc / vpto）
```

- **前处理**：物化并补齐 tile op 的 section 结构（`pto-materialize-tile-op-sections` 等）、校验 cube / vector 物理 section 边界、generic 形式 op 降级为具体形式。
- **共享 mainline** 按功能分六组：规范化 / pipe 降级（`pto-canonicalize-ir` 等）、布局推断（`pto-infer-layout`）、融合规划（`pto-fusion-plan` / `pto-fusion-region-gen` 等）、UB 内存规划（`pto-plan-memory`，explicit / level3 模式跳过）、**自动同步插入**（`pto-unified-sync` 等，互斥四选一，是 auto 模式正确性的关键）、收尾清理。

这段是三层共用的基础设施——它不理解也不关心"你写的是哪一层"，只负责把 IR 整理成后端可消费的形式。之后按 `--pto-backend` 分叉，三层的命运开始不同。

### 4.3 tileop → 最终实现

tile 指令（`pto.tadd` / `pto.tmatmul` 等）是**唯一两条后端都支持的层**，但两条的实现路线完全不同：

**emitc 路径：语义外包给 pto-isa，lowering 外包给 bisheng**

```text
pto.tadd ──EmitPTOManualPass──► emitc.call_opaque "TADD"（不展开）
        ──emitc::translateToCpp──► kernel.cpp（#include "pto/pto-inst.hpp"）
        ──bisheng（ptoas 外部，--enable-pto-tile-fusion）──► 硬件指令
```

`EmitPTOManualPass` 把 PTO / arith / scf / memref 整体转换为 emitc dialect，tile 指令一对一映射为 pto-isa 模板函数调用；指令语义由 pto-isa C++ 头文件库提供，"虚拟指令 → 硬件指令"的最后一段 lowering 由 bisheng 内置的 tile-fusion 后端完成。**ptoas 在这条路径上不做任何指令展开。**

**vpto 路径：编译器自己展开为微指令序列**

```text
pto.tadd ──pto-expand-tile-op（编译期回调 Python TileLib，机制见 4.7）──►
         vecscope + vlds / vadd / vsts（mi 层 IR）──► 汇入 4.5 公共尾段
```

展开策略（某 dtype / shape 下该生成哪段微指令序列）不实现于 C++，而是实现为 PTODSL Python 模板（`ptodsl/ptodsl/tilelib/`），pass 在编译过程中回调 Python 获取展开结果。展开产物是普通的 mi 层 IR，与 `mode="explicit"` 手写的结果同构，后续处理完全一致。

### 4.4 vmi → 最终实现

**emitc：不支持**——emitc 后端没有 vmi op 的 lowering，vmi 写法只能走 vpto。

**vpto：逻辑向量物理化，三步变成 mi 层 IR**

```text
pto.vmi.vadd ──vmi-mask-granularity-assignment──► 确定 mask 物理粒度（b8 / b16 / b32）
             ──vmi-layout-assignment──► 逻辑 vreg 分配物理寄存器布局
                                        （L 个 lane → K = ⌈L·bitwidth/2048⌉ 条 256B 物理寄存器）
             ──vmi-to-vpto──► 翻译为物理 op（pto.vadd 等 mi 层 IR）──► 汇入 4.5 公共尾段
```

这三步就是 vmi 层"只写逻辑、不管物理"的兑现点：作者在 IR 里只声明 lane 数与谓词，物理寄存器数量、布局、mask 粒度全部由编译器在此决定。

### 4.5 mi → 最终实现（vpto 公共尾段）

mi 层 IR（无论手写、tile 展开还是 vmi 物理化的产物）在 vpto 后端走统一的收尾路径。**emitc 同样不支持 mi 层**（`pto.mad` 等无法 legalize）。

```text
① vpto-split-cv-module      cube / vector section 拆分为子模块，分别编译
④ 发射前优化                 vpto-expand-wrapper-ops：wrapper op（含 cube bridge，
                             见下）展开为低层 pointer/raw op
                             vpto-optimize-vcvt / vpto-mask-simplify：物理指令级优化
                             pto-expand-soft-lib：软实现函数展开（回调 SoftLib，4.7）
                             pto-vpto-scheduler：A5 物理指令调度
                             pto-validate-vpto-emission-ir：发射前校验
⑤ VPTOCANN900LLVMEmitter    （非 pass）物理 op → llvm.hivm.* intrinsic，
                             cube / vector 各生成一个 LLVM module
⑥ ptoas 内部 bisheng        LLVM IR → cube.o / vector.o → ld.lld 合并 → fatobj
                             （ObjectEmission.cpp:524-584，命令见 5.2）
```

**cube bridge wrapper**：VPTO 物理 op 层有一类高层 wrapper op（`include/PTO/IR/VPTOOps.td:3168`："Cube bridge wrappers: fuse common register config sequences with the core cube transfer/load/store ops"），如 `pto.mte_gm_l1`、`pto.mte_l1_ub`——把"cube 传输前的寄存器配置序列"与传输 op 融合为一条（实现在 `lib/PTO/IR/VPTO/cbridge/`），发射前由 `vpto-expand-wrapper-ops` 展开。注意它属于 cube 数据通路搬移，与 CV 同步的 `tpush_to_aiv` 等 pipe op 不是同一机制；CV pipe op 目前仅 emitc 后端有 lowering（`PTOToEmitCCommon.cpp:1767-1782`），vpto 未实现（issue #1614）。

### 4.6 三层 lowering 去向总表

| 层 | tracing 生成的 op | emitc 后端 | vpto 后端 |
|---|---|---|---|
| tileop | `pto.tadd` | `emitc.call_opaque "TADD"` → pto-isa → bisheng（4.3） | `pto-expand-tile-op` 展开为 mi 层 IR（4.3） |
| vmi | `pto.vmi.vadd` | **不支持** | `vmi-*-assignment` + `vmi-to-vpto` 物理化为 mi 层 IR（4.4） |
| mi | `pto.vadd` | **不支持** | 本身即物理形式，直接进入 4.5 优化与发射 |

**三层构成一条 lowering 链：tileop 展开为 mi 层 IR，vmi 物理化为 mi 层 IR，mi 直接进入后端优化；三者最终都翻译为** `llvm.hivm.`***，经 ptoas 内部 bisheng 编成 fatobj。** emitc 后端不经过这条链——它只接 tileop，把语义与 lowering 分别交给 pto-isa 和 bisheng。

### 4.7 附：编译期回调机制（TileLib / SoftLib）

`pto-expand-tile-op` 与 `pto-expand-soft-lib` 需要的展开知识实现于 PTODSL Python 侧，C++ pass 在编译过程中反向调用 Python：

```text
C++ ExpandTileOp 处理 pto.tadd
   │  service->materialize(request, context, callback)
   ▼
NativeModule.cpp 中的 PythonTileLibService（_core 加载时已注册到 TileLibRuntime）
   │  ① 获取 GIL（C++ 执行 Python 代码的前提）
   │  ② import ptodsl.tilelib._compiler_runtime，调用其 materialize(target, op, 操作数规格, ...)
   ▼
Python TileLib 模板：在同一个 MLIRContext 中经 binding 现场构造展开后的 Module
   │  返回 (module, entry_symbol)
   ▼
C++ 校验（确认 Python 使用的 Context 与编译器一致，NativeModule.cpp:73-97）
   → clone 进编译单元 → pass 继续执行
```

实现要点（`tools/ptoas/NativeModule.cpp:49-167`）：`_core` 扩展加载时将 `PythonTileLibService` 注册到编译器全局 `TileLibRuntime`；pass 调用 `materialize`；C++经 pybind11 直接调用 Python 函数；Python 返回模块后由 C++ 校验并接收。

采用该设计的原因：① 展开策略属于高频演进的知识，Python 实现修改后无需重编编译器；② 模板作者使用 PTODSL 即可，无需掌握 MLIR C++ API；③ 展开产物是普通的 mi 层 IR，后续 pass 统一处理。

### 4.8 附：pass 编排与编译选项速查

**pass 编排**：pass 定义于 `include/PTO/Transforms/Passes.td`，实现于 `lib/PTO/Transforms/`，pipeline 编排于 `tools/ptoas/ptoas_pipeline.cpp`。

**编译选项机制**：选项是 `tools/ptoas/ptoas.cpp` 中的 `llvm::cl::opt<T>` 全局变量；`_core.main(argv)` → `runPTOASDriver` → `parseDriverInvocation`（`driver.cpp:1402-1432`，含注册、`ResetAllOptionOccurrences` 重置、`ParseCommandLineOptions` 解析）将值写入全局变量，pipeline 各处以普通 C++ 变量读取。选项对 pass 的控制有三种方式：**决定 pipeline 组成**（`--pto-backend`、`--enable-insert-sync`、`--pto-level=level3` 跳过内存规划等）、**决定输出模式与早停点**（三个 `--emit-*`）、**作为具体 pass 的行为参数**（`--vpto-scheduler`）。另有独立一层：pass 类中声明的 `Option<>` 成员经 `registerPassManagerCLOptions()` 接入。

**选项的五个类别**：

| 类别 | 代表选项 | 作用 |
|---|---|---|
| ① 目标与流水线配置 | `--pto-arch`、`--pto-backend`、`--pto-level` | 决定编到哪颗核、走哪条后端、pipeline 级别 |
| ② pass 功能开关与行为参数 | `--enable-insert-sync`、`--enable-plan-memory`、`--enable-op-fusion`、`--vmi-fastmath`、`--vpto-scheduler`(+`-remat`) 等 | 改变 IR 本身：决定哪些 pass 进 pipeline、pass 内部走哪条策略 |
| ③ 早停观测快照 | `--emit-pto-ir`、`--emit-vpto`、`--emit-vpto-llvm-ir` | 打印某深度 IR 到 `-o` 并退出，无后续消费者 |
| ④ 旁路调试输出 | `--vpto-print-ir`、`--vpto-scheduler-trace`、`--enable-unified-sync-debug`、`--emit-addptr-trace` | 信息打 stderr 或标注进产物，编译与产物不受影响 |
| ⑤ 测试 oracle 门 | `--check-sync-ids`、`--check-sync-interference`、`--dump/check-sync-coverage` 等 | 对本次编译结果做绝对/差分断言，服务于编译器自身回归测试 |

**观测选项的三种行为模式**：早停快照（三个 `--emit-*-ir`，对应 4.2 mainline 后 / 4.5 ⑤前 / hivm 翻译后三个深度，打印即退出，不需要 bisheng / 真机）；旁路观察（`--vpto-print-ir` 等打 stderr，编译继续）；产物标注（`--emit-addptr-trace` 只丰富 emitc 的 C++ 输出）。注意 `--emit-addptr-trace` 说明"emit-"前缀不等于早停。

```bash
ptoas --pto-arch=a5 --emit-pto-ir kernel.mlir -o stage1.mlir                  # 4.2 mainline 后：tile op 尚未展开
ptoas --pto-backend=vpto --emit-vpto kernel.mlir -o stage2.mlir               # 4.5 ⑤前：全部物理指令
ptoas --pto-backend=vpto --emit-vpto-llvm-ir kernel.mlir -o stage3.ll         # LLVM hivm 文本
```

**各场景下选项的去向**：PTODSL 跑 sim 时 ptoas 由 `native_build.py:_run_ptoas` 以 subprocess 内部调起——①② **仍在生效，只是换了入口**（`@pto.jit` 参数即上层映射：`backend` → `--pto-backend`，`mode="explicit"` → `--pto-level=level3`，`insert_sync=True` → `--enable-insert-sync`，`target` → 模块属性 `pto.target_arch`；未映射的经 `ptoas_overrides` 透传）；③④ 属调试阶段手动 CLI；⑤ 属编译器自身回归，与 sim 无关。

---

## 第五部分 · 产物、编译命令与运行

### 5.1 两条后端的产物对照

| | ptoas 直接输出 | 形态 | 后续步骤 |
|---|---|---|---|
| emitc | `kernel.cpp` | C++ 源文件（调用 pto-isa） | 由调用方用 bisheng 编译为 `.o` |
| vpto | `kernel.o` | fatobj ELF（host 空壳符号 + 嵌入 aibinary） | 直接进入链接 |

PTODSL 通过 `_is_elf_object` 检查 ELF 魔数判断 ptoas 实际走了哪条后端（`native_build.py:240-246`）。

### 5.2 编译命令

```bash
# ── 公共第一步：ptoas（native_build.py:58-73）──
# vpto：  ptoas --pto-arch=a5 --enable-insert-sync --enable-tile-op-expand kernel.mlir -o kernel.o
# emitc： ptoas --pto-arch=a5 --enable-insert-sync --enable-tile-op-expand kernel.mlir -o kernel.cpp
# （explicit 模式：附加 --pto-level=level3，去掉 --enable-insert-sync）

# ── vpto 后端的设备编译（在 ptoas 内部执行，ObjectEmission.cpp:524-584）──
bisheng --cce-aicore-arch=dav-c310-vec --cce-aicore-only -O2 \
        --cce-generic-addrspace=off -cce-bitcode-is-aicore -dc \
        --cce-long-scbz=true -mllvm -cce-dyn-kernel-stack-size=true \
        -c -x ir - -o vector.o                    # LLVM IR 经 stdin 输入；cube 同理
ld.lld -m aicorelinux -Ttext 0 -r --allow-multiple-definition cube.o vector.o -o merged.o
bisheng -cc1 -fcce-fatobj-compile -fcce-include-aibinary merged.o -x cce stub.cpp -o kernel.o

# ── emitc 后端：C++ → 目标码（在 ptoas 外部由调用方执行，native_build.py:190-209）──
bisheng -std=gnu++17 -O2 -fPIC -xcce -Xhost-start -Xhost-end \
        --cce-aicore-arch=dav-c310-vec -I <pto-isa>/include \
        -c kernel.cpp -o kernel.o

# ── 两条后端汇合：launch.o 与最终链接（native_build.py:168-237）──
bisheng -xcce --cce-aicore-arch=dav-c310-vec -c launch.cpp -o launch.o
bisheng -fPIC -shared --cce-fatobj-link -Wl,--no-undefined \
        -o libkernel.so launch.o kernel.o -lruntime          # 仿真时替换为 -lruntime_camodel
```

要点：**vpto 的设备编译在 ptoas 内部完成，emitc 的设备编译在 ptoas 外部由调用方完成。**

### 5.3 运行方式

launch.cpp 提供 `ptodsl_launch_*` 函数，使用 CCE host launch 语法 `kernel<<<grid, nullptr, stream>>>(...)`；Python 侧 `compiled[grid, stream](*ptrs)` 经 ctypes 调用该 `.so`。仿真模式下设置 `MSPROF_SIMULATOR_MODE=1`，链接 `libruntime_camodel` 并加载 camodel 的 ACL。**PTOAS 不包含 VPTO 字节码解释器；`ptobc` 是前端 PTO IR 的二进制编码格式，ptoas 解码后进入相同的 pipeline，不构成独立的执行后端。**

### 5.4 emitc 与 vpto 后端对比

| 维度 | EmitC | VPTO |
|---|---|---|
| 语义实现方式 | 调用 pto-isa C++ 库 | 编译器自行 lowering 至 LLVM hivm |
| 默认配置 | ptoas CLI 默认（`--pto-backend=emitc`） | `@pto.jit` 默认 |
| 支持的 mode | 仅 auto | auto + explicit（vmi / mi 必须使用此后端） |
| tile op 处理 | 不展开，映射为 pto-isa 调用（4.3） | `pto-expand-tile-op` 展开为微指令序列（4.3 / 4.7） |
| vmi / mi 支持 | 不支持（4.4 / 4.5） | 完整支持 |
| CV pipe op（`tpush_to_aiv` 等） | 有 lowering（`PTOToEmitCCommon.cpp:1767-1782`） | 未实现（issue #1614，见 4.5） |
| 中间形态 | emitc dialect → C++ 源文件 | 物理 VPTO IR → LLVM IR（`llvm.hivm.*`） |
| 调度 / 融合 | 交由 CCE 编译器 | `pto-fusion-region-gen` + `pto-vpto-scheduler` |
| ptoas 输出 | `.cpp` | fatobj ELF |
| 运行时 | 相同：fatobj + launcher .so + CANN / camodel | 相同 |
| 已知约束 | vmi / mi 写法不可用；`pto.mad` 无法 legalize | `tprint` 不支持；部分 tile 形态缺少 TileLib 模板；CV pipe op 未支持 |

---

## 第六部分 · 前端生态：TileLang 如何生成 PTODSL

> 本部分整合自 TileLang 前端文档 `tilelang_to_ptodsl.md`，原文（含 `ptodsl_dump/` 三个完整生成文件）已复制到本目录：`docs/tilelang_to_ptodsl.md`、`docs/ptodsl_dump/`；下文引用的 dump 文件均相对本目录。范围只到"TileLang 打印出 PTODSL Python 文本"为止，之后的流程即本文第三至第五部分。

前两条来源（PTODSL 手写、pto-func 裸 binding）都由人编写。第三条来源是 **TileLang 的 device codegen**：`target="pto"` 时，codegen 把最终 TIR **打印成 PTODSL Python 文本**——不是解释执行，也不是 ELF，也不经过 MLIR binding，而是直接生成调用 `pto.`* 的 Python 源码，再由 PTODSL tracing 成 MLIR。

教学顺序：**先确认 CV（文件外框）**，再填 Cube 侧和 Vector 侧（SIMD / SIMT），最后看 Mixed 如何把两边装进同一入口。

查看生成文本（不编设备码）：

```python
art = tilelang.lower(func, target="pto", enable_device_compile=False)
print(art.kernel_source)
```

下文的 PTODSL 生成文本是用 `$HOME/llvm-workspace/.venv` 对同名 kernel 执行下面代码得到的 `kernel_source` 原文，完整文件见本目录下的 `ptodsl_dump/`：

```python
from tilelang.ascend.target import normalize_pto_target
target = normalize_pto_target("pto")
with target:
    art = tilelang.lower(func, target=target, enable_device_compile=False)
```

UB 被 `MergeUBAllocations` 合成一块 `buf_dyn_shmem`：SIMD 整除例中 a/b/c 的元素偏移是 0 / 256 / 512。

### 6.1 产物是什么：两行 import，三个名字

生成文件固定两行 import：

```python
import tilelang.contrib.ptodsl as tl   # 生成代码用的兼容层
from ptodsl import pto                 # PTODSL 方言
```

| 名字 | 谁用 | 例子 |
|------|------|------|
| `T` | 你写 kernel | `T.gemm`、`T.SimdVF` |
| `pto` | 生成文本里的 ISA / 运行时 | `pto.vmi.vadd`、`pto.simt`、`pto.mte_*` |
| `tl` | 生成文本里 TileLang 补的 helper | `tl.PTOGemmL1Template`、`tl.finalize_mixed_kernel` |

`tl` 不是前端的 `import tilelang.ascend.language as T`。

### 6.2 先确认 CV：文件骨架

Ascend 一个 core group 上有 **AIC（Cube）** 和 **AIV（Vector）**。编译器看这个 PrimFunc 里两边各有没有活，投影成 1 或 2 个 SBlock，再决定 PTODSL 怎么包。

这是 **判定结果**，不是「每个 kernel 都要拆成 Mixed」。纯 Vector、纯 Cube 也是 CV 判定的结果（只有一边有活）。

| 结果 | TIR 上有什么 | PTODSL 外框 |
|------|----------------|-------------|
| 纯 Vector | 只有 `VECTOR`（或没有 CUBE 块） | `kernel_kind="vector"`，函数开头一次 `pto.init_core()` |
| 纯 Cube | 只有 `CUBE` / GEMM、L1 | `kernel_kind="cube"`，同样一次 `init_core`；常有 `insert_sync=False` |
| Mixed | **同时有** `CUBE` 和 `VECTOR` | **不写** `kernel_kind`；两段 section，各自 `init_core`，最后 `tl.finalize_mixed_kernel` |

分离发生在调度：`AssignCore` 给 task 贴核 mask，`LowerScheduledTIR` 把同一份 body 投影两遍。Codegen **只包装**，不再决定谁上哪颗核。

入口可以是：

- `T.Kernel` + 自动调度（`T.gemm` 和 Vector 写在一起，编译器自己拆）
- `T.MixedKernel`（多 `sid`）
- 手写 `T.Cube()` / `T.Vector()`（核边界你划）

三种入口，外框仍然只有上表三行。

对照同一套算子，看外框怎么变：

```python
# 只有 vecadd → 纯 Vector
with T.Kernel(1):
    T.copy(...)
    with T.SimdVF(lanes=128):
        for i in T.Parallel(256):
            c[i] = a[i] - b[i]

# 只有 gemm → 纯 Cube
with T.Kernel(1):
    T.copy(A, a_l1)
    T.copy(W, w_l1)
    T.gemm(a_l1, w_l1, c, transpose_B=True, clear_accum=True)
    T.copy(c, C)

# gemm + dual_copy epilogue → Mixed
with T.Kernel(32) as bx:
    T.copy(...)
    T.gemm(...)
    T.dual_copy(c_l0c, ub)
    T.dual_copy(ub, C)
```

Mixed 骨架（Cube / Vector 的转换都填进对应 `with`）：

```python
@pto.jit(name="main", target="a5", mode="explicit", insert_sync=True)
def main(...):
    with tl.mixed_kernel_section("cube"):
        pto.init_core()
        # ← Cube 侧填这里
    with tl.mixed_kernel_section("vector"):
        pto.init_core()
        # ← Vector 侧填这里

main = tl.finalize_mixed_kernel(main)
```

`finalize_mixed_kernel`：PTODSL 默认按 backend 切模块；合核必须改成扁平 AIC 布局，Cube/Vector 才能作为 **同一入口的两个物理 section**。公开 API 没有 `module_style=`，所以生成代码里补这一刀。Mixed **不能**再写一个 `kernel_kind`。

`vector_count==1` 时，Vector section 里还会有 `if pto.get_subblock_idx() == 0`：物理仍是双 AIV，只让 AIV0 跑这段（含跨核同步）。

### 6.3 Cube 侧怎么转

Cube 侧没有 SIMT / SIMD 那套 VF。典型是 L1 上的 `T.gemm`（以及 GM↔L1、L0 相关 copy）。

`T.gemm` **不会**在 `kernel_source` 里展开成一长串 `pto.mad`。Codegen 构造 helper，调用一次 `run_l1_tile`：

```python
@pto.jit(name="main", kernel_kind="cube", target="a5", mode="explicit", insert_sync=False)
def main(...):
    pto.init_core()
    a_l0_0 = pto.castptr(zero_addr, ...)
    b_l0_0 = pto.castptr(zero_addr, ...)
    _tl_gemm_l1 = tl.PTOGemmL1Template(tile_m, tile_n, tile_k, base_k, ...)
    # GM→L1：pto.mte_*
    _tl_gemm_l1.run_l1_tile(a_l1, b_l1, a_l0_0, b_l0_0, acc, ...)
```

量化路径是 `tl.PTOBlockscaledGemmL1Template`。DMA、指针、`init_core` 仍是 `pto.*`。

纯 Cube **会有** `tl.`*（GEMM template），**没有** `mixed_kernel_section` / `finalize_mixed_kernel`。

`run_l1_tile` 内部的 `pto.mte_l1_l0a` / `pto.mad` / `pto.set_flag` 是 **PTODSL 前端 tracing** 时才展开的，已经离开「打印 `kernel_source`」这一步。教转换时，看到 helper 调用就算 Cube 侧转换完成。

当前 L1 helper 要求 `transpose_B=True`（逻辑 B 为 `W[N,K]`）。这是 helper 契约，不是 CV 的一部分。

### 6.4 Vector 侧怎么转

AIV 上先选 **执行模型**，再谈指令。

| 模型 | 前端 | 含义 |
|------|------|------|
| SIMD | `T.SimdVF` | 无线程；一条向量指令打满 VL 个 lane |
| SIMT | `T.SimtVF` | N 个 workitem，每人跑一份（多为标量）代码 |

`T.copy` 进 UB 通常写在 VF **外面**，两边都是 `pto.mte_gm_ub`，不算第三种模型。

| | SIMD `T.SimdVF` | SIMT `T.SimtVF` |
|--|-----------------|-----------------|
| `T.Parallel` | 切成 VMI chunk | 绑到 `threadIdx` |
| PTO Parallel pass | `VerifyParallelToPTO` → `VectorizeParallelToPTO` | **整块跳过** |
| 产物 | `pto.vecscope` + `pto.vmi.*` | `with pto.simt(...)` + `pto.load` / `pto.store` |

`threads=128` ≠ `lanes=128`。

#### SIMD

三种写法，**同一** `T.SimdVF` **里不要把** `T.Parallel` **和手写** `T.vmi` **混在一起**。

`lanes` = 一条 VMI 的宽度 **L**（只允许 64 / 128 / 256，默认 128）。
`T.Parallel(E)` = 逻辑元素个数。
编译器做 `i = q * L + lane`，`q ∈ [0, ceil(E/L))`。
AscendC 忽略 `lanes`；只有 PTO 的 Parallel→VMI 用它。

**能整除：**`lanes=128`**，**`Parallel(256)` **→ 两个满 chunk**

```python
with T.Kernel(1):
    a_ub = T.alloc_shared((256,), "float32")
    b_ub = T.alloc_shared((256,), "float32")
    c_ub = T.alloc_shared((256,), "float32")
    T.copy(A, a_ub)
    T.copy(B, b_ub)
    with T.SimdVF(lanes=128):
        for i in T.Parallel(256):
            c_ub[i] = a_ub[i] - b_ub[i]
    T.copy(c_ub, C)
```

生成 PTODSL 原文（`ptodsl_dump/simd_div.ptodsl.py`）：

```python
@pto.jit(name="simd_div_kernel", kernel_kind="vector", target="a5", mode="explicit")
def simd_div_kernel(A: pto.ptr(pto.f32, "gm"), B: pto.ptr(pto.f32, "gm"), C: pto.ptr(pto.f32, "gm")):
  pto.init_core()
  bx = pto.cast(pto.get_block_idx(), pto.si32)
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  pto.mte_gm_ub(..., dst=ub+0, ...)    # A → a，1024B
  pto.mte_gm_ub(..., dst=ub+256, ...)  # B → b
  pto.set_flag("MTE2", "V", event_id=0)
  pto.wait_flag("MTE2", "V", event_id=0)
  with pto.vecscope():
    mask = pto.vmi.create_mask(128, size=128)   # 全程共用满 mask，没有 remaining
    for _tl_range_q in range(0, 2, 1):          # ceil(256/128) = 2
      q = pto.cast(_tl_range_q, pto.si32)
      vload_0 = pto.vmi.vload(pto.addptr(..., q * 128), 0, size=128)
      vload_1 = pto.vmi.vload(pto.addptr(..., (q * 128) + 256), 0, size=128)
      vsub_2 = pto.vmi.vsub(vload_0, vload_1, mask)
      pto.vmi.vstore(vsub_2, pto.addptr(..., (q * 128) + 512), 0, mask)
  pto.set_flag("V", "MTE3", event_id=0)
  pto.wait_flag("V", "MTE3", event_id=0)
  pto.mte_ub_gm(..., src=ub+512, C, 1024, ...)
```

要点：没有 `remaining`；`create_mask` 在 loop 外；两次 `vsub` 都用同一满 mask。DMA 与 VF 之间有 MTE2→V / V→MTE3 flag。

**不能整除：**`lanes=128`**，**`Parallel(200)` **→ 满 chunk + 尾巴**

逻辑上 200 个元素：第一拍 128，第二拍只该写 72。

- ISA 的 **load 不带 mask**（仍读满 128 lane）。
- **计算和 store 用动态 mask**，不写过 E。
- 调用方要保证 load 多读的 lane 可读（常见做法：UB 垫到 `ceil(E/L)*L`）。输出若只有 E 个元素，masked store 不会越写。

```python
with T.Kernel(1):
    a = T.alloc_shared((256,), "float32")   # 垫到 2*128，给尾块 vload 读
    b = T.alloc_shared((200,), "float32")   # 输出紧凑，正好 E
    T.copy(A, a)
    with T.SimdVF(lanes=128):
        for i in T.Parallel(200):           # E=200，不能整除 L
            b[i] = a[i] + a[i]
    T.copy(b, B)
```

```text
q=0: remaining=200 → mask 前 128 lane 有效 → 写 b[0:128]
     remaining -= 128 → 72
q=1: remaining=72  → mask 前 72 lane 有效
     vload 仍从 a[128] 读 128 个数
     vstore 只写 b[128:200]
```

生成 PTODSL 原文（`ptodsl_dump/simd_tail.ptodsl.py`）：

```python
  with pto.vecscope():
    remaining = pto.const(0, dtype=pto.si32)
    remaining = pto.const(200, dtype=pto.si32)   # 初值 = E
    for _tl_range_q in range(0, 2, 1):           # 仍是 2 个 chunk（含尾巴）
      q = pto.cast(_tl_range_q, pto.si32)
      mask = pto.vmi.create_mask(remaining, size=128)
      vload_0 = pto.vmi.vload(..., q * 128, 0, size=128)   # load 仍是满 128
      vload_1 = pto.vmi.vload(..., q * 128, 0, size=128)   # a[i]+a[i]，两次同一地址
      vadd_2 = pto.vmi.vadd(vload_0, vload_1, mask)
      pto.vmi.vstore(vadd_2, ..., (q * 128) + 256, 0, mask)  # b 从 UB 偏移 256 起
      remaining = pto.cast(remaining - 128, pto.si32)
  pto.mte_ub_gm(..., B, 800, nburst=(1, 800, 800), ...)    # 只写回 200 个 float32
```

和整除版的差别：循环前把 `remaining` 设成 E；**每拍** `create_mask(remaining, size=L)`；拍末 `remaining -= L`。写回 GM 的 burst 是 800 字节，不是垫过的 1024。

不要把 `Parallel` 写成 256 再自己 `if i < 200`：当前这条路径会拒带 guard 的 Parallel。契约见 `testing/ascend/language/pto/test_pto_parallel_vectorize_structure.py`（该文件主要断言 Vectorize 之后的 **TIR**，不是整份 PTODSL 文件）。

**SIMD 的另外两条路**

- **手写** `T.vmi.`*（同一 VF 无 `T.Parallel` 时，三个 PTO Parallel pass 跳过）：一对一变成 `pto.vmi.*`，没有编译器插入的 chunk / `remaining`，除非你自己写循环。`T.vmi.*` 只能写在 `T.SimdVF()` 里。
- **手写** `T.simd.`*：`T.simd.vld` / `vadd` / `vsts` → `pto.vlds` / `pto.vadd` 等，**不是** `pto.vmi`。自动向量化不会进这种手写块。

#### SIMT

和 SIMD 不是「换个参数」，是另一条 lowering。

```python
@T.prim_func
def main(A: T.Buffer((n,), "float32"), B: T.Buffer((n,), "float32"), C: T.Buffer((n,), "float32")):
    with T.Kernel(n // 128) as bx, T.SimtVF(threads=128):
        for i in T.Parallel(128):
            idx = bx * 128 + i
            C[idx] = A[idx] + B[idx]
```

流程：

1. `T.SimtVF` → SBlock `SIMT_VF`，body 开头绑 `thread_extent`（先 `threadIdx.x`；也可到 y/z，workitem ≤ 2048）。
2. Layout 把 `T.Parallel` 的 `i` 换成 **本线程编号**，并行圈消失，剩下标量（或短向量）访存。
3. `VerifyParallelToPTO` / `VectorizeParallelToPTO` **整块跳过** `SIMT_VF`。
4. Codegen 扫 `thread_extent`，包 `with pto.simt(tx, ty, tz):`。

生成 PTODSL 原文（`ptodsl_dump/simt_add.ptodsl.py`）。本例 `n=128`、`T.Kernel(1)`，`bx * 128 + i` 被折成只用 `tid_x`：

```python
@pto.jit(name="simt_add_kernel", kernel_kind="vector", target="a5", mode="explicit")
def simt_add_kernel(A: pto.ptr(pto.f32, "gm"), B: pto.ptr(pto.f32, "gm"), C: pto.ptr(pto.f32, "gm")):
  pto.init_core()
  bx = pto.cast(pto.get_block_idx(), pto.si32)
  with pto.simt(128, 1, 1):
    simtvf_tx = pto.get_tid_x()
    simtvf_ty = pto.get_tid_y()
    simtvf_tz = pto.get_tid_z()
    pto.store(pto.load(A, simtvf_tx) + pto.load(B, simtvf_tx), C, simtvf_tx)
```

对照：

```text
SIMD 产物:  with pto.vecscope():     pto.vmi.vload / vadd / vstore
SIMT 产物:  with pto.simt(128,1,1):  pto.get_tid_x(); pto.load / pto.store
```

- `pto.simt(128)` = 128 人各执行一遍缩进代码。
- `pto.vmi.*(size=128)` = 一条指令 128 lane。

`threads=(32, 4)` 会变成 `pto.simt(32, 4, 1)`，并用 `pto.get_tid_y()`。SIMT 里的短向量（`T.vectorized`）是 `pto.load(..., contiguous=k)`，仍然在 `pto.simt` 里面，不是 `pto.vmi`。

`SimtVF` 里 `Parallel` 的 extent 通常与 `threads` 对齐（或按 fragment 让每线程管一段）。

### 6.5 Mixed：把 Cube / Vector 装进同一外框

自动调度示例（不必手写 `T.Cube`）：

```python
with T.Kernel(32) as bx:
    res = T.alloc_l0c((256, 256), "float32")
    x_l1 = T.alloc_l1((256, 256), "bfloat16")
    temp = T.alloc_shared((128, 256), "float32")
    T.copy(...)
    T.gemm(...)
    T.dual_copy(res, temp)
    T.dual_copy(temp, C[...])
```

投影之后：Cube 段 = 6.3 的 helper + L1 DMA；Vector 段 = 6.4 的 SIMD / SIMT / DMA。PTODSL 就是 6.2 的两段 `mixed_kernel_section`。

跨核数据（L0C→UB、flag）在 **InsertSync** 已经写进 TIR；生成文本只是把 set/wait 打进对应 section。那是 **框与框怎么接**，不是第四种计算模型。

手写合核与自动拆核，**外框相同**，只是核边界是你划的还是调度推的。

### 6.6 对照表

| 你写的 | CV 结果 | Vector 模型 | 生成文本里一眼能认的片段 |
|--------|---------|-------------|--------------------------|
| `SimdVF(lanes=L)` + `Parallel(k*L)` | 纯 Vector | SIMD 满 chunk | `pto.vecscope`；loop 外 `create_mask(L)`；`range(0, k)`；`pto.vmi.*` |
| `SimdVF(lanes=L)` + `Parallel(E)`，E 不能整除 L | 纯 Vector | SIMD 带尾 | `remaining = pto.const(E)`；每拍 `create_mask(remaining, size=L)`；`remaining - L` |
| 手写 `T.vmi` | 纯 Vector | SIMD 手写 | `pto.vmi.*` 一对一，无编译器插入的 `remaining` |
| `T.simd.*` | 纯 Vector | SIMD MicroAPI | `pto.vlds` 等，无 `pto.vmi` |
| `SimtVF(threads=N)` + `Parallel` | 纯 Vector | SIMT | `with pto.simt(N,1,1):`；`pto.get_tid_x()`；`pto.load` / `pto.store` |
| 仅 `T.gemm` | 纯 Cube | （无 VF） | `kernel_kind="cube"`；`tl.PTOGemmL1Template` |
| `T.gemm` + Vector epilogue | Mixed | Cube + Vector | 无 `kernel_kind`；section + `finalize_mixed_kernel` |

转换流水线：

```text
TileLang
  → 共用 Ascend 调度 / 布局 / 同步
  → LowerScheduledTIR          【在这里确认 CV：1 块还是 2 块】
  → PTO 分叉                   【仅 SIMD+Parallel → VMI；SIMT / 手写 VMI 跳过】
  → LowerTileOp                【copy/gemm → intrinsic】
  → CodeGenTileLangPTO         【按 CV 选外框，按 Cube / SIMD / SIMT 填内容】
PTODSL 文本 ──► （接本文第三部分：PTODSL tracing → MLIR → ptoas）
```

### 6.7 最小记忆

1. 先问：**几块核？** 纯 Vector / 纯 Cube / Mixed → 三种 `@pto.jit` 骨架。
2. **Cube 填 helper**（`tl.PTOGemmL1Template`），DMA 用 `pto.`*。
3. **Vector 先选 SIMD 还是 SIMT**：SIMD 用 `lanes` + `pto.vmi`（或 `T.simd`）；SIMT 用 `threads` + `pto.simt`。
4. 看 SIMD 时再问 `E % L` 是不是 0——不是 0 就去对 `remaining` 那三段。
5. Mixed 只是把 Cube / Vector 放进两个 section，再 `finalize_mixed_kernel`。

### 6.8 实现与测试入口

| 主题 | 位置 |
|------|------|
| Ascend / PTO pass 分叉 | `tilelang/ascend/pipeline.py` |
| CUBE / VECTOR 投影 | `src/ascend/transform/lower_scheduled_tir.cc` |
| Parallel → VMI | `src/ascend/transform/vectorize_parallel_to_pto.cc` |
| PTODSL 打印 | `src/ascend/codegen/codegen_pto.cc` |
| GEMM helper | `tilelang/contrib/ptodsl/gemm.py` |
| Mixed 包装 | `tilelang/contrib/ptodsl/mixed_kernel.py` |
| SIMD 整除 / 尾巴 TIR 契约 | `testing/ascend/language/pto/test_pto_parallel_vectorize_structure.py` |
| SIMD 尾巴数值 | `testing/ascend/language/pto/test_vectorize_tail_mask_run.py` |
| SIMT `pto.load` / `pto.store` | `testing/ascend/language/pto/test_simtvf_local_integer_storage.py` |

---

## 第七部分 · 前端生态：PyPTO 与 PTO 工具链的两条建联路线

> PyPTO 侧证据基于 `/home/chenben/code/pypto`（0.3.0，Pro 模式 `pypto_pro`），引用格式为 `文件:行号`；PTOAS 侧证据即本文第四、五部分。

PTOAS README 将 **PyPTO** 列为与 PTODSL、CuTile 并列的前端框架（"支持 PyPTO、PTODSL、CuTile 等框架在 Python 端直接构建、操作和编译 PTO Bytecode"）。但 PyPTO 与前三个来源（3.2 / 3.3 / 第六部分）有一个本质差异：**它自带一套独立的 IR（PyPTO IR）与编译链**，不直接使用 PTOAS 的 Python binding。它与 PTO 工具链之间存在两条建联路线：

```text
                    用户 Python kernel（@pl.jit，section_cube / section_vector）
                                      │
                            parser → PyPTO IR（后端中立）
                            （block.* / system.* / ptr.*，225 个注册算子）
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼ 路线 A：pto-isa（pypto 仓内完整实现）               ▼ 路线 B：ptoas（设计预留，发射器不在 pypto 仓）
      BackendCCE + CCECodegen                             （PTO backend，缺失）
      逐算子翻译为 pto-isa C++ 文本                          PyPTO IR → PTO dialect MLIR（.pto）
            │                                                     │  ← 在此汇入本文 1.2 全景图的 MLIR 入口
            │                                                     ▼
            │                                              ptoas 编译器（本文第四部分）
            │                                              emitc 后端 / vpto 后端
            │                                                     │
      bisheng（--enable-pto-tile-fusion，                  bisheng（5.2，vpto 在 ptoas 内部、
      PTO tile-fusion 后端做指令 lowering）                 emitc 在外部）
            │                                                     │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
                    fatobj / .so → CANN runtime（真机）/ camodel（仿真）
```

### 7.1 路线 A：PyPTO → pto-isa C++ → bisheng

**转换流程（全部在 pypto 仓内，JIT 一次调用触发）**：

```text
① kernel[stream, block_dim](*args) 首次调用：绑定实参，确定编译签名
② parser：Python AST → PyPTO IR
③ IR passes（auto_mutex 的 allocate_mutex_ids / insert_sync、自动流水变换等）
④ CCECodegen.generate_single() 按 target（cube / vector）把每个 IR 算子
   翻译为一行 pto-isa C++：
     block.load   → TLOAD(dst.tile, src);
     block.add    → TADD(dst.tile, a.tile, b.tile);
     block.matmul → TMATMUL(dst.tile, l.tile, r.tile);
   （225 个翻译规则以 REGISTER_BACKEND_OP(BackendCCE, ...) 注册）
⑤ 组装 kernel.cpp：cube / vector 两段用 __DAV_CUBE__ / __DAV_VEC__ 宏合并，
   头部补 #include <pto/pto-inst.hpp>（jit.py:1858，头文件来自 cann-pto-isa 安装包）
```

**编译运行**：

```text
⑥ bisheng --npu-arch=dav-3510 -DREGISTER_BASE --enable-pto-tile-fusion ... kernel.cpp
   → bisheng 内置 PTO tile-fusion 后端把 TLOAD / TMATMUL 等虚拟指令 lowering 成硬件指令
   （compile_config.py:198-210）
⑦ 产出 .so（AICORE 二进制 + Host launcher），落在 build 目录
⑧ Python 加载 .so，ctypes 调 call_kernel(block_dim, stream, *abi_args) 下发 NPU
⑨ torch.npu.synchronize() 等待执行完成（kernel 启动对 Host 异步）
```

**路线 A 与 PTOAS emitc 后端是同构的两套独立实现**——两者最终都落到"pto-isa C++ + bisheng"，但发射器、同步插入、launcher 各自实现：

| 维度 | 路线 A（PyPTO 自有链路） | PTOAS emitc 后端（本文 4.3） |
|---|---|---|
| IR → C++ 发射器 | pypto 仓 `BackendCCE`（注册表 + 产码函数） | ptoas 仓 `EmitPTOManualPass`（PTO → emitc dialect → C++） |
| pto-isa 头文件 | `#include <pto/pto-inst.hpp>`（jit.py:1858） | `#include "pto/pto-inst.hpp"`（EmitPTOManualPass 插入） |
| 自动同步 | pypto 侧 `InsertSync` pass（auto_mutex，核内 mutex 令牌） | ptoas 侧 `pto-unified-sync`（`--enable-insert-sync`） |
| 指令 lowering | bisheng `--enable-pto-tile-fusion` | 同左（pto-isa 语义实现均在 pto-isa 库） |
| 设备编译位置 | pypto JIT 内部调 bisheng | ptoas 外部由调用方调 bisheng（5.2） |
| 产物与运行 | .so + ctypes `call_kernel` | fatobj + launch.o 链接 .so + ctypes |

### 7.2 路线 B：PyPTO IR → PTO MLIR → ptoas

路线 B 的汇合点即本文 1.2 全景图中的"PTO dialect 的 MLIR 模块"——PyPTO 若能将自己的 IR 翻译为 PTO MLIR，之后的流程与 PTODSL / TileLang 完全一致（第四、五部分）：

```text
① ～ ③ 与路线 A 相同（Python → PyPTO IR → passes，两路线共享）
④' PTO backend codegen（不在 pypto 仓）：PyPTO IR → PTO dialect MLIR
    映射路标（pypto 仓 IR 注册处注释）：
      ptr.addptr            → pto.addptr            （ptr_ops.cpp:91）
      ptr.make_tensor       → pto.make_tensor_view  （ptr_ops.cpp:206）
      block.make_mutex_tile → pto.alloc_tile        （memory.cpp:149，可省 init_memref pass）
      block.set_validshape  → pto.set_validshape    （out_memory.cpp:316）
      system.mutex_lock_dyn → pto.get_buf           （sync.cpp:293）
      system.sync_all       → pto.syncall           （sync.cpp:198）
⑤' ptoas 编译（即本文 5.2 的命令）：
     vpto：  ptoas --pto-arch=a5 --enable-insert-sync --enable-tile-op-expand kernel.mlir -o kernel.o
     emitc： ptoas --pto-arch=a5 --enable-insert-sync --enable-tile-op-expand kernel.mlir -o kernel.cpp
⑥' 之后与本文 5.2 / 5.3 完全相同：fatobj → launch.o 链接 .so → CANN / camodel 运行
```

**两侧的实现现状不对称**：

- **PTOAS 侧就绪**：PTO MLIR 的输入口（`.pto` 文本 / PTOBC）、ptoas CLI、emitc / vpto 双后端、launcher 链接与运行方式全部是现成的（本文第三至五部分）；README 还明确将 PyPTO 列为可经 Python binding 构造 PTO IR 的前端；
- **PyPTO 侧只有路标**：`PtrType` 对齐 `!pto.ptr<dtype>`（type.h:647）且带 ptoas 专用的 `base_ptr`/`offset` 注解（"those are ptoas-only"，backend_cce_ops.cpp:488）；codegen 基类按"C++ 或 MLIR SSA 名"双产物设计（codegen_base.h:27-32，`f_codegen_cce` / `f_codegen_pto` 命名约定）；动态 shape ABI 顺序 "mirrors ptoas codegen"（jit.py:656）；但**没有 MLIR 发射器、没有 `BackendPTO`、没有任何 ptoas 调用**——JIT 配置硬编码只接受 CCE 后端，现状注解为 `// Single-program CCE generation (skip ptoas)`（cce_codegen.cpp:449）。

即：路线 B 的前半程（Python → PyPTO IR）与路线 A 共用且现成，后半程（PTO MLIR → ptoas → 运行）由 PTOAS 提供且现成，**缺的只是中间"PyPTO IR → PTO MLIR"的发射器**。

### 7.3 同步语义：IR 层共享，lowering 层分叉

两条路线的分叉点在最后产码。同步语义（谁等谁、event id、slot 轮转）在 PyPTO IR 层已定死（auto_mutex 插 `system.mutex_lock/unlock_dyn`、自动流水插 `set/wait_cross_core`），同一条 IR 算子在两路分别落到：

| PyPTO IR | 路线 A（pto-isa C++） | 路线 B（pto dialect，注释标注） |
|---|---|---|
| `system.mutex_lock/unlock_dyn` | `get_buf(PIPE, id, 0)` / `rls_buf(...)` | `pto.get_buf` / `pto.rls_buf` |
| `system.sync_all` | `SYNCALL<...>()` / `AscendC::SyncAll()` | `pto.syncall` |
| `system.sync_src/dst_dyn` | `set_flag(PIPE,PIPE,id)` / `wait_flag(...)` | pto dialect flag ops（pypto 仓无标注） |
| `system.set/wait_cross_core` | `set_intra_block`（1:2 时 id+16）/ `ffts_cross_core_sync` / `wait_flag_dev` | pto dialect cross-core ops（pypto 仓无标注） |

注意 pto dialect 基本是 pto-isa 指令集的 MLIR 镜像（名字只差 `get_buf` ↔ `pto.get_buf` 这层包装），因此 CV 同步语义在两路之间不需要翻译。另见本文 5.4：CV pipe op（`tpush_to_aiv` 等）在 PTOAS 侧目前也只有 emitc 后端有 lowering。

### 7.4 两条路线对照与定位

| | 路线 A：pto-isa | 路线 B：ptoas |
|---|---|---|
| 中间载体 | pto-isa C++ 文本（kernel.cpp） | PTO dialect MLIR 文本（.pto / .mlir） |
| 发射器 | pypto 仓 `BackendCCE` + `CCECodegen`（现成） | PTO backend（缺失，pypto 仓仅有接口与标注） |
| 编译器 | bisheng PTO tile-fusion 后端 | ptoas（emitc / vpto）→ bisheng |
| 自动同步 | pypto `InsertSync`（auto_mutex） | ptoas `pto-unified-sync`（--enable-insert-sync） |
| 内存规划 | pypto `init_memref` pass + `TASSIGN` | ptoas `pto-plan-memory`（或 IR 直出 `pto.alloc_tile`） |
| 优化归属 | 调度 / 融合交给 bisheng | ptoas mainline（融合、调度、内存规划，见 4.2） |
| 现状 | 端到端产品化 | 两端就绪，中间发射器未实现 |

**定位**：路线 A 是 PyPTO 当前的产品化链路——自带一个与 PTOAS emitc 后端同构的 "IR → pto-isa C++" 发射器，把指令 lowering 交给 bisheng；路线 B 是架构上预留的汇聚链路——让 PyPTO 与 PTODSL / TileLang 一样汇入 PTO MLIR，把融合、内存规划、同步插入统一交给 ptoas 的 mainline（这正是 PTOAS"一份 IR、一套优化流水线"设计所面向的场景，见第八部分）。

---

## 第八部分 · 总结：PTOAS 的定位与职责

**问题背景**：昇腾 NPU 包含 Cube（矩阵）、Vector（SIMD）、SIMT（标量组）三个异构计算单元，以及 MTE 数据搬移引擎与共享 UB。直接硬件编程需要手工管理寄存器、mask 粒度、跨单元同步与 DMA，复杂度高；同时多个上层框架（PyPTO、PTODSL、CuTile、TileLang 等）都需要生成 NPU 代码。

**PTOAS 的方案**：定义硬件抽象的虚拟指令集 PTO，以 MLIR dialect 形式实现，使各前端汇聚到同一份 IR，由编译器统一完成验证、优化、同步插入与代码生成。具体承担四项职责：

1. **一份 IR，三层抽象入口**：tile 指令面向算子级编写，vmi 逻辑向量面向 SIMD 级编写，mi 物理微指令面向指令级编写；三者属于同一 dialect，编译器内部将高层逐级 lowering 到低层（tile→mi、vmi→mi，第四部分）。
2. **一个 Python 接口层，多条构造路径**：构建期生成的 Python binding 是 Python 构造 MLIR 的通道；PTODSL 在其上提供自动化封装（子命名空间适配 + tracing），pto-func 直接调用 binding，TileLang 以代码生成方式产出 PTODSL 文本；编译器在需要 Python 侧知识时（TileLib / SoftLib）经同一 binding 回调。
3. **一套优化流水线**：布局推断、UB 内存规划、自动同步插入、算子融合、tile op 展开、VMI 物理化、指令调度，由若干单一职责的 pass 组成；driver 层编译选项决定 pipeline 组成、输出模式与 pass 行为参数（4.8）。
4. **两条后端，同一产物形式**：emitc 将语义交由 pto-isa C++ 库实现（支持 tprint 调试，仅 tile 层 auto 模式）；vpto 由编译器自行 lowering 到 LLVM hivm（默认后端，支持全部三层）。两者最终均生成 CCE fatobj，经 launcher `.so` 在 CANN runtime（真机）或 camodel（仿真）上执行。

**总结**：三层抽象入口与多条前端路径汇聚为同一份 PTO IR，经一次编译生成可在 NPU 上执行的目标代码——这就是 PTOAS。
