---
title: CANN-Bench 精度验证模块分析
date: 2026-08-03
categories: CANN
tags:
  - Precision
  - CANN
mathjax: true
plugins:
  - mathjax
---

> 基于 `cann-bench/src/kernel_eval/` 源码和 `docs/design/precision_comparison_design.md` 设计文档分析。
>
> 项目地址：[CANN-Bench](https://gitcode.com/cann/cann-bench?source_module=search_project)

---

## 一、精度验证的输入数据是如何产生的？

精度验证涉及**三组数据**，分别来自**三个不同函数/执行路径**，但基于**同一组输入张量**：

### 数据流全景

```
                         CaseSpec (cases.yaml/csv)
                              │
                   ┌──────────┼──────────┐
                   ▼          ▼          ▼
            input_tensors (data_generator.py:generate_input_tensors_from_case)
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ ① AI 输出     │   │ ② Golden 输出 │   │ ③ Native 输出 │
  │ (NPU 上执行)  │   │ (FP64 CPU)   │   │ (同精度 CPU)  │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                  │                  │
         ▼                  ▼                  ▼
  ┌──────────────────────────────────────────────────────┐
  │              compare_tensors(a, g, native)            │
  │              (compare.py:680)                         │
  └──────────────────────────────────────────────────────┘
```

### ① AI 输出（被测对象）

- **来源**：AI 生成的 Ascend C 算子代码在 NPU 设备上执行后的输出
- **调用路径** (`eval/evaluator.py:284-286`):

```python
ai_result = self._run_ai_op_isolated(actual_ai_func, params, ...)
```

- **底层执行**：通过 `OpRunner.run()` → NPU kernel launch → `torch.as_strided` 或 `numpy.frombuffer` 转为 Tensor

### ② Golden 输出（FP64 高精度参考）

- **来源**：`golden.py` 中定义的 PyTorch 参考函数，用 **FP64 精度在 CPU** 上计算
- **调用路径** (`eval/evaluator.py:239-244`):

```python
oracle_func = self._oracle_reference_func(case.rel_path, golden_func)
golden_inputs = self._apply_golden_precision(input_tensors)  # → FP64
golden_result = self.op_runner.run(oracle_func, golden_params, ...)
```

- **Golden 策略** (`eval/evaluator.py:829-844`): 支持三种模式：

| 策略 | 精度 | 设备 | 说明 |
|------|------|------|------|
| `fp64_cpu`（默认） | float64 | CPU | 输入提升到 float64，避免 NPU 溢出污染 |
| `native_cpu` | 原始 dtype | CPU | 保持原始精度在 CPU 上计算 |
| `native_npu` | 原始 dtype | NPU | 保持原始精度在 NPU 上计算 |

### ③ Native 输出（同精度参考，用于小值域/相消对照）

- **来源**：与 golden **同一函数**，但输入保持**目标 dtype 精度**（不做 FP64 提升）
- **调用路径** (`eval/evaluator.py:313-323`):

```python
bench_func = self._bench_reference_func(case.rel_path, golden_func)
native_inputs = tensors_to_cpu(input_tensors)  # 保持原始精度
native_result = self.op_runner.run(bench_func, native_params, ..., to_device=False)
```

- 当 golden 策略本身是 `native_cpu`/`native_npu` 时，native 直接复用 golden 结果优化性能

### 关键结论

| 数据 | 执行函数 | 精度 | 设备 |
|------|---------|------|------|
| AI output | AI 生成的算子代码 | 原始 dtype | NPU |
| Golden output | `golden.py` / oracle 钩子 | FP64 | CPU |
| Native output | `golden.py` / bench 钩子 | 原始 dtype | CPU |

**三组数据来自两个不同函数（AI算子 vs golden参考），但同精度参考和FP64参考可以共用同一个 golden 函数，区别仅在于输入精度和计算设备。**

---

## 二、精度验证的具体逻辑与数学公式

系统支持两种 Checker：`RelativeErrorChecker`（**默认**，CANN 评测集选用）和 `AllCloseChecker`（StanfordBench 选用）。

### 2.1 总览：判定流程

整数类型走逐元素精确匹配，threshold=0 的浮点走字节级 bit-exact 比较，其余普通浮点（fp16/bf16/fp32/...）分两阶段：

> **阶段 1**：计算全局 MERE（平均相对误差）和 MARE（最大相对误差），两者都达标则直接 PASS——绝大多数 case 走这条路径。
>
> **阶段 2**：阶段 1 不通过时触发。将元素按 golden 绝对值大小分为三个互不重叠区域——小值域（golden 极小）、相消区（golden 在精度边界且 output≈0）、正常值域（其余）——每个区域独立做 NPU vs CPU 同精度对照。三个区域全部通过才算 PASS，任一不通过则 FAIL。

**核心思想**：阶段 2 用"CPU 同精度参考是否也产生同样的误差"来区分浮点精度不够导致的必然误差和算子算错。

---

### 2.2 阶段 1：全局相对误差

#### 2.2.1 预处理

FP64 Golden 先截断到算子输出精度再升回 FP64，保证双方在同等位宽下比较。

#### 2.2.2 逐元素相对误差

对每个有效位置（排除 NaN、Inf、双方同号 Inf），计算：

$$rel\_err_i = \frac{|output_i - golden_i|}{|golden_i| + 10^{-7}}$$

分母加 $10^{-7}$ 仅用于防除零。

#### 2.2.3 聚合指标

**MERE**（Mean Relative Error，平均相对误差）：所有有效位置 `rel_err` 的均值，反映整体偏离程度。

**MARE**（Max Relative Error，最大相对误差）：所有有效位置 `rel_err` 的最大值，反映最差单个元素的偏离程度。

#### 2.2.4 判定

若 $MERE < \theta$ 且 $MARE < 10 \times \theta$，则直接 PASS；否则进入阶段 2 兜底。

MARE 线设为 10 倍是因为平均值看整体趋势、最大值看个别元素，放宽 10 倍避免边缘舍入差异误杀。

#### 2.2.5 阈值 $\theta$

| dtype | $\theta$ | 尾数位数 | MARE 上限 $(10\theta)$ |
|-------|----------|----------|----------------------|
| float64 | $2^{-13} \approx 1.22 \times 10^{-4}$ | 52 | $10 \times 2^{-13} \approx 1.22 \times 10^{-3}$ |
| float32 | $2^{-13} \approx 1.22 \times 10^{-4}$ | 23 | $10 \times 2^{-13} \approx 1.22 \times 10^{-3}$ |
| hifloat32 | $2^{-11} \approx 4.88 \times 10^{-4}$ | 19 | $10 \times 2^{-11} \approx 4.88 \times 10^{-3}$ |
| float16 | $2^{-10} \approx 9.77 \times 10^{-4}$ | 10 | $10 \times 2^{-10} \approx 9.77 \times 10^{-3}$ |
| bfloat16 | $2^{-7} \approx 7.81 \times 10^{-3}$ | 7 | $10 \times 2^{-7} \approx 7.81 \times 10^{-2}$ |
| float8_e4m3fn | $2^{-3} \approx 1.25 \times 10^{-1}$ | 3 | $10 \times 2^{-3} \approx 1.25$ |
| float8_e5m2 | $2^{-2} \approx 2.50 \times 10^{-1}$ | 2 | $10 \times 2^{-2} \approx 2.50$ |

整数类型（int8/16/32/64, uint8/16/32/64）：$\theta = 0$，要求逐元素精确相等。

> 未知 dtype 回退 float32 阈值。

---

### 2.3 阶段 2：分区域兜底

阶段 1 不通过时触发。将元素按 $|golden_i|$ 大小分为三个互不重叠区域：

- **小值域**：$|golden_i| < \delta_{small}$。golden 极小，相对误差无意义，改用绝对误差判定。
- **相消区**：$\delta_{small} \leq |golden_i| < \tau_{cancel}$ 且 $|output_i| \approx 0$。疑似 Kahan 灾难性相消导致 output 归零。
- **正常值域**：其余有效位置。相对误差可信，正常判定。

每个区域采用统一的同精度对照模式：分别统计 NPU 和 CPU 在该区域的错误数，若满足以下条件则通过：

$$\frac{Err_{npu}}{\max(Err_{cpu},\ 1)} \leq 2$$

区别在于各区域的**错误定义**：

- **正常值域**：元素相对误差超过 $10\theta$ 即计入错误。额外规则——若 $Err_{npu} = 0$ 直接通过；若 $Err_{npu} > 0$ 但 $Err_{cpu} = 0$ 直接不通过（CPU 干净则 NPU 必有 bug）。
- **小值域**：元素绝对误差 $|output_i - golden_i|$ 超过 $\epsilon_{small}$ 即计入错误。
- **相消区**：同正常值域，相对误差超过 $10\theta$ 即计入错误。

最终判定：

$$PASS = \text{正常值域通过} \land \text{小值域通过} \land \text{相消区通过}$$

多输出算子每个输出独立执行上述流程，全部通过才算整体 PASS。

#### 2.3.1 小值域阈值

| dtype | $\delta_{small}$ | $\epsilon_{small}$ |
|-------|-----------------|-------------------|
| float64 | $2^{-14} \approx 6.10 \times 10^{-5}$ | $2^{-30} \approx 9.31 \times 10^{-10}$ |
| float32 | $2^{-14} \approx 6.10 \times 10^{-5}$ | $2^{-30} \approx 9.31 \times 10^{-10}$ |
| hifloat32 | $2^{-12} \approx 2.44 \times 10^{-4}$ | $2^{-28} \approx 3.73 \times 10^{-9}$ |
| float16 | $2^{-11} \approx 4.88 \times 10^{-4}$ | $2^{-16} \approx 1.53 \times 10^{-5}$ |
| bfloat16 | $2^{-8} \approx 3.91 \times 10^{-3}$ | $2^{-16} \approx 1.53 \times 10^{-5}$ |
| float8_e4m3fn | $2^{-4} \approx 6.25 \times 10^{-2}$ | $2^{-6} \approx 1.56 \times 10^{-2}$ |
| float8_e5m2 | $2^{-3} \approx 1.25 \times 10^{-1}$ | $2^{-5} \approx 3.13 \times 10^{-2}$ |

#### 2.3.2 相消阈值

物理背景是 Kahan 灾难性相消：两个大数相减得到小结果时浮点有效数字丢失。例如 fp32 下 $10000.5 - 10000.3$，精确结果 $=0.2$，但 fp32 约 7 位有效数字，$10000.5$ 已占 6 位，减法后最后一位不可靠——NPU 和 CPU 都可能输出 0 甚至符号错误。

阈值基于 IEEE 754 尾数位数设计：尾数越少，相消发生的值域越宽。

| dtype | $\tau_{zero}$（判定 $output \approx 0$） | $\tau_{cancel}$（golden 上限） |
|-------|-----------------------------------------|------------------------------|
| float64 | $2^{-8} \approx 3.91 \times 10^{-3}$ | $2^{-8} \approx 3.91 \times 10^{-3}$ |
| float32 | $2^{-8} \approx 3.91 \times 10^{-3}$ | $2^{-8} \approx 3.91 \times 10^{-3}$ |
| hifloat32 | $2^{-8} \approx 3.91 \times 10^{-3}$ | $2^{-8} \approx 3.91 \times 10^{-3}$ |
| float16 | $2^{-5} \approx 3.13 \times 10^{-2}$ | $2^{-5} \approx 3.13 \times 10^{-2}$ |
| bfloat16 | $2^{-3} \approx 1.25 \times 10^{-1}$ | $2^{-3} \approx 1.25 \times 10^{-1}$ |
| float8_e4m3fn | $2^{-1} \approx 5.00 \times 10^{-1}$ | $2^{-1} \approx 5.00 \times 10^{-1}$ |
| float8_e5m2 | $2^{0} = 1.00$ | $2^{0} = 1.00$ |

---

### 2.4 NaN/Inf 特殊处理

- NPU 和 Golden 同一位置都为 NaN → 视为匹配，不参与误差计算
- NaN 位置不一致 → 直接 FAIL（计算路径有差异）
- NPU 为 Inf、Golden 为有限值 → Inf 替换为 dtype 最大有限值后继续比较（fp16 饱和溢出场景）
- 双方为 Inf 且同号 → 视为匹配
- 双方为 Inf 但异号 → 直接 FAIL

---

### 2.5 AllCloseChecker（简化备选标准）

StanfordBench 等评测集使用，逻辑简单：对每个元素判断 $|output_i - golden_i| \leq atol + rtol \times |golden_i|$，所有元素都满足则 PASS。默认 $atol = rtol = 0.01$，不做分区域兜底。

---

### 2.6 阈值配置优先级

1. **算子级**（最高）：`proto.yaml` → `precision_thresholds`
2. **Bench 级**：`bench_config.dtype_tolerance_map`
3. **CLI 级**：`--precision-thresholds`
4. **默认表**（最低）：`utils/thresholds.py` → `PRECISION_THRESHOLDS`

---

## 三、验证失败时采取的措施

系统设计了**多层故障处理机制**：

### 3.1 详细诊断信息收集

**逐输出精度指标**（嵌入在 `CompareResult` 中，`utils/compare.py:88-114`）：

- MERE / MARE / max_diff / mean_diff
- mismatch_count / total_count / mismatch_ratio
- 正常值域/小值域/相消位置的各自 NPU 错误数、CPU 错误数、总数
- 各值域的兜底判定结果（`small_value_passed` / `cancel_passed` / `normal_passed`）

**格式化输出**（`checkers/relative_error_checker.py:95-154`）：每个输出有一行摘要，包含 MERE/MARE、阈值、以及如果触发了兜底判定则追加小值域/相消的 NPU/CPU 错误对比信息。

### 3.2 失败类型分类

在 `base/result.py:204-234` 中区分两种失败类型（影响评分权重分配，见 README 评分公式）：

| failure_type | 触发条件 | 分值影响 |
|-------------|---------|---------|
| `compile_runtime_error` | 形状不匹配/输出数量不一致/算子崩溃/golden 执行失败 | 整算子计 0 分（w_c + w_f + w_p 全扣） |
| `precision_mismatch` | 相对误差超标/MERE-MARE 不通过 | 仅扣 w_f + w_p 分（w_c 编译分保留） |
| `cascade_device` | NPU 设备不可恢复，级联跳过 | 不计入 failed（跳过标记） |
| `oom_killed` | 子进程被 OOM Kill | 按编译/运行时错误计分 |

分类逻辑（`base/result.py` `STRUCTURAL_FAILURE_MARKERS`）：

```python
STRUCTURAL_FAILURE_MARKERS = (
    "输出数量不匹配", "同精度输出数量不匹配", "形状不匹配",
    "shape mismatch", "is None", "AI算子执行失败",
    "TypeError:", "RuntimeError:", "ValueError:",
)
```

### 3.3 防作弊二次验证

当 `Config.enable_accuracy_retry=True` 时（`eval/evaluator.py:359-369`）：

1. 第一轮输入通过精度验证后，使用**偏移种子（seed+1）+微扰（所有浮点 Tensor +0.01）**生成全新输入
2. 用新输入重新执行 golden + AI 算子
3. **两轮都通过才算最终 PASS**
4. 二次验证异常只在 stdout 输出 WARN，不阻断评测流程（返回第一轮结果）

---

## 四、核心文件索引

| 文件 | 职责 |
|------|------|
| `src/kernel_eval/utils/compare.py` | 张量精度对比引擎（`compare_tensors` / `_compare_single_tensor`） |
| `src/kernel_eval/utils/thresholds.py` | 所有阈值表（精度/小值域/相消）及查询函数 |
| `src/kernel_eval/checkers/relative_error_checker.py` | RelativeErrorChecker 实现，封装 compare_tensors |
| `src/kernel_eval/checkers/allclose_checker.py` | AllCloseChecker 实现（StanfordBench 用） |
| `src/kernel_eval/eval/accuracy_eval.py` | AccuracyEvaluator：统一的精度评测入口 |
| `src/kernel_eval/eval/evaluator.py` | Evaluator：综合评测调度器（golden 执行 + AI 执行 + 精度对比 + 性能采集 + 故障恢复） |
| `src/kernel_eval/base/result.py` | 结果基类（AccuracyResult / OutputResult / PerfResult / failure_type 分类） |
| `src/kernel_eval/base/checker.py` | CorrectnessChecker 抽象基类 |
| `docs/design/precision_comparison_design.md` | 精度对比方案设计文档（MERE/MARE + 小值域 + 相消理论） |
