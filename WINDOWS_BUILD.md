# Windows 构建指南（已验证）

本分支为 `flash-attention-legacy` 增加了 Windows 构建支持。以下修改在一台真实的
Windows + Volta 机器上开发并验证，完整测试记录见[验证结果](#验证结果)。

## 验证环境

| 项目 | 版本 |
|---|---|
| 操作系统 | Windows 11 / x64 |
| Python | 3.12.10（便携版 python_embeded） |
| PyTorch | 2.10.0+cu128 |
| CUDA Toolkit | 12.8（nvcc 12.8.61） |
| 编译器 | MSVC 2022 BuildTools（cl.exe 19.44.35207） |
| GPU | Tesla V100-SXM2-16GB（SM 7.0 / Volta） |
| 构建后端 | setuptools + `torch.utils.cpp_extension`（ninja） |

## 修复内容

### 1. `setup.py` — MSVC 编译器参数（宿主代码此前无优化编译）

原代码把 GCC 风格参数传给宿主编译器：

```python
cxx_flags = ["-O3", "-std=c++17"]
```

`cl.exe` 会静默忽略未知参数（警告 **D9002**），导致 C++ 桥接代码
（`flash_attn_legacy.cpp`）以 **`/Od`（完全不优化）** 编译。修复为在 Windows
上使用 MSVC 原生参数：

```python
if sys.platform == "win32":
    cxx_flags = ["/O2", "/std:c++17"]
else:
    cxx_flags = ["-O3", "-std=c++17", "-fdiagnostics-color=always"]
```

说明：PyTorch 的 `BuildExtension` 也会注入 `/std:c++17`，这里显式写出是为了
意图清晰；真正的修复是 `/O2`。

### 2. `csrc/kernels/flash_attn_bwd.cu` — CUDA 内置函数加全局限定

`flash_attn_legacy` 命名空间内有 4 处调用 CUDA 内置函数 `atomicAdd` 和
`__shfl_down_sync` 时**缺少全局 `::` 前缀**。Windows 上的 nvcc 会报错：

```
error: no instance of overloaded function "flash_attn_legacy::atomicAdd" matches
```

（这些行在 Linux 上能编译只是侥幸通过宽松的名字查找。）修复为 4 处全部加
`::` 前缀，与文件内第 618 行 `::atomicAdd` 的既有写法保持一致。

## 构建与安装

```powershell
# 在 VS2022 x64 开发者命令行中执行（或先运行 vcvarsall.bat x64）
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:TORCH_CUDA_ARCH_LIST = "7.0"   # 可选；不设则按 setup.py 编译全部架构
pip install -e . --no-build-isolation
```

### 打包可复用 wheel

```powershell
pip wheel . --no-build-isolation --no-deps -w dist
```

生成 `flash_attn_legacy-0.5.0-cp312-cp312-win_amd64.whl`（约 3.5 MB）。
该 wheel 是自包含的（内含编译好的 `.pyd`），可在任何 **Python 3.12 +
PyTorch 2.10.0+cu128 + CUDA 12.8 运行时**的机器上免编译直接安装：

```powershell
pip install flash_attn_legacy-0.5.0-cp312-cp312-win_amd64.whl
```

## 验证结果

以下测试均在上表所述机器（Tesla V100，SM 7.0）上运行。

### 构建

- `python setup.py build_ext --inplace` → **退出码 0**，生成
  `flash_attn_legacy_cuda.cp312-win_amd64.pyd`
- `pip wheel . --no-build-isolation --no-deps` → **退出码 0**，wheel ZIP 校验通过

### 正确性（对比 eager attention，fp16，causal）

| 测试 | 结果 |
|---|---|
| MHA 前向（B=1,H=8,N=1024,d=64） | ✅ 与 eager 最大误差 **0.00098**（fp16 正常范围） |
| GQA（8 个 Q 头，2 个 KV 头） | ✅ 输出形状 (2, 8, 1024, 64) |
| `flash_attn_varlen_func`（packed） | ✅ 输出形状 (448, 8, 64) |
| `check_gpu_compatibility()` | ✅ "Tesla V100-SXM2-16GB is supported"（Volta WMMA 路径） |

### 性能（B=1, H=16, N=2048, d=64, fp16, causal, V100）

| 指标 | legacy | PyTorch SDPA | eager |
|---|---|---|---|
| 耗时 | 4.20 ms | 0.46 ms | 4.56 ms |
| 峰值显存 | **29.6 MB** | — | 578.9 MB |

legacy 比 eager 快约 1.09 倍，且显存节省 **95%**（29.6 MB vs 578.9 MB），
与 README 基准一致。在 V100 上，PyTorch 内置 SDPA（memory-efficient 后端）
纯速度更快；legacy 的价值在于：让硬性要求 `flash_attention_2` 的 HF 模型
（通过 `attn_implementation="flash_attn_legacy"`）在 V100 上可运行，以及
显著的显存节省。

### ComfyUI 回归测试

宿主 ComfyUI 0.30.1 环境安装 `flash-attn-legacy` 后仍正常启动
（18 秒监听端口，217 个节点加载）——无副作用。
