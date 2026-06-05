# Public Code

## English

This directory contains the public implementation path for the paper.

- `lasrc.py`: LASRC, the layer-adaptive stabilized residual composition operator.
- `scale_fg_main.py`: SCALE-FG primitive candidate construction and support-time selection data generation.
- `scale_fg_table.py`: result aggregation for the LASRC and SCALE-FG rows.
- `algorithm.py`: shared LoRA loading, optimization, composition, and inference helpers.
- `utils.py`: resource loading, few-shot splitting, seeding, and JSON helpers.

Resource paths are intentionally represented with placeholders such as `xx/xx`. Set `SCALE_RESOURCE_ROOT` or `RESOURCE_ROOT` to the local resource directory when using local resources.

## 中文

本目录包含论文公开代码路径。

- `lasrc.py`：LASRC，层自适应稳定残差组合算子。
- `scale_fg_main.py`：SCALE-FG 原始候选构造与 support-time 选择数据生成。
- `scale_fg_table.py`：LASRC 与 SCALE-FG 结果行汇总。
- `algorithm.py`：共享的 LoRA 加载、优化、组合与推理辅助函数。
- `utils.py`：资源加载、few-shot 划分、随机种子与 JSON 辅助函数。

资源路径已统一使用 `xx/xx` 等占位形式。使用本地资源时，可通过 `SCALE_RESOURCE_ROOT` 或 `RESOURCE_ROOT` 指向本地资源目录。
