# 调控 DNA 工作流（Phase 5 实验代码）

状态：真实 7B 生成验证已完成，完整应用实验尚未完成；本目录不构成应用加速或生物功能结论。采用 [Evo2 补充方法 A.6](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10176-5/MediaObjects/41586_2026_10176_MOESM1_ESM.pdf) 的 mm39 染色质可及性任务。搜索使用 Enformer + 四个 Flashzoi，最终用未参与搜索的 Borzoi mouse replicate-0 检查；不主张其训练数据独立。

## 准备

从仓库根目录运行，先按主 README 安装 Evo2 环境。本机验证配置为 GPU0 48 GB 生成、GPU1 24 GB 评分，评分 OOM 时统一改 GPU2 48 GB。评分依赖装入独立环境，保持已有 torch/CUDA。

```bash
python scripts/download_ckpt.py L27_g12_150M_s1 --dest benchmarks/phasec_ckpts
python -m venv --system-site-packages /tmp/evspark_phase5_env
/tmp/evspark_phase5_env/bin/python -m pip install -r scripts/regulatory_design/requirements-scorer.txt
PYTHONPATH=.:scripts python -m regulatory_design.prepare
```

将官方 Evo2 7B 权重置于 `models/evo2_7b/evo2_7b.pt`（可以链接本机 HF 缓存）。本轮 target SHA256：`c66645929dc1b9c631f5be656da8726f38946315dc9167000a615dd626fcecf4`；drafter SHA256：`be6ff9f8818a83533e39d112ee9ba0028a340fe3cead8dc342c2397baf32310a`。单项 `run` 可用 `--target-weights` / `--checkpoint` 覆盖路径；两臂哈希必须一致。

`assets.lock.json` 固定六个评分模型的 revision、LFS SHA256 和输入资产校验和。`reference/` 包含原始 UCSC mm39 响应及公开 targets 表，原 URL 和坐标保留在清单。UCSC JSON 含下载时间，提供原文件以保证跨机器校验和可复核。原文坐标按 0-based `[52051928,52123468)` 替换区间解释。

下载默认使用 `hf-mirror.com`；`--endpoint https://huggingface.co` 可指向原站，版本与校验和不变。不需要私有 HF token。下载实体和日志保存于 `~/.cache/evspark/phase5_downloads`，项目资产为符号链接；`EVSPARK_PHASE5_DOWNLOAD_CACHE` 可指定另一原生文件系统目录。

`EVSPARK_PHASE5_ARIA2` 可指向 aria2c，启用每文件 4 连接的分段续传。已有 `.aria2` 控制文件的断点必须继续使用该下载器，不能回退到将稀疏文件长度视为实际下载量的顺序传输。最终仍须完整 SHA256 校验通过才发布资产。

## 验证与运行

```bash
PYTHONPATH=.:scripts python -m pytest tests/test_regulatory_design.py -q
PYTHONPATH=.:scripts python -m regulatory_design.validate_generation --context 40960 --output benchmarks/phase5/generation_validation_40k.json
PYTHONPATH=.:scripts /tmp/evspark_phase5_env/bin/python -m regulatory_design.validate_scoring --gpu 1
PYTHONPATH=.:scripts python -u -m regulatory_design.launch --through all
```

本机队列要求 `sleep.target` 已 mask，启动前检查机器睡眠策略。也可用 `python -m regulatory_design.pipeline --through all` 串接资产、评分预检和实验；`pipeline_status.json` 每 30 秒更新阶段及下载进度。

应用单测还让三符号历史依赖小模型运行实际候选循环，检查完整序列概率、候选间联合分布、batch=4/1 一致性及提前结束状态隔离。计算在 CPU 上；Vortex 导入需要可见 CUDA 驱动，无可见驱动时该两项跳过。

Enformer 运行上游定量样例；正式设计输入 196,608 bp 并开启 TF gamma。Borzoi 上游 notebook 的 TF 参考数组尚未齐备，不能声称复现其逐位 TF 对拍；模型完整性、物种 head、形状和坐标验证分开记录。评分 batch 使用四条不同天然序列校准；归一化全轨道最大绝对差≤0.01 是实现验证界限，不是生物质量等价标准。

native 独立候选 batch=1/2/4/8，EvSpark batch=1/2/4；native 不加载 drafter 或维护其 hidden。两臂共用搜索、缓存和评分策略，在各设计长度按完整校准耗时选择最快可行 batch。先完成 8 个 pilot 设计；正确性通过、两臂检查模型平均 AUROC≥0.80 且完整流程配对中位加速≥1.25× 后才扩大。阈值不根据结果改变，校准种子与实验种子分离。

每次运行保存全部候选轨迹、最终 FASTA、协议、资产、环境、失败和阶段计时；加载/预热另列。断点恢复须重建前缀，结果不进入主耗时比较。合格仅表示搜索集成及检查 AUROC 均≥0.90，不表示实验功能已验证。固定预算只累计公共实测时间范围内、合格且精确去重的最终第一名输出。汇总生成报告、JSON 和四面板图，以整次设计为统计单位，不以不显著宣称等价。


## 实验完成后的入稿与数字核验

`launch --through all` 只在 pilot 投入门通过、正式与长设计数据全部齐备后调用 `regulatory_design.paper`。程序从原始配对结果重算统计量，保留每个来源的 SHA256，并生成 `paper_results.tex`、`paper_methods.tex` 和 `paper_materials.json`。存在主仓 `paper/latex_v3/main.tex` 时才更新该稿件并编译；公开复现仓只生成可复核片段。原稿先备份，匹配不到既有段落时停止以免覆盖其他人的改动。

可独立复核已有材料：

```bash
PYTHONPATH=.:scripts python -m regulatory_design.paper --audit-only
```

主仓 `scripts/check_paper_numbers.py` 还会核验新增片段、原始文件、配对耗时/质量点估计及主图；实验未完成时明确显示待执行，不填入预测数字。


在相同环境中只恢复已通过预检的实验队列，可用 `pipeline --application-only --through all`，它保留既有评分 batch，避免重做评分速度校准改变比较配置。若旧调度器因磁盘重挂失效但单项设计仍在计算，可指定 `--wait-pid PID --wait-result 该设计/result.json`；仅在这次设计完整成功并退出后接续，不能据此跳过尚未通过的验证。所有子进程显式使用项目绝对路径作为 cwd。

统计量仍按每次完整设计计算。因不同图案复用设计种子，置信区间以种子为组进行 10,000 次配对 bootstrap（固定重采样种子 2026091905），每次保留同一种子下所有图案，避免把共享早期随机轨迹当成独立设计重复。输出同时记录设计数和种子组数，正式实验为 24 对设计、8 组种子；pilot 与长设计各为 4 对设计、2 组种子。投入门只使用原定点估计，不因区间处理改变阈值。

本轮调度在每轮先完成独立候选生成，再批量评分并全局选取两个分支；生成与评分阶段按轮同步，未做跨阶段并行重叠。两臂执行相同调度，最快 baseline 指该工作流和预设可行 batch 网格中的实测最快配置。

执行范围可由结果目录的 `execution_scope.json` 限定，例如 `{"through":"pilot"}`。它只能收窄命令行范围：旧 `--through all` 重启命令不会越过用户设定的 pilot 上限。生成和统计协议不受此调度设置影响。

## 小规模 case study 的最终核验

8 次 pilot 和报告全部完成后，运行以下命令从所有搜索轨迹复核父分支、候选种子、全局选择、最终序列、损失、AUROC 与统计汇总：

```bash
PYTHONPATH=.:scripts python -m regulatory_design.audit_case
```

输出 `pilot_audit.json`（来源文件 SHA256）、`pilot_designs.tsv`（全部逐设计结果）和 `pilot_final_sequences.fasta`（八条最终输出）。此步骤只读已有结果，不启动生成或评分模型。
