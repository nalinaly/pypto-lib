本轮替换：Indexer 候选块池
=========================

index_pool_dsl.py 以单个 dsl_index_pool AIV SPMD 任务生成完整候选块池。
同一 Query 的归约、筛选、升序输出均由同一个 worker 完成；worker 0 独占发布计数向量。
Decode 和 Prefill 共享该实现及 CPU 参考，原导入接口保留，重复排序主体和私有辅助函数移除。

模型契约
--------
- FP32 分数每 8 个位置取最大值，下限为 -1e30，最新可见块设为 1e30。
- 最多选择 2048 个块，同分优先较小逻辑块号，输出按块号升序排列。
- 未使用位置为 -1；非有效 Query、负长度和空历史的计数为零。
- 短历史直接生成块号；长历史保持完整 FP32 排序精度，不改用 BF16 排序键。
- 保留 scores_ready 依赖，整个 SPMD 的完成句柄覆盖两个输出。
- 本地 AIV 计算，不涉及跨卡通信或全核屏障；两个框架均未修改。

实现
----
按 1024 个块扫描，用四轮 UINT32 保序字节直方图确定阈值，再按逻辑顺序稳定压缩输出。
分块 16 位直方图扩展为 32 位计数，覆盖超过 65535 个同分块，避免计数溢出。
固定大小的 raw/key/histogram/output 工作区位于 UB，不保留中间 GM 分数和归并数组。
参考 sample 的 FP32 保序键及直方图思路，未采用其 256 项预算或 Sink/Window 策略。
首次集成时动态 tile 宽度不符合 DSL 约束，已改为固定 TOPK 分块配合动态有效长度；
初次编译日志保留在 validation/initial-dynamic-tile.log.txt，最终动态入口已验证通过。

验证
----
- 10 组输入分别运行 DSL、原 Decode 和原 Prefill 核；块 ID、计数及保护区三方完全一致。
  覆盖短历史、2048 块边界、因果尾块、非有效行、FP32 近邻值、正负零、分数下限、
  65540 个同分块和选中块全部位于后段时的空前缀输出。
- 完整 C1A decode 短历史：72 token、12 请求、起始位置 8192，11 项输出通过原比较器。
- 完整 C1A decode 长历史：同样 72 token、起始位置 16384，实际触发超过 2048 块的筛选，
  11 项输出通过原比较器。块 ID 和数量均为零容差。
- 共用 Prefill C1A full 入口以 17 token 编译通过。
- 本轮未运行完整 Prefill 设备模型、整网、真实 checkpoint 或正式性能 A/B。

过程数据
--------
device-output/：compare_fn 回调捕获的真实设备输出。
fixtures/*/out：CPU 参考；in 为输入快照。编译目录复用时保留最后一组输入。
validation/：最终设备日志、原核差异统计及初次动态 tile 诊断。
generated/：DSL 设备代码、ABI 包装、编排及代表性的原核生成代码。
workflow/：替换前源码、真实输出对照脚本、队列执行和归档脚本。
reference/：本地固定 samples 版本的 Stem Indexer 源码和说明。
summary.json：版本、验证范围和文件清单，不重复归档整图权重。

复现
----
在固定 PyPTO、CANNBotDSL 0.8.1 和 CANN 环境下，经设备队列执行：
task-submit --timeout 0 --device auto --max-time 600 --run 'python models/deepseek_v4_1_flash/index_pool_dsl.py --case all -d "$TASK_DEVICE"'
task-submit --timeout 0 --device auto --max-time 600 --run 'python models/deepseek_v4_1_flash/decode_c1a_full.py -d "$TASK_DEVICE"'
长历史命令和三方对照见 workflow/run-suite.sh.txt；Torch 仅用于 CPU 数据准备和校验。
