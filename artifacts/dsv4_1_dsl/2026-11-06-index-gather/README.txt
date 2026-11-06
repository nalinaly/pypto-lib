本轮替换：C1A reindex 候选缓存收集
================================

index_gather_dsl.py 将 idx_gather_pool 替换为单个 dsl_idx_gather_pool AIV SPMD 任务。
保留原有 32 worker、每分块 32 个候选块/256 行，以及 cache_ready、pool_ready 两个依赖。
四个输出仍由原有 MIX 打分和 TopK 消费；路由判断与 runs 元数据规划继续使用原实现。

数据契约
--------
- FP4 payload 与 E8M0 scale 按字节复制，不重新量化或转换数值。
- 无效请求、逻辑块或物理页：payload=0、scale 字节=127、valid=0、physical=-1。
- 合法因果尾块仍复制完整八行原始数据，只屏蔽超出可见长度的 valid 和 physical。
- 重复候选块按原顺序复制，非有效 token 的输出保持原值。
- 固定 UB 工作区，本地 AIV 任务，不涉及跨卡通信或全核同步。
- 两个框架均未修改。

验证结果
--------
- 七组核心输入分别运行 DSL 与原 PyPTO 主体，四个真实设备输出及保护区全部逐字节一致。
  覆盖跨页、重复块、非法元数据、尾块、容量 1/32/33/65、零有效 token、worker 复用，
  以及完整的 0..255 payload/scale 字节编码；保留非有效行和一个额外输出分块。
- 完整 reindex 使用原生基线的固定输入与 CPU 参考，不改变原有比较器。
  起始位置 16384、6 个 Query；每个 Query 有 2049 个可见块，候选池只保留 2048 块，
  验证脚本先检查该条件，确认全部 Query 实际进入稀疏收集分支；11 项输出通过原比较器。
- 七组两种实现共 14 次编译和完整动态入口编译通过；编译与实际设备验证分别记录。
- 从 compare_fn 的 actual 参数保存真实设备输出；data/out 仅表示 CPU 参考。
- 具体通过记录见 summary.json。未运行整网、真实 checkpoint 或正式性能 A/B。

过程数据
--------
workflow/ 保存原实现、对照入口、设备队列命令和归档过程。
generated/ 保存设备代码、ABI 包装和编排；validation/ 保存编译及设备运行记录。
device-output/ 保存实际设备输出；fixtures/*/out 保存 CPU 参考。
编译目录复用时输入快照可能被后续同形状用例替换，全部输入可按固定规则重建。
初次对照脚本提取时的局部变量重名已修正，旧日志仅供追溯，最终全部设备对照已重跑通过。

复现
----
在固定 PyPTO、CANNBotDSL 0.8.1 和 CANN 环境中，经设备队列运行：
task-submit --timeout 0 --device auto --max-time 600 --run 'python models/deepseek_v4_1_flash/index_gather_dsl.py --case all -d "$TASK_DEVICE"'
task-submit --timeout 0 --device auto --max-time 600 --run 'python models/deepseek_v4_1_flash/decode_c1a_reindex.py --start-pos 16384 --query-lens 6 -d "$TASK_DEVICE"'
Torch 仅用于 CPU 输入准备和校验。设备维护期间只执行不占卡的编译和静态检查。
