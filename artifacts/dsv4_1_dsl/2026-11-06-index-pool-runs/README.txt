本轮替换：C1A reindex 稀疏候选池运行区间规划
=========================================

当前状态：实现、编译及 A5 设备验证全部通过。

实现
----
index_plan_dsl.py 新增 plan_pool_rows 设备函数和 plan_pool_runs 调度入口，
替换 decode_indexer.py 中的 idx_plan_pool_runs。
它仍是一个 AIV worker 的独立任务，只等待 pool_ready；候选缓存收集保留自己的依赖。
下游 MIX 打分同时等待收集、Query、权重及运行区间规划完成。
生成代码确认 block_num=1、显式 pool_ready 依赖，以及原有运行区间完成句柄的传递。

每个有效 Query 独立成组，四行 runs 元数据保持原契约：
- 起始行 token，结束行 token+1。
- 候选行数 max(0,min(pool_block_count,capacity))*8。
- Query 的组号等于 token，总组数等于 num_tokens。
- 零有效 Query 仅写总组数零；未使用的元数据及保护区保持原值。

原有请求分组功能保留在同一文件，默认 CLI 行为不变；新增 --mode pool 校验稀疏规划。
未修改 PyPTO、CANNBotDSL、通信代码、缓存格式或下游 MIX/TopK 的算法及阈值。

验证
----
- 五组核心用例各运行原 PyPTO 核与 DSL，完整有效布局的整数元数据逐项一致。
  覆盖零有效 Query、单 Query、17/72 行、部分有效行、零候选容量、负数和超容量计数。
- DSL 额外验证输入偏移、非连续输出视图及保护区，全部零容差通过。
  原核保持连续生产布局，避免把已知 PyPTO 标量写入视图行距问题作为数值参考。
- 原有请求分组的两种短序列模式回归通过，共 12 次核心设备运行。
- 完整 C1A reindex 使用已通过基线的固定输入和 CPU 参考，全部 11 项原有比较器通过。
  start-pos=16384，6 个 Query，每行可见 2049 块、候选池只保留 2048 块；
  路由检查确认全部 Query 实际进入本次修改的稀疏分支。
- 12 项核心入口与完整 reindex 入口编译通过，对应四组 A5 二进制编译通过。
- 四项预提交检查通过。

过程数据
--------
workflow/ 保存原实现快照、原核对照、固定输入回放、设备队列及归档脚本。
model-snapshot/ 保存本轮模型实现。
generated/ 保存 DSL 设备源码、ABI 包装和模型编排。
validation/ 保存编译、原核对照、设备运行和路由证据。
device-output/ 是比较器 actual 保存的真实设备输出；fixtures/*/out 是 CPU 参考。
同一动态编译目录保留最后一次输入，其余小用例可由固定构造器重建。
routing-inputs/ 保存决定稀疏分支的输入；完整权重沿用本地固定基线，不重复归档。

复现
----
在固定环境中，经 task-submit 分配 A5 后运行：

python models/deepseek_v4_1_flash/index_plan_dsl.py --mode pool --case all -d <设备>
python models/deepseek_v4_1_flash/index_plan_dsl.py --case mixed -d <设备>
python models/deepseek_v4_1_flash/decode_c1a_reindex.py --start-pos 16384 --query-lens 6 -d <设备>

精确固定输入回放过程保存在 workflow/run-reindex.py.txt。
Torch 仅用于 CPU 数据准备和校验，编译并发为 2，不进行文件 hash 校验。
未运行整网、真实 checkpoint 或正式性能 A/B。

当前 A5 Decode、TP1 的 C1A reindex Attention 子图只剩 idx_plan_pool 尚未替换。
