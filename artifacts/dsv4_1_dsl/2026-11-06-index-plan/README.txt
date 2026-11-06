本轮替换：Indexer 请求分组与运行区间规划
=====================================

当前状态：实现、编译及全部 A5 设备验证通过，代码和过程数据归入 dsv4.1-dsl 分支。

实现与契约
----------
index_plan_dsl.py 使用单个 AIV 任务替换 decode_indexer.py 的 idx_plan_req。
覆盖 C2A full、C1A full 和 C1A reindex 的稠密分支；后两种 reuse 路径没有这个任务。
同一请求的连续 Query 最多六行一组，原有起止行、最大可见长度、逐 Query 映射及组数保持不变。
C2A 模式中，可见长度不超过 INDEX_TOPK 的行按零参与组内最大值。
两个模式共用同一个 DSL 核函数，短序列标志改为有类型的运行时参数。
PyPTO/simpler 负责调用和计划完成依赖，后续 MIX+syncall 打分、TopK 和通信均未修改。

已经完成
--------
- 原核与 DSL 的 28 项入口编译通过；两种模式与动态行数可复用各自的动态程序。
- C1A full、C2A full、C1A reindex 三个完整入口编译通过。
- 两个核心程序及三个集成子程序的 A5 设备和编排二进制离线编译通过。
- DSL 的 14 组实际设备用例全部零容差通过。
  覆盖零有效行、单行、恰好六行、第七行拆组、全部短历史、请求切换与再次出现、
  72 行、负长度、宽度截断、非有效行、输入偏移及非连续输出保护区。
- 完整 C1A full 的 72 Query、12 个请求通过全部 11 项原有输出比较器。
- 原核连续布局的 14 项 CPU 模拟器检查全部零容差通过，证明调整后的对照脚本正常。
  这些结果明确标记为 a5sim，不计入 A5 原核逐项对照通过数。
- C1A reindex 的 CPU 输入预检通过：7 Query、候选池容量 544、每行 513 个连续候选块，
  全部满足稠密分支条件，并将产生六行加单行的分组；这不是设备集成结果。
- 四项预提交检查通过。

补齐的设备验证
--------------
- 按连续生产布局完成原核与 DSL 的 14 组设备对照，完整元数据逐项一致，不同元素数全部为零。
  两种实现共 28 次设备运行全部通过；DSL 额外验证非连续输出视图及保护区。
- 完整 C2A full 的 72 Query 通过原有全部输出比较器。
- 完整 C1A reindex 的 7 Query 通过全部 11 项原有输出比较器。
  候选块号、计数、可见长度和页映射检查确认全部 Query 进入稠密分支，实际执行本轮任务，
  并覆盖六行加单行的分组。

设备队列维护恢复后，原核对照与 C2A full 的排队任务完成，reindex 补交后通过。
CPU 模拟器、输入预检与 A5 实机结果分别保存，不混用通过数量。

原核扩展视图诊断及上游核对
------------------------
初次额外测试把原核输出也放在带父行距的切片中，发现 PyPTO 的 pl.write 按逻辑宽度寻址。
最小复现中，对 INT32[5,10] 的 [0:4,1:9] 切片写 [3,0]，A5 实际改写父张量 [2,5]，
预期为 [3,1]；直接写父张量的对照通过。此诊断不计入生产布局原核对照通过数。
生产 runs 一直是连续存储，因此原核对照恢复这一真实契约；DSL 仍独立验证非连续视图和保护区。
没有降低整数校验标准，也没有修改 PyPTO 或 CANNBotDSL 来掩盖问题。

已核对 hw-native-sys/pypto 的 main 7d4ef913 以及未合并的相关 PR #3208、#3191。
main 仍按逻辑 shape 计算标量偏移。PR #3208 包含 descriptor stride 寻址修改，
但未在其分支构建和执行本例，不能认定它已覆盖 outlined 输出参数丢失行距的情况。
同类问题已有 #3254，因此用中文补充本次完整最小复现和设备证据，没有重复建单：
https://github.com/hw-native-sys/pypto/issues/3254#issuecomment-6681261698

过程文件
--------
model-snapshot/ 保存实现，workflow/ 保存原核、校验脚本、最小复现和归档流程。
generated/ 保存 DSL 设备源码、ABI 包装及编排。
validation/ 区分编译、已完成设备运行、原核扩展视图诊断和上游核对。
device-output/ 保存通过的原核、DSL 输出和三条 Attention 集成实际输出，取自比较器 actual 参数。
routing-inputs/ 保留少量调度输入；完整权重由固定种子的模型构造器重建。

复现入口
--------
经 task-submit 分配 A5 后，在固定环境中运行：

python models/deepseek_v4_1_flash/index_plan_dsl.py --case all -d <设备>
python models/deepseek_v4_1_flash/decode_c1a_full.py -d <设备>
python models/deepseek_v4_1_flash/decode_c2a_full.py -d <设备>
python models/deepseek_v4_1_flash/decode_c1a_reindex.py --start-pos 4096 --query-lens 7 -d <设备>

原核对照由 workflow/compare-native.py.txt 保存，原 PyPTO 主体只在过程快照中保留。
Torch 仅用于 CPU 输入与校验。编译并发为 2，不进行文件 hash 校验。
未执行整网、真实 checkpoint 或正式性能 A/B。
