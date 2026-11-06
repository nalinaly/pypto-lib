本轮替换：C1A reindex 候选池稠密分支判定
=====================================

当前状态：实现、编译及 A5 设备验证全部通过。

实现与语义
----------
index_route_dsl.py 使用 classify_pool_rows 替换原 idx_plan_pool 核内主体。
保留 32 个 AIV worker、cache_ready 和 pool_ready 依赖，不增加全核屏障。
PyPTO/simpler 继续归约 32 个标记并选择稠密或稀疏路径。

每个 worker 按原轮转规则处理 Query，检查：
- 截断后的候选数量是否覆盖全部可见候选块。
- 可见长度是否处于分数宽度及页表覆盖范围内，请求号是否有效。
- 有效候选块号是否为从零开始的完整恒等映射。
- 被访问的页号是否低于缓存页数上界。

负候选数量和负可见长度按零处理，超容量计数按容量截断。
页号只保留原有上界检查：负页号不使判定失败，由后续 attention 屏蔽。
同一 worker 一旦发现不满足条件的 Query，最终标记保持为零；没有有效 Query 的 worker 输出一。
仅写每个 worker 对应行的第零列，其他行列保持原值。

候选块号按 1024 项、页表按 128 项分块加载，比较与归约在 VF 内完成。
固定 UB 工作区，不额外分配模型中间张量；原有缓存、MIX/TopK 和通信均未修改。

验证
----
- 七组核心用例各执行原 PyPTO 核与 DSL，共十四次 A5 运行。
  全部 32 个 worker 标记及完整输出保护区逐项一致，不同元素数均为零。
- 覆盖零有效 Query、部分有效行、72 行轮转、1/33/65/1025/2048 候选容量、
  1024 项分块尾部、重复/缺失块、负值及超容量计数、非法请求、页号上界和负页号规则。
- 完整稠密 reindex：固定已通过的输入和 CPU 参考，start-pos=4096、7 Query，
  每行有 513 个连续候选块，全部 11 项原有输出比较器通过。
- 完整稀疏 reindex：固定已通过的输入和 CPU 参考，start-pos=16384、6 Query，
  每行可见 2049 块但只保留 2048 块，全部 11 项原有输出比较器通过。
- 核心与两个集成入口编译通过，对应四组 A5 二进制编译通过。
- 四项预提交检查通过。

核入口盘点
----------
稠密及稀疏编译产物均注册 29 个核入口，全部来自 DSL SourceBundle，
没有原生核内源文件。核注册数包含同一程序中两个分支的入口，不表示每次都执行全部入口。
至此，当前 A5 Decode、TP1 的 C1A reindex Attention 子图所有核内任务均已切换为 DSL。
这不包含 MoE 通信协议、整网、Prefill 或其他 TP 配置。

过程文件
--------
workflow/ 保存原实现、原核对照、固定输入回放和归档脚本。
model-snapshot/ 保存本轮模型代码；generated/ 保存设备源码、ABI 包装和编排。
validation/ 保存编译、设备运行、分支证据和核注册盘点。
device-output/ 保存比较器 actual 参数取得的真实设备输出，fixtures/*/out 是 CPU 参考。
动态编译目录保留最后一次用例，其他小用例可由固定构造器重建。
routing-inputs/ 保存两条路径的调度输入，完整权重沿用本地固定参考，不重复归档。

初次公共比较辅助函数让 Mask 跨分支使用，被前端拒绝；现已将两类比较和归约分别
保留在自己的 VF 作用域内。该记录仅供追溯，不涉及 PyPTO 或 CANNBotDSL 框架修改。

复现
----
在固定环境中，经 task-submit 分配 A5 后运行：

python models/deepseek_v4_1_flash/index_route_dsl.py --case all -d <设备>
python models/deepseek_v4_1_flash/decode_c1a_reindex.py --start-pos 4096 --query-lens 7 -d <设备>
python models/deepseek_v4_1_flash/decode_c1a_reindex.py --start-pos 16384 --query-lens 6 -d <设备>

固定输入精确回放见 workflow/run-reindex.py.txt，原核逐项对照见 workflow/compare-native.py.txt。
Torch 仅用于 CPU 输入准备和校验，编译并发为 2，不进行文件 hash 校验。
未运行整网、真实 checkpoint 或正式性能 A/B，精度阈值保持不变。
