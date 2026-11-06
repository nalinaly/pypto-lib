本轮替换：C1A 压缩投影
======================

compressor_dsl.py 新增 C1A 生产投影路径，原 proj_cmp 由单个 dsl_c1a_proj_cmp
AIC SPMD 任务替代。计算为 BF16 [T,5120] × NZ BF16 [5120,512] -> BF16 [T,512]。
保持 M48/N64/K512 分块、FP32 依次累加及最终一次 BF16 舍入。
每个 worker 负责一个不重叠的输出块，保留输入依赖和缓存发布消费的完成句柄。
零有效 token 不写输出，非有效输出和保护行保持原值。

本轮直接复用 bf16_projection_dsl.py，不修改底座或 C2A 计算路径。
compressor.py 移除原 PyPTO 投影主体并导出共享 DSL 接口，decode/prefill 调用自动接入。
上轮完成的 AIV 缓存发布保持原有边界，C1A 压缩分支从投影到 FP4 缓存均使用 DSL。
不涉及跨卡通信或全核屏障；PyPTO 和 CANNBotDSL 框架均未修改。

验证结果
--------
- 六组 C1A 输入分别运行原 PyPTO 与 DSL，真实设备输出逐字节一致。
  覆盖 NZ 坐标、空输入、NaN 非有效输入、M48 尾块、正负 BF16 中点、72 token 和 40 worker。
- 结构化与中点数据对 CPU 参考零容差；随机数据使用 atol=1e-4、rtol=1/128、零异常比例。
  对原 PyPTO 的直接设备比较仍为逐字节一致；尾部和保护行始终精确检查。
- 原 C2A 投影五组用例通过既有比较器，FP32 误差阈值不变。
- 完整 C1A decode 回放既有固定输入和 CPU 参考，11 项输出通过原比较器。
- 共享 prefill C1A full 入口以 17 token 编译通过。
- 未执行完整 prefill 设备流程、整网、真实 checkpoint、正式性能 A/B 或 C2A 双卡整层重跑。

输出与过程数据
--------------
device-output/：从 compare_fn 的 actual 参数捕获的真实设备输出。
fixtures/*/out：CPU 参考输出，不能作为设备结果；in 为输入快照。
相同编译目录复用时只保留最后一组输入，全部输入可按源码中的固定种子重建。
generated/：DSL 设备代码、通用 ABI 包装、AIC 编排和原 PyPTO 生成代码。
workflow/：原实现、对照脚本、设备队列执行脚本及归档脚本。
validation/：设备运行、编译与预提交检查记录；summary.json 为版本和文件清单。
整图参考沿用上一轮已经通过的快照，输入规格、数学语义及 golden 均未改变。

复现
----
在固定的 PyPTO、CANNBotDSL 0.8.1 和 CANN 环境中，经设备队列运行：
task-submit --timeout 0 --device auto --max-time 600 --run 'python models/deepseek_v4_1_flash/compressor_dsl.py --op c1a -d "$TASK_DEVICE"'
task-submit --timeout 0 --device auto --max-time 600 --run 'python models/deepseek_v4_1_flash/compressor_dsl.py --op c2a -d "$TASK_DEVICE"'
task-submit --timeout 0 --device auto --max-time 600 --run 'python models/deepseek_v4_1_flash/decode_c1a_full.py -d "$TASK_DEVICE"'
默认 --op c2a 保留原命令行为，--op all 执行两种投影验证。
实际验证通过 workflow/run-suite.sh.txt 顺序占用一张 A5；Torch 仅用于 CPU 数据准备和校验。
