本轮替换：C1A 缓存发布
======================

models/deepseek_v4_1_flash/compressor_publish_dsl.py 将原生 c1a_compress
替换为单个 dsl_c1a_compress AIV 任务，继续由 PyPTO/simpler 调度。
任务内部完成 RMSNorm、保留未旋转 BF16 latent、尾部 64 维相邻对 RoPE、
group16 BF16-scale FP4 编码和物理缓存行写入。
没有拆出独立量化任务，没有跨卡通信或全核屏障；上游 C1A 投影保持现有实现。

保留的契约
----------
- 同一物理页归属同一 worker，按 token 顺序写入；重复槽位最后一个 token 生效。
- 负槽位仍生成 latent，但不发布缓存。非有效 latent 行和未写入缓存保持原值。
- C1A FP4 中点取偶数；C2A 仍取较小幅值，不能合并两者的舍入规则。
- scale 为 group16 BF16，保持 2**-9 下限；量化除以已舍入的 BF16 scale。
- 偶数元素在 FP4 payload 的低半字节；latent 在 RoPE 之前保存。
- 保留 cache_ready 和 source_ready 依赖，worker 数量为 max(min(num_tokens, 8), 1)。

共享与迁移
----------
compressor_vector_dsl.py 共用 RMSNorm、RoPE 和 FP4 编码，C1A/C2A 显式选择舍入模式。
C2A 环形状态读写、池化规则及请求归属不变。
decode 和 prefill 的共用 C1A 发布接口均切到 DSL；旧 C++ 核及私有头文件从模型移除，
仅在 workflow/ 留存以复现原核对照。PyPTO 和 CANNBotDSL 框架均未修改。

验证
----
- C1A 核心 5 组输入，分别运行 DSL 和原生核；3 项输出均与 CPU 参考零容差通过。
  从 compare_fn 回调捕获的真实设备输出逐字节一致，包括 payload、scale 和 latent。
  覆盖空输入、非有效行、跨页、重复/负槽位、FP4 中点、正负零和 worker 复用。
- C2A 共用实现 6 组设备用例通过，包括连续两步环形状态回放。
- 完整 C1A decode 使用替换前冻结的输入和 CPU 参考，11 项输出通过原比较器。
- C2A 双卡整层本轮未重跑：第一次排队超时，第二次仍未分配设备后主动取消。
  没有设备执行结果；共享算子的 6 组回归已覆盖本轮 C2A 代码改动。
- 共用 prefill C1A full 入口以 17 token 编译通过；未执行完整 prefill 设备流程。
- 不改变精度阈值；本轮未做整网、真实 checkpoint 或正式性能 A/B。

输出来源
--------
device-output/ 保存 compare_fn 的 actual 参数，即真实设备输出。
fixtures/*/out 保存 golden.run 的 CPU 参考，不能当作设备输出。
相同编译目录复用时 fixtures 只保留最后一组输入；各组输入可按固定种子重新生成。
validation/ 保存执行日志及原核差异统计；generated/ 保存设备代码、ABI 包装和编排。
summary.json 记录版本、范围与文件清单，不重复归档整层权重。

复现
----
在固定 PyPTO、CANNBotDSL 0.8.1 和 CANN 环境下，经设备队列运行：
task-submit --device auto --max-time 600 --run 'python models/deepseek_v4_1_flash/compressor_publish_dsl.py --case all -d "$TASK_DEVICE"'
task-submit --device auto --max-time 600 --run 'python models/deepseek_v4_1_flash/compressor_pool_dsl.py --case all -d "$TASK_DEVICE"'
task-submit --device auto --max-time 600 --run 'python models/deepseek_v4_1_flash/decode_c1a_full.py -d "$TASK_DEVICE"'
实际运行的有界编译脚本和原核对照脚本位于 workflow/；Torch 仅用于 CPU 准备和校验。
