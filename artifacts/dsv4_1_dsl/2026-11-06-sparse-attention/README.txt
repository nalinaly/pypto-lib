本轮替换：混合量化稀疏 Attention
================================

cann/cannbot-dsl 的 master samples 已由 48fa202 更新到 31bbd70，保持浅克隆。
参考 mixed_quant_sparse_flash_mla 的核内实现，模型使用的 DSL wheel 仍为官方 0.8.1。

实现位于 models/deepseek_v4_1_flash/sparse_attention_dsl.py。
原 sparse_gather_vf 的单个 MIX 任务已替换为 dsl_sparse_attn：
一个 AIC 和两个 AIV 在片上完成缓存收集、FP8/FP4 反量化、QK、在线 Softmax、PV 和输出。
每个 Query 的完整 Attention 由一个 block 独立处理，只有块内同步；跨卡通信任务未修改。
没有将反量化、Softmax 或归并拆成额外任务，没有新增全核屏障。

保留的模型契约
--------------
- FP8 窗口缓存：group-32 BF16 scale。
- FP4 压缩缓存：E2M1、group-16 BF16 scale、偶数元素在低半字节。
- payload 与 scale 分离存储，输入索引为物理缓存行。
- 按 128-key 更新在线状态，P 舍入为 BF16；sink 最后加入分母，不贡献 Value。
- 非有效输出行保持原值，无有效 KV 的 Query 输出为零。

验证结果
--------
- 8 组真实设备输出通过 atol=0.003、rtol=0.03 的原核对照；最大绝对差 0.000244140625，不同元素总数 4。
- SWA 共用反量化回归：4 组设备用例通过。
- C2A 第 2 层 EP2：固定输入和原比较器，16 项输出全部通过。
  x_next rel_l2=0.021193206，max_abs=0.65664196，与原基线一致。
- C1A 共用模型入口编译通过；完整 C1A、prefill、整网和性能 A/B 不在本轮验证范围。
- 本轮没有修改 PyPTO 或 CANNBotDSL 框架。

复现
----
在已配置 PyPTO、CANNBotDSL 0.8.1 和 CANN 的 A5 环境中，经设备队列运行：
task-submit --device auto --max-time 600 --run 'python models/deepseek_v4_1_flash/sparse_attention_dsl.py --case all -d "$TASK_DEVICE"'
SWA 回归入口为 sparse_attention.py --mode swa --swa-case all。
Torch 仅用于 CPU 输入准备和校验。原核对照脚本及三个原生源文件保存在 workflow/ 中。

summary.json 记录版本、范围和文件清单；validation/ 保存最终日志。
generated/ 保存 DSL 设备代码、包装与编排；fixtures/ 保存代表性小输入。
相同编译目录被复用时保留最后一组输入，全部用例可按源码中的固定种子重建。
reference/ 保留本轮参考 sample 的原文，workflow/ 保留替换前源码。

验证更正（2026-11-06）
---------------------
原对照脚本误读 data/out，将 CPU 参考当作设备输出。此前的设备逐元素一致声明作废。
原设备对 CPU 的精度检查和整层通过记录仍然有效，精度阈值没有改变。
已从 compare_fn 回调捕获真实输出，重跑全部 8 组 DSL/原生核对照，并精确检查尾部哨兵和全屏蔽零输出。
8 组真实设备输出通过 atol=0.003、rtol=0.03 的原核对照；最大绝对差 0.000244140625，不同元素总数 4。
validation/actual-device-comparison.log.txt 和 device-output/ 是修正后的设备证据。
带 invalid-cpu-reference 的旧记录及原 cases.log.txt 仅供追溯，不再作为设备对照证据。
fixtures/*/out 始终表示 CPU 参考。workflow/correct_comparison.py.txt 保存更正过程。
