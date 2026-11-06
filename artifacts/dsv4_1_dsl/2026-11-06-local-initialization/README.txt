DSV4.1 三类普通本地初始化任务迁移到 DSL（2026-11-06）

本次替换

1. decode_sp_zero_padding：仅清零 SWA 无效 token 对应的 BF16 行，保留有效行。
2. gate_pre_route：清零非有效路由的索引和权重，将所有分数行的补齐专家列
   填为原有的最小有限 FP32（-3.4028234663852886e38），保留实际专家列。
3. unused_attention_output：初始化非 SWA 层未使用的 Attention 输出端口。

前后两处 BF16 清零共用 zero_bf16_rows；路由初始化使用 initialize_routes。
二者均为 AIV DSL 核。PyPTO/simpler 负责循环、切片、任务提交和依赖。
SWA 仅提交无效行，未使用输出仍按最多八行分块，路由初始化仍为单个任务。
hc.zero_bf16_padding 保留导入接口，现有 Decode、Prefill、Drafter 调用不需修改。
通信、协议、框架和精度比较阈值均未修改。

验证

十三组原核与 DSL 对照，共二十六次 A5 设备运行。二十三个输出缓冲区
逐字节一致，包含有效区、额外保护列、正负零和有限 FP32 填充值。
覆盖零/部分/全部有效行、负数及超过容量的有效行数、八行尾部，
以及不补齐专家列、部分列补齐和跨多个 64 列块的补齐。

EP2 MoE 使用每卡八个专家、有效 token 数 72 和 17，验证了分数补齐与
后续投影、路由和专家计算的组合。三个输出全部通过原比较器。
TP1/EP2 layer 0（SWA）和 layer 2（C2A full）使用固定输入和原有比较器，
每层十六个输出全部通过。Torch 仅用于 CPU 数据准备和校验。

整层静态生成物覆盖六类 Attention 分支：75 个核注册中 67 个为 DSL，
剩余 8 个核注册全部属于通信或协议处理；详见 validation/kernel-inventory.json。
此前盘点的三类普通本地任务已经全部迁移。该结论限定当前 A5 Decode TP1
主路径，不扩展到 Prefill、TP>1 或未使用的备用入口。
源代码层面保留的通信协议任务可能被编译器消除或合并，源码任务种类数
与最终核注册数不同；这里报告的是实际生成物中的核注册数。

过程数据

model-snapshot/ 保存本轮模型代码；workflow/ 保存原实现及复现脚本。
generated/ 保存 DSL 导出 ABI、包装、设备源代码、原核对照及调度代码。
validation/ 保存编译、设备校验、代码检查和逐字节对照结果。
device-output/ 为比较器收到的实际设备输出；整层仅保存六个激活边界输出，
其余缓存输出的验证结果记录在日志中。fixtures/*/out 为 CPU 参考输出。
相同动态入口的 fixtures 保存最后一次运行的数据，其余输入可由固定构造器复现。
整层权重等大体积输入沿用已有固定数据，未在本次重复归档；对应位置见结果记录。
所有发布路径均已匿名化。不添加校验摘要；gzip 仅用于压缩张量数据。

复现

在已安装配套 PyPTO、CANNBotDSL 和 simpler 的 A5 环境中，先经 task-submit
申请设备，再运行模型入口：

  python models/deepseek_v4_1_flash/initialization_dsl.py -d <单卡设备>
  python models/deepseek_v4_1_flash/decode_moe.py --ep 2 --experts-per-rank 8 \
      --num-tokens-per-rank 72,17 -d <两卡设备>
  python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d <两卡设备>
  python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 2 -d <两卡设备>

完整原核对照与固定数据复现见 workflow/。未执行整网、真实 checkpoint
或正式性能 A/B；本轮目标是替换后功能稳定及语义保持。
