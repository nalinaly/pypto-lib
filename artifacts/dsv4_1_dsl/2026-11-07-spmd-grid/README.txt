DSV4.1 DSL 多块 SPMD 改造与上游契约同步（2026-11-07）

结果

完成本轮审视推荐的 40 处改造。生产路径共有 51 处显式 DSL SPMD 启动点：
原先 44 处单块、7 处多块；现在 47 处支持多块分工，4 处继续单块。
多块入口按工作量和硬件资源决定 block 数，输入很小时仍可退化为一个 block。
统计限定 A5 Decode TP1 的六类 Attention、mHC 和 EP MoE 主路径。

改造覆盖 Q/KV/O 投影、归一化与 RoPE、SWA、mHC、Gate、共享/路由专家、
本地 Dispatch/Combine、Indexer Q/K/W、缓存页发布、C2A 状态处理和初始化。
原来“外层按 tile 派发多个 spmd(1)”的入口改为一组 SPMD，由 DSL 使用
get_block_idx/get_block_num 以步进循环分配 tile；各块写互不重叠的区域。
需要同步启动的 MIX 保留 sync_start，并限制在可用 cluster 数内。
缓存与状态任务按页或状态块分工，同一所有者内部保持原 token 顺序。

继续单块的四处：
  dsl_idx_plan_req：请求分组及少量规划数据。
  dsl_idx_plan_pool_runs：候选池运行区间规划。
  dsl_gate_pre_route：少量路由输出及补齐列初始化。
  dsl_hc_collapse_tail：奇数尾部只有一行。

完整调用点、分工表达式及原建议见 validation/spmd-audit.json 和
validation/review-decisions.json。审视工具只用于列出调用关系，不把备用分支
误计为实际生成的生产核。29 个抽出的数学主体中 26 个保持原 AST；
另 3 个变化是用户确认的 Gate BF16 契约。详见 math-and-protocol-review.json。

对照上游

PyPTO-lib 主仓参考更新到 89d4c1aa，位于独立参考工作区；原 main 工作区
已有的十个未提交文件保持原样。cann/cannbot-dsl master 从 31bbd70
快进到 fcc6bfd。已安装的 CANNBotDSL 开发仓和 wheel 不在这次样例更新中变更。
本次按相关性同步修改，没有将 DSL 分支整体替换为主仓实现。

按用户确认同步：
  1. Gate 权重与 RMSNorm 中间值均采用 BF16，投影使用 BF16 操作数和 FP32 累加。
     调用者类型、数据构造器和参考计算一起更新；原 gate_from_fp32 改名为
     gate_from_bf16。该接口的数据准备也必须使用 BF16 权重。
  2. C2A STATE_CAPACITY 从 4 改为 8，与上游每请求 6 个候选 token 的容量要求一致。
     夹具和新参考结果采用容量 8，不沿用旧容量 4 的 C2A/整层输出。
  3. 同步上游 #1699：dispatch_wait 锚定 push，combine_wait 锚定 scatter，
     arrival signal 排除隐式依赖；本地 gather/归约同时等待 publish/notify 和 wait。
     通信访存、通知、等待和归约散发的核内主体保持不变，没有迁移为 DSL。

CANN 最新样例的 L0C 有效源范围修复与本分支已有投影 helper 一致。
mega_mla 的 544/320 字节打包缓存，以及 attn_prologue 的固定 scale=1、
INTERLEAVE_HALF RoPE，与当前分离 payload/scale 和相邻 RoPE 契约不同，
没有直接套用。完整差异见 validation/upstream-comparison.json。

验证

最终版本共 114 个设备用例通过，全部通过 task-submit 分配 A5 执行：
  103 个叶子用例：零/部分/全部有效行、尾块、保护区、页别名、共享状态块和两步回放。
  7 个 Attention 用例：SWA、C2A full/reuse、C1A full/reuse、C1A reindex 稠密/稀疏分支。
  2 个双卡 MoE 用例：每卡 8 个专家，有效行数 72/17 和 72/0。
  2 个 TP1/EP2 整层用例：layer 0 SWA、layer 2 C2A full，各 16 个输出通过；
    整层使用默认 384 个路由专家，新生成 BF16 权重和容量 8 的状态数据。
未放宽既有比较器。整层采用仓库原有分段校验，MoE 边界会用实际 Attention
结果重新计算参考；这不等于全链路逐元素精度保证。完整组合误差诊断保留在
summary.json 及日志中。未执行正式性能 A/B、整网、真实 checkpoint、Prefill
或 TP>1 的设备验证。Torch 仅用于 CPU 数据准备和参考计算。

整层生成物有 89 个核注册，其中 81 个 DSL 注册、8 个原生注册；MIX 的 AIC/AIV
成员及不同形状特化分别计数，不等于 SPMD 源码启动点数。剩余原生注册全部是
通信/协议处理，见 validation/kernel-inventory.json。

发现的问题及处理

1. Gate 路由尾块需保留无效行清零；已修复网格裁剪范围。
2. CANNBotDSL 整数索引保留单例轴。状态块先显式 view 为二维，再读取容量，
   修复原改造中取错容量的问题；容量 8 的六组池化/状态测试全部通过。
3. 动态维度别名导致外部调用编译失败：用同一长度表达式统一输入/输出视图。
   最小复现已补充到既有 PyPTO RFC：
   https://github.com/hw-native-sys/pypto/issues/2047#issuecomment-6689599033
4. 静态父容量被当成动态切片的精确长度：Indexer Key/Query/权重合并的网格
   显式接收有效行数，限定循环与尾块。独立复现及相关上游 #3177 的核对结果：
   https://github.com/hw-native-sys/pypto/issues/3324

这两个 PyPTO 问题实测于配套 PR #3231 的 45f03621，未宣称已在最新 main 重建复现。
本轮没有修改 PyPTO 或 CANNBotDSL 框架源码，也没有引入新的 hash 校验。

过程数据与复现

model-snapshot/：修改的模型文件。
generated/：75 个去重编译目录的调度代码，以及 112 份 DSL 导出 ABI、包装和设备源码。
validation/：最终结果、编译/设备日志、审视、上游对照和首次失败记录。
device-output/：比较器收到的实际输出，gzip 压缩；整层保留六个激活边界，
  Attention 保留计算结果、状态和规划数据。大块历史缓存不重复发布，
  其完整比较结果在日志中，缓存发布叶子的输出另有保存。
fixtures/：Gate、C2A 状态用例的输入/参考，以及整层变化契约的实际输入。
  大型模型权重通过固定构造器重建，不重复存入仓库。
workflow/：测试入口清单、运行脚本及两个独立复现；机器路径均匿名化。
  compile-layer-shapes.py 仅借用旧夹具形状加速前端编译，不用于设备输入或参考；
  无该本地夹具时，直接使用下列公开模型入口完成编译和运行即可。

在 summary.json 所列配套环境中，通过 task-submit 申请设备后运行：

  python models/deepseek_v4_1_flash/gate_dsl.py --stage all -d <单卡>
  python models/deepseek_v4_1_flash/compressor_pool_dsl.py -d <单卡>
  python models/deepseek_v4_1_flash/decode_swa.py -d <单卡>
  python models/deepseek_v4_1_flash/decode_c2a_full.py -d <单卡>
  python models/deepseek_v4_1_flash/decode_c2a_reuse.py -d <单卡>
  python models/deepseek_v4_1_flash/decode_c1a_full.py -d <单卡>
  python models/deepseek_v4_1_flash/decode_c1a_reuse.py -d <单卡>
  python models/deepseek_v4_1_flash/decode_c1a_reindex.py --start-pos 4096 --query-lens 7 -d <单卡>
  python models/deepseek_v4_1_flash/decode_c1a_reindex.py --start-pos 16384 --query-lens 6 -d <单卡>
  python models/deepseek_v4_1_flash/decode_moe.py --ep 2 --experts-per-rank 8 --num-tokens-per-rank 72,17 -d <两卡>
  python models/deepseek_v4_1_flash/decode_moe.py --ep 2 --experts-per-rank 8 --num-tokens-per-rank 72,0 -d <两卡>
  python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d <两卡>
  python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 2 -d <两卡>

单卡需传整数设备号；两卡需传逗号分隔的同组设备号。其余叶子参数见
workflow/unit-cases.json。C1A full 和 reindex 本次复用了已有固定输入/参考，
公开入口也可根据其固定种子重新构造；C2A full、Gate/MoE 和两层数据使用新契约。
