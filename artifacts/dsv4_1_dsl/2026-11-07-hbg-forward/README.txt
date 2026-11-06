DSV4.1 DSL：HBG 桥接与 fwic AICore 完整 decode 验证（2026-11-07）

结果

40 层完整 decode forward 已通过双卡设备执行：EP2、12 个请求、每请求
6 个有效 token，共 72 个有效 token，物理容量 80，每卡 64 个路由专家。
使用本模型入口默认的零权重/零激活夹具，覆盖 Encoder、Decoder、Engram、
Attention、MoE、最终归一化、LM head 和 greedy sample。
两卡均发布 AICore resident 模式：mode=1、publication=1，没有 legacy 回退。
验证输出有限、零权重结果以及无效行/填充列全零。初次执行和整理后的最终
版本均通过；最终结果见 validation/forward-result.json 和 device-final 日志。
本次证明整网执行链路可用，未验证实际 checkpoint 精度，也未做性能 A/B。

修改

1. PyPTO 原 PR #3231 已更新至 f3fe4058：按 PassContext 选择 TMR/HBG
   张量描述符和独立包装缓存；与 main b1054e0d 合并，采用上游外部写回
   属性和 CommContext ABI。动态切片使用实际描述符长度，覆盖 #3324。
   https://github.com/hw-native-sys/pypto/pull/3231
2. Indexer 建图不再读取设备产生的 runs/flags。分页打分工作区按 Query
   容量上界分片，每片不超过 100 个 panel；实际有效区间仍由 DSL 读取。
   reindex 的恒等和稀疏候选池统一使用已有的 gather + MIX TopK 路径，
   保留物理行映射。暂时去掉依赖设备标记的稠密快捷分支，未改变比较器。
3. MoE 分配直接使用常量 MOE_TOKENS；本地 gather 直接使用描述符的列宽，
   避免局部标量别名在新上游元数据中被误判为不精确长度。
4. CANNBotDSL 仓库与所有跨卡通信核均未修改。

验证范围

- 1948 个定向 PyPTO 单元用例通过。整批中新增测试的静态字段断言写错，
  修正后适配器全部 19 例单独重跑通过；保留原始失败和重跑日志。
- TMR/HBG 正常设备用例 12 个通过：部分 SPMD、0/3/17 行动态切片及尾部
  保护、全核 AIV/MIX 三轮交换和重复 syncall。
- HBG 非法部分核启动准确在 prepare 阶段返回 -5，独立进程验证通过。
- C1A reindex 的恒等/稀疏候选池分别通过原有 11 项输出比较，包括 TopK、
  物理缓存、输出激活和 mHC。复用上一轮固定输入/参考，容差未放宽。
- PyPTO 全部预提交项目已通过。第二轮全量检查期间执行提交改变了 Git
  差异，Pyright 被工具误标为“修改文件”；其结果为 0 错误、0 警告，随后
  在稳定工作区单独重跑 hook 通过。模型全量预提交检查通过。

仍存在的问题

fwic f1c21abf7 的 TMR 非法启动测试失败：编排已报告 INVALID_ARGS(-5)，
最终 Python 收到 AICPU 507018。独立进程可复现，未放宽 -5 断言，设备由
运行时确认清理。正常 TMR/HBG 与 HBG 非法启动通过。详见 invalid-tmr 日志。
动态局部形状在 outlining 往返时的变量作用域问题仍由 PyPTO #2047 跟踪；
本次只修复外部调用链的形状元数据，没有宣称解决通用标量别名问题。
https://github.com/hw-native-sys/pypto/issues/2047

依赖与复现

Linux ARM64、Python 3.12.15、CANNBotDSL 0.8.1（官方开发仓 8274a38b）、
PTOAS 0.72、PyPTO f3fe4058。验收使用 nalinaly/simpler f1c21abf7 的干净
源码快照和已安装的同版 HBG 二进制资源，TMR 库在本轮重新构建。PyPTO PR
仍保留上游 runtime pin 391b2fbc，没有将 fwic 分支强制写入上游依赖。
本地通用 DSL env.sh 已指向该配套环境；PYPTO_ROOT/runtime 的工作区版本
保留 f1c21abf7，因不同于 PR pin 会给出明确警告，没有关闭版本检查。

workflow/ 保存真实运行入口及环境选择代码，机器路径已替换为 <workspace>。
先运行 forward.py --compile-only 和 build-forward.py（CPU），再通过
task-submit --device auto --device-num 2 执行 run-forward.sh。
该包装只选择 HBG、限制编译线程、复用编译结果和记录调度模式，调用的仍是
models/deepseek_v4_1_flash/decode_fwd.py 的生产入口。
两个 reindex 入口可省略历史 --golden-data 参数，以原固定种子重新构造参考。

归档内容

generated/：完整 forward、两个 reindex 的编排和设备代码；52 份外部 DSL
导出目录，包含设备 ABI、HBG 包装和设备源代码。没有保存编译二进制。
validation/：编译、设备、回归与静态检查日志，最终结果和核注册清单。
device-output/：实际回传输出的 gzip 文件；Torch 仅处理 CPU 数据。
model-snapshot/：本轮模型源代码快照。

完整 forward 有 109 个核注册，其中 83 个 DSL、26 个原生；MIX 的 AIC/AIV
分别注册，注册数不等于任务数。原生核包含通信/协议、Engram、RoPE 表准备
与 LM head，不能将“整网跑通”解读为整个模型所有核都已经迁移为 DSL。
两个 reindex 子图均为 26 个 DSL 核注册。详见 kernel-inventory.json。
本轮没有增加或执行 hash 校验。
