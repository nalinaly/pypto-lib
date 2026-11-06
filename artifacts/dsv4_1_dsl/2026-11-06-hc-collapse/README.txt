本轮替换：MoE 入口的四路 mHC 合并
================================

当前状态：A5 设备验证全部通过。
代码和过程数据归入 dsv4.1-dsl 分支，具体记录见 summary.json。

实现
----
hc.py 的 hc_head 统一调用 hc_dsl.py 中的 collapse_tile。
每个 AIV 任务处理两行，奇数尾部使用实际的一行切片；两种形状共享同一个 DSL 核函数。
保留原来的 allow_early_resolve=True，PyPTO/simpler 继续负责数据依赖和任务调度。
hc_pre、MoE 和其他 hc_head 调用方使用同一实现，调用接口不变。

四路 FP32 分别乘以 pre_mix，按 (y0+y1)+(y2+y3) 求和，最终按最近偶数舍入为 BF16。
已有 collapse_norm_row 与新核函数复用四路加权合并步骤，后续归一化不变。
移除原 PyPTO 核内实现和无用分块常量。通信代码和两个框架均未修改。

已完成检查
----------
- 1、17、72 行的原核、DSL 合并、合并/RMSNorm 和 hc_pre 共 12 项入口编译通过。
  这些行数复用四个动态编译程序；全部明确使用 a5 设备平台。
- EP2 MoE 编排编译通过，配置为每卡 8 个专家，有效 token 数为 72、17。
- 四个核心程序和 MoE 子程序共五组 A5 设备及编排二进制离线编译通过。
- 四项预提交检查通过。
- CPU 检查确认新增样本能区分两两求和与顺序求和，并覆盖正负 BF16 舍入中点和零权重。

设备验证结果
------------
- 经 task-submit 执行 validate.py，12 项实际设备比较全部通过。
- 1、17、72 行的 DSL 合并输出与原 PyPTO 核逐位一致，不同元素数均为零。
- 合并/RMSNorm 和 hc_pre 在上述三种行数下全部通过。
- 经 task-submit 分配同组两张卡，EP2 MoE 的 next_pre_mix、x_mixed、x_next 全部通过原比较器。
- head 和 hc_pre 的合并输出采用零容差；其余已有阈值保持不变。
- 设备队列恢复后补齐全部验证，维护期间的编译结果与设备运行结果分别保存。

运行中遇到的校验脚本问题
----------------------
初次行数参数是未声明的临时标量，桥接无法推断其类型；当前改用实际行切片及张量形状。
初次独立校验脚本漏写 execution_mode=False，选到了模拟器；DSL 设备代码不支持该路径。
当前脚本已显式使用 A5 设备模式，并重新完成入口及设备二进制编译。
该模拟器尝试不计入设备验证，失败记录仅用于追溯。

过程文件
--------
workflow/ 保存原实现、校验脚本和编译归档流程；model-snapshot/ 保存本轮实现。
generated/ 保存 DSL 设备源码、ABI 包装和模型编排；validation/ 保存编译和检查记录。
device-output/ 保存从比较器 actual 参数取得的真实设备输出，fixtures/*/out 是 CPU 参考。
同一动态编译目录保存最后一次运行的 72 行输入，其他行数可按固定种子重新构造。
不包含整网、真实 checkpoint 或正式性能 A/B 结论。

标准入口复现
------------
使用现有固定环境，并通过 task-submit 分配设备后执行：

python models/deepseek_v4_1_flash/hc.py --op head --batch 1 -d <设备>
python models/deepseek_v4_1_flash/hc.py --op head --batch 17 -d <设备>
python models/deepseek_v4_1_flash/hc.py --op head --batch 72 -d <设备>
python models/deepseek_v4_1_flash/hc.py --op pre --batch 72 -d <设备>
python models/deepseek_v4_1_flash/hc.py --op head-norm --batch 17 -d <设备>
python models/deepseek_v4_1_flash/decode_moe.py --ep 2 --experts-per-rank 8 --num-tokens-per-rank 72,17 -d <设备组>

完整的三行数对照由 workflow/validate.py.txt 保存；原核仅留在过程快照中，不作为模型回退分支。
Torch 只用于 CPU 数据准备和校验；编译并发限制为 2，不进行文件 hash 校验。
