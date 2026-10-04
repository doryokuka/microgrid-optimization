# 数据与运行依赖

所有文件路径相对于 src。

| 模块 | 原入口所需私有文件 | 注意事项 |
| --- | --- | --- |
| 第一问 | 附件/附件1.xlsx、附件/附件5/result1.xlsx | 代码读取赛事模板标签，模板也不能随意删除后宣称可运行 |
| 第二问 | 第二问/data/attachment1.xlsx、attachment2.xlsx、calendar.npz、sha256.json，以及清单列出的文件 | common.py 在导入时读取 calendar；verify_data 遍历完整哈希清单；--retrain 也不跳过该校验 |
| 第二问默认预测 | 第二问/data/issued_forecasts.npz | 冻结预测缓存；不等于从原始数据重新训练 |
| 第二问参考核验 | 第二问/reference/q80.npz | verify_reference 需要；正常新优化不是读取该参考购电量决策 |
| 第三问 | 附件/附件1.xlsx、附件2.xlsx；第三问/data/centers.npz、startup.npz | 冻结预测和一月启动输入是运行必需数据，不应称为可随意删除的临时缓存 |
| 第三问现有 run 审计 | 第三问/data/baseline_reference.npz、reference.npz | run_pair 审计读取 baseline_reference，并调用 reference_check 读取两份全年参考结果 |
| 第四问 | 同层第二问最终版_Q80 的源码、calendar/issued_forecasts/sha256；附件2/3/4 | 现 ZIP 第二问目录名不匹配；Q4-3 另需 Q4-2 完整输出中的一月轨迹 |

原依赖版本仅是材料记录，未在本次安装验证。建议先为第一问和第二问分别做干净环境短跑，再决定是否公开第三、四问。不要删除代码中的哈希与因果性检查以掩盖缺失输入。更适合公开的后续改造是独立数据适配与合成演示入口，保留优化公式、约束和费用逻辑；不在本次任务中实施。

时间语义需复核：第一问显式循环移动午夜；第二问 common.py 文档称 right endpoints，而第一问和第四问文档包含 interval start 表述。不要因统一目录而擅自移动数组或模板表头。正式公开时应记录每问的时间约定。
