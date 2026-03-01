# myGpt 复刻 nanochat 路线图

## 总体原则

- 每个阶段独立可验证
- 先跑通再优化
- 保持代码简洁

---

## 阶段 1：GPT 模型核心（gpt.py）✅

目标：完成一个可以前向传播的完整 GPT 模型

### 1.1 基础组件
- [x] **RMSNorm** - 归一化层
  - 验证：对比 PyTorch LayerNorm 输出

### 1.2 注意力机制

- [x] **Rotary Embedding (RoPE)** - 旋转位置编码
  - 验证：可视化位置编码矩阵
- [x] **Attention** - 注意力层（支持因果掩码）
  - 验证：检查注意力权重形状、因果掩码是否正确

### 1.3 前馈网络
- [x] **MLP** - 使用 ReLU² 激活
  - 验证：检查输出形状

### 1.4 Transformer Block
- [x] **Block** - 组合 Attention + MLP + RMSNorm
  - 验证：单个 block 的前向传播

### 1.5 完整模型
- [x] **GPT** - 组合 Embedding + Blocks + LM Head
  - 验证：随机输入能输出 logits，形状正确

### 笔记
- `notes/01_rmsnorm.ipynb`
- `notes/02_rope.ipynb`
- `notes/03_attention.ipynb`
- `notes/04_mlp.ipynb`
- `notes/05_block.ipynb`
- `notes/06_gpt.ipynb`

---

## 阶段 2：分词器（tokenizer.py）✅

目标：能够将文本转换为 token ids，反之亦然

### 2.1 快速版本
- [x] 使用 HuggingFace tokenizers 库包装
  - 验证：encode/decode 往返测试

### 2.2 完整版本（可选，后期实现）
- [ ] 自己实现 BPE 训练和推理
- [ ] Rust 加速版本

### 笔记
- `notes/07_tokenizer.ipynb`

---

## 阶段 3：文本生成（engine.py）✅

目标：用训练好的模型生成文本

### 3.1 基础生成
- [x] **贪婪解码** - 每次选最大概率的 token
  - 验证：能生成连续文本（即使是乱码）

### 3.2 采样策略
- [x] **Temperature 采样**
- [x] **Top-k 采样**
- [ ] **Top-p (nucleus) 采样**（可选）
  - 验证：不同参数生成不同多样性的文本

### 3.3 KV Cache
- [x] 实现 KV Cache 加速推理
  - 验证：对比有无 cache 的生成速度

### 3.4 多样本生成
- [x] **num_samples** - 从同一 prompt 生成多个回复
- [x] **KV Cache 复制** - prefill 一次，decode 多份

### 笔记
- `notes/08_engine.ipynb`

---

## 阶段 4：数据加载（dataloader.py）✅

目标：能够加载和批处理训练数据

### 4.1 数据集处理
- [x] 理解文档拼接（BOS token 分隔）
- [x] Token buffer（deque 累积）
  - 验证：能读取处理后的数据

### 4.2 数据加载器
- [x] 实现基础 DataLoader
- [x] 理解 Parquet 格式
- [x] 理解 DDP 数据分配
- [x] 理解异步加载优化

### 笔记
- `notes/09_dataloader.ipynb`

---

## 阶段 5：优化器（adamw.py）⬅️ 当前

目标：实现训练所需的优化器

### 5.1 AdamW 基础
- [ ] **Adam 算法原理** - 一阶/二阶动量
- [ ] **AdamW vs Adam** - 权重衰减的正确位置
- [ ] **实现 AdamW**
  - 验证：对比 PyTorch AdamW 更新结果

### 5.2 参数分组
- [ ] **分离 decay 参数** - bias 和 norm 不加权重衰减
- [ ] **不同学习率** - embedding 可用不同 lr

### 5.3 学习率调度器
- [ ] **Warmup** - 训练初期逐渐增大 lr
- [ ] **Cosine Decay** - 余弦退火
- [ ] **组合调度器** - Warmup + Cosine
  - 验证：绘制 lr 曲线

### 5.4 Muon 优化器（可选进阶）
- [ ] 理解 Muon 的设计思想
- [ ] 矩阵参数用 Muon，其他用 AdamW

### 验证脚本
```python
# test_adamw.py
from picogpt.adamw import AdamW
import torch

model = torch.nn.Linear(10, 10)
opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

for _ in range(10):
    x = torch.randn(2, 10)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"Loss: {loss.item():.4f}")
```

### 笔记
- `notes/10_adamw.ipynb`

---

## 阶段 6：训练循环（train.py）

目标：能够训练模型

### 6.1 基础训练
- [ ] 实现训练循环
- [ ] 计算交叉熵损失
- [ ] 梯度裁剪
  - 验证：loss 能下降

### 6.2 梯度累积
- [ ] 支持小显存的梯度累积
- [ ] 正确处理 loss 缩放
  - 验证：累积 N 步 ≈ batch 放大 N 倍

### 6.3 混合精度训练
- [ ] **torch.autocast** - 自动混合精度
- [ ] **GradScaler** - 梯度缩放防止下溢
  - 验证：训练速度提升，精度不损失

### 6.4 日志和监控
- [ ] 打印训练指标（loss, lr, throughput）
- [ ] 支持 wandb（可选）

### 6.5 检查点管理
- [ ] 保存模型 + 优化器 + 调度器状态
- [ ] 加载并恢复训练
- [ ] 保存数据加载器状态（支持断点续训）
  - 验证：加载后能继续训练

### 验证脚本
```bash
# 小规模训练测试
python train.py --batch_size=4 --max_steps=100 --log_interval=10
```

### 笔记
- `notes/11_train.ipynb`

---

## 阶段 7：评估（eval.py）

目标：评估模型性能

### 7.1 困惑度评估
- [ ] 计算验证集困惑度
  - 验证：困惑度随训练下降

### 7.2 下游任务（可选）
- [ ] 实现简单的问答评估

### 笔记
- `notes/12_eval.ipynb`

---

## 阶段 8：分布式训练（可选）

### 8.1 DDP 基础
- [ ] 使用 PyTorch DDP 进行多 GPU 训练
- [ ] 理解 gradient all-reduce

### 8.2 分布式优化
- [ ] 分布式数据加载（每个 rank 读不同数据）
- [ ] 梯度同步策略

### 笔记
- `notes/13_ddp.ipynb`

---

## 阶段 9：Chat 微调（可选）

### 9.1 SFT
- [ ] 监督微调训练脚本
- [ ] 对话数据格式处理

### 9.2 对话格式
- [ ] 支持多轮对话格式
- [ ] Special tokens 处理

### 笔记
- `notes/14_sft.ipynb`

---

## 阶段 10：强化学习（可选进阶）

### 10.1 GRPO
- [ ] 理解 GRPO 算法
- [ ] 实现 reward 计算
- [ ] 实现 policy gradient

### 笔记
- `notes/15_rl.ipynb`

---

## 阶段 11：Web 服务（可选）

### 11.1 API 服务
- [ ] FastAPI 接口

### 11.2 Web UI
- [ ] 简单的聊天界面

---

## 建议的学习顺序

```
阶段 1 (gpt.py)           ✅
    ↓
阶段 2 (tokenizer.py)     ✅
    ↓
阶段 3 (engine.py)        ✅
    ↓
阶段 4 (dataloader.py)    ✅
    ↓
阶段 5 (adamw.py)         ⬅️ 当前
    ↓
阶段 6 (train.py)         ← 核心里程碑：能训练了！
    ↓
阶段 7+ (可选进阶)
```

## 参考文件对照

| myGpt 目标文件 | nanochat 参考文件 | 代码行数 |
|---------------|------------------|---------|
| gpt.py | nanochat/gpt.py | ~308 行 |
| tokenizer.py | nanochat/tokenizer.py | ~398 行 |
| engine.py | nanochat/engine.py | ~374 行 |
| dataloader.py | nanochat/dataloader.py | ~94 行 |
| adamw.py | nanochat/adamw.py | ~77 行 |
| train.py | scripts/base_train.py | ~399 行 |

## 当前进度

- [x] 阶段 1：GPT 模型核心（6 个 notebooks）
- [x] 阶段 2：分词器
- [x] 阶段 3：推理引擎
- [x] 阶段 4：数据加载
- [ ] 阶段 5：优化器 ⬅️
- [ ] 阶段 6：训练循环
- [ ] ...

---

## 里程碑

| 里程碑 | 目标 | 状态 |
|--------|------|------|
| M1 | 模型能前向传播 | ✅ |
| M2 | 能生成文本（用预训练权重） | ✅ |
| M3 | 能加载训练数据 | ✅ |
| M4 | 能训练模型（loss 下降） | ⬜ |
| M5 | 能在小数据集上过拟合 | ⬜ |
| M6 | 能完成完整训练 | ⬜ |

---

祝你复刻顺利！
