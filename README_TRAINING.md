# Llama3.2-3B Semantic Communication Training

这个项目已经配置为在4070ti显卡上训练Llama3.2-3B模型进行语义通信。

## 配置说明

### 硬件要求
- GPU: RTX 4070ti (12GB VRAM)
- RAM: 16GB+ 推荐
- 存储: 10GB+ 可用空间

### 关键优化
- **混合精度训练**: 使用FP16减少显存占用
- **梯度累积**: batch_size=8, accumulation_steps=4 (有效batch_size=32)
- **梯度裁剪**: 防止梯度爆炸
- **优化的数据集**: 1000训练样本，200验证样本

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 开始训练
```bash
python train_llama3_2.py
```

### 3. 配置文件
主要配置文件: `config/llama3_2_3b.yaml`

关键参数:
- `batch_size: 8` - 小batch size节省显存
- `gradient_accumulation_steps: 4` - 模拟更大batch
- `mixed_precision: true` - 启用混合精度训练
- `learning_rate: 1e-4` - 适合微调的学习率

## 显存使用情况

- **模型加载**: ~6GB (FP16)
- **训练过程**: ~8-10GB
- **峰值显存**: ~11GB (安全范围内)

## 训练输出

- 模型检查点: `checkpoints/best_model.pth`
- 训练日志: 实时显示loss和进度
- 验证结果: 每个epoch后显示验证损失

## 故障排除

### 显存不足 (OOM Error)
如果遇到 `torch.OutOfMemoryError` 错误:
1. **已修复**: 代码已修复模型重复加载导致的显存溢出问题
2. 如果仍有问题，可以:
   - 减少batch_size到4
   - 增加gradient_accumulation_steps到8
   - 确保关闭其他GPU程序

### 数据集缓存
CNN/DailyMail数据集会自动缓存到 `~/.cache/huggingface/datasets/`
- 第一次运行需要下载和缓存
- 后续运行会直接从缓存加载，速度很快

### 训练速度慢
- 检查GPU温度和功耗设置
- 确保使用CUDA而不是CPU
- 考虑使用更小的数据集进行测试

## 下一步

训练完成后，你可以:
1. 使用训练好的模型进行语义通信测试
2. 调整信道参数(信噪比等)
3. 在其他数据集上测试模型性能