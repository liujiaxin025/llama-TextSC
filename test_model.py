#!/usr/bin/env python3
"""
模型测试脚本
加载训练好的模型并测试其语义通信性能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.config import Config
from src.models.mlp_codec import ChannelCodec
from src.channel.awgn import AWGNChannel
from src.core.semantic_condenser import SemanticCondenser
from src.core.semantic_vectorizer import SemanticVectorizer
from src.training.train import load_news_fallback

class ModelTester:
    def __init__(self, config_path: str = "config/llama3_2_3b.yaml", model_path: str = "checkpoints/best_model.pth"):
        self.config = Config(config_path)
        self.device = self.config.get('training.device')

        print(f"Loading model from {model_path}...")
        self.load_model(model_path)
        print("✅ Model loaded successfully!")

    def load_model(self, model_path: str):
        """加载训练好的模型"""
        # 初始化模型结构
        encoder_config = self.config.get('mlp.encoder')
        decoder_config = self.config.get('mlp.decoder')

        self.codec = ChannelCodec(encoder_config, decoder_config)
        self.codec.to(self.device)

        # 初始化语义组件
        self.condenser = SemanticCondenser(
            self.config.get('model.llama3.model_name'),
            self.config.get('model.llama3.device'),
            self.config.get('model.llama3.torch_dtype')
        )
        self.vectorizer = SemanticVectorizer(
            self.config.get('model.sentence_bert.model_name'),
            self.config.get('model.sentence_bert.device')
        )

        # 初始化信道
        self.channel = AWGNChannel(self.device)

        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.codec.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.codec.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        # 设置为评估模式
        self.codec.eval()

    def preprocess_texts(self, texts: List[str]) -> torch.Tensor:
        """预处理文本，生成语义向量"""
        print("Pre-processing test texts...")
        semantic_vectors = []

        for text in tqdm(texts, desc="Extracting semantic vectors"):
            semantic_json = self.condenser.condense(text)
            vector = self.vectorizer.vectorize(semantic_json)
            semantic_vectors.append(vector.squeeze(0))

        return torch.stack(semantic_vectors)

    def test_snr_performance(self, test_vectors: torch.Tensor, snr_range: List[float]) -> Dict[float, Dict]:
        """测试不同信噪比下的模型性能"""
        results = {}

        with torch.no_grad():
            for snr_db in snr_range:
                print(f"\nTesting SNR: {snr_db} dB")

                total_loss = 0.0
                total_mse = 0.0
                total_cosine_sim = 0.0
                num_batches = 0

                # 批量处理
                batch_size = 32
                for i in range(0, len(test_vectors), batch_size):
                    batch_vectors = test_vectors[i:i+batch_size].to(self.device)

                    # 编码
                    encoded = self.codec.encode(batch_vectors)

                    # 添加噪声
                    noisy = self.channel.add_noise(encoded, snr_db)

                    # 解码
                    decoded = self.codec.decode(noisy)

                    # 计算指标
                    loss = torch.nn.functional.mse_loss(decoded, batch_vectors)
                    mse = self.channel.calculate_mse(batch_vectors, decoded)

                    # 计算余弦相似度
                    cosine_sim = torch.nn.functional.cosine_similarity(decoded, batch_vectors, dim=1)

                    total_loss += loss.item()
                    total_mse += mse
                    total_cosine_sim += cosine_sim.mean().item()
                    num_batches += 1

                avg_loss = total_loss / num_batches
                avg_mse = total_mse / num_batches
                avg_cosine_sim = total_cosine_sim / num_batches

                results[snr_db] = {
                    'mse_loss': avg_loss,
                    'channel_mse': avg_mse,
                    'cosine_similarity': avg_cosine_sim
                }

                print(f"  MSE Loss: {avg_loss:.6f}")
                print(f"  Channel MSE: {avg_mse:.6f}")
                print(f"  Cosine Similarity: {avg_cosine_sim:.6f}")

        return results

    def test_text_reconstruction(self, test_texts: List[str], snr_db: float = 10.0) -> List[Dict]:
        """测试文本重构质量"""
        print(f"\nTesting text reconstruction at SNR: {snr_db} dB")

        # 预处理文本
        test_vectors = self.preprocess_texts(test_texts)

        results = []

        with torch.no_grad():
            for i, (text, original_vector) in enumerate(zip(test_texts, test_vectors)):
                original_vector = original_vector.unsqueeze(0).to(self.device)

                # 编码传输
                encoded = self.codec.encode(original_vector)
                noisy = self.channel.add_noise(encoded, snr_db)
                decoded = self.codec.decode(noisy)

                # 计算相似度
                cosine_sim = torch.nn.functional.cosine_similarity(
                    decoded, original_vector, dim=1
                ).item()

                mse = torch.nn.functional.mse_loss(decoded, original_vector).item()

                results.append({
                    'original_text': text,
                    'cosine_similarity': cosine_sim,
                    'mse': mse
                })

                if i < 5:  # 显示前5个结果
                    print(f"\nExample {i+1}:")
                    print(f"Original: {text}")
                    print(f"Cosine Similarity: {cosine_sim:.6f}")
                    print(f"MSE: {mse:.6f}")

        return results

    def plot_performance(self, results: Dict[float, Dict]):
        """绘制性能曲线"""
        snr_values = sorted(results.keys())
        mse_losses = [results[snr]['mse_loss'] for snr in snr_values]
        cosine_sims = [results[snr]['cosine_similarity'] for snr in snr_values]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # MSE Loss
        ax1.plot(snr_values, mse_losses, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('MSE Loss vs SNR')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Cosine Similarity
        ax2.plot(snr_values, cosine_sims, 'r-o', linewidth=2, markersize=8)
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('Cosine Similarity vs SNR')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig('performance_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\n📊 Performance curves saved to 'performance_curves.png'")

    def run_comprehensive_test(self):
        """运行完整的测试"""
        print("=" * 60)
        print("🧪 Starting Comprehensive Model Testing")
        print("=" * 60)

        # 加载测试数据
        print("\n📝 Loading test data...")
        test_texts = load_news_fallback(100)  # 100个测试文本
        print(f"Loaded {len(test_texts)} test texts")

        # 预处理
        test_vectors = self.preprocess_texts(test_texts)
        print(f"Pre-processed {len(test_vectors)} semantic vectors")

        # 测试不同信噪比
        print("\n📡 Testing performance across different SNR values...")
        snr_range = [0, 5, 10, 15, 20, 25, 30]
        results = self.test_snr_performance(test_vectors, snr_range)

        # 测试文本重构
        print("\n🔄 Testing text reconstruction...")
        reconstruction_results = self.test_text_reconstruction(test_texts[:10], snr_db=15.0)

        # 绘制性能曲线
        print("\n📊 Generating performance plots...")
        self.plot_performance(results)

        # 生成报告
        print("\n📋 Generating test report...")
        self.generate_report(results, reconstruction_results)

        print("\n✅ Testing completed!")
        return results, reconstruction_results

    def generate_report(self, snr_results: Dict, reconstruction_results: List[Dict]):
        """生成测试报告"""
        report = []
        report.append("=" * 60)
        report.append("🧪 MODEL TEST REPORT")
        report.append("=" * 60)

        # SNR性能表
        report.append("\n📡 Performance vs SNR:")
        report.append("-" * 40)
        report.append(f"{'SNR (dB)':<10} {'MSE Loss':<12} {'Cosine Sim':<12}")
        report.append("-" * 40)

        for snr in sorted(snr_results.keys()):
            mse = snr_results[snr]['mse_loss']
            cos_sim = snr_results[snr]['cosine_similarity']
            report.append(f"{snr:<10} {mse:<12.6f} {cos_sim:<12.6f}")

        # 文本重构统计
        report.append(f"\n🔄 Text Reconstruction Summary (SNR=15dB):")
        report.append("-" * 40)
        cos_sims = [r['cosine_similarity'] for r in reconstruction_results]
        mses = [r['mse'] for r in reconstruction_results]

        report.append(f"Average Cosine Similarity: {np.mean(cos_sims):.6f}")
        report.append(f"Average MSE: {np.mean(mses):.6f}")
        report.append(f"Cosine Similarity Std: {np.std(cos_sims):.6f}")
        report.append(f"MSE Std: {np.std(mses):.6f}")

        # 保存报告
        report_text = '\n'.join(report)
        with open('test_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n📄 Test report saved to 'test_report.txt'")

def main():
    """主函数"""
    # 创建测试器
    tester = ModelTester()

    # 运行测试
    results, reconstruction_results = tester.run_comprehensive_test()

    print("\n🎉 All tests completed successfully!")
    print("Files generated:")
    print("  - performance_curves.png: Performance visualization")
    print("  - test_report.txt: Detailed test report")

if __name__ == "__main__":
    main()