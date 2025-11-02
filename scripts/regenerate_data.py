"""
重新生成数据集 - 使用2秒音频以适应显存限制
"""
import os
import shutil
import subprocess
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

def main():
    print("=" * 80)
    print("重新生成数据集(2秒音频)")
    print("=" * 80)
    print("由于显存限制，需要将音频长度从4秒减少到2秒")
    print()
    
    # 检查是否存在旧数据
    data_dir = os.path.join(project_root, 'data', 'processed', 'mixed')
    if os.path.exists(data_dir):
        print(f"发现旧数据目录: {data_dir}")
        response = input("是否删除并重新生成? (y/n): ")
        if response.lower() == 'y':
            print("删除旧数据...")
            shutil.rmtree(data_dir)
            print("✓ 旧数据已删除")
        else:
            print("取消操作")
            return
    
    # 生成新数据
    print("\n生成新数据集...")
    cmd = "python scripts/2_generate_mixtures.py"
    print(f"运行命令: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("\n✗ 数据生成失败")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("数据集重新生成完成!")
    print("=" * 80)
    print("\n配置已优化:")
    print("- 音频长度: 4秒 → 2秒")
    print("- Batch size: 2 → 1")
    print("- 编码器滤波器: 512 → 256")
    print("- 隐藏层通道: 512 → 256")
    print("- TCN块层数: 6 → 4")
    print("- Workers: 4 → 2")
    print("\n现在可以重新开始训练:")
    print("python scripts/3_train.py")

if __name__ == "__main__":
    main()
