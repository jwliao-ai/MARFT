import pandas as pd
import json
from pathlib import Path

def analyze_parquet(file_path, sample_rows=3, convert_json=False):
    """Parquet文件分析工具"""
    try:
        # 读取文件
        df = pd.read_parquet(file_path)
        print("✅ 文件读取成功")
    except Exception as e:
        print(f"❌ 读取失败: {str(e)}")
        return

    # 基础信息分析
    print("\n=== 文件结构 ===")
    print(f"总行数: {len(df):,}")
    print(f"列数: {len(df.columns)}")
    print("列名及数据类型:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")

    # 空值统计
    print("\n=== 空值统计 ===")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        print(f"  - {col}: {count} 空值")

    # # 数据预览
    # print(f"\n=== 前{sample_rows}行样例 ===")
    # print(df.head(sample_rows).to_markdown(index=False))

    # 转换为JSON
    if convert_json:
        output_path = Path(file_path).with_suffix('.json')
        try:
            df.to_json(output_path, orient='records', indent=2, force_ascii=False)
            print(f"\n🎉 转换完成 -> {output_path}")
        except Exception as e:
            print(f"❌ 转换失败: {str(e)}")

if __name__ == "__main__":
    file_path = "data/train-00000-of-00001.parquet"  # 修改为实际文件路径
    
    # 参数设置
    sample_size = 5      # 预览行数
    need_conversion = True  # 是否生成JSON文件
    
    # 执行分析
    analyze_parquet(file_path, sample_size, need_conversion)