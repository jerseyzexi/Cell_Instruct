import pandas as pd

# 1. 读取两个文件
#    假设第一个 CSV 是已经合并过 gene→srna_id 的表，列名里含有 'srna_id'
#    第二个 CSV 原始表里也有同样的 'srna_id' 列
df1 = pd.read_csv('../merged_with_srna.csv')    # 第一个 CSV，包含了一列 'srna_id'
df2 = pd.read_csv('../train.csv')   # 要被过滤的第二个 CSV，同样包含 'srna_id'

# 2. 取出第一个表里所有合法的 srna_id
valid_ids = set(df1['srna_id'].dropna())

# 3. 过滤第二个表：只保留那些 srna_id 在 valid_ids 中的行
df2_filtered = df2[df2['edit_prompt'].isin(valid_ids)]

# 4. （可选）把结果保存到新文件
df2_filtered.to_csv('second_filtered.csv', index=False)

# 5. 查看一下过滤后的前几行
print(df2_filtered.head())