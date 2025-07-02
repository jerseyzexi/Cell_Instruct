from Test.test import test_rxrx1_embedding

emb = test_rxrx1_embedding()
print(emb)

mask = emb != 1

# 2. nonzero 会返回所有 mask=True 的坐标索引（每行一个坐标向量）
#    as_tuple=False 返回形状为 (N, ndim) 的 LongTensor
non_one_indices = mask.nonzero(as_tuple=False)

# 3. 用 mask 直接索引得到所有不等于 1 的值
non_one_values = emb[mask]

# 4. 打印或收集成列表
for idx, val in zip(non_one_indices, non_one_values):
    # idx 是一个长度为 ndim 的张量，比如 [batch_i, seq_j, feat_k]
    # val 是 emb 在该位置的标量
    print(f"坐标 {idx.tolist()} 上的值 = {val.item()}")