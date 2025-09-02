from transformers import AutoTokenizer

MODEL_NAME = "E:\\workspace\\ai\\llm\\bge-large-zh-v1.5"

# MODEL_NAME = "E:\\workspace\\ai\\llm\\m3e-large"

# MODEL_NAME = "E:\\workspace\\ai\\llm\\bge-m3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
text = "，利用视觉信息和解析,?112233EE//()（）--++..><##@@"
# text = "视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技利用视觉信息和解析技析技"
tokens = tokenizer(text, return_tensors="pt")
token_count = len(tokens["input_ids"][0])  # 获取 token 数量
print(f"Token 数量: {token_count}")
print("-------------------------------")

from transformers import AutoModel

model = AutoModel.from_pretrained(MODEL_NAME)

inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
# 获取句子嵌入（通常取最后一层隐藏状态的平均）
embeddings = outputs.last_hidden_state.mean(dim=1)
print(embeddings)

"""
 简单版，
 1. 直接按token数量进行分割。
 2. 不考虑语义相关性。
 3. 可能会导致信息断层。
"""
def split_by_tokens(text, chunk_size=6):
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk))
    return chunks
"""
 增强版，
 1. 确保不切断中文字符（回退到最近的完整token）。
 2. 应用重叠区域（避免信息断层）。
 3. 优化了token计数逻辑，确保准确计算每个chunk的token数量。
"""
def semantic_chunking(text, max_tokens=512, overlap=20):
    # 首次完整编码获取所有token
    all_tokens = tokenizer.encode(text)
    chunks = []
    start_idx = 0
    
    while start_idx < len(all_tokens):
        end_idx = min(start_idx + max_tokens, len(all_tokens))
        
        # 确保不切断中文字符（回退到最近的完整token）
        while end_idx < len(all_tokens) and not tokenizer.convert_ids_to_tokens([all_tokens[end_idx]])[0].startswith('##'):
            end_idx += 1
        
        chunk = tokenizer.decode(all_tokens[start_idx:end_idx])
        chunks.append(chunk)
        
        # 应用重叠区域（避免信息断层）
        start_idx = max(0, end_idx - overlap) if overlap > 0 else end_idx
    
    return chunks


# 简单版使用示例
chunks = split_by_tokens(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} token count: {len(tokenizer.encode(chunk))}")



# 增强版使用示例
long_text = "您的长文本内容..."
chunks = semantic_chunking(long_text)
for i, chunk in enumerate(chunks):
    print(f"块{i+1} token数: {len(tokenizer.encode(chunk))}")