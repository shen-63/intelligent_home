import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 自动下载基座模型 & 读取你 G 盘解压好的 LoRA 权重
base_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = r"G:\models\checkpoint-400"  # G 盘权重绝对路径

print("⌛ 正在加载基座模型与 checkpoint-400 权重...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float32,
    device_map="cpu",  # 如果有独显可改为 "cuda" 或 "auto"
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# 2. 测试指令与系统提示词
system_prompt = "你是一个智能家居 NLU 解析助手，请将用户的控制指令解析为对应的意图和槽位 JSON。"
prompt = "帮我把卧室的空调打开，温度调到26度"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("🤖 正在本地解析意图...\n")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.1)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("💡 输入指令:", prompt)
print("📌 NLU 解析结果:\n", response)