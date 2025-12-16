from transformers import AutoModel
from torchinfo import summary
import os
from transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoConfig
import torch
def print_module(name, module, indent=0):
    prefix = " " * indent
    print(f"{prefix}{name}: {module.__class__.__name__}")

    # 子模块
    for child_name, child in module.named_children():
        print_module(child_name, child, indent + 4)

def print_structure(model):
    print("\n==================== EMBEDDING ====================")
    print_module("embed_tokens", model.model.embed_tokens)

    print("\n==================== 38 TRANSFORMER LAYERS ====================")
    for i, layer in enumerate(model.model.layers):
        print(f"\n----- Layer {i} -----")
        print_module(f"layer_{i}", layer)

    print("\n==================== FINAL RMS NORM ====================")
    print_module("norm", model.model.norm)

    print("\n==================== LM HEAD ====================")
    print_module("lm_head", model.lm_head)

    # 找 MTP 模块
    print("\n==================== MTP (NEXT-N-PREDICT) LAYERS ====================")

    mtp_attributes = ["nextn_predict_layers", "mtp_layers", "mtp_projs", "mtp_heads"]

    found = False
    for attr in mtp_attributes:
        if hasattr(model, attr):
            module = getattr(model, attr)
            print(f"\n>>> Found MTP module: {attr}")
            print_module(attr, module)
            found = True

    if not found:
        print("⚠️ 未找到任何 MTP 模块，请打印 model 以确认内部属性名。")

def hook(module, inp, out):
    print(f"[{module.__class__.__name__}] input: {inp[0].shape} output: {out.shape}")


# ----------------- MAIN ---------------------
model_path = "VITA-MLLM/VITA-Audio-Boost"

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, config=config, trust_remote_code=True, torch_dtype="auto"
)
for name, module in model.named_modules():
    module.register_forward_hook(hook)
print("模型加载成功！开始打印完整结构...\n")

print_structure(model)

#model = AutoModelForCausalLM.from_pretrained("VITA-MLLM/VITA-Audio-Boost",trust_remote_code=True,device_map="cpu") 
#model = AutoModel.from_pretrained("VITA-MLLM/VITA-Audio-Boost", device_map="cpu")
#print(summary(model, depth=5))
breakpoint()
print(model)
model(input_ids)
