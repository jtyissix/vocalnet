import torch
import os
import torchaudio
import torch.nn as nn
from vita_audio.tokenizer import get_audio_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from torch.fx import symbolic_trace
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
model_path = "VITA-MLLM/VITA-Audio-Boost"
audio_tokenizer_path='/home/jiangtianyuan/resource/voice/vita/models/THUDM/'
flow_path='/home/jiangtianyuan/resource/voice/vita/models/Decoder/'
audio_tokenizer = get_audio_tokenizer(
        audio_tokenizer_path, "glm4voice", flow_path=flow_path
    )
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="cuda",
    attn_implementation="flash_attention_2",
).eval()
model.generation_config = GenerationConfig.from_pretrained(
        model_path, trust_remote_code=True
    )

chat_template = """
{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n
"""
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,chat_template=chat_template,)

model.generation_config.max_new_tokens = 4096
model.generation_config.chat_format = "chatml"
model.generation_config.max_window_size = 8192
model.generation_config.use_cache = True
model.generation_config.do_sample = True
model.generation_config.pad_token_id = tokenizer.pad_token_id
class MTPConcatTracer:
    def __init__(self, model):
        self.model = model   # 注意：VITA 是 model.model 才有 mtp_xx
        self.hidden_cache = {}
        self.embed_cache = {}
        self.concat_cache = {}
        self.handles = []

    def register_hooks(self):

        # hook: hidden_norm
        for i, norm in enumerate(self.model.mtp_hidden_norms):
            h = norm.register_forward_hook(self._make_hidden_hook(i))
            self.handles.append(h)

        # hook: embed_norm
        for i, norm in enumerate(self.model.mtp_embed_norms):
            h = norm.register_forward_hook(self._make_embed_hook(i))
            self.handles.append(h)

        # hook: proj （concat 发生前）
        for i, proj in enumerate(self.model.mtp_projs):
            h = proj.register_forward_pre_hook(self._make_concat_hook(i))
            self.handles.append(h)

    def _make_hidden_hook(self, idx):
        def hook(m, inp, out):
            self.hidden_cache[idx] = out.detach()
        return hook

    def _make_embed_hook(self, idx):
        def hook(m, inp, out):
            self.embed_cache[idx] = out.detach()
        return hook

    def _make_concat_hook(self, idx):
        def hook(m, inp):
            x = inp[0].detach()       # shape = (B, T, 7168)
            concat = inp[0]

            h = self.hidden_cache.get(idx, None)  # (B, T, 3584)
            e = self.embed_cache.get(idx, None)

            if h is None or e is None:
                print(f"[MTP-{idx}] missing hidden/embed cache")
                return

            # 取第一 token 的前 10 维用于判断对齐
            concat_left  = concat[0, 0, :10]
            concat_right = concat[0, 0, 3584:3594]

            hidden_head = h[0, 0, :10]
            embed_head  = e[0, 0, :10]

            # 比较绝对差是否接近 0
            is_hidden_left = torch.allclose(concat_left, hidden_head, atol=1e-5)
            is_embed_left  = torch.allclose(concat_left, embed_head, atol=1e-5)

            print(f"\n========== MTP-{idx} Concat Check ==========")
            print(f"concat shape: {concat.shape}")
            print(f"hidden_norm shape: {h.shape}")
            print(f"embed_norm shape:  {e.shape}")

            if is_hidden_left:
                print(f"👉 concat = [ hidden_norm | embed_norm ]")
            elif is_embed_left:
                print(f"👉 concat = [ embed_norm | hidden_norm ]")
            else:
                print("⚠️  WARNING: Could not match concat left side!")

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

tracer = MTPConcatTracer(model)
tracer.register_hooks()
# 测试一次 forward
inputs=tokenizer.apply_chat_template([{"role":"user", "content":"convert the text to speech.\n 刘周祥你他妈真是个大傻逼！"}], tokenize=True,add_generation_prompt=True, return_tensors="pt")
#inputs = tokenizer("convert the text to speech.\n Hello world i want to know your structure clearly. can you tell it to me", return_tensors="pt")
responses=model.generate( input_ids=inputs.cuda(),
                # temperature=0.2,
                # top_p=0.8,
                # do_sample=False,
                # temperature=1.0,
                max_new_tokens=1024,
                min_new_tokens=1,)
response = responses[0][len(inputs[0]) :]
audio_offset = tokenizer.convert_tokens_to_ids("<|audio_0|>")
text_tokens = []
audio_tokens = []
for token_id in response:
    if token_id >= audio_offset:
        audio_tokens.append(token_id - audio_offset)
    else:
        text_tokens.append(token_id)

    if len(audio_tokens) == 0:
        continue

tts_speech = audio_tokenizer.decode(audio_tokens, source_speech_16k='/home/jiangtianyuan/resource/voice/vita/asset/2631296891109983590.wav')

#wav_path = os.path.join(output_dir, filename + ".wav")
#os.makedirs(os.path.dirname(wav_path), exist_ok=True)
torchaudio.save('/home/jiangtianyuan/resource/voice/vita/other_test/view_vita/demo.wav', tts_speech.unsqueeze(0), 22050, format="wav")
tracer.remove()
breakpoint()