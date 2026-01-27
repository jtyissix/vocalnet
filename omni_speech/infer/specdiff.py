import sys
import os
from accelerate import dispatch_model, infer_auto_device_map
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES']="0,2,4,5,6"
for p in sys.path:
        print(p)
sys.path.insert(0, '/home/jiangtianyuan/resource/voice/vocalnet/')
sys.path.append('/home/jiangtianyuan/resource/voice/vocalnet/')
#sys.path.insert(0, '/home/jiangtianyuan/resource/voice/vita/third_party/GLM-4-Voice/third_party/Matcha-TTS/')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print('hello')
import torch
from typing import Tuple, Callable
from omni_speech.model.builder import load_pretrained_model

from omni_speech.datasets.preprocess import preprocess_llama_3_v1, preprocess_qwen_2_5_v1
import whisper
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from functools import partial
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.model import CosyVoice2Model
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph
import logging
import librosa
import torchaudio
import json
import onnxruntime
import torchaudio.compliance.kaldi as kaldi
import re
import time
import argparse
import uuid
import glob
import pdb
import subprocess
import asyncio
TARGET_MODEL_PATH=""
DRAFT_MODEL_PATH=""
COSYVOICE_MODEL=""     ## CosyVoice2-0.5B       i.e. /workspace/CosyVoice/pretrained_models/CosyVoice2-0.5B-VocalNet
VOCALNET_MODEL = ""    ##  /root/speechllm_checkpoints/VocalNet-qwen25-7B  VocalNet speech LLM   i.e. ./checkpoints/VocalNet-1B
PROMPT_SPEECH="./omni_speech/infer/cn_prompt.wav"   
# PROMPT_SPEECH="./omni_speech/infer/common_voice_en_2586258.wav"   

try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    use_ttsfrd = False
def speculative_sample(
        self,
        draft_tokens: List[int],
        draft_logits: List[List[float]],  # 来自draft model的logits
        target_logits: List[List[float]],  # 来自verify_with_draft的logits
        temperature: float = 1.0,
        top_k: int = 1
    ) -> Tuple[List[int], int]:
        """
        执行speculative sampling
        
        Args:
            draft_tokens: K个draft生成的token
            draft_logits: K个draft位置的logits
            target_logits: K+1个target位置的logits (用于验证+bonus)
            
        Returns:
            (accepted_tokens, num_accepted)
        """
        accepted = []
        K = len(draft_tokens)
        
        for t in range(K):
            if t >= len(target_logits) or t >= len(draft_logits):
                break
                
            # 转换为tensor
            q_logits = torch.tensor(target_logits[t], device=self.device)
            p_logits = torch.tensor(draft_logits[t], device=self.device)
            
            # 应用temperature
            if temperature != 1.0:
                q_logits = q_logits / temperature
                p_logits = p_logits / temperature
            
            # 计算概率分布
            q_probs = F.softmax(q_logits, dim=-1)
            p_probs = F.softmax(p_logits, dim=-1)
            
            draft_token = draft_tokens[t]
            q_prob = q_probs[draft_token].item()
            p_prob = p_probs[draft_token].item()
            
            # 计算接受概率
            if p_prob < 1e-10:
                accept_prob = 1.0 if q_prob > 1e-10 else 0.0
            else:
                accept_prob = min(1.0, q_prob / p_prob)
            
            # 采样r ~ U[0,1]
            r = torch.rand(1).item()
            
            if r < accept_prob:
                # 接受draft token
                accepted.append(draft_token)
            else:
                # 拒绝，从(q-p)+重采样
                residual = torch.clamp(q_probs - p_probs, min=0)
                residual_sum = residual.sum()
                
                if residual_sum < 1e-10:
                    probs = q_probs
                else:
                    probs = residual / residual_sum
                
                # top-k
                if top_k > 0 and top_k < probs.shape[0]:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs)
                    probs.scatter_(0, top_k_indices, top_k_probs)
                    probs = probs / probs.sum()
                
                new_token = torch.multinomial(probs, 1).item()
                accepted.append(new_token)
                # 拒绝后退出循环
                return accepted, t + 1
        
        # 全部接受，采样bonus token
        if K < len(target_logits):
            bonus_logits = torch.tensor(target_logits[K], device=self.device)
            if temperature != 1.0:
                bonus_logits = bonus_logits / temperature
            bonus_probs = F.softmax(bonus_logits, dim=-1)
            
            if top_k > 0 and top_k < bonus_probs.shape[0]:
                top_k_probs, top_k_indices = torch.topk(bonus_probs, top_k)
                bonus_probs = torch.zeros_like(bonus_probs)
                bonus_probs.scatter_(0, top_k_indices, top_k_probs)
                bonus_probs = bonus_probs / bonus_probs.sum()
            
            bonus_token = torch.multinomial(bonus_probs, 1).item()
            accepted.append(bonus_token)
        
        return accepted, K + 1
class SpeechTokenizer:
    def __init__(self, speech_tokenizer_model: str, feat_extractor: Callable, get_tokenizer: Callable, 
                 campplus_model: str, allowed_special: str = 'all',device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = get_tokenizer()
        self.allowed_special = allowed_special
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model, sess_options=option, providers=providers
        )
        
        self.feat_extractor = feat_extractor
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
        )
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize('{}/pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)) is True, \
                'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyinvg')
        else:
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False)
            self.en_tn_model = EnNormalizer()
    
    def extract_speech_token(self, speech: torch.Tensor, max_duration_sec: int = 30):
        """
        Args:
            speech (torch.Tensor)
            max_duration_sec (int, optional)

        Returns:
            speech_token (torch.Tensor)
            speech_token_len (torch.Tensor)
        """
        sample_rate = 16000  
        max_samples = max_duration_sec * sample_rate

        total_samples = speech.shape[1]
        num_chunks = (total_samples + max_samples - 1) // max_samples  

        speech_tokens = []
        speech_token_lengths = []

        for i in range(num_chunks):
            start = i * max_samples
            end = min(start + max_samples, total_samples)
            chunk = speech[:, start:end] 
            feat = whisper.log_mel_spectrogram(chunk, n_mels=128)  

            input_name_0 = self.speech_tokenizer_session.get_inputs()[0].name
            input_name_1 = self.speech_tokenizer_session.get_inputs()[1].name
            inputs = {
                input_name_0: feat.detach().cpu().numpy(),
                input_name_1: np.array([feat.shape[2]], dtype=np.int32)
            }

            speech_token_output = self.speech_tokenizer_session.run(None, inputs)[0]
            chunk_speech_token = torch.tensor(speech_token_output.flatten().tolist(), dtype=torch.int32).unsqueeze(0).to(self.device)
            chunk_speech_token_len = torch.tensor([chunk_speech_token.shape[1]], dtype=torch.int32).to(self.device)
            speech_tokens.append(chunk_speech_token)
            speech_token_lengths.append(chunk_speech_token_len)

        speech_token = torch.cat(speech_tokens, dim=1)  
        speech_token_len = torch.cat(speech_token_lengths, dim=0).sum().unsqueeze(0)  

        return speech_token, speech_token_len

    def save_tokens(self, speech_token: torch.Tensor, speech_token_len: torch.Tensor, save_path: str):
        np.save(save_path, {'speech_token': speech_token.cpu().numpy(), 'speech_token_len': speech_token_len.cpu().numpy()})
        print(f"Tokens saved to {save_path}")

    def load_tokens(self, load_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(load_path, allow_pickle=True).item() 
        speech_token = torch.tensor(data['speech_token'])
        speech_token_len = torch.tensor(data['speech_token_len'])

        print(f"Tokens loaded from {load_path}")
        return speech_token, speech_token_len
    
    def extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,
                                              {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding
    
    def extract_text_token(self, text):
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        return text_token, text_token_len
    def text_normalize(self, text, split=True):
        text = text.strip()
        for token in self.tokenizer.special_tokens['additional_special_tokens']:
            if token in text:
                return text if split is False else [text]
        if contains_chinese(text):
            if self.use_ttsfrd:
                texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
                text = ''.join(texts)
            else:
                text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,、]+$', '。', text)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
        else:
            if self.use_ttsfrd:
                texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
                text = ''.join(texts)
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
        if split is False:
            return text
        return texts

class CosyvoiceVocoder:
    def __init__(self, frontend: SpeechTokenizer, model: CosyVoice2Model, sample_rate: int = 24000):
        frontend = frontend
        self.model = model
        self.sample_rate = sample_rate

    def inference_zero_shot(self, speech_token, prompt_token: torch.Tensor, prompt_feat: torch.Tensor, embedding: torch.Tensor, stream=False, speed=1.0, uuid=None, is_last_speech_chunk=False):
        speech_token = speech_token.squeeze(1) if speech_token.dim() == 3 else speech_token
        prompt_token = prompt_token.squeeze(1) if prompt_token.dim() == 3 else prompt_token

        model_input = {
            'speech_tokens': speech_token,
            'flow_embedding': embedding,
            'prompt_token': prompt_token,
            'prompt_feat': prompt_feat,
            'stream': stream,
            'speed': speed,
            'uuid': uuid,
            'is_last_speech_chunk': is_last_speech_chunk
        }

        for model_output in self.model.tts_direct_update(**model_input):
            yield model_output

class VocalNetModelStream:
    def __init__(self, draft_model_path: str,target_model_path:str, vocoder_path: str = COSYVOICE_MODEL, s2s: bool = True, **kwargs):
        self.s2s = s2s
        self.draft_model_path = draft_model_path
        self.target_model_path = target_model_path
        self.empty = True
        self.vocoder_path = vocoder_path

        self.temperature = kwargs.get('temperature', 0.1)
        self.num_beams = kwargs.get('num_beams', 1)
        self.max_new_tokens = kwargs.get('max_new_tokens', 512)
        self.top_p = kwargs.get('top_p', 0.1)
        self.top_p = kwargs.get('top_k', 0.0)
        self.streaming = kwargs.get('streaming', False)

        self.audio_dir = None
        self.empty = True

        self.txt_token_num = kwargs.get('txt_token_num', 5)
        self.speech_token_num = kwargs.get('speech_token_num', 15)
        self.reset_interval = kwargs.get('reset_interval', 50)

    def __initilize__(self):
        if self.empty:
            self.empty = False
            self.draft_tokenizer, self.draft_model, _ = load_pretrained_model(self.draft_model_path, s2s=self.s2s)
            self.draft_model.tokenizer = self.draft_tokenizer
            self.target_tokenizer, self.target_model, _ = load_pretrained_model(self.target_model_path, s2s=self.s2s)
            self.target_model.tokenizer = self.target_tokenizer
            device_map = infer_auto_device_map(self.target_model, max_memory={
            0: "45GiB", 2: "45GiB" ,4: "45GiB",5: "45GiB", 6: "45GiB"
                                                })
            self.target_model = dispatch_model(self.target_model, device_map=device_map)
            self.draft_model = dispatch_model(self.draft_model, device_map=device_map)
            self.txt_eos_emb=self.draft_model.get_model().embed_tokens(torch.tensor([[128009]], device=hidden_states.device))
            self.__init_vocoder__()


    def __init_vocoder__(self):
        model_dir = self.vocoder_path
        with open(f'{model_dir}/cosyvoice.yaml', 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

        frontend = SpeechTokenizer(
            speech_tokenizer_model=f"{model_dir}/speech_tokenizer_v2.onnx",
            feat_extractor=configs['feat_extractor'],
            get_tokenizer=configs["get_tokenizer"],
            allowed_special=configs['allowed_special'],
            campplus_model=f"{model_dir}/campplus.onnx"
        )

        model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'])

        model.load(
            f'{model_dir}/llm.pt',
            f'{model_dir}/flow.pt',
            f'{model_dir}/hift.pt'
        )

        self.cosy_vocoder = CosyvoiceVocoder(frontend=frontend, model=model)

        prompt_wav = PROMPT_SPEECH
        prompt_sr = 16000 

        prompt_speech_16k = self.postprocess(load_wav(prompt_wav, prompt_sr))

        prompt_token, prompt_token_len = frontend.extract_speech_token(prompt_speech_16k)
        resample_rate = 24000  
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=prompt_sr, new_freq=resample_rate)(prompt_speech_16k)

        speech_feat, speech_feat_len = frontend.extract_speech_feat(prompt_speech_resample)
        if resample_rate == 24000:
            token_len = min(int(speech_feat.shape[1] / 2), prompt_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
            prompt_token, prompt_token_len[:] = prompt_token[:, :token_len], token_len
        embedding = frontend.extract_spk_embedding(prompt_speech_16k)

        self.prompt_token = prompt_token
        self.speech_feat = speech_feat
        self.embedding = embedding
        

    def postprocess(self, speech, top_db=60, hop_length=220, win_length=440, max_val = 0.8):
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > max_val:
            speech = speech / speech.abs().max() * max_val
        speech = torch.concat([speech, torch.zeros(1, int(24000 * 0.2))], dim=1)
        return speech

    def set_audio_dir(self, output_dir):
        self.audio_dir = output_dir

    def __call__(self, messages: list):
        wav_file = messages[0]['path']

        global_uuid = str(uuid.uuid1())
        speech = whisper.load_audio(wav_file)

        raw_len = len(speech)
        speech = whisper.pad_or_trim(speech)
        padding_len = len(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0).unsqueeze(0)
        speech_length = round(raw_len / padding_len * 3000 + 0.5)
        speech_length = torch.LongTensor([speech_length])

        conversation = [{"from": "human", "value": "<speech>", "path": f"{wav_file}"}]
        if 'qwen' in self.model_name_or_path.lower():
            input_ids = preprocess_qwen_2_5_v1([conversation], self.tokenizer, True, 4096)['input_ids']
            input_ids = torch.cat([input_ids.squeeze(), torch.tensor([198, 151644, 77091, 198], device=input_ids.device)]).unsqueeze(0)
        else:
            input_ids = preprocess_llama_3_v1([conversation], self.tokenizer, True, 4096)['input_ids']
            input_ids = torch.cat([input_ids.squeeze(), torch.tensor([128006, 78191, 128007, 271], device=input_ids.device)]).unsqueeze(0)


        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensor = speech.to(dtype=torch.float16, device='cuda', non_blocking=True)
        speech_length = speech_length.to(device='cuda', non_blocking=True)

        self.draft_model.reset_streaming_state()
        self.draft_model.speech_generator.reset_streaming_cache()
        self.draft_model.speech_generator.set_last_chunk(is_last=False)
        self.draft_model.speech_generator.speech_token_num = self.speech_token_num
        self.target_model.speech_generator.reset_streaming_cache()
        self.target_model.speech_generator.set_last_chunk(is_last=False)
        self.target_model.speech_generator.speech_token_num = self.speech_token_num

        # 初始化speculative decoding专用状态
        self.draft_model.all_txt_ids_specdiff = []
        self.draft_model.punct_count_specdiff = 0
        self.draft_model.last_punct_reset_specdiff = 0
        self.draft_model.cur_hidden_states = []
        self.draft_model.cur_text = ""
        self.draft_model.units_preds = []
        self.draft_model.all_logits_specdiff = []
        draft_full_generated_text_idx = torch.empty([1, 0], device='cuda', dtype=torch.int64)
        all_corrected_generated_text_idx = torch.empty([1, 0], device='cuda', dtype=torch.int64)
        all_corrected_speech_list = torch.empty([1, 0], device='cuda', dtype=torch.int64)  # 存储验证后的正确tokens
        draft_units_pred_list=torch.empty([1, 0], device='cuda', dtype=torch.int64)
           
        sample_rate = None
        speech_list = []               
        with torch.inference_mode():
            is_finished = False
            
            while not is_finished:
                # ============================================================
                # STEP 1: 异步生成一组文字
                # ============================================================
                text_result = await self.draft_model.generate_text_tokens_one_step(
                    inputs=input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    txt_token_num=self.txt_token_num,
                    reset_interval=self.reset_interval,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    use_cache=True
                )


                draft_concat_ids, draft_special_flag, draft_is_last, draft_hidden_for_predict,draft_all_text_logit,draft_need_reset_speech_generator_streaming_cache = text_result

                # 更新文字
                if draft_concat_ids is not None:
                    draft_full_generated_text_idx = torch.cat([draft_full_generated_text_idx, draft_concat_ids], dim=-1)
                # ============================================================
                # STEP 2: 异步生成draft语音tokens (K个)
                # ============================================================
                     
                if draft_special_flag is None:
                    is_finished=False
                    self.draft_model.speech_generator.set_last_chunk(draft_is_last)
                    self.draft_model.speech_generator.reset_streaming_cache() if draft_need_reset_speech_generator_streaming_cache else None
                    get_draft_units_pred_list_task = asyncio.create_task(
                        self.draft_model.speech_generator.streaming_predict_mtp_async(
                            draft_hidden_for_predict,
                            infer_mtp_token_num=self.config.infer_mtp_token_num
                        )
                    )
                elif draft_special_flag==1:
                    is_finished=True
                    self.draft_model.speech_generator.set_last_chunk(draft_is_last)
                    self.draft_model.speech_generator.reset_streaming_cache() if draft_need_reset_speech_generator_streaming_cache else None
                    
                    get_draft_units_pred_list_task = None
                elif draft_special_flag==2:
                    is_finished=True
                    self.draft_model.speech_generator.set_last_chunk(draft_is_last)
                    self.draft_model.speech_generator.reset_streaming_cache() if draft_need_reset_speech_generator_streaming_cache else None
                    
                    get_draft_units_pred_list_task = asyncio.create_task(
                        self.draft_model.speech_generator.streaming_predict_mtp_async(
                            draft_hidden_for_predict,
                            infer_mtp_token_num=self.config.infer_mtp_token_num
                        )
                    )
                else:
                    is_finished=True
                    self.draft_model.speech_generator.set_last_chunk(draft_is_last)
                    self.draft_model.speech_generator.reset_streaming_cache() if draft_need_reset_speech_generator_streaming_cache else None
                    
                    get_draft_units_pred_list_task = asyncio.create_task(
                        self.draft_model.speech_generator.streaming_predict_mtp_async(
                            draft_hidden_for_predict,
                            infer_mtp_token_num=self.config.infer_mtp_token_num
                        )
                    )
                # ============================================================
                # STEP 3: 并行执行 - vocoder扩散 + 验证
                # 这是核心的并行优化!
                # ============================================================
                
                verify_task = asyncio.create_task(
                self.target_model.verify_text_with_draft_stream_async(
                    inputs_embeds=self.draft_model.inputs_embeds, 
                    attention_mask=self.draft_model.attention_mask,
                    position_ids=self.draft_model.position_ids,
                    draft_tokens=draft_full_generated_text_idx, 
                    audio_unit_list=None,
                )
                )
                temp=await get_draft_units_pred_list_task
                draft_units_pred_list=torch.cat([draft_units_pred_list,temp],dim=-1) if temp is not None else draft_units_pred_list
                vocoder_prediffuse_task_1=asyncio.create_task(
                    self.cosy_vocoder.inference_zero_shot(
                        speech_token=draft_units_pred_list,
                        prompt_token=self.prompt_token,
                        prompt_feat=self.speech_feat,
                        embedding=self.embedding,
                        stream=self.streaming,
                        speed=1.0,
                        uuid=global_uuid,
                        is_last_speech_chunk=is_finished
                    )
                )
                '''
                if draft_tokens is None or draft_tokens.shape[1] <= 2:
                    continue
                
                # draft tokens (不含SOS/EOS)
                draft_content = draft_tokens[:, 1:-1]
                '''
                
                
                # ============================================================
                # STEP 4: Speculative Sampling - 逐步更新tokens
                # 根据target logits验证并修正draft tokens
                # ============================================================
                corrected_tokens = draft_content[0].tolist()  # 默认使用draft
                
                if target_logits_list and len(target_logits_list) > 0:
                    flat_logits = target_logits_list[0]
                    draft_token_list = draft_content[0].tolist()
                    
                    if flat_logits and len(flat_logits) >= len(draft_token_list):
                        # 执行speculative sampling
                        corrected_tokens, num_accepted = self.sampler.speculative_sample(
                            draft_tokens=draft_token_list,
                            draft_logits=flat_logits,  # 近似：用target作为draft
                            target_logits=flat_logits,
                            temperature=self.temperature,
                            top_k=self.top_k
                        )
                        
                        # 记录接受率（可用于监控）
                        accept_rate = num_accepted / len(draft_token_list) if draft_token_list else 1.0
                
                # 保存corrected tokens
                all_corrected_speech.extend(corrected_tokens)
                
                # ============================================================
                # STEP 5: 收集vocoder输出
                # vocoder已经在并行执行，现在收集结果
                # 注意: vocoder输出的是基于draft的音频
                # 理想情况下，如果有rejection，应该用corrected重新生成
                # 这里为了效率，我们接受draft的输出（speculative的精神是大部分会accept）
                # ============================================================
                for out in vocoder_results:
                    speech_list.append(out['tts_speech'])
                    if sample_rate is None:
                        sample_rate = self.cosy_vocoder.sample_rate
                
                # ============================================================
                # STEP 6: 状态重置处理
                # ============================================================
                current_text = self.tokenizer.decode(
                    full_generated_text_idx.squeeze(0), skip_special_tokens=True
                )
                if current_text and current_text[-1] in ".!?":
                    if self.model.punct_count_specdiff - self.model.last_punct_reset_specdiff >= self.config.reset_interval:
                        self.model.speech_generator.reset_streaming_cache()
                        self.model.last_punct_reset_specdiff = self.model.punct_count_specdiff
                        self.model.units_preds = []
                        self.model.cur_hidden_states = []
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VocalNet infer')
    parser.add_argument('--query_audio', type = str, default="./omni_speech/infer/alpaca_cn_query.wav")
    parser.add_argument('--s2s', action='store_true', default=True)
    parser.add_argument('--save_dir', default="./generated_audio")
    args = parser.parse_args()

    audio_messages = [{"role": "user", "content": "<speech>","path": args.query_audio}]
    print("Initialized vocalnet")
    vocalnet = VocalNetModelStream(VOCALNET_MODEL, s2s=args.s2s)
    vocalnet.__initilize__()
    vocalnet.set_audio_dir(args.save_dir)

    response = vocalnet.__call__(audio_messages)
    print(response)


