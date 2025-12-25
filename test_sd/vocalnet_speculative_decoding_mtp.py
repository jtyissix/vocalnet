import sys
import os
from accelerate import dispatch_model, infer_auto_device_map
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "PCI_BUS_ID"
for p in sys.path:
    print(p)
sys.path.insert(0, '/home/jiangtianyuan/resource/voice/vocalnet/')
sys.path.append('/home/jiangtianyuan/resource/voice/vocalnet/')

print('hello')
import torch
import torch.nn.functional as F
from typing import Tuple, Callable, List, Dict
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
import argparse
from tqdm import tqdm
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

COSYVOICE_MODEL="/home/jiangtianyuan/resource/voice/vocalnet/cosyvoice/model/"
DRAFT_MODEL_PATH = "/home/jiangtianyuan/resource/voice/vocalnet/checkpoints/1B/"
TARGET_MODEL_PATH = "/home/jiangtianyuan/resource/voice/vocalnet/checkpoints/8B/"
PROMPT_SPEECH="./omni_speech/infer/cn_prompt.wav"

try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    use_ttsfrd = False


# 模型路径
DRAFT_MODEL_PATH = "/home/jiangtianyuan/resource/voice/vocalnet/checkpoints/1B/"
TARGET_MODEL_PATH = "/home/jiangtianyuan/resource/voice/vocalnet/checkpoints/8B/"
COSYVOICE_MODEL = "/home/jiangtianyuan/resource/voice/vocalnet/cosyvoice/model/"


class SpeculativeDecodingStats:
    """统计投机解码的指标"""
    def __init__(self):
        self.num_audio_tokens_list = []
        self.expected_accepted_list = []
        self.acceptance_rate_list = []
        self.draft_times = []
        self.verify_times = []
        
    def update(self, num_audio_tokens: int, expected_accepted: float, draft_time: float, verify_time: float):
        """更新统计信息"""
        self.num_audio_tokens_list.append(num_audio_tokens)
        self.expected_accepted_list.append(expected_accepted)
        acceptance_rate = expected_accepted / num_audio_tokens if num_audio_tokens > 0 else 0
        self.acceptance_rate_list.append(acceptance_rate)
        self.draft_times.append(draft_time)
        self.verify_times.append(verify_time)
    
    def get_summary(self) -> Dict:
        """获取统计摘要"""
        if len(self.acceptance_rate_list) == 0:
            return {
                'num_samples': 0,
                'avg_acceptance_rate': 0,
                'avg_expected_accepted': 0,
                'avg_num_audio_tokens': 0,
                'avg_draft_time': 0,
                'avg_verify_time': 0
            }
        
        return {
            'num_samples': len(self.acceptance_rate_list),
            'avg_acceptance_rate': np.mean(self.acceptance_rate_list),
            'std_acceptance_rate': np.std(self.acceptance_rate_list),
            'avg_expected_accepted': np.mean(self.expected_accepted_list),
            'std_expected_accepted': np.std(self.expected_accepted_list),
            'avg_num_audio_tokens': np.mean(self.num_audio_tokens_list),
            'std_num_audio_tokens': np.std(self.num_audio_tokens_list),
            'avg_draft_time': np.mean(self.draft_times),
            'avg_verify_time': np.mean(self.verify_times),
        }
    
    def print_summary(self):
        """打印统计摘要"""
        summary = self.get_summary()
        print("\n" + "="*70)
        print("Audio Token Acceptance Statistics (with MTP)")
        print("="*70)
        print(f"Number of samples: {summary['num_samples']}")
        print(f"Average Acceptance Rate: {summary['avg_acceptance_rate']*100:.2f}% (±{summary.get('std_acceptance_rate', 0)*100:.2f}%)")
        print(f"Average Expected Accepted: {summary['avg_expected_accepted']:.2f} (±{summary.get('std_expected_accepted', 0):.2f})")
        print(f"Average Audio Tokens: {summary['avg_num_audio_tokens']:.1f} (±{summary.get('std_num_audio_tokens', 0):.1f})")
        print(f"Average Draft Time: {summary['avg_draft_time']*1000:.2f}ms")
        print(f"Average Verify Time: {summary['avg_verify_time']*1000:.2f}ms")
        print("="*70 + "\n")


class VocalNetSpeculativeDecodingMTP:
    def __init__(self, draft_model_path: str, target_model_path: str, 
                 vocoder_path: str = COSYVOICE_MODEL, 
                 k: int = 5,
                 mtp_num: int = 3,  # MTP层数
                 s2s: bool = True, **kwargs):
        self.s2s = s2s
        self.draft_model_path = draft_model_path
        self.target_model_path = target_model_path
        self.vocoder_path = vocoder_path
        self.k = k  # draft model每次生成的text token数
        self.mtp_num = mtp_num  # MTP层数
        
        self.temperature = kwargs.get('temperature', 0)
        self.num_beams = kwargs.get('num_beams', 1)
        self.max_new_tokens = kwargs.get('max_new_tokens', 512)
        self.top_p = kwargs.get('top_p', 0.1)
        
        self.audio_dir = None
        self.empty = True
        self.stats = SpeculativeDecodingStats()

    def __initialize__(self):
        if self.empty:
            self.empty = False
            print("Loading draft model (1B) with MTP...")
            self.draft_tokenizer, self.draft_model, _ = load_pretrained_model(
                self.draft_model_path, s2s=self.s2s,device='cuda:0'
            )
            
            draft_device_map = infer_auto_device_map(self.draft_model, max_memory={
                0: "45GiB"
            })
            self.draft_model = dispatch_model(self.draft_model, device_map=draft_device_map)
            
            print("Loading target model (8B) with MTP...")
            self.target_tokenizer, self.target_model, _ = load_pretrained_model(
                self.target_model_path, s2s=self.s2s,device='cuda:0'
            )
            
            target_device_map = infer_auto_device_map(self.target_model, max_memory={
                0: "45GiB"
            })
            self.target_model = dispatch_model(self.target_model, device_map=target_device_map,)
            
            self.draft_model.eval()
            self.target_model.eval()
            
            print(f"Models loaded successfully. MTP layers: {self.mtp_num}")

    def set_audio_dir(self, audio_dir):
        self.audio_dir = audio_dir

    def _draft_generate(self, input_ids, speech_tensor, speech_length, num_tokens, **kwargs):
        """使用draft model生成候选tokens (使用MTP头)"""
        start_time = time.time()
        with torch.inference_mode():
            if self.s2s:
                outputs, segment_seq, units_pred_list, units_pred, tot_logit = self.draft_model.draft_generate(
                    input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    streaming_unit_gen=False,
                    do_sample=False,
                    num_beams=1,
                    max_audio_tokens=num_tokens,
                    use_cache=True,
                    pad_token_id=128004,
                    infer_mtp_token_num=self.mtp_num,  # 使用MTP头
                    streaming=False,
                    **kwargs
                )
            else:
                outputs = self.draft_model.generate(
                    input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=num_tokens,
                    use_cache=True,
                    pad_token_id=128004,
                )
                output_ids = outputs
                output_units = None
        
        draft_time = time.time() - start_time
        return outputs, segment_seq, units_pred_list, units_pred, tot_logit, draft_time

    def _verify_with_target(self, input_ids, speech_tensor, speech_length, draft_tokens, audio_unit_list):
        """用target model验证draft tokens (使用MTP头)"""
        start_time = time.time()
        
        with torch.inference_mode():
            test_logit = self.target_model.verify_with_draft(
                input_ids,
                speech_tensor,
                speech_length,
                draft_tokens=draft_tokens,
                audio_unit_list=audio_unit_list,
                infer_mtp_token_num=self.mtp_num,  # 使用MTP头
            )

        verify_time = time.time() - start_time
        return test_logit, verify_time

    def __call__(self, messages: list) -> dict:
        """使用投机解码进行推理"""
        audio_path = messages[0]['path']
        speech = whisper.load_audio(audio_path)
        
        # 准备输入
        if self.draft_model.config.speech_encoder_type == "glm4voice":
            speech_length = torch.LongTensor([speech.shape[0]])
            speech = torch.from_numpy(speech)
            speech = torch.nn.functional.layer_norm(speech, speech.shape)
        else:
            raw_len = len(speech)
            speech = whisper.pad_or_trim(speech)
            padding_len = len(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0).unsqueeze(0)
            speech_length = round(raw_len / padding_len * 3000 + 0.5)
            speech_length = torch.LongTensor([speech_length])
        
        conversation = [{"from": "human", "value": "<speech>", "path": f"{audio_path}"}]
        
        if 'qwen' in self.target_model_path.lower():
            input_ids = preprocess_qwen_2_5_v1([conversation], self.target_tokenizer, True, 4096)['input_ids']
            input_ids = torch.cat([input_ids.squeeze(), torch.tensor([198, 151644, 77091, 198], device=input_ids.device)]).unsqueeze(0)
        else:
            input_ids = preprocess_llama_3_v1([conversation], self.target_tokenizer, True, 4096)['input_ids']
            input_ids = torch.cat([input_ids.squeeze(), torch.tensor([128006, 78191, 128007, 271], device=input_ids.device)]).unsqueeze(0)

        input_ids = input_ids.to(device='cuda:0', non_blocking=True)
        speech_tensor = speech.to(dtype=torch.float16, device='cuda:0', non_blocking=True)
        speech_length = speech_length.to(device='cuda:0', non_blocking=True)
        #breakpoint()
        # 投机解码循环
        outputs, segment_seq, units_pred_list, units_pred, tot_logit, draft_time = self._draft_generate(
            input_ids, speech_tensor, speech_length, self.k, max_new_tokens=1024
        )
        
        if len(units_pred_list) <= 2:
            raise Exception("audio unit list did not generate any useful token!")
        
        # 去除sos和eos
        units_pred_list = [units_pred_list[i] for i in range(1, len(units_pred_list)-1)]
        test_logit, verify_time = self._verify_with_target(
            input_ids, speech_tensor, speech_length, outputs, units_pred_list
        )
        #breakpoint()
        # 计算接受率
        assert len(test_logit) == len(tot_logit)
        acceptance_probs = torch.zeros(self.k, device='cuda')
        offset = 0
        
        for i in range(len(test_logit)):
            assert len(test_logit[i]) == len(tot_logit[i])
            for j in range(len(test_logit[i])):
                if i > 0:
                    offset += len(test_logit[i-1])
                
                draft_probs = F.softmax(torch.tensor(tot_logit[i][j]), dim=-1)
                target_probs = F.softmax(torch.tensor(test_logit[i][j]), dim=-1)
                
                token_id = units_pred_list[i][0][j].item()
                q_x = draft_probs[token_id].item()
                p_x = target_probs[token_id].item()
                
                # 计算接受概率
                if q_x <= p_x:
                    accept_prob = 1.0
                else:
                    accept_prob = p_x / (q_x + 1e-10)
                acceptance_probs[offset+j] = accept_prob
        
        # 计算期望接受数
        expected_accepted = 0.0
        cumulative_prob = 1.0
        for i in range(self.k):
            cumulative_prob *= acceptance_probs[i].item()
            expected_accepted += cumulative_prob
        
        self.stats.update(
            num_audio_tokens=self.k,
            expected_accepted=expected_accepted,
            draft_time=draft_time,
            verify_time=verify_time
        )

        return {
            'num_audio_tokens': self.k,
            'expected_accepted': expected_accepted,
            'acceptance_rate': expected_accepted / self.k if self.k > 0 else 0
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VocalNet Speculative Decoding with MTP')
    parser.add_argument('--query_audio', type=str, default="./omni_speech/infer/llama_questions_42.wav")
    parser.add_argument('--s2s', action='store_true', default=True)
    parser.add_argument('--save_dir', default="./generated_audio", required=False)
    parser.add_argument('--k', type=int, default=5, help='Number of text tokens to draft per iteration')
    parser.add_argument('--mtp_num', type=int, default=3, help='Number of MTP layers')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    audio_messages = [{"role": "user", "content": "<speech>", "path": args.query_audio}]
    
    print("Initializing VocalNet with Speculative Decoding (MTP)...")
    print(f"Draft Model: {DRAFT_MODEL_PATH}")
    print(f"Target Model: {TARGET_MODEL_PATH}")
    print(f"Draft tokens per iteration (k): {args.k}")
    print(f"MTP layers: {args.mtp_num}")
    
    vocalnet = VocalNetSpeculativeDecodingMTP(
        draft_model_path=DRAFT_MODEL_PATH,
        target_model_path=TARGET_MODEL_PATH,
        k=args.k,
        mtp_num=args.mtp_num,
        s2s=args.s2s,
        max_new_tokens=args.max_new_tokens
    )
    
    vocalnet.__initialize__()
    vocalnet.set_audio_dir(args.save_dir)
    
    print("\nStarting inference with speculative decoding (MTP)...")
    response = vocalnet(audio_messages)
    
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    print(f"Audio tokens: {response['num_audio_tokens']}")
    print(f"Expected accepted: {response['expected_accepted']:.2f}")
    print(f"Acceptance rate: {response['acceptance_rate']*100:.2f}%")
    
    vocalnet.stats.print_summary()