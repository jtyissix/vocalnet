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


class CosyvoiceVocoder:
    def __init__(self, frontend: SpeechTokenizer, model: CosyVoice2Model, sample_rate: int = 24000):
        frontend = frontend
        self.model = model
        self.sample_rate = sample_rate

    def inference_zero_shot(self, speech_token, prompt_token: torch.Tensor, prompt_feat: torch.Tensor, embedding: torch.Tensor, stream=False, speed=1.0):
        if speech_token.dim() == 3 and speech_token.size(1) == 1:
            speech_token = speech_token.squeeze(1)
        elif speech_token.dim() != 2:
            raise ValueError(f"The dimension of speech_token should be 2D, but got {speech_token.dim()}D")

        if prompt_token.dim() == 3 and prompt_token.size(1) == 1:
            prompt_token = prompt_token.squeeze(1)
        elif prompt_token.dim() != 2:
            raise ValueError(f"The dimension of prompt_token should be 2D, but got {prompt_token.dim()}D")

        if prompt_feat.dim() != 3:
            raise ValueError(f"The dimension of prompt_feat should be 3D, but got {prompt_feat.dim()}D")

        model_input = {
            'speech_tokens': speech_token,
            'flow_embedding': embedding,
            'prompt_token': prompt_token,
            'prompt_feat': prompt_feat,
            'stream': stream,
            'speed': speed
        }

        for model_output in self.model.tts_direct(**model_input):
            yield model_output


class SpeculativeDecodingStats:
    """统计投机解码的指标"""
    def __init__(self):
        self.num_audio_tokens_list = []  # 每次的audio token数量
        self.expected_accepted_list = []  # 每次的期望接受数
        self.acceptance_rate_list = []    # 每次的接受率
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
        print("Audio Token Acceptance Statistics")
        print("="*70)
        print(f"Number of samples: {summary['num_samples']}")
        print(f"Average Acceptance Rate: {summary['avg_acceptance_rate']*100:.2f}% (±{summary.get('std_acceptance_rate', 0)*100:.2f}%)")
        print(f"Average Expected Accepted: {summary['avg_expected_accepted']:.2f} (±{summary.get('std_expected_accepted', 0):.2f})")
        print(f"Average Audio Tokens: {summary['avg_num_audio_tokens']:.1f} (±{summary.get('std_num_audio_tokens', 0):.1f})")
        print(f"Average Draft Time: {summary['avg_draft_time']*1000:.2f}ms")
        print(f"Average Verify Time: {summary['avg_verify_time']*1000:.2f}ms")
        print("="*70 + "\n")


class VocalNetSpeculativeDecoding:
    def __init__(self, draft_model_path: str, target_model_path: str, 
                 vocoder_path: str = COSYVOICE_MODEL, 
                 k: int = 5,  # 每次draft生成的token数
                 s2s: bool = True, **kwargs):
        self.s2s = s2s
        self.draft_model_path = draft_model_path
        self.target_model_path = target_model_path
        self.vocoder_path = vocoder_path
        self.k = k  # draft model每次生成的token数
        
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
            print("Loading draft model (1B)...")
            self.draft_tokenizer, self.draft_model, _ = load_pretrained_model(
                self.draft_model_path, s2s=self.s2s
            )
            
            # 为draft model分配GPU
            draft_device_map = infer_auto_device_map(self.draft_model, max_memory={
                0: "45GiB", 6: "45GiB" # draft model用较少的显存
            })
            self.draft_model = dispatch_model(self.draft_model, device_map=draft_device_map)
            
            print("Loading target model (8B)...")
            self.target_tokenizer, self.target_model, _ = load_pretrained_model(
                self.target_model_path, s2s=self.s2s
            )
            
            # 为target model分配GPU
            target_device_map = infer_auto_device_map(self.target_model, max_memory={
                0: "45GiB", 6: "45GiB"#, 6: "45GiB"
            })
            self.target_model = dispatch_model(self.target_model, device_map=target_device_map)
            
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

    def postprocess(self, speech, top_db=60, hop_length=220, win_length=440, max_val=0.8):
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > max_val:
            speech = speech / speech.abs().max() * max_val
        speech = torch.concat([speech, torch.zeros(1, int(24000 * 0.2))], dim=1)
        return speech

    def set_audio_dir(self, audio_dir):
        self.audio_dir = audio_dir

    def _draft_generate(self, input_ids, speech_tensor, speech_length, num_tokens,**kwargs):
        """使用draft model生成候选tokens"""
        start_time = time.time()
        with torch.inference_mode():
            if self.s2s:
                outputs,segment_seq,units_pred_list,units_pred,tot_logit = self.draft_model.draft_generate(
                    input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    streaming_unit_gen=False,
                    do_sample=False,  # 使用贪心解码
                    num_beams=1,
                    max_audio_tokens=num_tokens,
                    use_cache=True,
                    pad_token_id=128004,
                    infer_mtp_token_num=0,
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
        #tot_logit is draft model's logit
        draft_time = time.time() - start_time
        return outputs,segment_seq,units_pred_list,units_pred,tot_logit, draft_time
    def _verify_with_target(self, input_ids, speech_tensor, speech_length, draft_tokens,audio_unit_list):
        """
        用 target 的 forward logits 来验证 draft_tokens，返回：
        - accepted_count: 接受的前缀长度
        - fix_token: 第一个不匹配位置 target 应该输出的 token（用于纠错推进1步）
        - verify_time
        """
        start_time = time.time()

        

        with torch.inference_mode():
            test_logit=self.target_model.verify_with_draft(
            input_ids,
            speech_tensor,
            speech_length,
            draft_tokens=draft_tokens,
            #draft_seq_list,
            audio_unit_list=audio_unit_list,
            infer_mtp_token_num=0,)

        verify_time = time.time() - start_time
        return test_logit, verify_time

    

    def __call__(self, messages: list) -> dict:
        """
        使用投机解码进行推理
        """
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

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensor = speech.to(dtype=torch.float16, device='cuda', non_blocking=True)
        speech_length = speech_length.to(device='cuda', non_blocking=True)

        # 投机解码循环
        generated_tokens = []
        generated_units = [] if self.s2s else None
        #breakpoint()
        outputs,segment_seq,units_pred_list,units_pred,tot_logit, draft_time=self._draft_generate(input_ids, speech_tensor, speech_length, self.k,max_new_tokens=1024)
        if len(units_pred_list)<=2:
            raise Exception("audio unit list did not generate any useful token!")
        
        #cut sos and eos
        units_pred_list=[units_pred_list[i] for i in range(1,len(units_pred_list)-1)]
        test_logit,verify_time=self._verify_with_target(input_ids, speech_tensor, speech_length, outputs,units_pred_list)
        #breakpoint()
        assert len(test_logit)==len(tot_logit)
        acceptance_probs = torch.zeros(self.k, device='cuda')
        offset=0
        for i in range(len(test_logit)):
            assert len(test_logit[i])==len(tot_logit[i])
            for j in range(len(test_logit[i])):
                if i>0:
                    offset+=len(test_logit[i-1])
                draft_probs = F.softmax(torch.tensor(tot_logit[i][j]), dim=-1)
                target_probs = F.softmax(torch.tensor(test_logit[i][j]), dim=-1)
                #breakpoint()
                # 获取draft token的概率
                token_id = units_pred_list[i][0][j].item()
                q_x = draft_probs[token_id].item()
                p_x = target_probs[token_id].item()
                
                # 计算接受概率
                if q_x <= p_x:
                    accept_prob = 1.0
                else:
                    accept_prob = p_x / (q_x + 1e-10)
                acceptance_probs[offset+j] = accept_prob
        expected_accepted = 0.0
        cumulative_prob = 1.0
        for i in range(self.k):
            cumulative_prob *= acceptance_probs[i].item()
            expected_accepted += cumulative_prob
        self.stats.update(
            num_audio_tokens=self.k,
            expected_accepted=expected_accepted,
            draft_time=draft_time,  # 你已经记录的draft时间
            verify_time=verify_time  # 你已经记录的verify时间
        )

        # 返回结果
        return {
            'num_audio_tokens': self.k,
            'expected_accepted': expected_accepted,
            'acceptance_rate': expected_accepted / self.k if self.k > 0 else 0
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VocalNet Speculative Decoding')
    parser.add_argument('--query_audio', type=str, default="./omni_speech/infer/llama_questions_42.wav")
    parser.add_argument('--s2s', action='store_true', default=True)
    parser.add_argument('--save_dir', default="./generated_audio", required=False)
    parser.add_argument('--k', type=int, default=5, help='Number of tokens to draft per iteration')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of tokens to generate')
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 准备消息
    audio_messages = [{"role": "user", "content": "<speech>", "path": args.query_audio}]
    
    print("Initializing VocalNet with Speculative Decoding...")
    print(f"Draft Model: {DRAFT_MODEL_PATH}")
    print(f"Target Model: {TARGET_MODEL_PATH}")
    print(f"Draft tokens per iteration (k): {args.k}")
    
    # 创建投机解码模型
    vocalnet = VocalNetSpeculativeDecoding(
        draft_model_path=DRAFT_MODEL_PATH,
        target_model_path=TARGET_MODEL_PATH,
        k=args.k,
        s2s=args.s2s,
        max_new_tokens=args.max_new_tokens
    )
    
    vocalnet.__initialize__()
    vocalnet.set_audio_dir(args.save_dir)
    
    print("\nStarting inference with speculative decoding...")
    response = vocalnet(audio_messages)
    
    print("\n" + "="*50)
    print("Generated Response:")
    print("="*50)
    print(f"Text: {response['text']}")
    if 'audio' in response:
        print(f"Audio saved to: {response['audio']}")
    
    # 打印统计信息
    vocalnet.stats.print_summary()