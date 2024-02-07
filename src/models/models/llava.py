import copy
from typing import Dict, List, Sequence, Tuple
import os
import shutil
import warnings
import math
from matplotlib import pyplot as plt

import numpy as np
import torch
import torchvision.transforms as transforms
import transformers
import torchaudio
from llava import conversation as conversation_lib
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_TOKEN,
                             IGNORE_INDEX, IMAGE_TOKEN_INDEX,)
from llava.mm_utils import tokenizer_image_token
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.model.multimodal_encoder.builder import build_vision_tower 
from llava.conversation import conv_templates
from llava.model import *
from einops import repeat
from PIL import Image
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
import pandas as pd

from src.models.models.bases import MultimodalModel
from src.models.models.timesnet import TimesNetEncoder
from src.models.models.layers.embed import TSPerceiverResampler
from src.models.models import lagllama 


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation



def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def llava_preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def load_pretrained_llava(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map=None, device="cuda", **kwargs):
    kwargs["device_map"]=device_map
    
    if 'mpt' in model_name.lower():
        raise NotImplementedError("We only support LLaMA variants of LLaVA for now.")
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LLaVATS.from_pretrained(model_base, low_cpu_mem_usage=False, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = LLaVATS.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = LLaVATS.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=model.device)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len


class LLaVATS(LlavaLlamaForCausalLM):
    ''' Subclass the original LLaVA to make it work with time series encoders.
    '''
    def __init__(self, config, encoder_name="matplotlib", clip_path=None):
        
        if clip_path is not None:
            config.mm_vision_tower = clip_path

        # We move the intializiation of the vision_tower 
        # to here so that we can control it.
        if hasattr(config, "mm_vision_tower"):
            mm_vision_tower = config.mm_vision_tower
            del config.mm_vision_tower
        else:
            mm_vision_tower = None
    
        super().__init__(config) 
            
        if mm_vision_tower:
            config.mm_vision_tower = mm_vision_tower
            
            # I dont understand why `delay_load` is necessary, but the weights
            # are not properly initalized if we don't do it this way. I believe it has
            # something to do with the meta intialization 
            
            if encoder_name == "matplotlib":
                base_vision_tower = build_vision_tower(config, delay_load=True)
                mpl_encoder = MatplotlibEncoder(dim=base_vision_tower.config.image_size,)
                vision_tower = nn.Sequential(mpl_encoder, base_vision_tower)
                vision_tower.is_loaded = False
                vision_tower.load_model = base_vision_tower.load_model
            
            elif encoder_name == "spectrogram":
                base_vision_tower = build_vision_tower(config, delay_load=True)
                spec_encoder = SpectrogramEncoder(dim=base_vision_tower.config.image_size)
                vision_tower = nn.Sequential(spec_encoder, base_vision_tower)
                vision_tower.is_loaded = False
                vision_tower.load_model = base_vision_tower.load_model
                
            elif encoder_name == "timesnet":
                vision_tower = TimesNetEncoder(device=self.device)
                vision_tower.is_loaded = False
            
            elif encoder_name == "simple_cnn":
                vision_tower = SimpleCNNEncoder(device=self.device)
                vision_tower.is_loaded = False

            elif encoder_name == "lagllama":
                LAG_LLAMA_PATH = "/mmfs1/gscratch/bdata/mikeam/pytorch-transformer-ts/pretrained_checkpoints/epoch=199-step=20000.ckpt"
                vision_tower = LagLlamaEncoder(LAG_LLAMA_PATH,device=self.device, output_len=config.mm_hidden_size)
                vision_tower.is_loaded = False
            else:
                raise NotImplementedError(f"Unknown encoder name {encoder_name}")
            
            self.get_model().vision_tower = vision_tower
            self.get_model().mm_projector = build_vision_projector(config)


# This is the main entry point for the model
class LLaVA(MultimodalModel):
    
    def __init__(self, hf_name_or_path : str,
                       model_base : str = None,
                       model_name : str = None,
                       encoder_name : str = "matplotlib",
                       thaw_vision_encoder : bool = False,
                       clip_path : str = None,
                    **kwargs)  -> None:
            
        MultimodalModel.__init__(self, **kwargs)
        if model_name is None:
            self.model_name = hf_name_or_path.split("/")[-1]
        
        self.tokenizer, self.model, self.context_len \
            = load_pretrained_llava(hf_name_or_path, model_base, self.model_name,
                                    encoder_name=encoder_name, clip_path=clip_path)
        
        self.model.to(dtype=torch.bfloat16)
        
        self.model.get_model().requires_grad_(False)

        for p in self.model.get_model().mm_projector.parameters():
             p.requires_grad = True

        if encoder_name in ["timesnet","simple_cnn", "lagllama"] or thaw_vision_encoder:
            for p in self.model.get_model().vision_tower.parameters():
                p.requires_grad = True
        
        self.model.enable_input_require_grads()
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        
        self.save_hyperparameters()
    
        
    def prepare_inputs(self, context: List[str], label: List[str], ts: List[np.array]):
        
        ts = self._prepare_ts(ts)
        context = [f"{DEFAULT_IMAGE_TOKEN}\n" + x  for x in context]

        sources = []
        for context, label in zip(context, label):
            sources.append([
                {"from" : "human", "value": context},
                {"from" : "gpt", "value": label},
            ])

        data_dicts = llava_preprocess(sources,self.tokenizer,has_image=True)
        
        return data_dicts["input_ids"].to(self.device), data_dicts["labels"].to(self.device), ts
    
    def get_class_logprobs(input_ids: torch.LongTensor,
                        attention_mask: torch.Tensor,
                        labels_to_tokens: Dict[str, torch.Tensor],
                        model,
                        normalize_length: bool = False) -> torch.FloatTensor:
        
        overall_probs = []
        with torch.no_grad():
            for label, classname_tokens in labels_to_tokens.items():

                # TODO(jpgard): make sure num_tokens_in_classname is correct w/multi-token labels
                num_tokens_in_classname = classname_tokens.shape[1]
                classname_tokens = repeat(classname_tokens, "b s -> (repeat b) s", repeat=len(input_ids))
                _input_ids = torch.cat((input_ids, classname_tokens), dim=1)
                _attention_mask = torch.cat([attention_mask, torch.ones_like(classname_tokens).bool()],
                                            dim=1)
                logits = model(_input_ids, attention_mask=_attention_mask,).logits
                logprobs = torch.log_softmax(logits, dim=-1)

                # Extract the probabilities for only the classname tokens
                gen_probs = logprobs[
                            :, -num_tokens_in_classname - 1: -1, :
                            ]  # (B, num_tokens_in_classname, vocab_len)
                gen_probs = torch.gather(gen_probs, 2, classname_tokens[:, :, None]).squeeze(-1)

                # Aggregate probabilities over tokens in the classname
                if normalize_length:
                    class_prob = torch.mean(gen_probs, dim=1)
                else:
                    class_prob = torch.sum(gen_probs, dim=1)
                overall_probs.append(class_prob)  # (B, 1)

        return torch.vstack(overall_probs).T  # [B, num_classes]
    
    def forward(self,context: List[str], label: List[str], ts: List[np.array], compute_class_probs=None, **kwargs):
        if np.max(np.abs(ts)) > 1e10:
            return None, None
        input_ids, label_ids , ts_emb = self.prepare_inputs(context, label, ts)
        outputs = self.model(input_ids=input_ids, labels=label_ids, images = ts_emb)
        return outputs.loss, outputs.logits[..., :-1, :]

    def handle_options(self, batch):
        if "options" in batch and isinstance(batch["options"][0][0], str):
            options = batch["options"]
            results = []
            for b in options:
                new_options = [x+ self.tokenizer.eos_token for x in b]
                results.append(self.tokenizer(new_options,return_tensors="pt", padding=True)["input_ids"])
            
            batch["options"] = results

    def setup(self, stage):
        self.model.to(self.device)
        self.model.get_model().vision_tower.to(self.device)
        self.model.get_model().mm_projector.to(self.device)

    def get_log_probs_for_class(self, context: str, ts: np.array, class_name: str, **kwargs):
        context = context + "\nReply with the correct letter only."
        input_ids, class_ids, ts_emb = self.prepare_inputs([context], [class_name], [ts])
        class_token_length = class_ids.shape[1]
        logits = self.model(input_ids=input_ids, labels=class_ids, images = ts_emb).logits
        logprobs = torch.log_softmax(logits, dim=-1)
        # Hack for now
        return logprobs[-1][-2][class_ids[0,-2]]
        
    def generate(self, context: List[str], label: List[str], ts: List[np.array], **kwargs):
        _input_ids, _label_ids, ts_emb = self.prepare_inputs(context, label, ts)
        context = [f"{DEFAULT_IMAGE_TOKEN}\n" + c for c in  context]
        if "v1" in self.model_name.lower():
            conv = conv_templates["default"].copy()
        else:
            conv = conv_templates["llava_llama_2"].copy()

        conv.append_message(conv.roles[0], context[0])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        # Only generate after 
        output = self.model.generate(
                    images=ts_emb,
                    input_ids=input_ids,
                    max_new_tokens = 200,
                    do_sample=True,
                    temperature=0.5,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        output_str =  self.tokenizer.decode(output[0, input_ids.shape[1]:]).strip()
        res_gen = zip(context, [output_str], label, ts)
        return [{"context": c, "result": r, "label": l, "ts" : list(t)} for c, r, l, t in res_gen]


class SpectrogramEncoder(nn.Module):
    """ 
    This encoder converts the time series data into a spectrogram image.
    """ 

    def __init__(self, dim: int = 64,
                 pad_val: int = 0,
                 output_shape: Tuple[int] = (3, 224, 224),
                 *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.output_shape = output_shape
        self.pad_val = pad_val

    
    

    def forward(self, ts: List[torch.Tensor]) -> torch.Tensor:
        to_return = []
        for t in ts:
            nfft = min(ts.shape[1], max(256, 2 ** math.ceil(math.log2(t.shape[-1]))))
            transform = torchaudio.transforms.Spectrogram(n_fft=nfft).to(t.device)
            spectrogram = transform(t)
            spectrogram = torch.log1p(spectrogram)
            spectrogram = torch.unsqueeze(spectrogram, 0)

            # Pad the spectrogram to match the output shape
            pad = nn.ZeroPad2d((0, self.output_shape[2] - spectrogram.shape[2], 0, self.output_shape[1] - spectrogram.shape[1]))
            spectrogram = pad(spectrogram)

            # Tile the spectrogram to have three channels
            spectrogram = spectrogram.repeat(1, 3, 1, 1)

            to_return.append(spectrogram)
        return torch.cat(to_return, dim=0)
 
class MatplotlibEncoder(nn.Module):
    """ 
    This encoder plots the ts as a matplotlib figure and then encodes the figure as an image.
    """ 

    def __init__(self, dim : int = 64,
                pad_val : int = 0,
                output_shape : Tuple[int] = (3,224,224),
                use_ticks: bool = True,
                agg:SpectrogramEncoder = "seperate",
                *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.output_shape = output_shape
        self.pad_val = pad_val
        self.use_ticks = use_ticks
        self.aggregator = agg
    
    def forward_seperate(self, ts: List[torch.Tensor]) -> torch.Tensor:
        to_return = []
        for t in ts:
            fig, ax = plt.subplots()
            ax.plot(t.cpu().numpy())
            if not self.use_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            fig.canvas.draw()
            fig.tight_layout(pad=0)
            width, height = fig.get_size_inches() * fig.get_dpi()
            width, height = int(width), int(height)
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
            plt.close(fig)
            
            # Resize the image
            image = Image.fromarray(image)
            image = transforms.Resize((self.dim, self.dim))(image)
            image = transforms.ToTensor()(image)
            to_return.append(image.type(t.dtype).to(t))
        return torch.stack(to_return, dim=0)
    
    def forward_plot_together(self, ts: List[torch.Tensor]) -> torch.Tensor:
        to_return = []
        fig, ax = plt.subplots()
        for t in ts:
            ax.plot(t.cpu().numpy())
        if not self.use_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.canvas.draw()
        fig.tight_layout(pad=0)
        width, height = fig.get_size_inches() * fig.get_dpi()
        width, height = int(width), int(height)
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        plt.close(fig)
        
        # Resize the image
        image = Image.fromarray(image)
        image = transforms.Resize((self.dim, self.dim))(image)
        image = transforms.ToTensor()(image)
        to_return.append(image.type(t.dtype).to(t))
        return torch.stack(to_return, dim=0)
    
    def forward(self, ts: List[torch.Tensor]) -> torch.Tensor:
        if self.aggregator == "seperate":
            return self.forward_seperate(ts)
        elif self.aggregator == "plot_together":
            return self.forward_plot_together(ts)
        else:
            raise ValueError(f"Unknown aggregator {self.aggregator}")


class SimpleCNNEncoder(nn.Module):

    def __init__(self, d_model :int = 1024, device=None, *args, **kwargs) -> None:
        
        super().__init__()
        self.resampler = TSPerceiverResampler(dim=d_model, depth=2)
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 1, 3, padding=1)
        self.device = device


    def forward(self, ts: List[torch.Tensor]) -> torch.Tensor:
        if len(ts) > 1:
            raise ValueError(f"Expected batch size of one time series for now, got {len(ts)}")
        ts = self.resampler(ts[0])
        ts = self.conv1(ts)
        ts = self.conv2(ts)
        ts = ts.view(ts.shape[0], -1).unsqueeze(0)  
        return ts 

    
        # Is required when training as part of LLaVA for mytical reasons. 
    def load_model(self):
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            
            for param in layer.parameters():
                param.requires_grad = True



def pad_and_reshape_array(array, MAX_LENGTH):
    current_length = array.shape[0]
    padding_needed = (MAX_LENGTH - current_length % MAX_LENGTH) % MAX_LENGTH
    padded_array = np.pad(array, (0, padding_needed), 'constant', constant_values=(0,))
    new_shape = (padded_array.size // MAX_LENGTH, MAX_LENGTH)
    reshaped_array = padded_array.reshape(new_shape)
    return reshaped_array



def simple_collate_fn(batch, device=None):
    collated_batch = {}
    
    # Assuming all dictionaries have the same keys
    for key in batch[0]:
        values = [d[key] for d in batch]
        
        # Check if values are tensor-compatible (e.g., numeric types)
        if all(isinstance(v, (int, float, list, torch.Tensor, np.ndarray)) for v in values):
            try:
                # Convert values to tensor, handling cases where values are already tensors
                tensor_values = torch.tensor(values) if not isinstance(values[0], torch.Tensor) else torch.stack(values)
            except ValueError as e:
                # Handle cases where tensor conversion is not directly possible (e.g., lists of different lengths)
                print(f"Warning: Could not convert values for key '{key}' to a tensor: {e}")
                tensor_values = values  # Leave as is or handle specially
            collated_batch[key] = tensor_values.to(device,torch.float32)
        else:
            # For non-numeric types, you can decide how to handle them (e.g., leave as list)
            collated_batch[key] = values
    
    return collated_batch

class LagLlamaEncoder(nn.Module):

    def __init__(self, ckpt_path:str, device=None, output_len = 256*4, *args, **kwargs) -> None:
        # For now we're only going to support loading the model from the weights
        super().__init__()
        self.ckpt_path = ckpt_path
        self.model = lagllama.LagLlamaModel.from_checkpoint(ckpt_path, return_embeddings=True)
        self.transformation = lagllama.create_transformation()
        self.device = device
        self.final_mapping = nn.Linear(256, output_len)
        # Hardcoded from training
        self.max_context_length = 1349

    
    def forward(self, ts: List[torch.Tensor]) -> torch.Tensor:
        input_device = ts[0].device
        reshaped_ts = [pad_and_reshape_array(x.cpu().numpy(), self.max_context_length) for x in ts]
        # This gets the time series into the shape that the model expects
        windowed_ts = []
        embs = []
        for ts in reshaped_ts:
            to_transform = [{"start": pd.Period("2020-02-02", freq="1W"), "past_target": x} for x in ts]
            values = list(self.transformation(to_transform,False))
            to_model = simple_collate_fn(values,device=input_device)
            to_model["future_time_feat"] = None
            to_model["past_time_feat"] = to_model["past_time_feat"][...,:-1].transpose(1,2)
            del to_model["start"]
            embs.append(self.model(**to_model).mean(dim=0).mean(dim=0))
        return self.final_mapping(torch.stack(embs, dim=0)).unsqueeze(0)
    

    def load_model(self):
        self.model = lagllama.LagLlamaModel.from_checkpoint(self.ckpt_path, return_embeddings=True)
        
        for layer in self.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
