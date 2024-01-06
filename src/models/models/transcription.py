from typing import List, Optional, Tuple, Union
from types import MethodType

from transformers import (AutoModelForSpeechSeq2Seq, AutoTokenizer, 
                            WhisperFeatureExtractor, AutoFeatureExtractor, WhisperModel)
from transformers.modeling_outputs import BaseModelOutput

from transformers.audio_utils import spectrogram, window_function
import torch
from torch import nn

import numpy as np

from src.models.models.bases import MultimodalModel


def _patch_whisper_feature_extractor(feature_extractor: WhisperFeatureExtractor):
    
    def _np_extract_fbank_features(self: WhisperFeatureExtractor, waveform: np.array) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
        implementation with 1e-5 tolerance.
        """
        log_spec = spectrogram(
            waveform,
            window_function(self.n_fft, "hann"),
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            mel_filters=self.mel_filters,
            log_mel="log10",
        )
        # This is the old line, which won't work when there's only one timestep
        # log_spec = log_spec[:, :-1]
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
    
    feature_extractor._np_extract_fbank_features = MethodType(_np_extract_fbank_features, feature_extractor)


def _patch_whisper_encoder(model: WhisperModel):
     
     def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight[: inputs_embeds.size(1)]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
     
     model.encoder.forward = MethodType(forward, model.encoder)

class HFTranscriptionModel(MultimodalModel):
    def __init__(self, hf_name_or_path : str,
                 **kwargs)  -> None:
        
        MultimodalModel.__init__(self, **kwargs)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(hf_name_or_path)
        self.model.to(self.device)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(hf_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name_or_path)
        self.objective = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        if isinstance(self.model.model, WhisperModel):
            _patch_whisper_encoder(self.model.model)

        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            _patch_whisper_feature_extractor(self.feature_extractor)

    def forward(self, ts: List[List], context: torch.Tensor, label: torch.Tensor,
                **kwargs) -> torch.Tensor:
        
        x = self.feature_extractor(ts, return_tensors='pt', padding=True, sampling_rate=16000,)
        x = x['input_features'].to(self.device)
        x = x[:,:self.model.config.max_target_positions:]
        
        context_ids = self.tokenizer(context, return_tensors="pt", padding="longest", truncation=True, max_length=self.model.config.max_target_positions)["input_ids"].to(self.device) 
        label_ids = self.tokenizer(label, return_tensors="pt", padding="longest", truncation=True, max_length=self.model.config.max_target_positions)["input_ids"].to(self.device)
        

        max_len = max(context_ids.shape[1], label_ids.shape[1])
        context_ids = torch.nn.functional.pad(context_ids, (0, max_len - context_ids.shape[1]), value=self.tokenizer.pad_token_id)
        label_ids = torch.nn.functional.pad(label_ids, (0, max_len - label_ids.shape[1]), value=self.tokenizer.pad_token_id)
        
        logits = self.model(x,decoder_input_ids = context_ids).logits
        loss = self.objective(logits.view(-1, logits.size(-1)), label_ids.view(-1))


        return loss, logits
    
    def generate(self, ts: List[List], context: torch.Tensor,
                label: torch.Tensor, **kwargs) -> torch.Tensor:
        
        x = self.feature_extractor(ts, return_tensors='pt', padding=True, sampling_rate=16000)
        x = x['input_features'].to(self.device)
        context_ids = self.tokenizer(context, return_tensors="pt", padding="longest", truncation=True)["input_ids"].to(self.device) 
        output = self.model.generate(x, decoder_input_ids=context_ids)
        
        output_strs = [self.tokenizer.decode(o, skip_special_tokens=True) for o in output]
        context_strs = [self.tokenizer.decode(c, skip_special_tokens=True) for c in context_ids]
        # label_strs = [self.tokenizer.decode(l, skip_special_tokens=True) for l in label]

        res_gen = zip(context_strs, output_strs, label, ts)
        return [{"context": c, "result": r, "label": l, "ts" : list(t)} for c, r, l, t in res_gen]
    
