from transformers import T5ForConditionalGeneration
import copy
import torch
import torch.nn as nn


class CopyGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.prob_proj = nn.Linear(config.d_model * 2, 1)

    def forward(self, src, decode_output, decode_attn, memory, gen_logits):
        decode_attn = torch.mean(decode_attn, dim=1)
        batch_size, steps, seq = decode_attn.size()
        src = src.unsqueeze(1).repeat([1, steps, 1])
        # vocab
        copy_logits = torch.zeros_like(gen_logits)
        context = torch.matmul(decode_attn, memory)
        copy_logits = copy_logits.scatter_add(2, src, decode_attn)
        prob = self.prob_proj(torch.cat([context, decode_output], -1)).sigmoid()

        gen_logits = prob * gen_logits.softmax(-1)
        copy_logits = (1 - prob) * copy_logits.softmax(-1)
        final_logits = gen_logits + copy_logits
        return final_logits


class T5Copy(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.generator = CopyGenerator(config)

    def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids, model_kwargs, model_input_name=None):
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
            }
            new_kwargs = copy.deepcopy(encoder_kwargs)
            new_kwargs.pop('src')
            model_kwargs["encoder_outputs"] = encoder(input_ids, return_dict=True, **new_kwargs)
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        res = super().prepare_inputs_for_generation(input_ids=input_ids,
                                                    past=past,
                                                    attention_mask=attention_mask,
                                                    head_mask=head_mask,
                                                    decoder_head_mask=decoder_head_mask,
                                                    cross_attn_head_mask=cross_attn_head_mask,
                                                    use_cache=use_cache,
                                                    encoder_outputs=encoder_outputs,
                                                    **kwargs)
        res['src'] = kwargs['src']
        return res

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            src=None
    ):
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  decoder_input_ids=decoder_input_ids,
                                  decoder_attention_mask=decoder_attention_mask,
                                  encoder_outputs=encoder_outputs,
                                  past_key_values=past_key_values,
                                  inputs_embeds=inputs_embeds,
                                  decoder_inputs_embeds=decoder_inputs_embeds,
                                  labels=labels,
                                  use_cache=use_cache,
                                  output_attentions=True,
                                  output_hidden_states=True,
                                  return_dict=True)

        memory = outputs.encoder_last_hidden_state
        decode_attn = outputs.cross_attentions[-1]
        decode_output = outputs.decoder_hidden_states[-1]
        gen_logits = outputs.logits
        if self.training:
            prob = self.generator(input_ids, decode_output, decode_attn, memory, gen_logits)
        else:
            if src is not None:
                prob = self.generator(src, decode_output, decode_attn, memory, gen_logits)
            else:
                prob = self.generator(input_ids, decode_output, decode_attn, memory, gen_logits)
        outputs.logits = prob
        return outputs
