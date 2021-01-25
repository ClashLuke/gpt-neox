from deepspeed.runtime.pipe import schedule
import torch
from .datasets import SinglePromptDataset
import torch.nn.functional as F
import torch.distributed as dist
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# nucleus

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def inference_batch(self, prompt, tokenizer, seq_len=100, temperature = 1., filter_logits_fn = top_k, filter_thres = 0.9):
        self.module.eval()
        self.total_loss = None
        # Use the provided data iterator
        train_iterator = self.data_iterator
        train_loss_fn = self.module.loss_fn
        self.module.loss_fn = None
        
        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)

        out = torch.squeeze(tokenizer.encode(prompt, return_tensors='pt').long())
        t = out.shape

        with torch.no_grad():
            for _ in range(seq_len):
                print('----', _, '----')
                x = out[-self.module.seq_len:]
                dataset = SinglePromptDataset(x, tokenizer, 'with_labels')
                self._build_data_iter(dataset)
                self._exec_schedule(sched)
                if self.is_last_stage():
                    outputs = self.pipe_buffers['outputs'] # outputs = tuple([partitioned_tensor_metadata, partitioned_tensor_data, ? outputs[1] ?])
                    logits = outputs[1][:, :,-1]
                    filtered_logits = filter_logits_fn(logits, thres = filter_thres)
                    probs = F.softmax(filtered_logits / temperature, dim=-2)
                    sample = torch.multinomial(probs, 1)
                    out = torch.cat((out, sample), dim=-1)
                    out = out[0, :] # needs to be this shape for now until we figure out how to do input correctly...
                    if tokenizer.eos_token is not None and (sample == tokenizer.eos_token).all():
                        break

        out = out[:, t:]
        # Restore the training iterator and loss fn
        self.set_dataiterator(train_iterator)
        self.module.loss_fn = train_loss_fn
        return out if self.is_last_stage() else None