"""E031 streaming exact BF16-to-FP32 loader; user-authorized 512MiB commit reserve."""
import gc
import json
import math
import struct
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForCausalLM,GenerationConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from e012.common import Pause
from e014.memory import commit_available

def load(path,progress=None):
    path=Path(path);cfg=AutoConfig.from_pretrained(path,local_files_only=True)
    if cfg.model_type!='qwen3':raise ValueError('Qwen3 only')
    cfg._attn_implementation='eager'
    # Meta parameters allocate no weight storage and perform no random CPU init.
    with torch.device('meta'):
        model=AutoModelForCausalLM.from_config(cfg,attn_implementation='eager')
    expected=dict(model.named_parameters(remove_duplicate=False))
    checkpoint=path/'model.safetensors'
    with checkpoint.open('rb') as stream:
        header_bytes=struct.unpack('<Q',stream.read(8))[0]
        if not 0<header_bytes<16*2**20:raise ValueError('Invalid safetensors header')
        header=json.loads(stream.read(header_bytes))
    keys=sorted(k for k in header if k!='__metadata__')
    if set(expected)-set(keys) != ({'lm_head.weight'} if cfg.tie_word_embeddings else set()):
        raise RuntimeError('Unexpected missing checkpoint keys')
    if set(keys)-set(expected):raise RuntimeError('Unexpected checkpoint keys')
    previous_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.no_grad(), checkpoint.open('rb',buffering=0) as stream:
            for i,key in enumerate(keys):
                spec=header[key];shape=tuple(spec['shape']);start,end=spec['data_offsets']
                dtype={'BF16':torch.bfloat16,'F32':torch.float32}.get(spec['dtype'])
                if dtype is None or shape!=tuple(expected[key].shape):raise RuntimeError('Invalid tensor '+key)
                count=math.prod(shape);itemsize=2 if spec['dtype']=='BF16' else 4
                if end-start!=count*itemsize or start<0 or 8+header_bytes+end>checkpoint.stat().st_size:
                    raise ValueError('Invalid tensor offsets '+key)
                # No mmap / copy-on-write mapping: one writable byte buffer,
                # then independent FP32 storage. Guard the process commit limit.
                if commit_available()<(end-start)+count*4+2**29:
                    raise Pause('Insufficient commit headroom before loading '+key)
                payload=bytearray(end-start);stream.seek(8+header_bytes+start)
                if stream.readinto(payload)!=len(payload):raise IOError('Truncated tensor '+key)
                source=torch.frombuffer(payload,dtype=dtype).reshape(shape)
                value=source.to(dtype=torch.float32,device='cpu',copy=True)
                del source,payload
                module_name,param_name=key.rsplit('.',1)
                model.get_submodule(module_name)._parameters[param_name]=torch.nn.Parameter(value,requires_grad=False)
                del value
                if progress is not None:progress(i+1,len(keys),key)
    finally:
        torch.set_num_threads(previous_threads)
    # Meta context also created rotary buffers: replace them from their exact
    # config formula on CPU, rather than materializing uninitialized buffers.
    model.model.rotary_emb=Qwen3RotaryEmbedding(config=cfg,device='cpu')
    model.tie_weights()
    if any(p.is_meta or p.dtype!=torch.float32 for p in model.parameters()):raise RuntimeError('Unloaded/non-FP32 parameter')
    if any(b.is_meta for b in model.buffers()):raise RuntimeError('Uninitialized meta buffer')
    if cfg.tie_word_embeddings and model.lm_head.weight is not model.model.embed_tokens.weight:raise RuntimeError('Embedding tie lost')
    if (path/'generation_config.json').exists():
        model.generation_config=GenerationConfig.from_pretrained(path,local_files_only=True)
    gc.collect()
    return model.eval().requires_grad_(False)
