import json
from pathlib import Path
from PIL import Image
import torch

def encode_qwen_rows(processor,tokenizer,rows):
    from qwen_vl_utils import process_vision_info
    data=[];lengths=[];image_id=int(tokenizer.convert_tokens_to_ids('<|image_pad|>'))
    for row in rows:
        with Image.open(row['image_path']) as src:image=src.convert('RGB')
        user={'role':'user','content':[{'type':'image','image':image},{'type':'text','text':row['prompt']}]}
        answer={'role':'assistant','content':[{'type':'text','text':row['answer']}]}
        full=processor.apply_chat_template([user,answer],tokenize=False,add_generation_prompt=False)
        prefix=processor.apply_chat_template([user],tokenize=False,add_generation_prompt=True)
        images=process_vision_info([user])[0]
        item=processor(text=full,images=images,return_tensors='pt')
        pre=processor(text=prefix,images=images,return_tensors='pt')
        n=pre.input_ids.shape[1]
        if not torch.equal(item.input_ids[0,:n],pre.input_ids[0]):raise ValueError('Assistant prefix mismatch')
        labels=item.input_ids.clone();labels[:,:n]=-100
        item['labels']=labels
        data.append(dict(item));lengths.append(item.input_ids.shape[1]);image.close()
    longest=max(lengths);pad=tokenizer.pad_token_id
    if pad is None:pad=tokenizer.eos_token_id
    for item,n in zip(data,lengths):
        for key,value in [('input_ids',pad),('attention_mask',0),('labels',-100)]:
            item[key]=torch.nn.functional.pad(item[key],(0,longest-n),value=value)
        item['use_cache']=False
    vision=torch.cat([d['input_ids'].eq(image_id) for d in data]);caption=torch.cat([d['labels'].ne(-100) for d in data])
    if not vision.any(1).all() or not caption.any(1).all():raise ValueError('Missing vision or answer mask')
    print('[128 calibration] sequence length',longest,'valid lengths',min(lengths),max(lengths),flush=True)
    return rows,data,vision,caption

def encode_internvl_rows(processor,tokenizer,rows):
    data=[];lengths=[];image_id=int(tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>'))
    if image_id!=151671:raise ValueError('InternVL native visual token ID changed')
    for row in rows:
        with Image.open(row['image_path']) as src:image=src.convert('RGB')
        user={'role':'user','content':[{'type':'image'},{'type':'text','text':row['prompt']}]}
        answer={'role':'assistant','content':[{'type':'text','text':row['answer']}]}
        full=processor.apply_chat_template([user,answer],tokenize=False,add_generation_prompt=False)
        prefix=processor.apply_chat_template([user],tokenize=False,add_generation_prompt=True)
        item=processor(text=[full],images=[image],return_tensors='pt')
        pre=processor(text=[prefix],images=[image],return_tensors='pt')
        n=pre.input_ids.shape[1]
        if not torch.equal(item.input_ids[0,:n],pre.input_ids[0]):raise ValueError('InternVL assistant prefix mismatch')
        labels=item.input_ids.clone();labels[:,:n]=-100;item['labels']=labels
        data.append(dict(item));lengths.append(item.input_ids.shape[1]);image.close()
    longest=max(lengths);pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for item,n in zip(data,lengths):
        for key,value in [('input_ids',pad),('attention_mask',0),('labels',-100)]:item[key]=torch.nn.functional.pad(item[key],(0,longest-n),value=value)
        item['use_cache']=False
    vis=torch.cat([d['input_ids'].eq(image_id) for d in data]);cap=torch.cat([d['labels'].ne(-100) for d in data])
    if not vis.any(1).all() or not cap.any(1).all():raise ValueError('Missing native visual/answer tokens')
    print('InternVL native encoding',len(rows),'samples; padded tokens',longest,'visual counts',sorted(set(vis.sum(1).tolist())),flush=True)
    return rows,data,vis,cap

def calibration_records(processor, name, data_root, manifest):
    rows = [json.loads(line) for line in Path(manifest).read_text().split('\n') if line.strip()]
    if len(rows) != 128 or len({r['question_id'] for r in rows}) != 128:
        raise ValueError('Exactly 128 unique calibration records required')
    rows = [dict(r, image_path=str(Path(data_root) / r['source_image'])) for r in rows]
    encoder = encode_internvl_rows if name == 'internvl35_8b' else encode_qwen_rows
    return encoder(processor, processor.tokenizer, rows)
