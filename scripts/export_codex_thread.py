from __future__ import annotations
import argparse,json
from pathlib import Path

def texts(payload):
    out=[]
    for item in payload.get('content',[]):
        if item.get('type') in {'input_text','output_text','text'} and item.get('text'):
            out.append(item['text'])
    return '\n'.join(out).strip()

def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args();src=Path(a.input);dst=Path(a.output);rows=[]
    with src.open(encoding='utf-8') as f:
        for line in f:
            obj=json.loads(line);stamp=obj.get('timestamp','')
            if obj.get('type')!='response_item':continue
            payload=obj.get('payload',{});role=payload.get('role')
            if payload.get('type')!='message' or role not in {'user','assistant'}:continue
            text=texts(payload)
            if text:rows.append((stamp,role,text))
    lines=['# Codex 对话导出','',f'- 来源：`{src}`',f'- 可见消息数：{len(rows)}','']
    for stamp,role,text in rows:
        lines += [f'## {"用户" if role=="user" else "Codex"}  {stamp}','',text,'']
    dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text('\n'.join(lines),encoding='utf-8');print(dst,len(rows),dst.stat().st_size)
if __name__=='__main__':main()