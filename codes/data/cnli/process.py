import json

def _norm(x):
    return x.replace('\t', ' ')

keys = ['train', 'dev']
for key in keys:
    writer = open(f'./{key}.tsv', 'w')
    data = open(f'../raw_data/CMNLI/{key}.json')
    for d in data:
        d = json.loads(d)
        sen1, sen2 = d['sentence1'], d['sentence2']
        sen1 = _norm(sen1)
        sen2 = _norm(sen2)
        label = 1 if d['label'] == 'contradiction' else 0
        writer.write(f'{sen1}\t{sen2}\t{label}\n')
        
    data = open(f'../raw_data/OCNLI/{key}.json')
    for d in data:
        d = json.loads(d)
        sen1, sen2 = d['sentence1'], d['sentence2']
        sen1 = _norm(sen1)
        sen2 = _norm(sen2)
        label = 1 if d['label'] == 'contradiction' else 0
        writer.write(f'{sen1}\t{sen2}\t{label}\n')
    writer.close()
