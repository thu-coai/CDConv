import json
import random

random.seed(13)

lines = open('../raw_data/cdconv.txt').readlines()
random.shuffle(lines)
dev_size = test_size = int(0.2 * len(lines))
train, dev, test = lines[dev_size + test_size:], lines[:dev_size], lines[dev_size: dev_size + test_size]

for name, data in zip(['train', 'dev', 'test'], [train, dev, test]):
    # 2 class
    with open(f'./2class_{name}.tsv', 'w') as f:
        for d in data:
            d = json.loads(d)
            row = [d['u1'], d['b1'], d['u2'], d['b2'], str(int(d['label'] != 0))]
            row = [e.replace('\t', '') for e in row]
            f.write('\t'.join(row) + '\n')

    # 4 class
    with open(f'./4class_{name}.tsv', 'w') as f:
        for d in data:
            d = json.loads(d)
            row = [d['u1'], d['b1'], d['u2'], d['b2'], str(d['label'])]
            row = [e.replace('\t', '') for e in row]
            f.write('\t'.join(row) + '\n')

