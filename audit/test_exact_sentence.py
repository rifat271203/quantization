import sys
import re

sys.stdout.reconfigure(encoding="utf-8")

text = (
    "Since August 25, 2017, extreme violence in Rakhine State, Myanmar, has driven an estimated 727,0001 "
    "people from the Rohingya community across the border into the Cox’s Bazar District of Bangladesh. "
    "This exodus brings the total number of Displaced Rohingya Population (DRP) in the district to about 919,0002 "
    "in what is one of the fastest developing forced displacement crises in the world. 85% of the DRP are living in "
    "collective sites, 13% in collective sites with host communities, and 2% in dispersed sites in host communities. "
    "In Ukhia and Teknaf, the two Upazilas where most of the DRP have settled, they outnumber the host community by over a factor of three."
)

def split_sentences_exact(t):
    masked = list(t)
    for m in re.finditer(r'\b(e\.g\.|i\.e\.|viz\.|etc\.|al\.|Mr\.|Mrs\.|Dr\.|No\.|Fig\.|Vol\.|pp\.|sec\.)', t, re.I):
        for i in range(m.start(), m.end()):
            if masked[i] == '.':
                masked[i] = '@'
    for m in re.finditer(r'\b\d+\.\d+\b', t):
        for i in range(m.start(), m.end()):
            if masked[i] == '.':
                masked[i] = '@'
    masked_str = ''.join(masked)
    splits = [m.end() for m in re.finditer(r'[.!?]+[\'"’”)]?(?:\s+|$)', masked_str)]
    if not splits or splits[-1] < len(t):
        splits.append(len(t))
    sentences = []
    prev = 0
    for idx in splits:
        s = t[prev:idx].strip()
        if s:
            sentences.append(s)
        prev = idx
    return sentences

sents = split_sentences_exact(text)
print(f"Total sentences: {len(sents)}")
for i, s in enumerate(sents):
    print(f"S{i+1} ({len(s.split())}w): {s}")

cur = []
for s in sents:
    cur.append(s)
    w = sum(len(x.split()) for x in cur)
    if 80 <= w <= 180:
        print(f"\nPassage ({w}w, {len(cur)}s):")
        print(' '.join(cur))
