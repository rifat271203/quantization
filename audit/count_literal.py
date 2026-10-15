import json, re, collections
pat = re.compile(r'^(agri_mech|agri_seed|disaster_sod|disaster_nap|disaster_emcrp|emcrp|disaster_act|nctb_sci|edu_nep)_\d\d(__bn|__en)?$')
files = [
    ('eval_candidate_pool.jsonl', 'question_id'),
    ('data/train_bilingual.jsonl', 'id'),
    ('data/eval_bangla.jsonl', 'id'),
    ('data/eval_english.jsonl', 'id'),
]
for fn, key in files:
    c = collections.Counter()
    total = 0
    for l in open(fn, encoding='utf-8'):
        r = json.loads(l)
        total += 1
        if pat.match(r[key]):
            c[r['doc_id']] += 1
    print(fn, 'rows:', total, 'literal-ID rows:', sum(c.values()), dict(c))