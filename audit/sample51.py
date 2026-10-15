import csv, random, sys
sys.stdout.reconfigure(encoding='utf-8')
rows = list(csv.DictReader(open('data_v2/screening_v2.csv', encoding='utf-8-sig')))
random.seed(7)
done = set()
for d in ['agriculture', 'disaster', 'education']:
    pool = [r for r in rows if r['domain'] == d]
    done |= {r['scr_id'] for r in random.sample(pool, min(17, len(pool)))}
rest = [r for r in rows if r['scr_id'] not in done]
half = (len(rest) + 1) // 2
for name, part in [('rest_a.txt', rest[:half]), ('rest_b.txt', rest[half:])]:
    with open('data_v2/' + name, 'w', encoding='utf-8') as f:
        for r in part:
            f.write(f"### {r['scr_id']} | {r['doc_id']} | p.{r['page']} | {r['words']}w\n")
            f.write(f"BEFORE: {r['context_before'][:150]}\n")
            f.write(f"TEXT: {r['text']}\n\n")
    print(name, len(part))