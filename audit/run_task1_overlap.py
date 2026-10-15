import sys
import os
import json
from collections import defaultdict
import pypdfium2 as pdfium

sys.stdout.reconfigure(encoding="utf-8")

DOCS = {
    'agri_mechanization_2020': 'data/raw_sources/agriculture/en/agri_mechanization_policy_2020_en.pdf',
    'agri_seed_rules_2020': 'data/raw_sources/agriculture/en/seed_rules_2020_en.pdf',
    'disaster_sod_2019': 'data/raw_sources/disaster/en/sod_2019_en.pdf',
    'disaster_displacement_nap_2022': 'data/raw_sources/disaster/en/nap_displacement_2022_2042_en.pdf',
    'disaster_displacement_strat_2021': 'data/raw_sources/disaster/en/strategy_displacement_2021_en.pdf',
    'disaster_emcrp_rpf': 'data/raw_sources/disaster/en/rpf_emcrp_en.pdf',
    'disaster_emcrp_esmf': 'data/raw_sources/disaster/en/esmf_emcrp_en.pdf',
    'edu_nep_2010': 'data/raw_sources/education/en/national_education_policy_2010_en.pdf',
}

with open('data_v2/en_candidates.jsonl', 'r', encoding='utf-8') as f:
    sampled = [json.loads(line) for line in f]

# Group sampled passages by doc_id
by_doc = defaultdict(list)
for c in sampled:
    by_doc[c['doc_id']].append(c)

print("=" * 80)
print("TASK 1: OVERLAP ANALYSIS OF 220 SAMPLED PASSAGES")
print("=" * 80)

# Locate each passage by word offsets (page, start_word, end_word)
doc_overlap_stats = {}

for doc_id, path in DOCS.items():
    pdf = pdfium.PdfDocument(path)
    passages = by_doc.get(doc_id, [])
    
    # Map each passage to (page, start_idx, end_idx)
    located = []
    for c in passages:
        p_no = c['page'] - 1
        page_text = pdf[p_no].get_textpage().get_text_range()
        p_words = page_text.split()
        c_words = c['text'].split()
        k = len(c_words)
        match_idx = -1
        for i in range(len(p_words) - k + 1):
            if p_words[i:i+k] == c_words:
                match_idx = i
                break
        assert match_idx != -1, f"Could not locate {c['candidate_id']} on page {c['page']}"
        located.append({
            'candidate_id': c['candidate_id'],
            'page': c['page'],
            'start': match_idx,
            'end': match_idx + k,
            'words': k
        })
        
    # Check pairwise overlaps
    num_passages = len(located)
    overlapping_pairs = 0
    overlapping_pair_details = []
    for i in range(num_passages):
        for j in range(i + 1, num_passages):
            pA = located[i]
            pB = located[j]
            if pA['page'] == pB['page']:
                # overlap condition on word spans
                ov_start = max(pA['start'], pB['start'])
                ov_end = min(pA['end'], pB['end'])
                if ov_start < ov_end:
                    overlapping_pairs += 1
                    overlapping_pair_details.append((pA['candidate_id'], pB['candidate_id'], pA['page'], ov_end - ov_start))
                    
    # Track word coverage per page
    # page -> array of coverage counts for each word position
    page_word_coverage = defaultdict(lambda: defaultdict(int))
    total_sampled_words = sum(p['words'] for p in located)
    
    for p in located:
        for w_pos in range(p['start'], p['end']):
            page_word_coverage[p['page']][w_pos] += 1
            
    # Calculate words covered by more than one passage
    # Definition 1: Sum of tokens across sampled passages that fall on multi-covered positions
    multicovered_token_instances = 0
    for p in located:
        for w_pos in range(p['start'], p['end']):
            if page_word_coverage[p['page']][w_pos] > 1:
                multicovered_token_instances += 1
                
    frac_multicovered_tokens = (multicovered_token_instances / total_sampled_words) if total_sampled_words > 0 else 0.0
    
    # Definition 2: Unique word positions covered by >= 2 passages / unique word positions covered
    all_covered_unique_positions = sum(len(positions) for positions in page_word_coverage.values())
    multi_covered_unique_positions = sum(sum(1 for cnt in positions.values() if cnt > 1) for positions in page_word_coverage.values())
    frac_unique_multi = (multi_covered_unique_positions / all_covered_unique_positions) if all_covered_unique_positions > 0 else 0.0
    
    doc_overlap_stats[doc_id] = {
        'sampled_count': num_passages,
        'total_words': total_sampled_words,
        'overlapping_pairs': overlapping_pairs,
        'multicovered_token_instances': multicovered_token_instances,
        'frac_multicovered_tokens': frac_multicovered_tokens,
        'unique_covered': all_covered_unique_positions,
        'unique_multi': multi_covered_unique_positions,
        'frac_unique_multi': frac_unique_multi,
        'pair_details': overlapping_pair_details
    }
    
    print(f"\nDoc ID: {doc_id}")
    print(f"  Sampled passages: {num_passages}")
    print(f"  Total sampled words: {total_sampled_words:,}")
    print(f"  Overlapping passage pairs: {overlapping_pairs}")
    print(f"  Sampled words covered by >1 passage: {multicovered_token_instances:,} / {total_sampled_words:,} ({frac_multicovered_tokens:.4f} or {frac_multicovered_tokens*100:.2f}%)")
    print(f"  Unique page word positions covered: {all_covered_unique_positions:,} (multi-covered: {multi_covered_unique_positions:,}, {frac_unique_multi*100:.2f}%)")
    if overlapping_pair_details:
        print(f"  First 3 overlapping pairs:")
        for id1, id2, pg, ov_len in overlapping_pair_details[:3]:
            print(f"    - {id1} & {id2} on page {pg}: {ov_len} words overlap")

print("\n" + "=" * 80)
print("OVERLAP SUMMARY TABLE")
print("=" * 80)
print(f"{'Doc ID':<32} {'Sampled':<8} {'Overlapping Pairs':<18} {'Total Words':<12} {'Multi-covered Words':<20} {'Overlap Fraction':<16}")
print("-" * 110)
for doc_id, s in doc_overlap_stats.items():
    print(f"{doc_id:<32} {s['sampled_count']:<8} {s['overlapping_pairs']:<18} {s['total_words']:<12} {s['multicovered_token_instances']:<20} {s['frac_multicovered_tokens']*100:.2f}%")
