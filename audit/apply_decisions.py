import csv, sys
sys.stdout.reconfigure(encoding='utf-8')

N = {
 'scr_0021': 'starts mid-list, ends in another section',
 'scr_0026': 'two chapters mixed',
 'scr_0068': 'starts mid-list, ministry named only earlier',
 'scr_0044': 'indicator list only',
 'scr_0077': 'mid-list fragment',
 'scr_0048': 'consultant ToR fragment',
 'scr_0054': 'untitled bullets, topic unclear',
 'scr_0093': 'partisan political narrative',
 'scr_0092': 'rhetoric, few specific facts',
 'scr_0001': 'generic conclusion, few facts',
 'scr_0011': 'dangling ending, list is in next passage',
 'scr_0033': 'mid-list start, no heading',
 'scr_0040': 'untitled objectives list',
 'scr_0042': 'flattened table text',
 'scr_0051': 'too short after removing footnote, list tail',
 'scr_0052': 'mid-list start, cluster name missing',
 'scr_0053': 'says three activities, lists two, cut off',
 'scr_0060': 'starts mid-list at (v)',
 'scr_0075': 'indicator list only',
 'scr_0082': 'starts mid-bullet list',
 'scr_0096': 'mid-list start, outdated terms',
 'scr_0099': 'general exhortation, few facts',
 'scr_0102': 'preface, partisan narrative',
 'scr_0126': 'three unrelated items',
}
TRIM = {
 'scr_0034': 'delete last sentence "In providing government incentives, the following strategies will be followed."',
 'scr_0038': 'delete heading 14.0 and everything after; end at 13.2',
 'scr_0027': 'start at "6.1.3 Providing Loans", drop the leading bullets',
 'scr_0055': 'start at "3.3 Relevant International Laws", drop the leading list',
 'scr_0078': 'delete trailing "7. Selection Procedure and Form of Contract."',
 'scr_0128': 'stop after item 4, drop "Secondary and College Education" part',
 'scr_0018': 'delete from "10.3 Hilly and Barind areas" to the end',
 'scr_0023': 'delete first sentence "Besides, necessary assistance...", start at 6.1.4',
 'scr_0025': 'delete last sentence "In this case, the following policy will be taken."',
 'scr_0032': 'delete last sentence "The following policies will be taken..."',
 'scr_0070': 'delete first two bullets, start at "2.3 Take Initiatives..."',
 'scr_0071': 'fix the ". ." (two dots)',
 'scr_0083': 'delete first sentence "(9) Support...", start at 4.3',
 'scr_0095': 'stop after item 8, delete "Higher Education 9..."',
 'scr_0111': 'delete first sentence "Necessary steps will be taken to conserve..."',
}
FINAL60 = """
0003 0013 0012 0009 0008 0002 0007 0006 0016 0023 0031 0029 0036 0010 0030 0037 0014 0019 0015 0039
0043 0058 0083 0070 0056 0046 0080 0045 0072 0050 0059 0067 0061 0049 0066 0084 0063 0073 0057 0047
0090 0089 0100 0115 0116 0119 0098 0094 0104 0106 0111 0091 0107 0108 0124 0121 0097 0112 0117 0129
""".split()
FINAL60 = ['scr_' + x for x in FINAL60]

rows = list(csv.DictReader(open('data_v2/screening_v2.csv', encoding='utf-8-sig')))
fields = list(rows[0].keys())
ids = {r['scr_id'] for r in rows}
for k in list(N) + list(TRIM) + FINAL60:
    if k not in ids:
        print('NOT FOUND:', k)
for k in FINAL60:
    if k in N:
        print('ERROR: final60 contains rejected', k)

for r in rows:
    s = r['scr_id']
    if s in N:
        r['keep'], r['reject_reason'] = 'n', N[s]
    else:
        r['keep'], r['notes'] = 'y', TRIM.get(s, '')

def write(path, rs):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rs)

write('data_v2/screening_decided.csv', rows)
final = [r for r in rows if r['scr_id'] in set(FINAL60)]
write('data_v2/final60.csv', final)

from collections import Counter
print('rows:', len(rows), 'keep=y:', sum(r['keep'] == 'y' for r in rows), 'keep=n:', sum(r['keep'] == 'n' for r in rows))
print('final60 rows:', len(final), Counter(r['domain'] for r in final))