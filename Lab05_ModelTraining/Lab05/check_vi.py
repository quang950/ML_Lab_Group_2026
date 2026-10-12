import json, re, glob

bad = re.compile(r'[A-Za-z]\?[A-Za-z]')        # dạng chữ?chữ, dấu hiệu mất dấu
nonascii = re.compile(r'[^\x00-\x7F]')           # có ký tự tiếng Việt thật

for nb in sorted(glob.glob('notebooks/*.ipynb')):
    cells = json.load(open(nb, encoding='utf-8'))['cells']
    src = lambda c: ''.join(c['source'])
    md_bad = sum(1 for c in cells if c['cell_type'] == 'markdown' and bad.search(src(c)))
    code_bad = sum(1 for c in cells if c['cell_type'] == 'code' and bad.search(src(c)))
    has_vi = any(nonascii.search(src(c)) for c in cells)
    print(f"{nb[10:]:30s} md_hỏng={md_bad:3d}  code_hỏng={code_bad:3d}  có_chữ_có_dấu={has_vi}")