import json
import sys

base_path, *candidate_paths = sys.argv[1:]
with open(base_path, encoding="utf-8") as f:
    base = json.load(f)
base_ids = [x["token_ids"] for x in base]
failed = False
for path in candidate_paths:
    with open(path, encoding="utf-8") as f:
        got = json.load(f)
    got_ids = [x["token_ids"] for x in got]
    if got_ids != base_ids:
        print(f"FAIL: {path} differs from connector-free full recomputation")
        for i, (a, b) in enumerate(zip(base_ids, got_ids)):
            if a != b:
                print(f"  request {i}: baseline={a} recovered={b}")
        failed = True
    else:
        print(f"PASS: {path} exactly matches connector-free full recomputation")
raise SystemExit(1 if failed else 0)
