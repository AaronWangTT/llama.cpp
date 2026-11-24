#!/usr/bin/env python3
"""Token list for --spec-draft-shortlist: tokenize text files with a running llama-server and write the token ids
that occur, most frequent first, one per line.

    spec-draft-shortlist.py --url http://127.0.0.1:8080 --out shortlist.txt file...

The server must serve the model the list is for (same vocabulary). Only /tokenize is used, special tokens are not parsed.
"""
import argparse
import json
import urllib.request


def tokenize(url, text):
    body = json.dumps({'content': text, 'add_special': False, 'parse_special': False}).encode()
    req = urllib.request.Request(url + '/tokenize', data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())['tokens']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    ap.add_argument('--out', required=True)
    ap.add_argument('--chunk', type=int, default=262144, help='characters per request, whole lines')
    ap.add_argument('files', nargs='+')
    args = ap.parse_args()

    count = {}
    n_tokens = 0

    def flush(buf):
        nonlocal n_tokens
        if buf:
            for t in tokenize(args.url, ''.join(buf)):
                count[t] = count.get(t, 0) + 1
                n_tokens += 1

    for path in args.files:
        buf = []
        size = 0
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                buf.append(line)
                size += len(line)
                if size >= args.chunk:
                    flush(buf)
                    buf = []
                    size = 0
        flush(buf)

    order = sorted(count, key=lambda t: (-count[t], t))
    with open(args.out, 'w') as f:
        f.write('# %d tokens, %d distinct, most frequent first\n' % (n_tokens, len(order)))
        for t in order:
            f.write('%d\n' % t)
    print('%d tokens, %d distinct -> %s' % (n_tokens, len(order), args.out))


if __name__ == '__main__':
    main()
