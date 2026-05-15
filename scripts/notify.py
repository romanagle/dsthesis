#!/usr/bin/env python3
"""
notify.py — send a push notification via ntfy.sh.

Setup (one-time):
    Subscribe at https://ntfy.sh/roma-pipeline-notifications in your browser
    or install the ntfy app on your phone and subscribe to the same topic.

Usage:
    python scripts/notify.py "Pipeline done" "gnn_film training finished."
    python scripts/notify.py "Pipeline FAILED" "train_gnn.py exited with code 1"
"""

import argparse
import sys
import urllib.request

NTFY_TOPIC = "roma-pipeline-notifications"
NTFY_URL   = f"https://ntfy.sh/{NTFY_TOPIC}"

parser = argparse.ArgumentParser()
parser.add_argument("subject")
parser.add_argument("body", nargs="?", default="")
args = parser.parse_args()

title   = f"[roma] {args.subject}"
message = args.body or args.subject

try:
    req = urllib.request.Request(
        NTFY_URL,
        data=message.encode(),
        headers={"Title": title, "Priority": "default"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()
    print(f"notify.py: sent → ntfy.sh/{NTFY_TOPIC}", file=sys.stderr)
except Exception as e:
    print(f"notify.py: failed to send ({e})", file=sys.stderr)
