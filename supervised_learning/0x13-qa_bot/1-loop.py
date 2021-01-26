#!/usr/bin/env python3
"""1-loop.py"""

in_q = ''
while (in_q is not None):
    """1-loop.py"""
    in_q = input("Q: ")
    if in_q.lower() in ['bye', 'exit', 'quit', 'goodbye']:
        print("A: Goodbye")
        break
    print("A: ")
