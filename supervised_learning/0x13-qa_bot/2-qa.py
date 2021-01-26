#!/usr/bin/env python3
"""2-qa.py"""


qa = __import__('0-qa').question_answer


def answer_loop(reference):
    """2-qa.py"""
    in_q = ''
    while (in_q is not None):
        """Create a basic Q/A input loop"""
        in_q = input("Q: ")
        if in_q.lower() in ['bye', 'exit', 'quit', 'goodbye']:
            print("A: Goodbye")
            break
        answer = qa(in_q, reference)
        if answer == '':
            print("A: Sorry, I do not understand your question.")
            continue
        print("A: {}".format(answer))
