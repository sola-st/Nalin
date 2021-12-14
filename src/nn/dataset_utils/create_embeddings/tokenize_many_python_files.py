"""

Created on 05-May-2020
@author Jibesh Patra

Tokenize many python files so as to learn embeddings later, using fastText.
As parts of the tokens, skip comments and doc-strings.
"""
import tokenize
from typing import List
from pathlib import Path
from tqdm import tqdm


def get_tokens_of_a_file(in_file_path: str) -> List:
    skip_token_nums = {tokenize.ENCODING, tokenize.INDENT, tokenize.NEWLINE, tokenize.COMMENT, tokenize.DEDENT,
                       tokenize.NL}
    tokens_in_file = []
    last_seen_token = None
    try:
        with open(in_file_path, 'rb') as f:
            tokens = tokenize.tokenize(f.readline)
            for toknum, tokval, _, _, _ in tokens:
                if toknum not in skip_token_nums and last_seen_token != tokenize.INDENT:
                    is_doc_string = toknum == 3 and ((tokval.startswith('"""') and tokval.endswith('"""')) or (
                        (tokval.startswith("'''") and tokval.endswith("'''"))))
                    if len(tokval) and not is_doc_string:
                        tokens_in_file.append(tokval + ' ')
                    elif is_doc_string:
                        tokens_in_file.append('"" ')
                last_seen_token = toknum
    except:
        return []
    return tokens_in_file


if __name__ == '__main__':
    in_dir = 'benchmark/python_scripts'
    out_file = 'benchmark/all_tokens'
    file_list = list(Path(in_dir).glob('**/*.py'))
    with open(out_file, 'w+') as out_f:
        for file in tqdm(file_list, desc='Writing tokens'):
            tokens = get_tokens_of_a_file(in_file_path=file)
            out_f.writelines(tokens)
