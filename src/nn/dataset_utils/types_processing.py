"""

Created on 17-June-2020
@author Jibesh Patra

The types extracted during runtime usually look something like --> <class 'numpy.ndarray'> or
<class 'seaborn.palettes._ColorPalette'> change them to --> ndarray, ColorPalette
"""

import re

remove_chars = re.compile(r'>|\'|<|(class )|_|(type)')


def process_types(tp: str) -> str:
    cleaned_type = remove_chars.sub('', tp)
    cleaned_type = cleaned_type.split('.')[-1].strip()
    return cleaned_type
