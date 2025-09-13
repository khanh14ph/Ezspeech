import re
from collections import defaultdict
from typing import Dict, List, Optional

from importlib_resources import files

DELIMITER = "▁"
VOWELS = "aăâeêioôơuưy"
TONE_CHARS = "àằầèềìòồờùừỳáắấéếíóốớúứýảẳẩẻểỉỏổởủửỷạặậẹệịọộợụựỵãẵẫẽễĩõỗỡũữỹ"
TONE_MARKS = ["1_", "2_", "3_", "4_", "5_"]
SPECIAL_SUBWORDS = [
    "uôc",
    "uych",
    "uyn",
    "uynh",
    "uyp",
    "uyt",
    "uyên",
    "uyêt",
    "i",
    "in",
    "iêt",
    "iêu",
    "iêng",
    "uôc_",
    "uych_",
    "uyn_",
    "uynh_",
    "uyp_",
    "uyt_",
    "uyên_",
    "uyêt_",
    "i_",
    "in_",
    "iêt_",
    "iêu_",
    "iêng_",
]
