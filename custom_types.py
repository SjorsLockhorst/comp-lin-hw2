"""
FILE: custom_types.py

File containing any custom types.

Authors: Tijn, Gaby, Felix, Sjors
"""

from typing import Tuple, Callable, Any, List

TaggedWord = Tuple[str, str]
FeatureMap = Callable[[List[TaggedWord], int, List[str]], Any]
