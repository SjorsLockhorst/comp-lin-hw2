# Computationele Linguistiek HW2
## NER

### Style guide

Write extra feature extractors in the following manner.

Write them in util.py

Their signature should be something like

```python
def extract_some_feature(sentence, i, history, **kwargs):
    return "Anything"
```

kwargs here refer to any optional key word arguments you might want to add.

Then, in features.py, you can import your features, and add them to a feature set,
by returning them in a dictionary like so:

```python
from util import extract_some_feature

def test_feature_set(sentence, i, history, **kwargs):
    return {
        "some feature name": extract_some_feature(sentence, i, history, optional_arguments_here="Anything")
    }
```
