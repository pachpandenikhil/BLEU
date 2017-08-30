# BLEU
BLEU score calculator for evaluating the accuracy of machine translation of different languages.

- Implemented a BLEU score calculator for evaluating the accuracy of machine translation of different languages, as defined in the [paper](http://www.aclweb.org/anthology/P02-1040.pdf).
- Calculated BLEU score fully matches the true BLEU score.

Core Technology: Python.

# Data

Many reference translations can be found from [EUROPARL corpus](http://www.statmt.org/europarl/archives.html).

The candidate translations can be obtained by taking the corresponding English sentences from the reference translations and running them through [Google Translate](https://translate.google.com/).

# Program

*calculatebleu.py* takes two paramaters:

  1. path to the candidate translation (a single file).
  
  2. path to the reference translations (either a single file, or a directory if there are multiple reference translations).

```
> python calculatebleu.py /path/to/candidate /path/to/reference
```

# Output

BLEU score of the candidate translation relative to the set of reference translations is written to an output file called *bleu_out.txt*.
