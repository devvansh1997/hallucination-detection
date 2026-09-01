"""methods/ -- the detector registry.

Adding a method is two steps:
    1. write methods/<name>.py with a Method subclass (see halluguard.py)
    2. add one line to REGISTRY below

Nothing else changes. The runner (56_run_method.py) handles data, splitting, AUROC and output
format identically for every entry, which is the only reason numbers from different methods can go
in the same table.
"""

from .base import DataBundle, Method, canonical, load_numbered

REGISTRY = {}


def register(cls):
    assert cls.name, "%s must set a name" % cls.__name__
    assert cls.granularity in ("beam", "question"), \
        "%s.granularity must be 'beam' or 'question', got %r" % (cls.__name__, cls.granularity)
    assert cls.name not in REGISTRY, "duplicate method name: %s" % cls.name
    REGISTRY[cls.name] = cls
    return cls


from .eigenscore import EigenScore                    # noqa: E402
from .lexical_similarity import LexicalSimilarity    # noqa: E402
from .ln_entropy import LNEntropy                    # noqa: E402
from .perplexity import Perplexity                   # noqa: E402

register(Perplexity)
register(LNEntropy)
register(LexicalSimilarity)
register(EigenScore)

# HalluGuard is DROPPED FROM THE COMPARISON TABLE (decision, 2026-08-31). The code and the
# findings stay: PROJECT_LOG 2.9-2.12 record that its released implementations disagree with its
# own paper, and that under the authors' rebuttal spec it reduces to counting distinct answers.
# That is a reproducibility result, not a baseline row, and it does not belong in a column of
# AUROCs. Re-enable by uncommenting; methods/halluguard.py is unchanged and still self-tests.
# from .halluguard import HalluGuard                 # noqa: E402
# register(HalluGuard)


def get(name):
    if name not in REGISTRY:
        raise SystemExit("unknown method %r. Available: %s"
                         % (name, ", ".join(sorted(REGISTRY))))
    return REGISTRY[name]()


def describe():
    return [(n, c.granularity, c.description) for n, c in sorted(REGISTRY.items())]


__all__ = ["Method", "DataBundle", "REGISTRY", "register", "get", "describe",
           "canonical", "load_numbered"]
