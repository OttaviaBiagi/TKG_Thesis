from .tgn   import TGNClassifier,  train as train_tgn,   evaluate as eval_tgn,   make_model as make_tgn
from .dyrep import DyRepEPC,        train as train_dyrep, evaluate as eval_dyrep, make_model as make_dyrep
from .tgat  import TGAT_EPC,        train as train_tgat,  evaluate as eval_tgat,  make_model as make_tgat

MODEL_REGISTRY = {
    'TGN':   {'make': make_tgn,   'train': train_tgn,   'eval': eval_tgn},
    'DyRep': {'make': make_dyrep, 'train': train_dyrep, 'eval': eval_dyrep},
    'TGAT':  {'make': make_tgat,  'train': train_tgat,  'eval': eval_tgat},
}
