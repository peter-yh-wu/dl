'''
Based on https://github.com/CorentinJ/Real-Time-Voice-Cloning

Peter Wu
peterw1@andrew.cmu.edu
'''

from .tacotron import Tacotron
from .tacotron_ph import Tacotron_ph
from .tacotron_ph_demo import Tacotron_ph_demo
from .tacotron_ph_rand import Tacotron_ph_rand


def create_model_ph(name, hparams):
  if name == "Tacotron":
    return Tacotron_ph(hparams)
  else:
    raise Exception("Unknown model: " + name)
