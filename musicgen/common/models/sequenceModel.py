import abc
from common.models.model import Model

"""
Abstract base class for sequence models
This currently doesn't do anything, but I have a hunch that having this in the inheritance
   hierarchy later on will be helpful.
"""
class SequenceModel(Model):

	__metaclass__ = abc.ABCMeta