import pytest
from unittest import TestCase
import utils.model_utils as mutils

class Test_model(TestCase):

    def test_load_embed_txt(self):
        embed_dict, embed_size = mutils.load_embed_txt('data/w2v_without_entity.vec')
        self.assertTrue(embed_dict['ORG'][0] == 0.0054711)

    