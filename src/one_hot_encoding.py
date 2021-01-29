import numpy as np
import pdb
from numpy import argmax


class one_hot_encoding():

	_alphabet = []
	_char_to_int = {}
	_int_to_chat = {}
	_onehot_encode = {}
	_inverse_encode = {}

	def __init__(self, alphabet):
		self._alphabet=alphabet
		self._predefine_encoding()

	def _predefine_encoding(self):
		# define a mapping of chars to integers
		self._char_to_int = dict((c, i) for i, c in enumerate(self._alphabet))
		self._int_to_char = dict((i, c) for i, c in enumerate(self._alphabet))

	def create_encoding(self,data):
		# integer encode input data
		integer_encoded = [self._char_to_int[f] for f in data]
		onehot_encoding = self._get_onehot_encode(integer_encoded)
		return onehot_encoding


	def _get_onehot_encode(self,integer_encoded):
		onehot_encoded = list()
		for value in integer_encoded:
			letter = [0 for _ in range(len(self._alphabet))]
			letter[value] = 1
			onehot_encoded.append(letter)
		return onehot_encoded

	def get_inverted_encode(self,data):
		return self._int_to_char[argmax(data)]


if __name__ == "__main__":
	pdb.set_trace()
	# define input string
	data = 'ACEFFGGIH'
	print(data)

	# define universe of possible input values
	# alphabet = 'abcdefghijklmnopqrstuvwxyz '
	alphabet = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq']
	hh = one_hot_encoding(alphabet)
	encode = hh.create_encoding(data)
	print(encode)
	inverted = hh.get_inverted_encode(encode[0])
