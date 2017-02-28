class BaseDataLoader(object):
	def __init__(self, config):
		self._config = config

	def data_iter(self):
		#a generator to go through the dataset in a loop
		pass

	def train(self, do_shuffle=True):
		#Go into training mode
		pass

	def validate(self):
		#Go into Validation mode
		pass

	def test(self):
		#Go into test mode
		pass

	def shuffle(self):
		#Shuffle the dataset
		pass
