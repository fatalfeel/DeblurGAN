from .model_gan import ConditionalGAN
from .model_test import TestModel

def create_model(opt):
	model = None

	if opt.model == 'test':
		assert (opt.dataset_mode == 'single')
		#from .test_model import TestModel
		model = TestModel( opt )
	else:
		model = ConditionalGAN(opt)

	print("model [%s] was created" % (model.name()))

	return model
