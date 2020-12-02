import torch
import torch.nn as nn
import torch.nn.functional as f
from models.resnets import ResNet
from models.patchDiscriminator import PatchDiscriminator

import torchvision.models as models

def pretrainedVGG(opt):
	"""
	"""
	nth_feature = opt.perceptual_depth-1
	layer_num=[0,2,5,7,19,12,14,17,19,21,24,26,28]
	n = layer_num[nth_feature]
	vgg16 = models.vgg16(pretrained=True)
	vgg16_conv = nn.Sequential(*list(vgg16.children())[0][:n+1])

	for param in vgg16_conv.parameters():
		param.requires_grad = False

	print("load vgg16 network")
	return vgg16_conv

def getG(opt):

	"""
	Define ResNet for cycleGAN Generator
	"""

	net = opt.G
	in_channel = opt.channel
	out_channel = opt.channel
	num_filter = opt.G_filter
	norm_type = opt.norm
	n_blocks = opt.G_blocks
	use_dropout = opt.dropout
	use_bias = opt.bias
	padding_type = opt.paddingtype
	
	init_type = opt.initalize
	init_gain = opt.init_gain



	if net == 'resnet':
		model = ResNet(in_channel, out_channel, num_filter,
			norm_type = norm_type,
			n_blocks = n_blocks,
			use_dropout = use_dropout,
			use_bias = use_bias,
			padding_type = padding_type)
	else:
		raise NotImplementedError("Check model name")

	print('load Generator({})'.format(net), end=', ')
	model_initalize(model, init_type, init_gain)

	return model


def getD(opt):
	"""
	Define(Get) PatchGAN Discriminator
	
	Params:
		in_channel : parmas of patchDiscriminator. 
		num_filter : 
		num_layers :
		norm_type :

		init_type(str) : initalize type (xavier | normal | kaiming)
		init_gain(float) : initalize parameter

	"""

	in_channel = opt.channel
	num_filter = opt.D_filter
	num_layers = opt.D_layers
	norm_type = opt.norm
	init_type = opt.initalize
	init_gain = opt.init_gain


	model = PatchDiscriminator(in_channel, num_filter, num_layers, norm_type)
	print('load Markovian Discriminator', end=', ')
	model_initalize(model, init_type, init_gain)
	return model


def model_initalize(model, init_type ='xavier', init_gain=0.02):
	"""
	Initalize network weights.

	Params:
		net : network
		init_type(str) : initalziation method : normal, xavier, kaiming
		init_gain(float) : scaling factor for normal, xavier

	"""	
	init_layer_list = [nn.Conv2d, nn.ConvTranspose2d]
	init_norm_list = [nn.BatchNorm2d]

	@torch.no_grad()
	def initalize_func(m):
		if type(m) in init_layer_list:
			if init_type == "normal":
				nn.init.normal_(m.wieght.data, 0.0, init_gain)
			elif init_type == "xavier":
				nn.init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == "kaiming":
				nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
			else:
				raise NotImplementedError("Check initalzation method")

		elif type(m) in init_norm_list: # batch norm
			nn.init.normal_(m.weight.data, 1.0, init_gain)
			nn.init.constant_(m.bias.data, 0.0)

	print('initalize network with {}'.format(init_type))
	model.apply(initalize_func)

if __name__ == "__main__":
	print(models.vgg16(pretrained=True))