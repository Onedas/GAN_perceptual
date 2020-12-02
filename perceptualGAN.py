import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import itertools
import wandb
from ModelLoader import getG, getD, pretrainedVGG


class myModel(pl.LightningModule):

	def __init__(self, opt):
		super().__init__()
		self.opt = opt

		self.G = getG(opt).to(self.device)
		self.D = getD(opt).to(self.device)

		self.featureModel = pretrainedVGG(self.opt).to(self.device)

	def forward(self, x):
		return self.G(x)

	def configure_optimizers(self):
		lr = self.opt.lr
		b1= self.opt.beta1
		b2 = self.opt.beta2

		opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(b1,b2))
		opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(b1,b2))
		return [opt_g, opt_d], []

	def training_step(self, batch, batch_idx, optimizer_idx):
		
		A, B = batch

		# train G
		if optimizer_idx == 0:

			fake_B = self.G(A)
			D_fake_B = self.D(fake_B)

			ones = torch.ones_like(D_fake_B).to(self.device)

			loss_gan =F.mse_loss(D_fake_B, ones)

			# perceptual loss
			real_feature = self.featureModel(B)
			fake_feature = self.featureModel(fake_B.detach())

			loss_percep = F.l1_loss(fake_feature, real_feature) * self.opt.percep_lambda

			loss_G = loss_gan + loss_percep

			self.log('loss_gan',loss_gan)
			self.log('loss_percep',loss_percep)
			self.log('loss_G',loss_G)
			return loss_G

		# train D
		if optimizer_idx == 1:
			fake_B = self.G(A)
			D_fake_B = self.D(fake_B)
			D_B = self.D(B.detach())

			zeros = torch.zeros_like(D_B).to(self.device)
			ones = torch.ones_like(D_B).to(self.device)
			
			loss_D_real = F.mse_loss(D_B, ones)
			loss_D_fake = F.mse_loss(D_fake_B, zeros)

			loss_D = (loss_D_real + loss_D_fake)*0.5

			self.log('loss_D',loss_D)
			return loss_D
	
	def validation_step(self, batch, batch_idx):
		pass

if __name__ == "__main__":
	from config import get_arguments
	parser = get_arguments()
	opt = parser.parse_args()
	print(opt)
	model =myModel(opt)