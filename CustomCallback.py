import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb


def output2wandbImage(output):
	return wandb.Image(output[0].detach().cpu().numpy().transpose((1,2,0)))


class ImagePredictionLogger(Callback):
	"""
		callback for log cyclegan validation images at wandb
	"""

	def __init__(self, val_data):
		super().__init__()
		self.val_data = val_data


	def on_validation_epoch_end(self, trainer, pl_moudle):
		for idx, (A,B) in enumerate(self.val_data):
			A=A.to(pl_moudle.device)
			B=B.to(pl_moudle.device)

			batch_fake_B = pl_moudle.forward(A)

			for batch, fake_B in enumerate(batch_fake_B):
				trainer.logger.experiment.log({
					"fake_B{:02}{:02}".format(idx,batch) : output2wandbImage(fake_B),
					})
