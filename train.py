from config import get_arguments
from perceptualGAN import myModel
from DataLoader import load_horse2zebra
from CustomCallback import ImagePredictionLogger

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
	parser = get_arguments()
	opt = parser.parse_args()
	print(opt)
	# model
	model = myModel(opt)
	# data load
	train_loader, valid_loader = load_horse2zebra(opt)

	# logger
	if opt.use_wandb:
		logger = WandbLogger(project=opt.wandb_project, offline=opt.offline)
	else:
		logger = None

	# trainer
	trainer = Trainer(logger = logger, 
					  gpus=1, 
					  max_epochs=opt.epochs,
					  callbacks=[ImagePredictionLogger(valid_loader)])
	
	# train
	trainer.fit(model, train_loader,valid_loader)
	print('train done')
