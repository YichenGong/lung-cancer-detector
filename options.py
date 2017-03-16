# Conatins a class which handles the options to run the experiment
import argparse

def parse():
	parser = argparse.ArgumentParser(prog="Lung Cancer Detector")

	parser.add_argument("-model", help="Model to run", type=str, default="")
	parser.add_argument("-d", "--data", help="Dataloader file", type=str, default="stage1")
	parser.add_argument("-p", "--pre-process", help="Pre-processing hook to be applied to all input", type=str, default="")
	parser.add_argument("-n", "--name", help="Name of the experiment to be suffixed on all outputs", type=str, default="")
	parser.add_argument("-val", "--validation-ratio", help="Percentage of data to be used for validation", type=float, default=0.1)
	parser.add_argument("-lr", "--learning-rate", help="Initial Learning Rate", type=float, default=0.1)
	parser.add_argument("-ldr", "--decay-rate", help="Learning rate exponential decay rate", type=float, default=0.96)
	parser.add_argument("-lm", "--momentum", help="Momentum for momentum based learning", type=float, default=0.9)
	parser.add_argument("-e", "--epochs", help="Number of epochs to run", type=int, default=100)
	parser.add_argument("-b", "--batch", help="Batch size", type=int, default=32)
	parser.add_argument("--threads", help="Number of threads", type=int, default=1)
	parser.add_argument("--false-negative-weight", help="Weight on false negative predictions", type=float, default=1.0)
	parser.add_argument("-s", "--size", help="Size of images DEPTH * HEIGHT * WIDTH", nargs=3, type=int, default=[128, 128, 128])
	parser.add_argument("--original", help="Use original size of the images", action="store_true")
	parser.add_argument("--seed", help="Random Generator Seed", type=int, default=0)
	parser.add_argument("--model-save-path", help="Model Saving Path", type=str, default="summaries/")

	# Loader specific options
	parser.add_argument("--padded-images", help="Make all images of the same size by zero padding", action="store_true")
	parser.add_argument("--upscale-batch", help="Upscale every batch to have equal number of positive and negative labels", action="store_true")

	#Training specific arguments
	parser.add_argument("--no-train", help="No training, only testing from saved model", action="store_true")
	parser.add_argument("--no-test", help="No testing", action="store_true")
	parser.add_argument("--no-validation", help="No validation, Only train", action="store_true")
	
	
	parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")

	#Multi-Head-Specific
	parser.add_argument("--amhu2-luna-train", help="Aggressive Multi-Head Unet 2D, Train Luna", action="store_true")
	parser.add_argument("--amhu2-lidc-train", help="Aggressive Multi-Head Unet 2D, Train LIDC", action="store_true")
	parser.add_argument("--amhu2-luna-lidc-train", help="Aggressive Multi-Head Unet 2D, Train on intermixed set of LUNA and LIDC segmentation data", action="store_true")
	parser.add_argument("--amhu2-sample-train", help="Aggressive Multi-Head Unet 2D, Train Sample Kaggle", action="store_true")
	parser.add_argument("--amhu2-stage1-train", help="Aggressive Multi-Head Unet 2D, Train Stage1 Kaggle", action="store_true")
	parser.add_argument("--amhu2-nodule-cancer-train", help="Aggressive Multi-Head Unet 2D, Train Cancer head using data from LIDC/LUNA", action="store_true")
	parser.add_argument("--amhu2-infer", help="Aggressive Multi-Head Unet 2D, Infer Stage1 Test Dataset", action="store_true")

	return parser.parse_args()

if __name__ == "__main__":
	print(parse())
