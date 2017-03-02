from models.tf_unet import unet
from utils.luna_preprocessed_load_data import DataLoad
import tensorflow as tf


if __name__ == "__main__":
    flags = tf.app.flags
    flags.DEFINE_integer("width", 512, "width")
    flags.DEFINE_integer("height", 512, "height")
    # flags.DEFINE_integer("layers", 512, "layers")
    flags.DEFINE_integer("batch_size", 4, "batch size")
    flags.DEFINE_float("train_ratio", 1.0, "train validation ratio")

    flags.DEFINE_string("data_path", "data/LUNA/1_1_1mm_slices_nodule/", "The path to slices")
    flags.DEFINE_string("mask_path", "data/LUNA/1_1_1mm_slices_lung_masks/", "The path to nodule mask of the slices")
    flags.DEFINE_string("original_data_path", "data/LUNA/original_lungs/", "original lung path")
    flags.DEFINE_string("annotation_file_path", "data/LUNA/csv/annotations.csv" , 'path for annotation')
    flags.DEFINE_integer("exp_id", 0, "The experiment id")
    flags.DEFINE_string("output_path", "out/LUNA/unet/", "output path")
    config = flags.FLAGS

    data_provider = DataLoad(config=config)
    data_provider.train()



    output_path = config.output_path
    net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)
    trainer = unet.Trainer(net, batch_size=config.batch_size)
    path = trainer.train(data_provider, output_path, training_iters=32, epochs=100, restore=True)


