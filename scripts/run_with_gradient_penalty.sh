# To train with gradient penalty
python train.py --dataset_name MNIST --image_channels 1 --use_gradient_penalty --logs_dir logs_gp --save_dir models_gp --optimizer Adam --lr 1e-4 --n_epochs 5