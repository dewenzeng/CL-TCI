################################# U-Net backbone ######################################
# train pretext rotation
# python train_pretext_rotation.py --batch_size 16 --device cuda:0 --classes 4 --epoch 500 --model_name unet --data_dir /data/data/share/dewen/data/cxr/contrastive  \
# --lr 1e-1 --min_lr 1e-3 --patch_size 256 --experiment_name contrast_bch_rotation_unet_ --pretext_method rotation

# train pretext pirl
# python3 train_pretext_pirl.py --batch_size 32 --device cuda:0 --classes 512 --epoch 500 --model_name unet --data_dir /data/data/share/dewen/data/cxr/contrastive  \
# --lr 1e-1 --min_lr 1e-3 --patch_size 256 --experiment_name contrast_bch_pirl_unet_ --pretext_method pirl

# train vanilla simclr
# python train_contrastive_simclr.py --batch_size 32 --device cuda:0 --classes 512 --epoch 500 --temp 0.1 --model_name unet --data_dir /data/data/share/dewen/data/cxr/contrastive  \
# --lr 1e-1 --min_lr 1e-3 --patch_size 256 --experiment_name contrast_bch_vanilla_simclr_unet_ --do_contrast --use_vanilla

# train vanilla moco
# python train_contrastive_moco.py --batch_size 16 --device cuda:0 --classes 512 --epoch 500 --temp 0.1 --model_name unet --data_dir /data/data/share/dewen/data/cxr/contrastive  \
# --lr 1e-1 --min_lr 1e-3 --patch_size 256 --experiment_name contrast_bch_vanilla_moco_unet_ --do_contrast --use_vanilla

# train CL-TCI-SimCLR
# python train_contrastive_simclr.py --batch_size 16 --device cuda:0 --classes 512 --epoch 500 --temp 0.1 --model_name unet --data_dir /data/data/share/dewen/data/cxr/contrastive  \
# --lr 1e-1 --min_lr 1e-3 --patch_size 256 --experiment_name contrast_bch_cltci_simclr_unet_ --do_contrast

# train CL-TCI-MoCo
# python train_contrastive_moco.py --batch_size 16 --device cuda:0 --classes 512 --epoch 500 --temp 0.1 --model_name unet --data_dir /data/data/share/dewen/data/cxr/contrastive  \
# --lr 1e-1 --min_lr 1e-3 --patch_size 256 --experiment_name contrast_bch_cltci_moco_unet_ --do_contrast

################################# Deeplab V3+ backbone ######################################
# just change the model_name to deeplab


################################# fine-tune ######################################
# finetune unet on jsrt dataset using 10 samples, using 5 fold cross validation
# python train_supervised_cv.py --device cuda:0 --batch_size 10 --epochs 200 --data_dir /data/users/dewenzeng/data/jsrt/ --lr 1e-4 --min_lr 1e-6 --model_name unet --dataset jsrt --experiment_name supervised_jsrt_cltci_simclr_unet_s10_ \
# --classes 3 --restart --pretrained_model_path /path/to/model.pth --enable_few_data --sampling_k 10

# finetune unet on jsrt dataset using all samples, using 5 fold cross validation
# python train_supervised_cv.py --device cuda:0 --batch_size 10 --epochs 200 --data_dir /data/users/dewenzeng/data/jsrt/ --lr 1e-4 --min_lr 1e-6 --model_name unet --dataset jsrt --experiment_name supervised_jsrt_cltci_simclr_unet_sall_ \
# --classes 3 --restart --pretrained_model_path /path/to/model.pth

# finetune deeplab on montgomery dataset using 10 samples, using 5 fold cross validation
# python train_supervised_cv.py --device cuda:0 --batch_size 10 --epochs 200 --data_dir /data/users/dewenzeng/data/montgomery/ --lr 1e-4 --min_lr 1e-6 --model_name deeplab --dataset montgomery --experiment_name supervised_montgomery_cltci_simclr_deeplab_s10_ \
# --classes 3 --restart --pretrained_model_path /path/to/model.pth --enable_few_data --sampling_k 10