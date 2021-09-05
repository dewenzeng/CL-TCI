#!/bin/bash 
#JSUB -J cxr_shape_prior
#JSUB -q tensorflow_sub
#JSUB -gpgpu "num=1"
#JSUB -R "span[ptile=1]" 
#JSUB -n 1
#JSUB -o logs/output.%J 
#JSUB -e logs/err.%J 
##########################Cluster environment variable###################### 
if [ -z "$LSB_HOSTS" -a -n "$JH_HOSTS" ]
then
        for var in ${JH_HOSTS}
        do
                if ((++i%2==1))
                then
                        hostnode="${var}"
                else
                        ncpu="$(($ncpu + $var))"
                        hostlist="$hostlist $(for node in $(seq 1 $var);do printf "%s " $hostnode;done)"
                fi
        done
        export LSB_MCPU_HOSTS="$JH_HOSTS"
        export LSB_HOSTS="$(echo $hostlist|tr ' ' '\n')"
fi

nodelist=.hostfile.$$
for i in `echo $LSB_HOSTS`
do
    echo "${i}" >> $nodelist
done

ncpu=`echo $LSB_HOSTS |wc -w`
##########################Software environment variable#####################
module load python/3.6.10
module load cuda/cuda10.1 
module load pytorch/pytorch1.5.1 

# python3 train_shape_prior2.py --device cuda:0 --batch_size 10 --epochs 200 --data_dir /data/users/dewenzeng/data/cxr/supervised/ --lr 1e-4 --min_lr 1e-6 --model_name unet --experiment_name supervised_cxr_random_unet_center_aligned_epoch200_worotation_ \
# --initial_filter_size 32 --classes 3 --use_prior

# python3 train_shape_prior2.py --device cuda:0 --batch_size 10 --epochs 200 --data_dir /data/users/dewenzeng/data/cxr/supervised/ --lr 1e-4 --min_lr 1e-6 --model_name unet --experiment_name supervised_cxr_random_unet_branch_aligned_epoch200_worotation_ \
# --initial_filter_size 32 --classes 3 --use_prior

# python3 train_bb.py --device cuda:0 --batch_size 10 --epochs 500 --data_dir /data/users/dewenzeng/data/cxr/supervised/ --lr 1e-3 --min_lr 1e-6 --experiment_name supervised_cxr_bb_aug_if16_

# python3 train_supervised_cv.py --device cuda:0 --batch_size 10 --epochs 200 --data_dir /data/data/share/dewen/data/jsrt/ --lr 1e-4 --min_lr 1e-6 --model_name unet --dataset cxr --experiment_name supervised_cxr_unet_s150_ --classes 4 

python3 train_supervised_cv.py --device cuda:0 --batch_size 10 --epochs 200 --data_dir /data/users/dewenzeng/data/jsrt/ --lr 1e-4 --min_lr 1e-6 --model_name unet --dataset jsrt --experiment_name supervised_jsrt_cltci_simclr_unet_s10_ \
--classes 3 --restart --pretrained_model_path /data/data/share/zhuorui/model/contrast_cxr_ours_unet_simclr_2021-09-04_10-09-40/model/latest.pth --enable_few_data --sampling_k 10

# python3 train_supervised_cv.py --device cuda:0 --batch_size 10 --epochs 200 --lr 1e-4 --min_lr 1e-6 --model_name unet --experiment_name supervised_cxr_ours_chexpert_v2_unet_epoch200_ --classes 3 \
# --initial_filter_size 32 --classes 3 --restart --pretrained_model_path /data/users/dewenzeng/code/cxr_segmentation_rmyy/results/contrast_chexpert_ours_2021-05-19_12-03-49/model/latest.pth --enable_few_data --sampling_k 150

# python3 train_supervised_cv.py --device cuda:0 --batch_size 10 --epochs 200 --data_dir /data/users/dewenzeng/data/montgomery/ --lr 1e-4 --min_lr 1e-6 --model_name deeplab --dataset mont --experiment_name supervised_mont_ours_deeplab_v4_ \
# --initial_filter_size 32 --classes 3 --restart --model_version v4