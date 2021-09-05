#!/bin/bash 
#JSUB -J cxr_contrastive
#JSUB -q tensorflow_sub
#JSUB -gpgpu "num=2"
#JSUB -R "span[ptile=2]" 
#JSUB -n 2
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

# python3 train_contrastive_simclr.py --batch_size 16 --device cuda:0 --classes 512 --epoch 500 --temp 0.1 --model_name unet --data_dir /data/data/share/dewen/data/cxr/contrastive  \
# --lr 1e-1 --min_lr 1e-3 --patch_size 256 --experiment_name contrast_bch_vanilla_simclr_unet_ --do_contrast

python3 train_pretext_pirl.py --batch_size 32 --device cuda:0 --classes 512 --epoch 500 --model_name unet --data_dir /data/data/share/dewen/data/cxr/contrastive  \
--lr 1e-1 --min_lr 1e-3 --patch_size 256 --experiment_name contrast_bch_pirl_unet_ --pretext_method pirl