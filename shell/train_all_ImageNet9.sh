# algorithms="OURS ERM AugMix RandAugment CutOut MixUp CutMix RSC MEADA ACVC PixMix"
#  
# algorithms="ERM OURS AugMix RandAugment CutOut MixUp CutMix RSC MEADA ACVC PixMix"
# algorithms="AugMix RandAugment PixMix MEADA CutOut MixUp CutMix RSC ACVC OURS"

# algorithms="OURS"
algorithms="MixUp CutMix"
# trails="test1 test2 test3"
# trails="Inpaint"
# trails="Contrastive/handcrafted/1 Contrastive/handcrafted/2 Contrastive/handcrafted/3 Contrastive/handcrafted/4 Contrastive/handcrafted/5"
trails="Crossval/1 Crossval/3 Crossval/4 Crossval/5"
loss="ce"

#SD Inpaint
# linkpath='/homes/55/jianhaoy/projects/EKI/link/imagenet9_SD_template_link/original_sd_inpaint_full_link.json'
# SD
linkpath='/homes/55/jianhaoy/projects/EKI/link/imagenet9_SD_template_link/original_sd_link.json'
# VQGAN
# linkpath='/homes/55/jianhaoy/projects/EKI/link/imagenet9_SD_template_link/original_vqgan_link.json'

# /homes/55/jianhaoy/projects/EKI/results_ImageNet9/E30_PRET_CE

# python -u shell_single_imagenet9_train.py -d=original -g=0 -a=$algo -ic=$loss -lp=$linkpath

for algo in $algorithms; do
    for out in $trails; do
    echo "$algo $out"
    nohup python -u shell_single_imagenet9_train.py -d=original -g=5 -a=$algo -ic=$loss -lp=$linkpath > ./results_ImageNet9/$out/$algo/original.out 
    done
done