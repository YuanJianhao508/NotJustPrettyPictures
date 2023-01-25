# algorithms="OURS ERM AugMix RandAugment CutOut MixUp CutMix RSC MEADA ACVC PixMix"
#  
# algorithms="ERM OURS AugMix RandAugment CutOut MixUp CutMix RSC MEADA ACVC PixMix SN SRN"
# algorithms="AugMix RandAugment PixMix MEADA CutOut MixUp CutMix RSC ACVC OURS"

algorithms="MixUp CutMix"
# algorithms="AugMix RandAugment CutOut MixUp CutMix RSC"
# algorithms="OURS ERM ACVC AugMix CutMix MixUp CutOut MEADA PixMix RandAugment SRN SN L2D RSC"
# trails="E30_PRET_CE1 E30_PRET_CE2 E30_PRET_CE3 E30_PRET_CE4 E30_PRET_CE5"
# trails='Contrastive/handcrafted/1 Contrastive/handcrafted/2 Contrastive/handcrafted/3 Contrastive/handcrafted/4 Contrastive/handcrafted/5'
trails="Crossval/1 Crossval/2 Crossval/3 Crossval/4 Crossval/5"
loss="ce"

#VQGAN
# linkpath='/homes/55/jianhaoy/projects/EKI/link/Texture/original_vqgan_link_v4.json'

#SD
# linkpath='/homes/55/jianhaoy/projects/EKI/link/Texture/original_sd_link.json'
linkpath='/homes/55/jianhaoy/projects/EKI/link/Texture/original_sd_v1_link.json'


# python -u shell_single_texture_train.py -d=original -g=0 -a="OURS" -ic="ce" -lp='/homes/55/jianhaoy/projects/EKI/link/Texture/original_sd_link.json'

for algo in $algorithms; do
    for out in $trails; do
    echo "$algo $out"
    nohup python -u shell_single_texture_train.py -d=original -g=2 -a=$algo -ic=$loss -lp=$linkpath > ./results_Texture/$out/$algo/original.out 
    done
done