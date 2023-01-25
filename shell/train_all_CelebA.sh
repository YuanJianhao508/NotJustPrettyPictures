# algorithms="OURS ERM ACVC AugMix CutMix MixUp CutOut MEADA PixMix RandAugment SRN SN L2D RSC"
algorithms="MixUp CutMix"
# trails="E30_PRET_CE2 E30_PRET_CE3 E30_PRET_CE4 E30_PRET_CE5"
# trails="Contrastive/handcrafted/1 Contrastive/handcrafted/2 Contrastive/handcrafted/3 Contrastive/handcrafted/4 Contrastive/handcrafted/5"
trails="Crossval/1 Crossval/2 Crossval/3 Crossval/4 Crossval/5"
loss="ce"

#SD
# linkpath='/homes/55/jianhaoy/projects/EKI/link/CelebA/original_sd_link.json'
#VQGAN
# linkpath='/homes/55/jianhaoy/projects/EKI/link/CelebA/original_vqgan_link.json'
#Controlled
# linkpath="/homes/55/jianhaoy/projects/EKI/link/CelebA/original_control_link.json"
#Reversed
linkpath="/homes/55/jianhaoy/projects/EKI/link/CelebA/original_sd_reversed_link.json"

for algo in $algorithms; do
    for out in $trails; do
    echo "$algo $out"
    nohup python -u shell_single_celeba_train.py -d=original -g=5 -a=$algo -ic=$loss -lp=$linkpath > ./results_CelebA/$out/$algo/original.out 
    done
done

# nohup python -u shell_single_celeba_train.py -d=original -g=1 -a=ERM -ic='ce' -lp='/homes/55/jianhaoy/projects/EKI/link/CelebA/original_sd_link.json'