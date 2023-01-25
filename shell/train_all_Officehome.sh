# algorithms="AugMix RandAugment PixMix MEADA CutOut MixUp CutMix RSC ACVC"
# algorithms="MixUp CutMix RSC"
algorithms="AugMix RandAugment CutOut MixUp CutMix RSC"
# trails="E50_PRET_CE1 E50_PRET_CE2 E50_PRET_CE3 E50_PRET_CE4 E50_PRET_CE5"
trails="Crossval/1"
loss="ce"

# For dumb VQGAN_CLIP
# linkpath="/homes/55/jianhaoy/projects/EKI/link/officehome/offichome_vc_dumb_link.json"
# For hand SD
# linkpath="/homes/55/jianhaoy/projects/EKI/link/officehome/offichome_sd_dumb_link.json"
# For conservative
# linkpath="/homes/55/jianhaoy/projects/EKI/link/officehome/officehome_sd_conservative_autoprompt_link.json"
# For moderate
# linkpath="/homes/55/jianhaoy/projects/EKI/link/officehome/officehome_sd_moderate_autoprompt_link.json"
# For Minimal SD
# linkpath='/homes/55/jianhaoy/projects/EKI/link/officehome/officehome_sd_minimal_link.json'
# For Minimal VQGAN
linkpath='/homes/55/jianhaoy/projects/EKI/link/officehome/officehome_vc_minimal_link.json'

for algo in $algorithms; do
    for out in $trails; do
    echo "$algo $out"
    python -u shell_single_officehome_train.py -d=Art -g=4 -a=$algo -ic=$loss -lp=$linkpath > ./results_Officehome/$out/$algo/Art.out &
    python -u shell_single_officehome_train.py -d=Clipart -g=3 -a=$algo -ic=$loss -lp=$linkpath > ./results_Officehome/$out/$algo/Clipart.out &
    python -u shell_single_officehome_train.py -d=Product -g=4 -a=$algo -ic=$loss -lp=$linkpath > ./results_Officehome/$out/$algo/Product.out &
    python -u shell_single_officehome_train.py -d=Real_World -g=1 -a=$algo -ic=$loss -lp=$linkpath > ./results_Officehome/$out/$algo/Real_World.out
    done
done

