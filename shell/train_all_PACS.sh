
# algorithms="OURS ERM AugMix RandAugment CutOut MixUp CutMix RSC MEADA ACVC PixMix"
# algorithms="AugMix RandAugment CutOut MixUp CutMix RSC"
algorithms="ACVC L2D"
# algorithms="ERM OURS AugMix RandAugment CutOut MixUp CutMix RSC MEADA ACVC PixMix L2D"
# trails="Contrastive/handcrafted/1 Contrastive/handcrafted/2 Contrastive/handcrafted/3 Contrastive/handcrafted/4 Contrastive/handcrafted/5"
# trails="E30_PRET_CE1 E30_PRET_CE2 E30_PRET_CE3 E30_PRET_CE4 E30_PRET_CE5"
trails="Rebuttal"
loss="ce"

# For dumb Stable Diffusion
# linkpath="/homes/55/jianhaoy/projects/EKI/link/pacs/pacs_sd_dumb_link.json"

# For auto-prompt moderate
# linkpath="/homes/55/jianhaoy/projects/EKI/link/pacs/pacs_sd_autoprompt_link.json"

# For auto-prompt conservative
# linkpath="/homes/55/jianhaoy/projects/EKI/link/pacs/pacs_sd_conservative_autoprompt_link.json"

# For dumb VQGAN_CLIP
# linkpath="/homes/55/jianhaoy/projects/EKI/link/pacs/pacs_vc_dumb_link.json"

# For hand-crafted SD
linkpath="/homes/55/jianhaoy/projects/EKI/link/pacs/pacs_sd_engi_link.json"

# For text inversion
# linkpath="/homes/55/jianhaoy/projects/EKI/link/pacs/pacs_sd_text_inv_8_link.json"



for algo in $algorithms; do
    for out in $trails; do
    echo "$algo $out"
    python -u shell_single_train.py -d=photo -g=1 -a=$algo -ic=$loss -lp=$linkpath > ./results/$out/$algo/photo.out &
    python -u shell_single_train.py -d=cartoon -g=5 -a=$algo -ic=$loss -lp=$linkpath > ./results/$out/$algo/cartoon.out &
    python -u shell_single_train.py -d=art_painting -g=3 -a=$algo -ic=$loss -lp=$linkpath > ./results/$out/$algo/art_painting.out &
    python -u shell_single_train.py -d=sketch -g=2 -a=$algo -ic=$loss -lp=$linkpath > ./results/$out/$algo/sketch.out
    done
done
