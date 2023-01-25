# algorithms="ERM OURS AugMix RandAugment CutOut MixUp CutMix RSC MEADA ACVC PixMix SN SRN L2D"
algorithms="ERM OURS"
trails="Test_adap"

for algo in $algorithms; do
    for out in $trails; do
    echo "$algo $out"
    python -u shell_test.py -d=photo -g=0 -a=$algo > ./results/$out/$algo/photo.out &
    python -u shell_test.py -d=cartoon -g=1 -a=$algo> ./results/$out/$algo/cartoon.out &
    python -u shell_test.py -d=art_painting -g=0 -a=$algo > ./results/$out/$algo/art_painting.out &
    python -u shell_test.py -d=sketch -g=2 -a=$algo > ./results/$out/$algo/sketch.out
    done
done

# python -u shell_test.py -d=photo -g=2 -a=OURS

# Texture
# for algo in $algorithms; do
#     for out in $trails; do
#     echo "$algo $out"
#     python -u shell_test.py -d=original -g=1 -a=$algo > ./results_Texture/$out/$algo/original.out
#     done
# done

# python -u shell_test.py -d=original -g=0 -a=ERM
