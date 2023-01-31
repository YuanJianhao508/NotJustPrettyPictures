algorithms="ERM OURS"
trails="Trail1"
loss="ce"

# For dumb Stable Diffusion
linkpath="./link/pacs_sd_dumb_link.json"

for algo in $algorithms; do
    for out in $trails; do
    echo "$algo $out"
    python -u shell_single_train.py -d=photo -g=1 -a=$algo -ic=$loss -lp=$linkpath > ./results/$out/$algo/photo.out &
    python -u shell_single_train.py -d=cartoon -g=2 -a=$algo -ic=$loss -lp=$linkpath > ./results/$out/$algo/cartoon.out &
    python -u shell_single_train.py -d=art_painting -g=3 -a=$algo -ic=$loss -lp=$linkpath > ./results/$out/$algo/art_painting.out &
    python -u shell_single_train.py -d=sketch -g=0 -a=$algo -ic=$loss -lp=$linkpath > ./results/$out/$algo/sketch.out
    done
done
