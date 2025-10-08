
for seed in 0 1 2 3

do
  # python3 main_blend.py --seed ${seed}
  # python3 main_badnet.py --seed ${seed}
  # python3 main_badnet_sq.py --seed ${seed}
  python3 main_cl.py --seed ${seed}
  # python3 main_sig.py --seed ${seed}
done

