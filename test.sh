for seed in 3

do
  # python3 main_blend.py --seed ${seed}
  # python3 main_ctrl.py --seed ${seed}
  # python3 main_trojan.py --seed ${seed}
  python3 main_badnet.py --seed ${seed}
  # python3 main_badnet_sq.py --seed ${seed}
  # python3 main_refool.py --seed ${seed}
  python3 main_sig.py --seed ${seed}
done