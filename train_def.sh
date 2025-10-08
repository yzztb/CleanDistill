
for seed in 0 1 2 3

do
  python3 main_blend_def.py --seed ${seed}
  python3 main_badnet_sq_def.py --seed ${seed}
  python3 main_badnet_def.py --seed ${seed}
  python3 main_cl_def.py --seed ${seed}
  python3 main_sig_def.py --seed ${seed}
done

