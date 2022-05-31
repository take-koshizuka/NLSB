# Neural Lagrangian Schrödinger bridge

# Train

1. OU-sde data (artificial)
   ```
   python3 train.py -c config/ou-sde/[config path] -d [save dir]
   ```
2. sc-RNA data (real)
   ```
   python3 train.py -c config/rna/[config path] -d [save dir]
   ```
3. uniform (artificial)
   ```
   python3 train.py -c config/uniform/[config path] -d [save dir]
   ```

# Evaluate

## OU-sde

1. MDD
   ```
   python3 eval_mdd.py -c config/eval_mdd.json -p [checkpoint path]
   ```
2. CDD
   ```
   python3 eval_cdd.py -c config/eval_cdd.json -p [checkpoint path]
   ```

3. plot
   ```
   python3 plot_ou-sde.py -c config/plot.json -p [checkpoint path]
   ```

## sc-RNA

1. MDD
   ```
   python3 eval_mdd.py -c config/eval_mdd.json -p [checkpoint path]
   ```

2. CDD
   ```
   python3 eval_lmt_traj.py -c config/eval_lmt_traj.json -p [ckpt] -pl [ckpt2]
   ```

3. plot on 2D-PCA space
   ```
   python3 plot_rna.py -c config/plot.json -p [ckpt] -o [out_dir]
   ```



