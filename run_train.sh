 CUDA_VISIBLE_DEVICES=0,1 python main.py \
                          -b 30 \
                          --epochs 250 \
                          --lr 0.003 \
                          --snapshot-fname-prefix exps/snapshots/ppn \
                          #2>&1 | tee exps/logs/ppn.log \
                            
                            
