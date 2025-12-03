python evaluate.py --protocol_files in_the_wild sonar fake_or_real asvspoof_2019 \
                   --batch_size 128 \
                    --models tcm_add xlsr_sls nes2net_x wav2vec2_aasist \
                    --fix_length \
                    --augment_type noise \
                    --num_workers 8

python evaluate.py --protocol_files in_the_wild sonar fake_or_real asvspoof_2019 \
                   --batch_size 128 \
                    --models tcm_add xlsr_sls nes2net_x wav2vec2_aasist \
                    --fix_length \
                    --augment_type perturb \
                    --num_workers 8