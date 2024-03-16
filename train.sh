## CIFAR100 classes 20
# python main_prompt.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#           --partition noniid-labeluni --n_parties 100 --beta 0.01 --cls_num 10 --device cuda:3 \
#          --batch-size 40 --comm_round 60  --test_round 50 --sample 0.05 --moment 0.5 --rho 0.9 --alg Final-SGPT-leaky\
#         --dataset cifar100 --lr 0.01 --epochs 5 --key_prompt 20 --avg_key --initial_g --leaky\
#         --share_blocks 0 1 2 3 --share_blocks_g 4 5 6 

### CIFAR100 classes 50
# python main_prompt.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
#           --partition noniid-labeluni --n_parties 100 --beta 0.01 --cls_num 50 --device cuda:1 \
#          --batch-size 40 --comm_round 60  --test_round 50 --sample 0.05 --moment 0.5 --rho 0.9 --alg Final-SGPT-relu-56-noint \
#         --dataset cifar100 --lr 0.01 --epochs 5 --key_prompt 20 --avg_key \
#         --share_blocks 0 1 2 3 4 --share_blocks_g  5 6 

# office dataset
python main_prompt.py  --pretrained_dir checkpoints/imagenet21k_ViT-B_16.npz --model_type ViT-B_16 \
          --n_parties 4 --cls_num 10 --device cuda:2 \
         --batch-size 50 --comm_round 40  --test_round 30 --sample 1 --rho 0.9 --alg Final-SGPT-leaky\
        --dataset office --lr 0.01 --epochs 5 --key_prompt 4 --avg_key --initial_g --moment 0 --leaky\
        --share_blocks 0 1 2 3 4 --share_blocks_g   5 6