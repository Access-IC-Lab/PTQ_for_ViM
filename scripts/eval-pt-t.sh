CUDA_VISIBLE_DEVICES=$1 python main.py --eval --resume "/home/common/SharedModelWeight/Mamba/vim_t_midclstok_76p1acc.pth" --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path "/home/common/SharedDataset/ImageNet-1k" --device cuda --batch-size 256