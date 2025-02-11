CUDA_VISIBLE_DEVICES=$1
python main.py --eval --resume "/home/common/SharedModelWeight/Mamba/vim_t_midclstok_76p1acc.pth" --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path "/home/common/SharedDataset/ImageNet-1k" --device cuda --batch-size 128 --quantization-config temp_PTQ_1 --calibration-size 256 --quantization

python main.py --eval --resume "/home/common/SharedModelWeight/Mamba/vim_t_midclstok_76p1acc.pth" --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path "/home/common/SharedDataset/ImageNet-1k" --device cuda --batch-size 128 --quantization-config temp_PTQ_2 --calibration-size 256 --quantization

python main.py --eval --resume "/home/common/SharedModelWeight/Mamba/vim_t_midclstok_76p1acc.pth" --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path "/home/common/SharedDataset/ImageNet-1k" --device cuda --batch-size 128 --quantization-config temp_PTQ_3 --calibration-size 256 --quantization

python main.py --eval --resume "/home/common/SharedModelWeight/Mamba/vim_t_midclstok_76p1acc.pth" --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path "/home/common/SharedDataset/ImageNet-1k" --device cuda --batch-size 128 --quantization-config temp_PTQ_4 --calibration-size 256 --quantization

# python main.py --eval --resume "/home/common/SharedModelWeight/Mamba/vim_t_midclstok_76p1acc.pth" --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path "/home/common/SharedDataset/ImageNet-1k" --device cuda --batch-size 128 --quantization-config temp_PTQ_5 --calibration-size 256 --quantization

# python main.py --eval --resume "/home/common/SharedModelWeight/Mamba/vim_t_midclstok_76p1acc.pth" --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path "/home/common/SharedDataset/ImageNet-1k" --device cuda --batch-size 128 --quantization-config temp_PTQ_6 --calibration-size 256 --quantization

# python main.py --eval --resume "/home/common/SharedModelWeight/Mamba/vim_t_midclstok_76p1acc.pth" --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path "/home/common/SharedDataset/ImageNet-1k" --device cuda --batch-size 128 --quantization-config temp_PTQ_7 --calibration-size 256 --quantization

# python main.py --eval --resume "/home/common/SharedModelWeight/Mamba/vim_t_midclstok_76p1acc.pth" --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path "/home/common/SharedDataset/ImageNet-1k" --device cuda --batch-size 128 --quantization-config temp_PTQ_8 --calibration-size 256 --quantization

# python main.py --eval --resume "/home/common/SharedModelWeight/Mamba/vim_t_midclstok_76p1acc.pth" --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path "/home/common/SharedDataset/ImageNet-1k" --device cuda --batch-size 128 --quantization-config temp_PTQ_9 --calibration-size 256 --quantization