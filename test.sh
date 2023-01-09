python3 inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--raw_image $1 \
# --image_path img.jpg \
# --mask_path mask.jpg \
# --reference_path man.png \
# --seed 10 \
--scale 5

# python3 inference.py \
# --plms --outdir results \
# --config configs/v1.yaml \
# --ckpt checkpoints/model.ckpt \
# --image_path examples/image/example_1.png \
# --mask_path examples/mask/example_1.png \
# --reference_path examples/reference/example_1.jpg \
# --seed 321 \
# --scale 5