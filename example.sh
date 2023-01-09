python3 inference.py \
--plms --outdir results \
--config configs/v1.yaml \
--ckpt checkpoints/model.ckpt \
--image_path examples/image/example_$1.png \
--mask_path examples/mask/example_$1.png \
--reference_path examples/reference/example_$1.jpg \
--seed 321 \
--scale 5