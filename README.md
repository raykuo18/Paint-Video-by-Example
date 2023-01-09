# Paint-Video-by-Example
## Requirements
A suitable [conda](https://conda.io/) environment named [`my_env`] can be created
and activated with:

```
./setup.sh [my_env]
```
## Pretrained Model
We provide the checkpoint ([Google Drive](https://drive.google.com/file/d/15QzaTWsvZonJcXsNv-ilMRCYaQLhzR_i/view?usp=share_link) | [Hugging Face](https://huggingface.co/Fantasy-Studio/Paint-by-Example/resolve/main/model.ckpt)) that is trained on [Open-Images](https://storage.googleapis.com/openimages/web/index.html) for 40 epochs. By default, we assume that the pretrained model is downloaded and saved to the directory `checkpoints`.

## Testing and Visualization
Simply run:
```
test.sh [input_image]
```
The output image will be  [input_image]_grid.png