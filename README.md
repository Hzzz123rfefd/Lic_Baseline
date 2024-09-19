# Lic_Baseline
Training and inference of models related to learnning image compress network

## Installation
```bash
conda create -n lic python=3.10
conda activate lic
git clone https://github.com/Hzzz123rfefd/Lic_Baseline.git
cd lic
pip install -r requirements.txt
pip install -e .
```
## Usage
### Model Config
1、You need to first configure the __init__ parameter of the image compression model in <cof> dir.
2 、You need to register your model in <compressai/zoo/__init__.py> and name it {"name of your model", "model class"}

### Trainning
An examplary training script with a rate-distortion loss is provided in train.py.

1、prepare trainning data,using npy array,size of array is [data_size, image_channel, image_height, image_weight]. If you want to directly process image data, please convert it to npy format

* sh
```bash
python train.py --model_name {name of your model} \
                --datasets_path {path of data} \
                --data_size {image number of dataset}  \
                --image_channel 3  \
                --image_weight 512  \
                --image_height 512  \
                --lamda 0.0001  \
                --batch_size 2  \
                --lr 0.0001  \
                --epoch 1000  \
                --clip_max_norm 0.5  \
                --factor 0.3  \
                --patience 8  \
                --save_model_dir {fold dir to save model}  \
                --device cuda
```
* example
```bash
python train.py --model_name stf
                --datasets_path "data/shuffled_data_512.npy" \
                --labels_path "data/shuffled_labels_512.npy"  \
                --data_size 600  \
                --image_channel 3  \
                --image_weight 512  \
                --image_height 512  \
                --lamda 0.0001  \
                --batch_size 2  \
                --lr 0.0001  \
                --epoch 1000  \
                --clip_max_norm 0.5  \
                --factor 0.3  \
                --patience 8  \
                --save_model_dir "./model/"  \
                --device cuda
```
### Image Codec
#### compress
Now,you can compress your raw image data through LIC
TODO: 
1、Currently, only PNG files are supported for reading, and in the future, the raw files of images should be directly read
2、After training a sufficient number of bitrate models, the user should specify the compression quality and automatically select the compression model without specifying the model path
* sh
```bash
python example/codec.py encode \
            --model_name {name of your model} \
            --image_path {path of your image (only png format now)} \ 
            --output_path_y {path of bitstream(y.bin)} \
            --output_path_z {path of bitstream(z.bin)} \ 
            --pre_model_path {path of model}} \
            --device "cpu"
```
* example
```bash
python example/codec.py encode \
            --model_name stf  \
            --image_path "data/image.png" \
            --output_path_y "data/y.bin" \
            --output_path_z "data/z.bin" \
            --pre_model_path "model/stf/lambda = 0.0001/model.pth" \
            --device "cpu"
```
#### decompress
Now, you can extract your images from the binary stream
```bash
python example/codec.py decode \
            --model_name {name of your model} \
            --path_y "data/y.bin" \
            --path_z "data/z.bin" \
            --output_image_path "data/re.png" \
            --pre_model_path "model/stf/lambda = 0.0001/model.pth" \
            --device "cpu"
```
* example
```bash
python example/codec.py decode \
            --model_name stf  \
            --path_y {path of bitstream(y.bin)} \
            --path_z {path of bitstream(z.bin)} \
            --output_image_path {path of image to save} \
            --pre_model_path {path of model} \
            --device "cpu"
```
