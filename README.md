# MaskMorpher: Generating Video Transitions with Line Matches
### Mia Kan

MSc Computer Graphics, Vision, and Imaging Thesis, Sep 2025

## Results

<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input starting frame</td>
        <td>Input ending frame</td>
        <td>Inbetweening results</td>
    </tr>
  <tr>
  <td>
    <img src=example/real/003/00.png width="250">
  </td>
  <td>
    <img src=example/real/003/24.png width="250">
  </td>
  <td>
    <img src=example/real/003/out.gif width="250">
  </td>
  </tr>
  <tr>
  <td>
    <img src=example/real/002/00.png width="250">
  </td>
  <td>
    <img src=example/real/002/24.png width="250">
  </td>
  <td>
    <img src=example/real/002/out.gif width="250">
  </td>
  </tr>
  <tr>
  <td>
    <img src=example/animation/003/00.jpg width="250">
  </td>
  <td>
    <img src=example/animation/003/24.jpg width="250">
  </td>
  <td>
    <img src=example/animation/003/out.gif width="250">
  </td>
  </tr> 
  <tr>
  <td>
    <img src=example/animation/002/00.png width="250">
  </td>
  <td>
    <img src=example/animation/002/24.png width="250">
  </td>
  <td>
    <img src=example/animation/002/out.gif width="250">
  </td>
  </tr> 
</table>



## Startup Guide
#### 1. Setup conda environment

```bash
$ git clone https://github.com/kan32501/MaskMorpher.git
$ cd MaskMorpher
```

```bash
$ conda create -n MaskMorpher python=3.10.14
$ conda activate MaskMorpher
$ pip install -r requirements.txt
```

#### 2. Download required base models

1. Download the [Gluestick](https://github.com/cvg/GlueStick) weights and put them in `./models/resources/weights`.

```bash
$ wget https://github.com/cvg/GlueStick/releases/download/v0.1_arxiv/checkpoint_GlueStick_MD.tar -P models/resources/weights
```

2. Download the  [DWPose](https://github.com/IDEA-Research/DWPose) pretrained weights `dw-ll_ucoco_384.onnx` and `yolox_l.onnx` [here](https://drive.google.com/drive/folders/1Ftv-jR4R8VtnOyy38EVLRa0yLz0-BnUY?usp=sharing), then put them in `./checkpoints/dwpose`. 

3. Download the FCVG model [here](https://drive.google.com/drive/folders/1qIvr9WO8qk3NUdztxweTmexfkHt8oRDB?usp=sharing), put them in `./checkpoints`

#### 3. Run the inference script

Run. Decrease `-n` to < 10 if receing `CUDA error: out of memory`
```bash
$ python main.py -n 12 -va "./example/video/turtle" -vb "./example/video/tiger" -ma "./example/masks/turtle-60_mask.png" -mb "./example/masks/tiger-0_mask.png"
```

4. Download a model from SEARAFT model files [here](https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW). Then move it into `./models/searaft/models` . Then set `--SEARAFT_path` in `arguments.py` as `./models/SEARAFT/models/<MODEL-NAME>.pth`

See `arguments.py` to customize input parameters.

>   -n : number of intermediate frames, default=12\
>   -a : path to directory of frames in videoA\
>   -b : path to directory of frames in videoB\
>   -ma: path to mask for last frame in videoA\
>   -mb: path to mask for first frame in videoB\
>   --height : output frames height, default is 576\
>   --width: output frames width, default is 1024\


## Acknowledgements

Thanks for the work of [Frame-wise Conditions-driven Video Generation](https://github.com/Tian-one/FCVG) &  [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT?tab=readme-ov-file). Our code is based on the implementation of them.
