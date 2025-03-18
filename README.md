# TF-TI2I: Training-Free Text-and-Image-to-Image Generation via Multi-Modal Implicit-Context Learning in Text-to-Image Models

### Augment your T2I models with arbitrary number of images as references in a Training-Free manner 🔥🔥🔥

![paperteaser](./demo_out/paper_teaser.jpg)
### Text-and-Image-to-Image Generation: Generating both prompt-following📃 and references-following🖼️ image.
### Key Features of TF-TI2Iy:
* ✅ **Training-Free Text-and-Image-to-Image (TF-TI2I)**: State-of-the-art method for text-and-image-to-image generation without training. 
* ✅ **Imaplicit Context Learning**: Disconver the Implicit Context Learning capability of textual tokens to extract visual details from vision tokens. 
* ✅ **Contextual Tokens Sharing (CTS)**: By sharing contextual tokens, we can effectively aggregate the visual informations from multiple images.
* ✅ **References Contextual Masking (RCM)**: To reduce the confliction between images, we propose References Contextual Masking to restrict the information learned from references.
* ✅ **Winnter-Takes-All (WTA)**: We propose Winner-Takes-All module to address the distribution shift and features confliction, by assigning each vision tokens with most sailent contextual tokens.
## 🦦0. Preparation
```
conda create -n tfti2i python=3.9 -y
conda activate tfti2i
pip install -r requirements.txt
```
## 🐾1. Run

With the environment installed, directly run the following script, to interactively utilizing the FreeCond framework
### 1.1 Gradio Interface
* 👍User-friendly, direct image generation
* 👎Limit controlbilty

```
# gradio app support
python freecond_app.py
```
### 1.2 Jupyternote Book
* 👍Intuitive, flaxible
* 👎Not suit for large scale evaluation

```
# ipynb support
TF-TI2I.ipynb
```
### 1.3 Python Script
* 👍Suit for evaluation, parameter searching
* 👎Lacking flaxibilty

## 🤓2. For Research
### 👀2-1. Visualization
![visualization](./demo_out/self_attn_multi.png)
![visualization2](./demo_out/CI_visualization.png)

Due to code optimizations, certain random seed-related functionalities may behave differently compared to our development version 😢. As a result, some outputs might slightly differ from the results reported in our research paper.
```
# 👀Visualization
self_attention_visualization.ipynb
CI_visualization.ipynb
```
The *self_attention_visualization* is designed for better understanding the feature distribution of masked area (How much from inner mask area and how much from outer mask area⚖️)
This repository includes two Jupyter notebooks for visualizing key aspects of the inpainting process:

#### `self_attention_visualization.ipynb`
This notebook provides insights into the feature distribution within the masked area during inpainting.
- Specifically, it helps visualize, the proportion of attention originating from the inner mask area versus the outer mask area. ⚖️

#### Key Observation:
- Successful inpainting is often associated with significantly stronger self-attention within the inner mask region.
- This aligns with the intuitive expectation that the generated object should focus more on itself than on the background.

#### `CI_visualization.ipynb`
This notebook introduces a **Channel Influence Indicator**, which helps identify the role of latent mask inputs in the cross-attention layers during training.

#### Key Insights:
- Certain feature channels become highly adapted to mask inputs, amplifying cross-attention within the inner mask area.
- This selective amplification enhances the model's ability to apply prompt instructions specifically to the masked region.

### 📏2-2. Metrics evaluation
As mentioned earlier, this repository integrates existing state-of-the-art (SOTA) text-guided inpainting methods. We use this repository to evaluate these methods under various formulations of **FreeCond Control**, as detailed in our research paper, particularly in the appendix section.

Our evaluation metrics are adapted from [BrushBench](https://github.com/TencentARC/BrushNet) and enhanced with a novel **IoU score**. This score automatically calculates the mask-fitting degree of the generated object, providing a more comprehensive assessment of inpainting performance.

The included metrics are categorized as follows:

#### 1. **Image Quality**
- **IR (Image Reward)**  
- **HPS (Human Perceptive Score)**  
- **AS (Aesthetic Score)**  

#### 2. **Background Preservation**
- **LPIPS (Learned Perceptual Image Patch Similarity)**  
- **MSE (Mean Squared Error)**  
- **PSNR (Peak Signal-to-Noise Ratio)**  

#### 3. **Instruction Following**
- **CLIP (Contrastive Language–Image Pretraining)**  
- **IoU Score (Intersection over Union by SAM)**  

These metrics collectively evaluate the performance of the inpainting methods across key aspects, ensuring a thorough comparison and analysis.
```
# 📏Metrics evaluation
freecond_evaluation.py \
--method "sd" \
# Currently support ["sd", "cn", "hdp", "pp", "bn"]. Defaults to "sd". \
--variant "sd15" \
# (optional) Mainly designed for SDs currently support ["sd15", "sd2", "sdxl", "ds8"]. Defaults to "sd15". \
--data_dir "./data/demo_FCIBench" \
# Root directory for data_csv and corresponding image sources. \
--data_csv "FCinpaint_bench_info.csv" \
# CSV file that specifies the path of image sources and corresponding prompt instructions. \
--inf_step=50 \
# Inference step \
--tfc=25 \
# Freecond_control time: uses setting_1 before tfc, setting_2 after tfc \
--fg_1=1 \
# The inner mask scale before tfc (default: 1) \
--fg_2=1.5 \
# The inner mask scale after tfc (default: 1) \
--bg_1=0 \
# The outer mask scale before tfc (default: 0) \
--bg_2=0.2 \
# The outer mask scale after tfc (default: 0) \
--qth=24 \
# The high-frequency threshold (default: 32). Threshold 32 corresponds to the highest frequency component of 64x64 VAE latent space. \
--hq_1=0 \
# The scale of high-frequency component before tfc (default: 1) \
--hq_2=1
# The scale of high-frequency component after tfc (default: 1)
```
The implementation of FCinpaint_bench_info.csv should be formulated as following
```
prompt,image,mask
"A fluffy panda juggling teacups, in watercolor style",FC_images/img_0_0.jpg,FC_masks/mask_0_0.png
"A fluffy panda juggling teacups, in watercolor style",FC_images/img_0_1.jpg,FC_masks/mask_0_1.png
"A fluffy panda juggling teacups, in watercolor style",FC_images/img_0_2.jpg,FC_masks/mask_0_2.png
"A golden retriever wearing astronaut gear, in cyberpunk style",FC_images/img_1_0.jpg,FC_masks/mask_1_0.png
"A golden retriever wearing astronaut gear, in cyberpunk style",FC_images/img_1_1.jpg,FC_masks/mask_1_1.png
"A golden retriever wearing astronaut gear, in cyberpunk style",FC_images/img_1_2.jpg,FC_masks/mask_1_2.png
```
