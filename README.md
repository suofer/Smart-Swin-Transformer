# Smart-Efficient-Transformer
  <strong>Smart Mask</strong>, the proposed method, replaces the Reverse Mask of the Swin Transformer for utilizing the different channel-wise features fully. The Smart Mask-based Attention is called <strong>S</strong>hifted <strong>S</strong>mart <strong>W</strong>indow <strong>M</strong>ulti-Head <strong>S</strong>elf-<strong>A</strong>ttention (<strong>SSW-MSA</strong>) module, and the Transformer embedded in <strong>SSW-MSA</strong> is regarded as <strong>Smart Swin Transformer</strong>. <strong>SSW-MSA</strong> improves the efficacy of the Self-Attention module by filtering out the associations that are irrelevant to the target task with a mask in one channel. Information about the distribution and the local environment is deduced from these dependencies. Our U-Shaped model based on the proposed methods is called <strong>S</strong>mart <strong>S</strong>win <strong>Trans</strong>former <strong>Net</strong>work (<strong>SSTrans-Net</strong>). <br />
# Datasets
[Synapse](https://github.com/microsoft/Swin-Transformer) <br />
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) <br />
# train
python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --max_epochs 300 --img_size 224 --base_lr 0.05 --batch_size 24 <br />
# test
python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_savenii --max_epoch 300 --base_lr 0.05 --img_size 224 --batch_size 24 <br />
# Contributions guidelines for this project 
[Swin Transformer](https://github.com/microsoft/Swin-Transformer) <br />
[Swin UNet](https://github.com/HuCaoFighting/Swin-Unet) <br />
