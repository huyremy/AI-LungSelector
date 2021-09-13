import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
from lungs_segmentation.pre_trained_models import create_model
import lungs_segmentation.inference as inference

model = create_model("resnet34")
device = torch.device('cpu')
model = model.to(device)

plt.figure(figsize=(20,40))
plt.subplot(1,1,1)
image, mask = inference.inference(model,'koxuong.png', 0.2)
img = inference.img_with_masks(image, [mask[0], mask[1]], alpha = 0.1)
imageio.imwrite('test.png', img)
