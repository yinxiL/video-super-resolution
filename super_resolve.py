from __future__ import print_function
import argparse
import glob
import torch
import os
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np

# ===========================================================
# Argument settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input', type=str, required=False, default='', help='input image to use')
parser.add_argument('--model', type=str, default='model_path.pth', help='model file to use')
parser.add_argument('--output', type=str, default='test.jpg', help='where to save the output image')
args = parser.parse_args()
print(args)


# ===========================================================
# input image setting
# ===========================================================
img_path = glob.glob(os.path.join(os.getcwd(),'dataset/Test/*.bmp'))
for i in range(len(img_path)):
	GPU_IN_USE = torch.cuda.is_available()
	img = Image.open(img_path[i]).convert('YCbCr')
	y, cb, cr = img.split()


	# ===========================================================
	# model import & setting
	# ===========================================================
	model = torch.load(os.path.join(os.getcwd(), 'output/checkpoint/',args.model), map_location=lambda storage, loc: storage)
	data = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
	if GPU_IN_USE:
		model.cuda()
		data = data.cuda()
		cudnn.benchmark = True
		model = torch.load(args.model)


	# ===========================================================
	# output and save image
	# ===========================================================
	out = model(data)
	out = out.cpu()
	out_img_y = out.data[0].numpy()
	out_img_y *= 255.0
	out_img_y = out_img_y.clip(0, 255)
	out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

	out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
	out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
	out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

	out_img.save('output/sample/%03d.jpg'%i)
	print('output image saved to %03d.jpg'%i)
