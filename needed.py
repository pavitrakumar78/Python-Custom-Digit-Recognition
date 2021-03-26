from PIL import Image
from numpy import asarray


def imresize(arr,size):
	img=Image.fromarray(arr)
	img=img.resize(size)
	return asarray(img)
