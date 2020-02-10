import numpy as np
from mnist import MNIST
from scipy import sparse
from scipy import stats
from scipy.stats import truncnorm
import random
from scipy import stats

def readMNIST(dataset = "training", data_path = "./data/MNIST" ,debug=0):
	mndata = MNIST(data_path)

	if dataset is "training":
		data, labels = mndata.load_training()
	elif dataset is "testing":
		data, labels = mndata.load_testing()

	imgs=np.asfarray(data)
	imgs=imgs/255
	imgs=imgs+0.00001
	if debug==1:
		print(imgs[0])
		print(np.max(imgs[0]))
		print(np.shape(data))
		show_img(imgs[0],28,28)

	return imgs


def add_noise(trainData, noise_prob=0):
	data_shape=trainData.shape
	is_flipped =(np.random.rand(data_shape[0],data_shape[1])>noise_prob)
	noisy_trainData=np.multiply(trainData,is_flipped)
	return noisy_trainData


def genSparseMtx(rows,cols,prob):
	rvs = stats.norm(loc=0, scale=0.3).rvs
	S = sparse.random(rows,cols, density=prob, data_rvs=rvs,format='lil')
	return S.tocsr()
