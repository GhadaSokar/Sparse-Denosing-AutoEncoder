import numpy as np
from utils import *
from scipy import sparse
import math
import time
import random
class DAE(object):


	def __init__(self, input_size, hidden_size,activation_type ,lr, momtem,dropout,eta,lamda,is_Denoise=False,noise_prob=0.25,is_saprse_Evol=False,run_highDim=False,debug=0):
	    
		self.run_highDim=run_highDim
		#network paramters
		if(run_highDim==True):
			self.input_size=250000
		else:
			self.input_size = input_size        # number of input units
		self.hidden_size = hidden_size      # number of hidden units
		
		#training paramters
		self.activation_type=activation_type #hidden layer activation
		self.lr=lr                          # learning rate
		self.dropout=dropout
		# momuntem
		self.momtem = momtem                # momuntem parameter
		self.prev_grad_W1=sparse.lil_matrix((self.input_size, self.hidden_size)).tocsr()
		self.prev_grad_W2=sparse.lil_matrix((self.hidden_size,self.input_size)).tocsr()
		self.prev_grad_b1 = np.zeros(self.hidden_size)
		self.prev_grad_b2 = np.zeros(self.input_size)

		# DAE or AE
		self.is_Denoise=is_Denoise          # if True DAE else AE
		self.noise_prob=noise_prob

		#Sparse evolution algo paramters
		self.is_saprse_Evol=is_saprse_Evol
		self.eta=eta                        # param for prob of inital sparse weights
		self.lamda=lamda                    # prob of Evolu removed weights        
		self.W1_rmv_num=0
		self.W2_rmv_num=0

		if is_saprse_Evol:
			self.W_P_density= (eta*(self.hidden_size+self.input_size))*1.0/(self.hidden_size*self.input_size)
		else:
			self.W_P_density=1

		self.W1=genSparseMtx(self.input_size,self.hidden_size,self.W_P_density)
		self.W2=genSparseMtx(self.hidden_size,self.input_size,self.W_P_density)
		self.b1 = np.zeros(self.hidden_size)
		self.b2 = np.zeros(self.input_size)
		if debug==1:
			print("intial connection prob%f Eta %i" %(self.W_P_density, eta))
			print("count of non_zero elements",self.W1.count_nonzero())
			print("inital Weights", self.W1.todense())


	def train(self, train_data, val_data, max_epoch=100, tol=1e-3,debug=0):

		data_shape= train_data.shape
		nsample=data_shape[0]

		## training
		(train_loss, val_loss) = ([], [])
		(diff_loss, nEpoch) = (tol+1, 0)	
		LastEpoch=False
		
		while (nEpoch < max_epoch): #and (diff_loss > tol):
			print("Epoch #%i" %(nEpoch))
			start_epoch_time=time.time()
			if self.is_Denoise:
				#Mask bits
				noisy_traindata = add_noise(train_data, self.noise_prob)
			else:
				noisy_traindata = train_data 
						
			## Shuffle dataset
			it=0
			idx_order = np.random.permutation(nsample)
			for idx in idx_order:
				#if(it%1000==0):
				#	print(it)
				## update gradient using one sample (sochastic)
				sample = train_data[idx, :]
				noisy_sample = noisy_traindata[idx, :]
				if(self.run_highDim):
					sample = np.reshape(np.resize(np.reshape(sample,(28,28)),(500,500)),(250000)) 
					noisy_sample = np.reshape(np.resize(np.reshape(noisy_sample,(28,28)),(500,500)),(250000))
				self.gradientDescent(sample, noisy_sample,nEpoch)				
				it=it+1
			print("Weight updated")	
			## cal cross entropy loss each 5 epochs
			if(nEpoch%1==0):
				train_loss += [self.cross_entropy_loss(train_data)]
				val_loss += [self.cross_entropy_loss(val_data)]
        
			## stopping criteria
			if (nEpoch > 1):
			 	diff_loss = abs(train_loss[nEpoch-1] - train_loss[nEpoch])

			if(nEpoch==max_epoch-1):# or diff_loss < tol):
				LastEpoch=True		 	
			if self.is_saprse_Evol:
				print("rmv small weights")
				#start=time.time()
				nRemoved_W1=self.rmv_small_weights(0) #0 for W1
				nRemoved_W2=self.rmv_small_weights(1) #1 for W2
				#end=time.time()
				#print("Time for connection removal: %i"%(end-start));
				if(not LastEpoch):
					print("add random weights")
					self.add_rnd_weights(nRemoved_W1,0) #0 for W1
					self.add_rnd_weights(nRemoved_W2,1) #1 for W2
					print("random weights added")

			end_epoch_time=time.time()
			total_epoch_time=end_epoch_time-start_epoch_time
			if(nEpoch%5==0):
				print("Epoch #%i time: %f training_loss:%f validation_loss:%f" %(nEpoch,total_epoch_time,train_loss[nEpoch],val_loss[nEpoch]))
			#Save Network Weights
			#if(nEpoch%10==0):
			#	sparse.save_npz("%i.weights1.txt"%(nEpoch), self.W1)
			#	sparse.save_npz("%i.weights2.txt"%(nEpoch), self.W2)
			#	np.savetxt("%i.b1.txt"%(nEpoch), self.b1,newline=" ")
			#	np.savetxt("%i.b2.txt"%(nEpoch), self.b2,newline=" ")
			nEpoch += 1
		sparse.save_npz("latest.weights1.txt", self.W1)
		sparse.save_npz("latest.weights2.txt", self.W2)
		np.savetxt("last.b1.txt", self.b1,newline=" ")
		np.savetxt("last.b2.txt", self.b2,newline=" ")
		return (train_loss, val_loss) 

	def test(self, test_data,debug=0):						
		test_loss = self.cross_entropy_loss(test_data)
		return test_loss     
	def sigmoid(self,x):
		return (1 / (1 + np.exp(-x)))

	def FeedForward(self, noisy_data,debug=0):
		
		hidden_op=self.sigmoid(self.b1 + noisy_data@self.W1)
		if(self.dropout>0):
			 hidden_op=np.multiply(hidden_op,(np.random.rand(self.hidden_size)>self.dropout)*1.0)
       
		predicted_x=self.sigmoid(self.b2 + hidden_op@self.W2)
			
		if debug==1:
			print("hidden op",hidden_op)
			print("x with sigm",predicted_x)
		return (hidden_op, predicted_x)

	def gradientDescent(self, data_x, noisy_data_x,nEpoch,debug=0):
		
		(hidden_op, pred_x) = self.FeedForward(noisy_data_x)

		delta_op = pred_x - data_x
		tmp_compHidden=1-hidden_op
		h_hcomp = np.multiply(hidden_op,tmp_compHidden)
		
		# calculate gradient for existing weights only		
		delta = (delta_op@self.W2.transpose()) * h_hcomp
		self.W1_mask=sparse.csr_matrix.copy(self.W1)
		self.W1_mask.data[:]=1
		self.W2_mask=sparse.csr_matrix.copy(self.W2)
		self.W2_mask.data[:]=1

		sparse_noisy_data_x=self.W1_mask.multiply(sparse.csr_matrix(noisy_data_x.reshape(self.input_size,1)))
		dw1=sparse_noisy_data_x.multiply(sparse.csr_matrix(delta))

		sparse_hidden_op=self.W2_mask.multiply(sparse.csr_matrix(hidden_op.reshape(self.hidden_size,1)))
		dw2=sparse_hidden_op.multiply(sparse.csr_matrix(delta_op))
				
		# Weight update
		self.prev_grad_W1=dw1*self.lr+self.prev_grad_W1*self.momtem
		self.prev_grad_W2=dw2*self.lr+self.prev_grad_W2*self.momtem
		self.prev_grad_b1=self.lr * delta+self.prev_grad_b1*self.momtem
		self.prev_grad_b2=self.lr * delta_op+self.prev_grad_b2*self.momtem

		self.W1-=self.prev_grad_W1
		self.W2-=self.prev_grad_W2
		self.b1-=self.prev_grad_b1
		self.b2-=self.prev_grad_b2

	def cross_entropy_loss(self, data):
		loss = 0
		data_shape= data.shape
		nsample=data_shape[0]
		for n in range(nsample):
			sample_x = data[n, :]
			if(self.run_highDim):
				sample_x = np.reshape(np.resize(np.reshape(sample_x,(28,28)),(500,500)),(250000)) 
			(hidden_op, pred_x) = self.FeedForward(sample_x)
			## loss = - (x log(p) + (1-x) log (1-p))
			loss -= np.sum(np.multiply(sample_x, np.ma.log(pred_x)).filled(0))/nsample
			loss -= np.sum(np.multiply((1-sample_x), np.ma.log(1-pred_x)).filled(0))/nsample
		return loss		


	def rmv_small_weights(self,W_idx,debug=0):		
		# remove fraction lamda of the smallest postive weights
		if(W_idx==0):
			pos_elems=sparse.csr_matrix(self.W1.multiply((self.W1>0)*1.0))
		else:
			pos_elems=sparse.csr_matrix(self.W2.multiply((self.W2>0)*1.0))
		rmv_pos_cnt=int(self.lamda*pos_elems.count_nonzero())
		if debug==1:
			print("Pos_elems before removal:",pos_elems.todense())
			print("rmv_cnt:",rmv_pos_cnt)
		if(rmv_pos_cnt>0):	
			rmv_pos_idx =pos_elems.data.argpartition(rmv_pos_cnt)[:rmv_pos_cnt]
			pos_elems.data[rmv_pos_idx] = 0
		if debug==1:
			print("removed idx",rmv_pos_idx)
			print("Pos_elem after removal:",pos_elems.todense())
		
		# remove fraction lamda of the largest negative weights
		if(W_idx==0):
			neg_elems=sparse.csr_matrix(self.W1.multiply((self.W1<0)*1.0))
		else:
			neg_elems=sparse.csr_matrix(self.W2.multiply((self.W2<0)*1.0))
		rmv_neg_cnt=int(self.lamda*neg_elems.count_nonzero())
		if(rmv_neg_cnt>0):
			rmv_neg_idx =neg_elems.data.argpartition(-rmv_neg_cnt)[-rmv_neg_cnt:]
			neg_elems.data[rmv_neg_idx] = 0

		if(W_idx==0):
			self.W1=sparse.csr_matrix(pos_elems+neg_elems)
			self.W1.eliminate_zeros()
		else:
			self.W2=sparse.csr_matrix(pos_elems+neg_elems)
			self.W2.eliminate_zeros()

		return rmv_pos_cnt+rmv_neg_cnt


	def add_rnd_weights(self,nRemoved,W_idx,debug=0):
		if W_idx==0:
			self.W_mask=sparse.csr_matrix.copy(self.W1)
			self.prev_grad_W1=self.prev_grad_W1.multiply(self.W_mask)
		else:
			self.W_mask=sparse.csr_matrix.copy(self.W2)
			self.prev_grad_W2=self.prev_grad_W2.multiply(self.W_mask)
		self.W_mask.data[:]=1
			
		while nRemoved>0:
			prob=self.W_P_density
			if W_idx==0: 
				random_add_pos = genSparseMtx(self.input_size,self.hidden_size,prob)
			else:
				random_add_pos = genSparseMtx(self.hidden_size,self.input_size,prob)
			random_add_pos_intersect= random_add_pos.multiply(self.W_mask)
			random_add_pos=random_add_pos-random_add_pos_intersect
			if(random_add_pos.nnz>nRemoved):
				random_add_pos.data[nRemoved:]=0
			if W_idx==0: 	
				self.W1=self.W1+random_add_pos
				self.W_mask=sparse.csr_matrix.copy(self.W1)
			else:
				self.W2=self.W2+random_add_pos
				self.W_mask=sparse.csr_matrix.copy(self.W2)	
			self.W_mask.data[:]=1
			nRemoved=nRemoved-random_add_pos.nnz
		
		if(debug==1):
			print("W after added connection:",self.W1)
	
		
def ExcuteDAE():
    ## Paramters ##
	input_size   = 784      # size of input vector
	hidden_size   = 1000     # size of hidden layer vector of first autoencoder
	
	activation_type="Sigmoid"
	lr=0.01
	momuntem=0.9
	dropout=0
	max_epoch = 50         # number of optimization iterations
	tol=1e-3

	is_saprse_Evol=True
	eta=20                 # initial sparse connection
	lamda=0.3              # precentage of removed connection

	is_Denoisy=True
	noise_prob=0.3

	run_highDim=False      # this paramter is used as proof of concept that the algo can be run on 500x500 on CPU only
						   # MNIST is rescaled to 500x500

	#Load MNIST training and testing
	train_data    = readMNIST("training")
	test_data  = readMNIST("testing")
	print("Dataset is loaded")
	n_validation_samples=12000
	val_data=train_data[0:n_validation_samples,:]
	train_data=train_data[n_validation_samples:train_data.shape[0],:]
	debug=0
	if debug==1:
			noise_prob=0.3
			momuntem=0.9
			run_highDim=False
			is_saprse_Evol=True
			max_epoch=20
			eta=20 	
			hidden_size   = 1000
			input_size   = 784
			lr=0.01
			val_data=train_data[0:100,:]
			train_data=train_data[100:200,:]
			

	#create DAE network  
	DAE_network = DAE(input_size, hidden_size,activation_type,lr,momuntem,dropout,eta,lamda,is_Denoisy,noise_prob,is_saprse_Evol,run_highDim)
	start=time.time()
	(train_loss,val_loss)=DAE_network.train(train_data,val_data, max_epoch, tol)
	end=time.time()
	print("Time spent ",end-start)
	print("train_loss:",train_loss)
	print("val_loss",val_loss)  
	print("Calculate loss on test set")
	test_loss=DAE_network.test(test_data)
	print("test_loss:",test_loss) 
	
ExcuteDAE()
