import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import ast
import cv2
import os
from glob import glob
import re
import pandas as pd
from PIL import Image, ImageDraw 
from dask import bag
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import model_from_json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_data(fname,mode):
	print('Processing:',fname,'\n');
	import cv2;
	import numpy as np
	import matplotlib.pyplot as plt
	def wind(I,r,c):
		if((c+16 > I.shape[1]) | (c-16 < 0) | (r+16 > I.shape[0]) | (r-16 < 0) ):
			print("error out of bounds, skipping");
			return I[1:32,1:32],0;
		else:
			ll = I[r-16:r+16,c-16:c+16];
			if((ll.shape[0] == 32)&(ll.shape[1] == 32)):
				return ll,1;
			else:
				print("misshapen, skipping");
				return ll,0;
	def wind_new(TI,I,r,c,R,C,mode):
		if((c+16 > I.shape[1]) | (c-16 < 0) | (r+16 > I.shape[0]) | (r-16 < 0) ):
			print("error out of bounds, skipping");
			return I[1:32,1:32],0;
		ll = I[r-16:r+16,c-16:c+16];
		if((ll.shape[0] != 32)|(ll.shape[1] != 32)):
				return ll,0;
		final_i = None;	
		for i,g in enumerate(R):
			if (R[i] == r) & (C[i] == c):
				final_i = i;
				break;
		if(final_i == None):
			print('indexing error, skipping');
			return I[1:32,1:32],0;
		newr = R[final_i-16:final_i+16];
		newc = C[final_i-16:final_i+16];
		mask = np.zeros((TI.shape[0],TI.shape[1]),'uint8');
		mask[newr,newc] = 255;
		if mode == 'stroke':
			return mask[r-16:r+16,c-16:c+16].copy(),1;
		elif mode == 'bgr':
			mask = np.zeros((TI.shape[0],TI.shape[1],3),'bool');
			mask[newr,newc,:] = True;
			l = np.zeros(I.shape,I.dtype) + 255;
			np.copyto(l,I,'same_kind',mask);
			return l[r-16:r+16,c-16:c+16],1;
		else:
			print("Some error!")
			return I[1:32,1:32],0;
	I = cv2.imread(fname);
	IG = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY);
	th2,TG = cv2.threshold(IG,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	L = cv2.findNonZero(TG)
	c = L[:,0,0];
	r = L[:,0,1];
	rr = [];
	cc = [];
	for i,C in enumerate(c):
		if(i%64 == 0):
			rr.append(r[i]);
			cc.append(C);
	if mode == 'stroke':
		fin_arr = [];
		for i,m in enumerate(rr):
			iii,mm = wind_new(TG,I,m,cc[i],r,c,mode);
			print(iii.shape)
			cv2.waitKey();
			if(mm == 0):
				continue;
			if i == 0:
				fin_arr = np.expand_dims(iii,2);
				continue;
			if ( (i > 0) & (len(fin_arr) == 0)):
				fin_arr = np.expand_dims(iii,2);
				continue;
			fin_arr = np.concatenate((fin_arr,np.expand_dims(iii,2)),2);
		fin_arr = np.reshape(fin_arr,(fin_arr.shape[2],32,32,1));
		return fin_arr/255;
	if mode == 'bgr':
		fin_arr = [];
		for i,m in enumerate(rr):
			iii,mm = wind_new(TG,I,m,cc[i],r,c,mode);
			if(mm == 0):
				continue;
			if i == 0:
				fin_arr = np.expand_dims(iii,3);
				continue;
			if ((i > 0) & (len(fin_arr) == 0)):
				fin_arr = np.expand_dims(iii,3);
				continue;
			fin_arr = np.concatenate((fin_arr,np.expand_dims(iii,3)),3);
		fin_arr = np.reshape(fin_arr,(fin_arr.shape[3],32,32,3));
		return fin_arr/255;

dat = get_data('/home/vonnegut/Keras/plots/0007.jpg','stroke');

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

dname = "~/Keras/plots/";
for fname in os.listdir(dname):
	ttvlist = []
	dat = get_data(dname+fname,'stroke')
	testpreds = model.predict(dat, verbose=1)
	ttvs = np.argsort(-testpreds)[:, 0:10]  # top 10
	preds_df = pd.DataFrame({'first': ttvs[:,0], 'second': ttvs[:,1], 'third': ttvs[:,2], 'fourth': ttvs[:,3], 'fifth': ttvs[:,4], 'sixth': ttvs[:,5], 'seventh': ttvs[:,6], 'eighth': ttvs[:,7], 'ninth': ttvs[:,8], 'tenth': ttvs[:,9]});
	preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']+ " " + preds_df['fourth']+ " " + preds_df['fifth']+ " " + preds_df['sixth']+ " " + preds_df['seventh']+ " " + preds_df['eighth']+ " " + preds_df['ninth']+ " " + preds_df['tenth'];
	print(preds_df);
