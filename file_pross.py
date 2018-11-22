import sys
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb

def get_max_pad(I,TI,R,C,M,rr,cc):	
	lt = 0; lb = 0;	ll = 0;	lr = 0;	
	for i in range(0,len(R)):		
		if( (R[i]-M < 0) ):
			if(np.abs(R[i]-M) > lt):
				lt = np.abs(R[i]-M);				
		if( (C[i]-M < 0)):
			if(np.abs(C[i]-M) > ll):
				ll = np.abs(C[i]-M);				
		if(R[i]+M > I.shape[0]): 
			if(np.abs(I.shape[0] - (R[i]+M) + 1) > lb):
				lb = np.abs(I.shape[0] - (R[i]+M) + 1);
		if(C[i]+M > I.shape[1]):
			if(np.abs(I.shape[1] - (C[i]+M) + 1) > lr):
				lr = np.abs(I.shape[1] - (C[i]+M) + 1);					
	I = cv2.copyMakeBorder(I,lt,lb,ll,lr,cv2.BORDER_CONSTANT,0);
	TI = cv2.copyMakeBorder(TI,lt,lb,ll,lr,cv2.BORDER_CONSTANT,0);
	R = list(np.asarray(R) + lt);
	C = list(np.asarray(C) + ll);
	rr = list(np.asarray(rr) + lt);
	cc = list(np.asarray(cc) + ll);
	return I,TI,R,C,rr,cc,lt,ll;
def add_rows_cols(I,N):
	lr = np.abs(N - I.shape[0]);
	lc = np.abs(N - I.shape[1]);
	I = cv2.copyMakeBorder(I,0,lr,lc,0,cv2.BORDER_CONSTANT,0);
	return I;
def get_data(fname,mode,N,reshap):
	if(N%2!=0):
		N = N-1;
	if (N <= 0):
		print('Please make sure window size >=2, window size too small, exiting');
		sys.exit(1);
	print('\nProcessing:',fname,'\t');
	print('Window Size:', N+1,'x',N+1,'\n');
	def wind_new(TI,I,r,c,R,C,mode,K,window_size):
		M = int(window_size/2);
		final_i = None;	
		for i,g in enumerate(R):
			if (R[i] == r) & (C[i] == c):
				final_i = i;
				break;
		if(final_i == None):
			print('indexing error, skipping');
			return I[1:32,1:32],0,[],[];
		lor = final_i-M;
		upp = final_i+M;
		
		if(lor < 0):
			lor = 0;
		
		newr = R[lor:upp];
		newc = C[lor:upp];
		colo = list(np.random.choice(range(256),size=3));
		while(colo==[255,255,255]):
			colo = list(np.random.choice(range(256),size=3));
		K[newr,newc,:] = colo;
		
		nr = []; nc = [];
		nr = newr[round(len(newr)/2)];
		nc = newc[round(len(newr)/2)];
		
		if mode == 'stroke':
			mask = np.zeros(TI.shape,'uint8');
			mask[newr,newc] = 255;
			#cv2.imwrite('/home/vonnegut/Keras/res/'+ str(globi[0]) + '_segment.jpeg',mask[r-M:r+M,c-M:c+M]);
			#globi[0] = globi[0] + 1;
			#if(mask[r-M:r+M,c-M:c+M].shape[0] == 99):
				#pdb.set_trace();
			return mask[r-M:r+M,c-M:c+M],1,nr,nc;
		elif mode == 'bgr':
			mask = np.zeros(I.shape,'bool');
			mask[newr,newc,:] = True;
			l = np.zeros(I.shape,I.dtype) + 255;
			np.copyto(l,I,'same_kind',mask);
			return l[r-M:r+M,c-M:c+M],1,nr,nc;
		else:
			print("Some error!")
			return I[1:32,1:32],0,[],[];
	I = cv2.imread(fname);
	IG = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY);
	th2,TG = cv2.threshold(IG,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	r = [];
	c = [];	
	for i in range(0,TG.shape[1]):
		if(sum(TG[:,i] != 0) != 0):
			tem = list(np.where(TG[:,i] != 0)[0]);
			r+= tem;
			c += [i]*len(tem);
	rr = [];
	cc = [];
	for i,C in enumerate(c):
		if(i%N == 0):
			rr.append(r[i]);
			cc.append(C);
	# pdb.set_trace();		
	I,TG,rr,cc,r,c,lt,ll = get_max_pad(I,TG,rr,cc,int(N/2),r,c);
	K = np.zeros(I.shape,I.dtype) + [255,255,255];
	L = np.zeros(I.shape,'uint8');
	L[rr,cc,:] = [255,255,255];
	NR = [];
	NC = [];
	
	# pdb.set_trace();
	if mode == 'stroke':
		fin_arr = [];
		for i,m in enumerate(rr):
			iii,mm,nr,nc = wind_new(TG,I,m,cc[i],r,c,mode,K,N);
			nr = nr - lt;
			nc = nc - ll;
			
			if((iii.shape[0] != N) | (iii.shape[1] != N)):
				iii = add_rows_cols(iii,N);				
			iii = cv2.resize(iii,(reshap,reshap));
			#iii = np.zeros((32,32)); mm = 1;
			if(mm == 0):
				continue;
			if i == 0:
				fin_arr = np.expand_dims(iii,2);
				NR.append(nr);
				NC.append(nc);
				continue;
			if ( (i > 0) & (len(fin_arr) == 0)):
				fin_arr = np.expand_dims(iii,2);
				continue;
			#if(iii.shape[0] == 0):
				# pdb.set_trace();
			fin_arr = np.concatenate((fin_arr,np.expand_dims(iii,2)),2);
			NR.append(nr);
			NC.append(nc);
		cv2.imwrite('strokes'+ str(N) +'_.jpg',K);	
		fin_arr = np.rollaxis(fin_arr,2,0);
		fin_arr = np.expand_dims(fin_arr,3);
		return fin_arr/255,NR,NC;
	if mode == 'bgr':
		fin_arr = [];
		for i,m in enumerate(rr):
			iii,mm,nr,nc = wind_new(TG,I,m,cc[i],r,c,mode,K,N);
			nr = nr - lt;
			nc = nc - ll;
			
			if((iii.shape[0] != N) | (iii.shape[1] != N)):
				iii = add_rows_cols(iii,N);	
			iii = cv2.resize(iii,(reshap,reshap));
			if(mm == 0):
				continue;
			if i == 0:
				fin_arr = np.expand_dims(iii,3);
				NR.append(nr);
				NC.append(nc);
				continue;
			if ((i > 0) & (len(fin_arr) == 0)):
				fin_arr = np.expand_dims(iii,3);
				continue;		
			fin_arr = np.concatenate((fin_arr,np.expand_dims(iii,3)),3);
			print(i);
			NR.append(nr);
			NC.append(nc);			
		cv2.imwrite('strokes'+ str(N) +'_.jpg',K);
		fin_arr = np.rollaxis(fin_arr,3,0);
		return fin_arr/255,NR,NC;
def draw_labels(labs,P,NR,NC):
	I = cv2.imread(P);
	fnam, ext = os.path.splitext(os.path.basename(P));
	for i,m in enumerate(labs):
		cv2.putText (I,str(ord(m)),(NC[i],NR[i]),1, 0.8, (0,0,255),1,2);
	cv2.imwrite(fnam + '_labels.jpeg',I);
	
			
data,nr,nc = get_data('plots/0019.jpg','bgr',100,32);
#labs = list(map(chr,range(0,len(nr))));
symbols = ['а','б','в','г','д','е','ё','ж','з','и','й','к','л','м','н','о','п','р','с','т','у','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я']
labs = ['б','в','г','д','е','ё','ж']
nr = [40,100,38,28,93,49,28];
nc = [100,34,83,92,87,24,59];
draw_labels(labs,'plots/0019.jpg',nr,nc)

"""
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(data[1,:,:].reshape(1,32,32,1));

def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='hsv')
            activation_index += 1;

plt.imshow(data[1,:,:].reshape(32,32))
display_activation(activations,4,8,1);
plt.show();
"""

