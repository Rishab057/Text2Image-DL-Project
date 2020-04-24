import os
import pickle
import csv

def load_filenames(data_dir):
	filepath = os.path.join(data_dir, 'filenames_old.pickle')
	with open(filepath, 'rb') as f:
		filenames = pickle.load(f)
	print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
	return filenames

def main():
	x = load_filenames('../data/birds/train/')
	print(x); exit(0)
	#x = [i.split('_')[-1] for i in x]
	
	#folder = '../data/birds/CUB_200_2011/*/'
	l = []
	with open("../data/birds/CUB_200_2011/images.txt") as f:
		c = csv.reader(f, delimiter=' ', skipinitialspace=True)
		for line in c:
			l.append(line[1])
	#print(l)
	y =[]
	cnt=0
	for i in x:
		#print(cnt,"/",len(x))
		cnt+=1
		if i+'.jpg' in l:
			y.append(i+'.jpg')

	print(len(y)); print(y)

	pickle_out = open("../data/birds/train/filenames.pickle","wb")
	pickle.dump(y, pickle_out)

main()