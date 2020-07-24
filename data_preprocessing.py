import os
import pickle

def load_filenames(data_dir):
	filepath = os.path.join(data_dir, 'filenames_old.pickle')
	with open(filepath, 'rb') as f:
		filenames = pickle.load(f)
	print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
	return filenames

def main():
	x = load_filenames('../data/coco/train/')
	x = [i.split('_')[-1] for i in x]
	# print(x)
	folder = '../data/coco/images/'
	l = os.listdir(folder)
	y =[]
	cnt=0
	for i in x:
		print(cnt,"/",len(x))
		cnt+=1
		# print(l[0])
		# print(i)
		# exit(0)
		if i+'.jpg' in l:
			y.append(i+'.jpg')

	print(y)

	pickle_out = open("../data/coco/train/filenames.pickle","wb")
	pickle.dump(y, pickle_out)

main()