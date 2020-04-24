import os
import pickle

def load_filenames(data_dir):
	filepath = os.path.join(data_dir, 'char-CNN-RNN-embeddings.pickle')
	with open(filepath, 'rb') as f:
		filenames = pickle.load(f)
	print(filenames)
	print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
	return filenames

def main():
	x = load_filenames('../data/birds/test/')

main()