# train.txt
with open(os.path.join(root_dir, "train.txt"), 'w') as f:
	f.write('\n'.join(train) + '\n')

# valid.txt
with open(os.path.join(root_dir, "valid.txt"), 'w') as f:
	f.write('\n'.join(valid) + '\n')