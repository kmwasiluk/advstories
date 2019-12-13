with open("ROCStories_samples.txt", "r") as f:
	o = f.read()
	L = o.split("<|endofsample|>")
	print(len(L))
#	for l in L:
#		L2 = l.split("<|endoftext|>")
#		if(len(L2) > 1):
#			n = sum(i == "<" for i in L2[1])
#			if(n > 10): #gets rid of blocks of a bunch of tags and also stories with excessive UNKs
#				continue
#			if(len(L2[1]) > 150):
#				print(len(L2[1]))
#				print(L2[1] + "<|endoftext|>")