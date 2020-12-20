def Indexing(Z,des,numSeg):
	Z = Intra_Norm(Z,numSeg)
	x = torch.split(des,numSeg,1)
	y = torch.split(Z,numSeg,1)
	for i in range(numSeg):
		size_x = x[i].shape[0]
		size_y = y[i].shape[0]
	xx = x[i].view(-1,1)
	xx = torch.tile(xx,torch.stack([1,1,size_y]))
	yy = y[i].view(-1,1)
	yy = torch.tile(yy,torch.stack([1,1,size_x]))
	yy = torch.transpose(yy,2,1,0)
	diff = torch.sum(torch.multiply(xx,yy),1)

	arg = torch.argmax(diff,1)
	max_idx = arg.reshape(-1,1)

	if i == 0: quant_idx = max_idx
	else: quant_idx = torch.cat((quant_idx,max_idx),1)

	return quant_idx


