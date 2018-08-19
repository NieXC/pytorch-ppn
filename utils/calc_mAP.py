import numpy as np
import scipy.io as sio
import collections

def calc_mAP(gt_path = './dataset/mpi/val_gt/mpi_val_groundtruth.mat', pred_path = './exps/preds/mat_results/pred_keypoints_mpii_multi.mat'):

	thresh = 0.5

	gtDir, partNames, name, predFilename, colorName = getExpParams(0)

	data = loadmat(gt_path)
	annolist_test_multi = data['annolist_test_multi']

	data = loadmat(pred_path)
	pred = data['pred']
	
	assert len(annolist_test_multi) == len(pred), 'incompatible length: annolist_test_multi & pred'
	scoresAll, labelsAll, nGTall = assignGTmulti(pred, annolist_test_multi, thresh)

	ap = np.zeros(nGTall.shape[0]+1)
	for j in range(nGTall.shape[0]):
		scores = np.array([])
		labels = np.array([])

		for imgidx in range(len(annolist_test_multi)):
			scores = np.append(scores,scoresAll[j][imgidx])
			labels = np.append(labels,labelsAll[j][imgidx])

		precision, recall, sorted_scores, sortidx, sorted_labels = getRPC(scores,labels,np.sum(nGTall[j,:]))

		ap[j] = VOCap(recall,precision) * 100

	ap[-1] = np.mean(ap[0:-1])

	columnNames = partNames
	genTableAP(ap,name)

	sio.savemat(predFilename, {'ap':ap, 'columnNames':columnNames})

	return ap

def genTableAP(ap,name):
	print(' '*(len(name)+1)+'& Head & Shoulder & Elbow & Wrist & Hip & Knee & Ankle & Total \\ \n')
	print('%s & %1.1f & %1.1f     & %1.1f  & %1.1f & %1.1f & %1.1f & %1.1f  & %1.1f %s\n'%(name,
						(ap[12]+ap[13])/2,(ap[8]+ap[9])/2,(ap[7]+ap[10])/2,(ap[6]+ap[11])/2,(ap[2]+ap[3])/2,
							(ap[1]+ap[4])/2,(ap[0]+ap[5])/2,ap[-1],'\\'))	

def VOCap(rec,prec):

	mrec = np.zeros(len(rec)+2)
	mrec[0] = 0
	mrec[1:-1] = rec
	mrec[-1] = 1

	mpre = np.zeros(len(prec)+2)
	mpre[0] = 0
	mpre[1:-1] = prec
	mpre[-1] = 0

	for i in range(len(mpre)-2,-1,-1):
		mpre[i] = max(mpre[i],mpre[i+1])

	i = np.nonzero(mrec[1:] != mrec[0:-1])[0] + 1
	ap = sum((mrec[i]-mrec[i-1]) * mpre[i])

	return ap

def getRPC(class_margin, true_labels, totalpos):
	N = len(true_labels)
	ndet = N

	npos = 0
	sorted_scores = np.sort(class_margin)
	sortidx = np.argsort(class_margin)

	sorted_labels = true_labels[sortidx]

	recall = np.zeros(ndet)
	precision = np.zeros(ndet)

	for ridx in range(ndet-1,-1,-1):
		if(sorted_labels[ridx] == 1):
			npos += 1

		precision[ndet - ridx - 1] = npos/float(ndet - ridx)
		recall[ndet - ridx -1] = npos/totalpos
 
	return precision, recall, sorted_scores, sortidx, sorted_labels

def assignGTmulti(pred,annolist_gt,thresh):
    nJoints = 14
    #LSP to MPII format map
    jidxMap = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 8, 9]
	
    scoresAll = [[[] for i in range(len(annolist_gt))] for j in range(nJoints)]
    #positive / negative labels
    labelsAll = [[[] for i in range(len(annolist_gt))] for j in range(nJoints)]

    #number of annotated GT joints per image
    nGTall = np.zeros((nJoints,len(annolist_gt)))

    for imgidx in range(len(annolist_gt)):
        # distance between predicted and GT joints
        len_pred_annorect = len(pred[imgidx].annorect) if isinstance(pred[imgidx].annorect, collections.Iterable) else 1
        len_annolist_gt_annorect = len(annolist_gt[imgidx].annorect) if isinstance(annolist_gt[imgidx].annorect, collections.Iterable) else 1
        dist = np.full((len_pred_annorect,len_annolist_gt_annorect,nJoints),np.inf)

        #score of the predicted joint
        score = np.full((len_pred_annorect,nJoints),np.nan)

        #body joint prediction exist
        hasPred = np.full((len_pred_annorect,nJoints),False,dtype = bool)

        #body joint is annotated
        hasGT = np.full((len_annolist_gt_annorect,nJoints),False,dtype = bool)

        for ridxPred in range(len_pred_annorect):
            #predicted pose
            rectPred = pred[imgidx].annorect[ridxPred] if isinstance(pred[imgidx].annorect, collections.Iterable) else pred[imgidx].annorect
            pointsPred = rectPred.annopoints.point

            #iterate over GT poses
            for ridxGT in range(len_annolist_gt_annorect): #GT
                # GT pose
                rectGT = annolist_gt[imgidx].annorect[ridxGT] if isinstance(annolist_gt[imgidx].annorect, collections.Iterable) else annolist_gt[imgidx].annorect

                # compute reference distance as head size
                refDist = util_get_head_size(rectGT)
                pointsGT = rectGT.annopoints.point

                #iterate over all possible body joints
                for i in range(nJoints):
                    #predicted joint in LSP format
                    ppPred = util_get_annopoint_by_id(pointsPred, jidxMap[i])

                    if (type(ppPred) is not type(None)):
                        score[ridxPred,i] = ppPred[0].score
                        hasPred[ridxPred,i] = True

                    #GT joint in LSP format
                    ppGT = util_get_annopoint_by_id(pointsGT, jidxMap[i])
                    if (type(ppGT) is not type(None)):
                        hasGT[ridxGT,i] = True

                    #compute distance between predicted and GT joint locations
                    if (hasPred[ridxPred,i] and hasGT[ridxGT,i]):
                        dist[ridxPred,ridxGT,i] = np.linalg.norm(np.array([ppGT[0].x, ppGT[0].y]) - np.array([ppPred[0].x, ppPred[0].y])) / refDist;

        nGT = np.zeros((len_pred_annorect,len_annolist_gt_annorect))
        for i in range(len_pred_annorect):
            for j in range(len_annolist_gt_annorect):
                nGT[i,:] = np.sum(hasGT,axis = 1)

        match = np.zeros((len_pred_annorect,len_annolist_gt_annorect,nJoints))  
        match[(dist <= thresh)] = 1

        pck = np.sum(match, axis = 2) / nGT

        if pck.shape[0] == 0:
            continue

        val = np.amax(pck,axis = 1)
        idx = np.argmax(pck,axis = 1)
        for ridxPred in range(idx.shape[0]):
            for j in range(pck.shape[1]):
                if(j != idx[ridxPred]):
                    pck[ridxPred,j] = 0

        val = np.amax(pck,axis = 0)
        predToGT = np.argmax(pck,axis = 0)
        predToGT[(val==0)] = -1

        for ridxPred in range(len_pred_annorect):
            if ridxPred in predToGT:
                ridxGT = np.nonzero(predToGT == ridxPred)[0]
                s = np.squeeze(score[ridxPred,:])
                m = np.squeeze(match[ridxPred,ridxGT,:])
                hp = np.squeeze(hasPred[ridxPred,:])
                idxs = np.nonzero(hp)[0]
                for i in range(idxs.shape[0]):
                    scoresAll[idxs[i]][imgidx] = np.append(scoresAll[idxs[i]][imgidx],s[idxs[i]])
                    labelsAll[idxs[i]][imgidx] = np.append(labelsAll[idxs[i]][imgidx],m[idxs[i]])
            else:
                s = np.squeeze(score[ridxPred,:])
                m = np.full(nJoints,False,dtype = bool)
                hp = np.squeeze(hasPred[ridxPred,:])
                idxs = np.nonzero(hp)[0]
                for i in range(idxs.shape[0]):
                    scoresAll[idxs[i]][imgidx] = np.append(scoresAll[idxs[i]][imgidx],s[idxs[i]])
                    labelsAll[idxs[i]][imgidx] = np.append(labelsAll[idxs[i]][imgidx],m[idxs[i]])

        for ridxGT in range(len_annolist_gt_annorect):
            hg = hasGT[ridxGT,:]
            nGTall[:,imgidx] = nGTall[:,imgidx] + np.transpose(hg)

    return scoresAll, labelsAll, nGTall

def util_get_annopoint_by_id(points, id_):
	for i in range(len(points)):
		if (points[i].id == id_):
			point = points[i]
			ind = i
			return point, ind

def util_get_head_size(rect):

	SC_BIAS = 0.6
	headSize = SC_BIAS*np.linalg.norm(np.array([rect.x2, rect.y2]) - np.array([rect.x1, rect.y1]))

	return headSize

def getExpParams(predidx):
	gtDir = './ground_truth/'
	colorIdxs = [1, 1]
	partNames = ['right ankle','right knee','right hip','left hip','left knee','left ankle','right wrist','right elbow','right shoulder','left shoulder','left elbow','left wrist','neck','top head','avg full body']

	if(predidx == 0):
		name = 'PPN'
		predFilename = './exps/preds/temp/predictions.mat'
		colorIdxs = [7, 1]

	colorName = getColor(colorIdxs)
	colorName = np.array(colorName)
	colorName = colorName / 255

	return gtDir, partNames, name, predFilename, colorName

def getColor(cidxs):

	color = [[0]*6]*9
	#qualitative
	color[1][1] = [55,126,184]
	color[2][1] = [255,127,0]
	color[3][1] = [255,255,51]
	color[4][1] = [0,0,0]
	color[5][1] = [77,175,74]
	color[6][1] = [228,26,28]
	color[7][1] = [152,78,163]
	color[8][1] = [247,129,191]

	#sequential
	color[1][2] = [253,141,60]
	color[1][3] = [254,204,92]
	color[1][4] = [255,255,178]
	color[1][5] = [254,240,217]

	color[2][2] = [107,174,214]
	color[2][3] = [189,215,231]
	color[2][4] = [117,107,177]

	color[3][2] = [194,230,153]

	color[4][2] = [140,150,198]
	color[4][3] = [179,205,227]
	color[4][4] = [237,248,251]

	return color[cidxs[0]][cidxs[1]]

def loadmat(filename):
	'''
	this function should be called instead of direct spio.loadmat
	as it cures the problem of not properly recovering python dictionaries
	from mat files. It calls the function check keys to cure all entries
	which are still mat-objects
	'''
	data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
	return _check_keys(data)

def _check_keys(dict):
	'''
	checks if entries in dictionary are mat-objects. If yes
	todict is called to change them to nested dictionaries
	'''
	for key in dict:
		if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
			dict[key] = _todict(dict[key])
	return dict        

def _todict(matobj):
	'''
	A recursive function which constructs from matobjects nested dictionaries
	'''
	dict = {}
	for strg in matobj._fieldnames:
		elem = matobj.__dict__[strg]
		if isinstance(elem, sio.matlab.mio5_params.mat_struct):
			dict[strg] = _todict(elem)
		else:
			dict[strg] = elem
	return dict

if __name__ == '__main__':
    print('Calculate mAP on MPII dataset')

