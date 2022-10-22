import numpy as np
from sklearn.decomposition import PCA
from operator import truediv
import matplotlib.pyplot as plt

def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values=0)
    return new_matrix

def zscores(data):
    '''
    For matrix data, z-scores are computed using the mean and standard deviation along each row of data.
    returns a centered, scaled version of each sample, (X-MEAN(X)) ./ STD(X)
    input: data with the shape of [samples_number,feature]
    This function performs well in ELM algorithm, but not well in 1D-CNN
    '''
    new_data=np.zeros([data.shape[0],data.shape[1]])
    for j in range(data.shape[0]):
        new_data[j,:] = (data[j,:]-np.mean(data[j,:]))/np.std(data[j,:],ddof=1)
    return new_data
def NormalizationEachBand(raw_data, unit=False):
    '''
    normalize the whole data to [0,1]
    '''
    new_data=np.zeros([raw_data.shape[0],raw_data.shape[1],raw_data.shape[2]])
    for i in range(raw_data.shape[2]):
        temp = raw_data[:,:,i]
        MAX = np.max(temp.ravel()).astype('float32')
        MIN = np.min(temp.ravel())
        new_data[:,:,i] = (temp - MIN)/(MAX - MIN)
    if unit:
        new_data = new_data.reshape(np.prod(new_data.shape[:2]),np.prod(new_data.shape[2:]))
        new_data = zscores(new_data)
        new_data=new_data.reshape(raw_data.shape[0],raw_data.shape[1],raw_data.shape[2])
    return new_data

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def indexToAssignment(Row_index, Col_index, pad_length):
    new_assign = {}
    for counter in range(Row_index.shape[0]):
        assign_0 = Row_index[counter] + pad_length
        assign_1 = Col_index[counter] + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def AA_andEachClassAccuracy(confusion_matrix):
    confusion_matrix = np.mat(confusion_matrix)
    OA = np.sum(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
    Po = OA
    xsum = np.sum(confusion_matrix, axis=1)
    ysum = np.sum(confusion_matrix, axis=0)
    Pe = float(ysum*xsum)/(np.sum(confusion_matrix)**2)
    Kappa = float((Po-Pe)/(1-Pe))

    list_diag = np.diag(confusion_matrix)
    each_acc = np.nan_to_num(truediv(list_diag, ysum))
    average_acc = np.mean(each_acc)
    precision = np.nan_to_num(truediv(list_diag, xsum.T))
    average_precision = np.mean(precision)

    return OA, Kappa, each_acc, average_acc, precision, average_precision

def generate_map(prediction,idx,gt):
    maps=gt.reshape(gt.shape[0]*gt.shape[1],)
    labeled_loc = np.squeeze(np.array(np.where(maps>0)),axis=0)
    tr_test = maps[labeled_loc]
    tr_test[idx] = prediction
    maps[labeled_loc] = tr_test
    maps.reshape(gt.shape[0], gt.shape[1])
    return maps

def DrawResult(labels, image_name):
    # ID=1:Pavia University
    # ID=2:Indian Pines
    # ID=6:KSC
    # ID=7:HU2012
    global palette
    global row
    global col
    num_class = int(labels.max())
    if image_name == 'PaviaU':
        row = 610
        col = 340
        palette = np.array([[216, 191, 216],
                            [0, 255, 0],
                            [0, 255, 255],
                            [45, 138, 86],
                            [255, 0, 255],
                            [255, 165, 0],
                            [159, 31, 239],
                            [255, 0, 0],
                            [255, 255, 0]])
        palette = palette * 1.0 / 255
    elif image_name == 'IndianP':
        row = 145
        col = 145
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [238, 0, 0]])
        palette = palette * 1.0 / 255

    elif image_name == 'KSC':
        row = 512
        col = 614
        palette = np.array([[94, 203, 55],
                            [255, 0, 255],
                            [217, 115, 0],
                            [179, 30, 0],
                            [0, 52, 0],
                            [72, 0, 0],
                            [255, 255, 255],
                            [145, 132, 135],
                            [255, 255, 172],
                            [255, 197, 80],
                            [60, 201, 255],
                            [11, 63, 124],
                            [0, 0, 255]])
        palette = palette * 1.0 / 255

    elif image_name == 'HU2012':
        row = 349
        col = 1905
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [139, 0, 0],
                            [100, 0, 255],
                            [255, 255, 0],
                            [238, 154, 0],
                            [85, 26, 139],
                            [255, 127, 80]])
        palette = palette * 1.0 / 255
    elif image_name == 'HU2018':
        row = 601
        col = 2384
        palette = np.array([[0, 208, 0],
                            [128, 255, 0],
                            [50, 160, 100],
                            [0, 143, 0],
                            [0, 76, 0],
                            [160, 80, 40],
                            [0, 236, 236],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [192, 180, 170],
                            [114, 133, 124],
                            [170, 0, 0],
                            [80, 0, 0],
                            [237, 164, 24],
                            [255, 255, 0],
                            [250, 190, 21],
                            [245, 0, 245],
                            [0, 0, 236],
                            [179, 197, 222]])
        palette = palette * 1.0 / 255

    X_result = np.zeros((labels.shape[0], 3))
    for i in range(1, num_class + 1):
        X_result[np.where(labels == i), 0] = palette[i - 1, 0]
        X_result[np.where(labels == i), 1] = palette[i - 1, 1]
        X_result[np.where(labels == i), 2] = palette[i - 1, 2]

    X_result = np.reshape(X_result, (row, col, 3))
    plt.axis("off")
    plt.imshow(X_result)
    return X_result



