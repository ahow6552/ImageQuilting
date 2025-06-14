
import copy
import math
import numpy as np

# def minimumCostPathOnArray(arr):
#     """
#     Standard array 'arr' is traversed top to bottom in minimum cost path
#     Return value: arr_mask
#     """
#     arr_mask = np.ones(np.array(arr).shape)

#     rows = len(arr)
#     cols = len(arr[0])

#     for i in range(1,rows):
#         arr[i][0] = arr[i][0] + min(arr[i-1][0], arr[i-1][1])
#         for j in range(1, cols-1):
#             arr[i][j] = arr[i][j] + min(arr[i-1][j-1], arr[i-1][j], arr[i-1][j+1])
#         arr[i][cols-1] = arr[i][cols-1] + min(arr[i-1][cols-2], arr[i-1][cols-1])

#     min_index = [0]*rows
#     min_cost = min(arr[-1])
#     for k in range(1,cols-1):
#         if arr[-1][k] == min_cost:
#             min_index[-1] = k

#     for i in range(rows-2, -1, -1):
#         j = min_index[i+1]
#         lower_bound = 0
#         upper_bound = 1 # Bounds for the case j=1

#         if j==cols-1:
#             lower_bound = cols-2
#             upper_bound = cols-1
#         elif j>0:
#             lower_bound = j-1
#             upper_bound = j+1

#         min_cost = min(arr[i][lower_bound:upper_bound+1])
#         for k in range(lower_bound, upper_bound+1):
#             if arr[i][k] == min_cost:
#                 min_index[i] = k


#     path = []
#     for i in range(0, rows):
#         arr_mask[i,0:min_index[i]] = np.zeros(min_index[i])
#         path.append((i+1, min_index[i]+1))
#     # print("Minimum cost path is: ")
#     # print(path)
#     return arr_mask

# def minimumCostMask(Ref, B1, B2, overlap_type, overlap_size):
#     """
#     B1, B2, Ref are numpy arrays
#     B1 and B2 are already present, we're trying to add Ref
#     Regions of overlap will have best of Ref and other block
#     To highlight the parts of Ref lost, the numpy.ones() array
#     ref_mask will denote those pixels as 0.
#     Placement is as follows:
#         __ B2
#         B1 Ref
#     overlap_type: Type of overlap
#     overlap_size: Number of layers to overlap
#     Return value: ref_mask
#     """
#     ref_mask = np.ones(Ref.shape)
#     #vertical
#     if overlap_type=='v':
#         arr = np.power(B1[:,-overlap_size:]-Ref[:,0:overlap_size], 2).tolist()
#         ref_mask[:,0:overlap_size] = minimumCostPathOnArray(arr)

#     #horizontal
#     elif overlap_type=='h':
#         arr = np.power(B2[-overlap_size:, :]-Ref[0:overlap_size, :], 2)
#         arr = arr.transpose()
#         arr = arr.tolist()
#         ref_mask[0:overlap_size,:] = minimumCostPathOnArray(arr).transpose()
#     #both
#     elif overlap_type=='b':
#         # Vertical overlap
#         arrv = np.power(B1[:,-overlap_size:]-Ref[:,0:overlap_size], 2).tolist()
#         ref_mask[:,0:overlap_size] = minimumCostPathOnArray(arrv)
#         # Horizontal overlap
#         arrh = np.power(B2[-overlap_size:, :]-Ref[0:overlap_size, :], 2)
#         arrh = arrh.transpose()
#         arrh = arrh.tolist()
#         ref_mask[0:overlap_size,:] = ref_mask[0:overlap_size,:]*(minimumCostPathOnArray(arrh).transpose())
#         # To ensure that 0's from previous assignment to ref_mask remain 0's
#     else:
#         print("Error in min path")

#     return ref_mask

def minimumCostPathOnArray(arr):
    """
    Given a 2D numpy array `arr`, returns a binary mask indicating the optimal
    top-down minimum-cost path. Used in texture quilting seam computation.
    """
    arr = np.array(arr, dtype=np.float32)
    rows, cols = arr.shape
    cost = arr.copy()
    backtrack = np.zeros_like(cost, dtype=np.int32)

    # Dynamic programming to build cost table
    for i in range(1, rows):
        for j in range(cols):
            # define candidate indices (avoid out-of-bounds)
            prev = cost[i-1, max(j-1, 0):min(j+2, cols)]
            min_idx = np.argmin(prev)
            actual_j = max(j-1, 0) + min_idx
            cost[i, j] += cost[i-1, actual_j]
            backtrack[i, j] = actual_j

    # Find min-cost endpoint at bottom row
    path = np.zeros(rows, dtype=np.int32)
    path[-1] = np.argmin(cost[-1])
    for i in range(rows-2, -1, -1):
        path[i] = backtrack[i+1, path[i+1]]

    # Generate mask (left of path is 0, right of path is 1)
    arr_mask = np.ones_like(arr)
    for i in range(rows):
        arr_mask[i, :path[i]] = 0  # left of seam = 0

    return arr_mask

def minimumCostMask(Ref, B1, B2, overlap_type, overlap_size):##自己修改
    """
    Generates a binary mask for seamless texture stitching using minimum cost paths.
    Ref, B1, B2: image blocks as NumPy arrays (assumed grayscale or single-channel).
    overlap_type: 'v' (vertical), 'h' (horizontal), or 'b' (both)
    overlap_size: number of overlapping pixels
    Returns:
        ref_mask: mask same shape as Ref, where 1 means keep Ref pixel, 0 means take from existing
    """
    ref_mask = np.ones_like(Ref,dtype=np.float32)

    if overlap_type == 'v':
        diff = np.square(B1[:, -overlap_size:] - Ref[:, :overlap_size])
        ref_mask[:, :overlap_size] = minimumCostPathOnArray(diff)

    elif overlap_type == 'h':
        diff = np.square(B2[-overlap_size:, :] - Ref[:overlap_size, :])
        mask_h = minimumCostPathOnArray(diff.T).T
        ref_mask[:overlap_size, :] = mask_h

    elif overlap_type == 'b':
        # vertical overlap
        diff_v = np.square(B1[:, -overlap_size:] - Ref[:, :overlap_size])
        mask_v = minimumCostPathOnArray(diff_v)
        ref_mask[:, :overlap_size] = mask_v

        # horizontal overlap
        diff_h = np.square(B2[-overlap_size:, :] - Ref[:overlap_size, :])
        mask_h = minimumCostPathOnArray(diff_h.T).T
        ref_mask[:overlap_size, :] *= mask_h  # element-wise AND

    else:
        raise ValueError("Invalid overlap_type, must be 'v', 'h', or 'b'")

    return ref_mask

#Uncomment below lines to run this as stand-alone file

# arr = np.random.rand(15,15)
# print(minimumCostMask(arr, arr, arr, 'b', 7))