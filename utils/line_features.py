import numpy as np
from utils.util import *
from utils.image_operations import get_mask_bounding_box

def cartesian_to_polar(cart_lines):
    """
    Convert two endpoints into an infinite polar line form

    Args
        cart_lines (np.array) : lines defined by two endpoints. shape (n_lines, 2, 2)

    Returns
        polar_lines (np.array) : shape (n_lines, 1, 2). lines defined by
                                - (r) the shortest distance to the line from the origin
                                - (theta) and the angle from the x axis
                            
    """
    # get the number of lines
    n_lines = cart_lines.shape[0]

    # initialize list of polar form lines
    polar_lines = np.zeros(shape=(n_lines, 1, 2))

    # get the polar form for each line
    for line in range(n_lines):
        # get line
        x_a = cart_lines[line, 0, 0]
        y_a = cart_lines[line, 0, 1]
        x_b = cart_lines[line, 1, 0]
        y_b = cart_lines[line, 1, 1]

        # get m
        m = (y_b - y_a) / (x_b - x_a) 
        # if the line is vertical
        if x_b - x_a == 0.0:
            # positive or negative infinity
            m = np.inf if y_b - y_a > 0 else -1 * np.inf

        # get b (if the line is not vertical)
        b = y_a - m * x_a 

        # convert to r, theta
        theta = np.arctan(-1.0 / m)
        rho = b * np.sin(theta)
        # handle the inf gradient cases
        if m == np.inf:
            theta = np.pi / 2 # positive inf gradient
            rho = x_a
        if m == -1 * np.inf:
            theta = -1 * np.pi / 2 # negative inf gradient
            rho = x_a
    
        # book keep the polar line parameters
        polar_lines[line, 0, 0] = rho
        polar_lines[line, 0, 1] = theta

    return polar_lines

def get_midpoints(cart_lines):
    """
    Return the midpoint of a line defined by two endpoints

    Args
        cart_lines (np.array) : lines defined by two endpoints. shape: (n_lines, 2, 2)

    Returns
        midpoints (np.array)
    """
    # get the number of lines
    n_lines = cart_lines.shape[0]

    # initialize the list of n midpoints
    midpoints = np.zeros(shape=(n_lines, 1, 2))

    # get the average x and y coords
    midpoints[:, 0, 0] = np.average(cart_lines[:, :, 0], axis=1) # x
    midpoints[:, 0, 1] = np.average(cart_lines[:, :, 1], axis=1) # y

    return midpoints

def normalize_lines_by_mask(mask, lines):
    """
    Normalize line endpoints with respect to the bounding box of a mask.

    The normalized coordinates has the origin at the center of the bounding box.

    Args
        mask (PIL.Image) : PIL Image binary mask, type "1"
        lines (np.array) : size (n_lines, 2, 2)

    Returns
        lines_normalized (np.array) : size (n_lines, 2, 2) normalized [-1,1] w.r.t bounding box
    """
    # mask image to array
    mask_np = np.expand_dims(np.asarray(mask), axis=2)

    # get bounding box
    min_x, min_y, max_x, max_y = get_mask_bounding_box(mask_np)

    # get ranges and origin
    mask_width, mask_height = max_x - min_x, max_y - min_y
    origin_x, origin_y = (max_x + min_x) / 2, (max_y + min_y) / 2

    # normalization function
    def normalize(point):
        # assume point is a (2,) numpy array
        norm_x = (point[0] - origin_x) / (mask_width / 2)
        norm_y = (origin_y - point[1]) / (mask_height / 2) # the y starts at the top and goes down. so reverse.

        # START HERE: create an array with the new coordinates
        normalized_point = np.asarray([norm_x, norm_y]) # (2,) array

        return normalized_point

    # intialize output
    lines_normalized = np.zeros_like(lines)

    # normalize each coordinate
    n_lines = lines.shape[0]
    for line in range(n_lines):
        # normalize both endpoints
        norm_start_pt = normalize(lines[line, 0, :])
        norm_end_pt = normalize(lines[line, 1, :])

        # bookkeep
        lines_normalized[line, 0, :] = norm_start_pt
        lines_normalized[line, 1, :] = norm_end_pt

    return lines_normalized

def greedy_match_lines(norm_hough0, norm_houghN, norm_midpoints0, norm_midpointsN, with_flow, flow0_midpts=None, flowN_midpts=None):
    """
    Greedily match the lines based on lowest L2 DISTANCE of

    - theta
    - normalized midpoint (x, y)
    - normalized optical flow

    Calculate the L2 distance in R^4 space and produce a 1:1 mapping

    Args
        norm_hough0 (np.array) : size (n_lines, 1, 2) [rho, theta] pairs for frame0, based on normalized lines
        norm_houghN (np.array) : size (n_lines, 1, 2) [rho, theta] pairs for frameN, based on normalized lines
        norm_midpoints0 (np.array) : size (n_points, 1, 2) midpoints for frame0, normalized by mask
        norm_midpointsN (np.array) : size (n_points, 1, 2) midpoints for frameN, normalized by mask
        flow0_midpts (np.array) : size (n_points, 1, 2) [x, y] optical flow for frame0 at midpoints
        flow0_midpts (np.array) : size (n_points, 1, 2) [x, y] optical flow for frameN at midpoints

    Returns
        matched_indices0 (np.array) : size (min_lines,) indices of the matched lines
        matched_indicesN (np.array) : size (min_lines,) indices of the matched lines
    """
    # get number of lines from both
    n_lines0 = norm_hough0.shape[0]
    n_linesN = norm_houghN.shape[0]

    # the set of lines with less lines chooses
    min_lines = min(n_lines0, n_linesN)

    # initialize matched lines list & all other parameters
    matched_indicesA = np.zeros(shape=(min_lines,), dtype=np.uint8)
    matched_indicesB = np.zeros(shape=(min_lines,), dtype=np.uint8)

    
    if not with_flow:
        # concatenate the hough line parameters [THETA ONLY IN INDEX 1] and the midpoints
        line_paramsA = np.concatenate((np.expand_dims(norm_hough0[:, :, 1], axis=2), norm_midpoints0), axis=2) if n_lines0 < n_linesN else np.concatenate((np.expand_dims(norm_houghN[:, :, 1], axis=2), norm_midpointsN), axis=2)
        line_paramsB = np.concatenate((np.expand_dims(norm_houghN[:, :, 1], axis=2), norm_midpointsN), axis=2) if n_lines0 < n_linesN else np.concatenate((np.expand_dims(norm_hough0[:, :, 1], axis=2), norm_midpoints0), axis=2)
    else:
        # concatenate the midpoints of the line with the optical flow
        line_paramsA = np.concatenate((norm_midpoints0, flow0_midpts), axis=2) if n_lines0 < n_linesN else np.concatenate((norm_midpointsN, flowN_midpts), axis=2)
        line_paramsB = np.concatenate((norm_midpointsN, flowN_midpts), axis=2) if n_lines0 < n_linesN else np.concatenate((norm_midpoints0, flow0_midpts), axis=2)
        # # only use optical flow
        # line_paramsA = flow0_midpts.copy() if n_lines0 < n_linesN else flowN_midpts.copy()
        # line_paramsB = flowN_midpts.copy() if n_linesN < n_linesN else flow0_midpts.copy()

    # shuffle the order of the indices so that each line has an equal chance of choosing first
    rand_indices = np.arange(min_lines)
    np.random.shuffle(rand_indices)

    # for each line in the frame0, match to the closest line in the second image
    for i in range(min_lines):
        # get random index
        rand_index = rand_indices[i]

        # bookkeep the random index
        matched_indicesA[i] = rand_index

        # get the line as a point in R^4 space
        linesA_i = line_paramsA[rand_index]

        # compute the squared distances from this "point" in R^4 to all the line "points" in R^4 from frameN
        L2dists = np.square(np.linalg.norm(line_paramsB - linesA_i, axis=2)) # numpy.linalg.norm along the row

        # get the line in imageB that has the minimum distance to the current line in frame0
        matchB_i = np.argmin(L2dists) # index of min. dist line

        # bookkeep the matched line and its parameters
        matched_indicesB[i] = matchB_i
        
        # **NOTE: what if it's not greedy ???
        # remove the matched line from the options by setting the parameter entries to 0
        # (this removes the need to resize the numpy arrays and deal with changed indices)
        line_paramsB[matchB_i, :, :] = np.inf

    # return the matching
    if n_lines0 < n_linesN:
        return matched_indicesA, matched_indicesB
    else:
        return matched_indicesB, matched_indicesA

def interpolate_lines(matched_lines0, matched_linesN, num_frames=13):
    """
    Generate the frame-wise pose/edge conditions that will be passed to ControlNext

    Args
        matched_lines0 (np.array) : size (n_lines, 2, 2), lines in frame0, matched by index
        matched_linesN (np.array) : size (n_lines, 2, 2), lines in frameN, matched by index
        num_frames (int) : # of interpolated frames to generate

    Returns
        conditions_images (list) : list of framewise pose/edge conditions as numpy images
    """
    # linearly interpolate between the two matched line segments, in line space
    interped_lines = [matched_lines0]
    # from 0 to the number of inbetween frames
    for i in range(num_frames):
        # lambda weight 
        frac = i / (num_frames - 1)
        # interped is a numpy array, size (n_lines, 2, 2) for each endpoint
        interped = interpolate_matches_linear(matched_lines0, matched_linesN, frac) # return the interpolated lines
        interped_lines.append(interped)
    # add the last frame
    interped_lines.append(matched_linesN)

    return interped_lines

def filter_lines_in_mask(mask, lines):
    """
    Args:
        mask (PIL.Image, mode "1") : binary mask on source image
        lines (np.array) : size (n, 2, 2), lines in source image defined by endpoints

    Returns
        mask_lines (np.array) : size (x, 2, 2) lines in source image that are inside the mask
    """
    # convert mask to array of 0/1
    mask_arr = np.array(mask, dtype=np.uint8) 
    h, w = mask_arr.shape

    # check each line
    valid_lines = []
    for line in lines:
        # extract coordinates
        (x1, y1), (x2, y2) = line.astype(int)

        # skip if endpoints are outside image bounds
        if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
            continue

        # keep line only if both endpoints lie in mask
        if mask_arr[y1, x1] == 1 and mask_arr[y2, x2] == 1:
            valid_lines.append(line)

    # return filtered lines or empty array if none valid
    return np.array(valid_lines) if valid_lines else np.empty((0, 2, 2), dtype=lines.dtype)