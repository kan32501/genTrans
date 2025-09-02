import os
import cv2
from PIL import Image, ImageOps
import numpy as np
from models.gluestick.drawing import plot_color_line_matches_opencv

def crop_and_resize(image, size=(1024, 576)):
    """
    Crops and resizes an image to target size

    Args
        image (PIL.Image.Image): input image
        size (tuple, optional): target size as (width, height)

    Returns
        PIL.Image.Image: The cropped and resized image.
    """
    target_width, target_height = size
    original_width, original_height = image.size

    target_ratio = target_width / target_height
    original_ratio = original_width / original_height

    if original_ratio > target_ratio:
        new_width = int(original_height * target_ratio)
        left = (original_width - new_width) // 2
        right = left + new_width
        top = 0
        bottom = original_height
    else:
        new_height = int(original_width / target_ratio)
        top = (original_height - new_height) // 2
        bottom = top + new_height
        left = 0
        right = original_width

    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize(size)

    return resized_image

def get_input_frames_by_index(video0_frames_dir, videoN_frames_dir, frame0_mask_path, frameN_mask_path, width, height, inputs_dir, save=True):
    """
    Return the filepaths and the corresponding images for the indices given as PIL.Image objects.
    Preprocesses the images to fit the argument width & height

    Function assumes that the directory contains PNGs and PNGs only.

    Args
        video0_frames_dir (string) : the directory that has all the start video frames expanded
        videoN_frames_dir (string) : the directory that has all the end video frames expanded
        width (int) : width of processed image
        height (int) : height of processed image
        inputs_dir (string) : directory to save the input frames
        save (bool) : whether to save the requested frames or not

    Returns
        inputs: dictionary containing {"image_x" : [filepath (string), image (PIL.Image)] } records
    """
    # get all the filepaths
    video0_frames_list = os.listdir(video0_frames_dir)
    videoN_frames_list = os.listdir(videoN_frames_dir)
    frame0_index = len(video0_frames_list) - 1
    frameN_index = 1
    
    # get full filepaths
    frame0_prev_path = video0_frames_dir + "/" + video0_frames_list[frame0_index - 1]
    frame0_path = video0_frames_dir + "/" + video0_frames_list[frame0_index]
    frameN_path = videoN_frames_dir + "/" + videoN_frames_list[frameN_index]
    frameN_next_path = videoN_frames_dir + "/" + videoN_frames_list[frameN_index + 1]

    # get images
    frame0_prev = Image.open(frame0_prev_path).convert('RGB')
    frame0 = Image.open(frame0_path).convert('RGB')
    frameN = Image.open(frameN_path).convert('RGB')
    frameN_next = Image.open(frameN_next_path).convert('RGB')

    frame0_mask = Image.open(frame0_mask_path).convert('RGB')
    frameN_mask = Image.open(frameN_mask_path).convert('RGB')

    # preprocess images
    frame0_prev = crop_and_resize(frame0_prev, (width, height))
    frame0 = crop_and_resize(frame0, (width, height))
    frameN = crop_and_resize(frameN, (width, height))
    frameN_next = crop_and_resize(frameN_next, (width, height))

    frame0_mask = crop_and_resize(frame0_mask, (width, height))
    frame0_mask_bin = load_mask(frame0_mask_path, width, height, as_numpy=False)
    frame0_mask_bin_inv = load_mask(frame0_mask_path, width, height, as_numpy=False, invert=True)
    frame0_masked = apply_mask(frame0, frame0_mask_bin)
    frame0_masked = crop_and_resize(frame0_masked, (width, height))
    frame0_masked_inv = apply_mask(frame0, frame0_mask_bin_inv)
    frame0_masked_inv = crop_and_resize(frame0_masked_inv, (width, height))

    frameN_mask = crop_and_resize(frameN_mask, (width, height))
    frameN_mask_bin = load_mask(frameN_mask_path, width, height, as_numpy=False)
    frameN_mask_bin_inv = load_mask(frameN_mask_path, width, height, as_numpy=False, invert=True)
    frameN_masked = apply_mask(frameN, frameN_mask_bin)
    frameN_masked = crop_and_resize(frameN_masked, (width, height))
    frameN_masked_inv = apply_mask(frameN, frameN_mask_bin_inv)
    frameN_masked_inv = crop_and_resize(frameN_masked_inv, (width, height))

    # save the PNGs
    if save:
        frame0_prev.save(inputs_dir + "/0_start_prev.png")
        frame0.save(inputs_dir + "/1_start.png")
        frameN.save(inputs_dir + "/2_end.png")
        frameN_next.save(inputs_dir + "/3_end_next.png")

        frame0_mask.save(inputs_dir + "/4_start_mask.png")
        frame0_masked.save(inputs_dir + "/5_start_masked.png")
        frame0_masked_inv.save(inputs_dir + "/6_start_masked_inv.png")

        frameN_mask.save(inputs_dir + "/7_end_mask.png")
        frameN_masked.save(inputs_dir + "/8_end_masked.png")
        frameN_masked_inv.save(inputs_dir + "/9_end_masked_inv.png")

    # return a dictionary of the filepaths & images
    inputs = dict()
    inputs["frame0_prev"] = [frame0_prev_path, frame0_prev]
    inputs["frame0"] = [frame0_path, frame0]
    inputs["frameN"] = [frameN_path, frameN]
    inputs["frameN_next"] = [frameN_next_path, frameN_next]

    return inputs

def save_out_frames(inbetween_images, 
                    out_frames_name, out_frames_dir, gif_dir,
                    bef_and_aft=True, 
                    video0_frames_dir=None, frame0_index=-1, 
                    videoN_frames_dir=None, frameN_index=-1,
                    width=-1, height=-1):
    """
    Saves the list of frame0s (optional) + inbetween frames + frameNs (optional) PIL images as

    - individual PNG files
    - an animated GIF.

    Option to add the before & after frames from the input clips into the output frames.

    Args
        inbetween_images (List[PIL.Image.Image]): List of images to save
        out_frames_name (string) : name of out frames, eg. "name_00.gif"
        out_frames_dir (string) : directory to save the final output PNG frames
        gif_dir (string) : directory to save the gif
        video0_frames_dir (string) : directory of the PNG frames from start video
        frame0_index (int) : index of the selected frame in start video
        videoN_frames_dir (string) : directory of the PNG frames from end video
        frameN_index (int) : index of the selected frame in end video
        bef_and_aft (bool): whether to save the before & after frames or not

    Returns
        gif_path (string) : filepath of gif
        out_frames (List) : list of PIL.Image() objects containing all the frames 
    """
    # create out frames directory if it doesnt exist
    if not os.path.exists(out_frames_dir): os.mkdir(out_frames_dir)
    
    # out frame index
    out_frame_index = 0

    # out frames container
    out_frames = []

    if bef_and_aft: 
        # convert all preceding frames from start video up until frame0_prev_index into PIL Images
        video0_frames_list = os.listdir(video0_frames_dir) # all filepaths in start video
        video0_frames_PIL = [Image.open(os.path.join(video0_frames_dir, video0_frame_path_i)).convert('RGB')
                        for video0_frame_path_i in video0_frames_list[:frame0_index + 1]]
        # save in out_frames_dir
        for video0_frame in video0_frames_PIL:
            # frame filename is xx.png
            frame_path = os.path.join(out_frames_dir, out_frames_name + '_{:02d}.png'.format(out_frame_index))
            video0_frame.save(frame_path)

            # append to output frames
            out_frames.append(crop_and_resize(video0_frame, size=(width, height)))

            # increment frame count
            out_frame_index += 1

    # save each image in the inbetween list
    for inbetween_image in inbetween_images:
        # frame filename is xx.png
        frame_path = os.path.join(out_frames_dir, out_frames_name + '_{:02d}.png'.format(out_frame_index))
        inbetween_image.save(frame_path)

        # append to output frames
        out_frames.append(inbetween_image)

        # increment frame count
        out_frame_index += 1

    if bef_and_aft:
        # convert all frames from end video after frameN_next_index into PIL Images
        videoN_frames_list = os.listdir(videoN_frames_dir) # all filepaths in start video
        videoN_frames_PIL = [Image.open(os.path.join(videoN_frames_dir, videoN_frame_path_i)).convert('RGB') 
                    for videoN_frame_path_i in videoN_frames_list[(frameN_index + 1):]]
        # save in out_frames_dir
        for videoN_frame in videoN_frames_PIL:
            # frame filename is xx.png
            frame_path = os.path.join(out_frames_dir, out_frames_name + '_{:02d}.png'.format(out_frame_index))
            videoN_frame.save(frame_path)

            # append to output frames
            out_frames.append(crop_and_resize(videoN_frame, size=(width, height))) 

            # increment frame count
            out_frame_index += 1

    # make images into gif and save
    gif_path = os.path.join(gif_dir, out_frames_name + '.gif')
    duration = 100 # if not bef_and_aft else 150
    out_frames[0].save(gif_path, save_all=True, append_images=out_frames[1:], loop=0, duration=duration)

    return gif_path, out_frames

def load_mask(mask_path, width, height, as_numpy=True, invert=False):
    """
    Load the mask image into a binary numpy array (or PIL Image)

    Args
        mask_path (string) : path of mask image
        width (int) : width of image
        height (int) : height of image
        as_numpy (bool) : set output type as numpy
        invert (False) : invert the mask

    Returns
        mask_img () : mask image, size (width, height, 1) as numpy or PIL
    """
    # load image as the arguement size
    mask_PIL = crop_and_resize(Image.open(mask_path), size=(width, height))
    mask_PIL = mask_PIL.convert("1")

    # invert the mask
    if invert:
        mask_PIL_L = mask_PIL.convert("L")
        mask_PIL = ImageOps.invert(mask_PIL_L).convert("1")

    # return the image
    if not as_numpy: 
        return mask_PIL
    else:
        # reduce to 1 channel
        mask_img = np.expand_dims(np.round(np.asarray(mask_PIL)[:, :, 0]).astype(np.uint8), axis=2)
    
    return mask_img

def apply_mask(image, mask, save_path=None):
    """
    Use PIL.Image.composite function to mask the image

    Args
        image (PIL.Image) : source image
        mask (PIL.Image) : mask image
        save_path (string) : location to save image

    Returns
        image_masked (PIL.Image) : masked image
    """
    # create black image for the 0 pixels
    width, height = image.size
    black = Image.new(mode="RGB", size=(width, height))

    # mask image
    image_masked = Image.composite(image, black, mask)

    # save image
    if save_path:
        image_masked.save(save_path)

    return image_masked

def get_mask_bounding_box(mask):
    """
    Get the bounding box of a mask

    Args
        mask (np.array) : size (width, height, 1) binary mask

    Returns
        min_x (int) : x value with leftmost pixel of the mask
        min_y (int) : y value with the highest pixel of the mask
        max_x (int) : x value with the rightmost pixel of the mask
        max_y (int) : y value with the lowest pixel of the mask
    """
    # convert from True / False to 1/0
    mask_01 = mask.astype(np.uint8)

    # get dimensions
    height, width, _ = mask_01.shape

    # find the leftmost and rightmost occurence
    min_x = np.inf
    max_x = -np.inf
    for y in range(height):
        # get in-mask pixels in row
        row = mask_01[y, :].squeeze(axis=1) # (width,)
        in_mask_cols = np.argwhere(row == 1)

        # get first & last col with 1
        first_col = in_mask_cols[0,0] if len(in_mask_cols) != 0 else np.inf
        last_col = in_mask_cols[-1,0] if len(in_mask_cols) != 0 else -np.inf
        
        # update the leftmost occurrence
        min_x = first_col if first_col < min_x else min_x
        # update the rightmost occurence
        max_x = last_col if last_col > max_x else max_x

    # find the highest and lowest occurence
    min_y = np.inf
    max_y = -np.inf
    for x in range(width):
        # get in-mask pixels col
        col = mask_01[:, x].squeeze(axis=1) # (height,)
        in_mask_rows = np.argwhere(col == 1)

        # get first & last col with 1
        first_row = in_mask_rows[0,0] if len(in_mask_rows) != 0 else np.inf
        last_row = in_mask_rows[-1,0] if len(in_mask_rows) != 0 else -np.inf
        
        # update the leftmost occurrence
        min_y = first_row if first_row < min_y else min_y
        # update the rightmost occurence
        max_y = last_row if last_row > max_y else max_y

    return min_x, min_y, max_x, max_y

def get_mask_dimensions(mask):
    """
    Return the dimensions of a binary mask

    Params
    mask (PIL.Image) : binary mask image type "1"

    Args
    mask_width (int) : width of mask
    mask_height (int) : height of mask
    """
    # mask image to array
    mask_np = np.expand_dims(np.asarray(mask), axis=2)

    # get bounding box
    min_x, min_y, max_x, max_y = get_mask_bounding_box(mask_np)

    # get ranges and origin
    mask_width, mask_height = max_x - min_x, max_y - min_y

    return mask_width, mask_height

def plot_condition_imgs(image, interped_lines, lw=1, save=True, out_dir=None):
    """
    Plot the interpolated lines onto black images

    Args
        image (PIL.Image) : base image 
        interped_lines (List[np.array]) : list of interpolated lines
        lw (int) : line width on image. default is 1

    Returns
        conditions_images (List[PIL.Image]) : list of framewise conditions as images
    """
    # convert images to numpy arrays
    image_np = np.asarray(image)

    # get number of frames
    num_frames = len(interped_lines)
    
    # if there are no matched lines, add all black images
    if interped_lines is None:
        img = np.zeros_like(image_np)
        conditions_images = []
        for i in range(num_frames + 2):
            conditions_images.append(img)
        return conditions_images

    conditions_images = plot_color_line_matches_opencv(image_np, interped_lines, lw=lw, save_path=False, number=False)
    conditions_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in conditions_images]

    if save:
        num_imgs = len(conditions_images)
        for i in range(num_imgs):
            conditions_image_PIL = Image.fromarray(conditions_images[i])
            conditions_image_PIL.save(os.path.join(out_dir, 'condition{:02d}.png'.format(i)))

    return conditions_images