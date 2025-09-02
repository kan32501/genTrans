import cv2
import numpy as np
import os
from PIL import Image
import seaborn as sns
from utils.image_operations import crop_and_resize, apply_mask

def plot_flow(image, points, flow, magnitude=100.0):
    """
    Plot optical flow vectors onto an image at the specified points.

    Args:
        image (PIL.Image): The input image to paint
        points (np.array): size (n, 1, 2). [x, y] origin points of the flow vectors 
        flow (np.array): size (n, 2). [u, v] flow vectors corresponding to the points.
    Returns:
        flow_image (PIL.Image): The image with flow vectors plotted.

    """
    # Convert PIL.Image to numpy (RGB -> BGR) for OpenCV
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # plot each flow vector as an arrow
    n = points.shape[0]
    if flow.shape[0] != n: raise ValueError("Number of points and flow vectors don't match.")
    for i in range(n):
        # get entries
        x, y = points[i, 0, 0], points[i, 0, 1]
        u, v = flow[i, 0, 0], flow[i, 0, 1]

        # endpoints
        start = (int(x), int(y))
        end = (int(x + magnitude * u), int(y + magnitude * v))

        # plot arrow
        color = (0, 230, 0)
        cv2.arrowedLine(img_bgr, start, end, color, 1, tipLength=0.3)

    # Convert back to PIL.Image
    flow_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return flow_image

# get extended endpoints of line
def get_extended_endpoints(matched_line, midpoint, width, height):
    # get the gradient vector as (run, rise)
    run = matched_line[1][0] - matched_line[0][0]
    rise = matched_line[1][1] - matched_line[0][1]

    # normalize gradient vector
    mag = np.sqrt(rise ** 2 + run ** 2)
    run /= mag
    rise /= mag

    # vector scalar - this amplifies the gradient vector
    scalar = max(height, width) * 2

    # extend endpoints
    x_a = int(midpoint[0] + scalar * run)
    y_a = int(midpoint[1] + scalar * rise)
    x_b = int(midpoint[0] - scalar * run)
    y_b = int(midpoint[1] - scalar * rise)

    return (x_a, y_a), (x_b, y_b)

def plot_lines(image, lines, extend_lines=False, color=True, number=True):
    """
    Plot lines onto an image

    Args:
        image (PIL.Image): The input image to paint
        lines (np.array): size (n, 2, 2). [[x_a, y_a] , [x_b, y_b]] line segments to plot
        color (bool) : color each line differently
    Returns:
        line_image (PIL.Image): The image with lines plotted
    """
    # Convert PIL.Image to numpy (RGB -> BGR) for OpenCV
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # choose color palette for lines
    n = lines.shape[0]
    line_colors = sns.color_palette(n_colors=n)

    # plot each line segment
    for i in range(n):
        # get endpoints
        pt1, pt2 = tuple(map(int, lines[i][0])), tuple(map(int, lines[i][1]))
        midpoint = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)

        # extend lines if needed
        if extend_lines:
            pt1, pt2 = get_extended_endpoints(lines[i], midpoint, image.width, image.height)

        # plot line
        color = (line_colors[i][2] * 255, line_colors[i][1] * 255, line_colors[i][0] * 255) if color else (255, 0, 0)
        cv2.line(img_bgr, pt1, pt2, color, 2)

        # plot number
        if number:
            org = (midpoint[0] + 2, midpoint[1] + 2)
            cv2.putText(img_bgr, str(i), tuple(map(int, org)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

    # Convert back to PIL.Image
    line_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return line_image

def plot_points(image, points, coordinates=False, color=False):
    """
    Plot points onto an image

    Args:
        image (PIL.Image): The input image to paint
        points (np.array): size (n, 1, 2). [x, y] coords to plot
        coordinates (bool): If True, display the coordinates of each point next to it.
    Returns:
        point_image (PIL.Image): The image with points plotted
    """
    # Convert PIL.Image to numpy (RGB -> BGR) for OpenCV
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # choose color palette for points
    n = points.shape[0]
    line_colors = sns.color_palette(n_colors=n)

    # plot each point
    for i in range(n):
        x, y = points[i, 0, 0], points[i, 0, 1]
        point = (int(x), int(y))
        cv2.circle(img_bgr, point, radius=2, color=(0, 0, 255), thickness=-1)  # red circle

        # plot coordinates slightly diagonally below the point
        if coordinates:
            # choose color for text
            color = (line_colors[i][2] * 255, line_colors[i][1] * 255, line_colors[i][0] * 255) if color else (0, 0, 255)
            
            # put text slightly shifted
            x_shift = 5
            y_shift = -5

            # plot coordinates
            cv2.putText(img_bgr, f"({x:.1f}, {y:.1f})", (int(x) + x_shift, int(y) + y_shift), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Convert back to PIL.Image
    point_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return point_image

def visualize_image_features(height, width,
                             matched_lines0, matched_linesN, 
                             midpoints0, midpointsN,
                             output_dir,
                             flow0_midpts=None, flowN_midpts=None,
                             frame0_mask=None, frameN_mask=None,
                             bg_white=True):
    # initialize base white images
    img0 = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * 255) if bg_white else Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    imgN = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * 255) if bg_white else Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    
    # apply masks if provided
    if frame0_mask is not None:
        img0 = apply_mask(img0, frame0_mask)
    if frameN_mask is not None:
        imgN = apply_mask(imgN, frameN_mask)

    # plot line segments, colored
    img0_lines_colored = plot_lines(img0, matched_lines0, extend_lines=False, color=True, number=False)
    imgN_lines_colored = plot_lines(imgN, matched_linesN, extend_lines=False, color=True, number=False)
    img0_lines_colored.save(os.path.join(output_dir, "frame0_lines_colored.png"))
    imgN_lines_colored.save(os.path.join(output_dir, "frameN_lines_colored.png"))

    # plot line segments, colored, with midpoints
    if flow0_midpts is None:
        img0_lines_colored_midpts = plot_points(img0_lines_colored, midpoints0, coordinates=False, color=False)
        imgN_lines_colored_midpts = plot_points(imgN_lines_colored, midpointsN, coordinates=False, color=False)
        img0_lines_colored_midpts.save(os.path.join(output_dir, "frame0_lines_colored_midpts.png"))
        imgN_lines_colored_midpts.save(os.path.join(output_dir, "frameN_lines_colored_midpts.png"))

    # plot line segments, colored, with flow
    if flow0_midpts is not None:
        img0_lines_colored_flow = plot_flow(img0_lines_colored, midpoints0, flow0_midpts)
        imgN_lines_colored_flow = plot_flow(imgN_lines_colored, midpointsN, flowN_midpts) 
        img0_lines_colored_flow.save(os.path.join(output_dir, "frame0_lines_colored_flow.png"))
        imgN_lines_colored_flow.save(os.path.join(output_dir, "frameN_lines_colored_flow.png"))
      
    # plot line segments, colored, with midpoints and flow
    if flow0_midpts is not None:
        img0_lines_colored_midpts_flow = plot_points(img0_lines_colored_flow, midpoints0, coordinates=False, color=False)
        imgN_lines_colored_midpts_flow = plot_points(imgN_lines_colored_flow, midpointsN, coordinates=False, color=False)
        img0_lines_colored_midpts_flow.save(os.path.join(output_dir, "frame0_lines_colored_midpts_flow.png"))
        imgN_lines_colored_midpts_flow.save(os.path.join(output_dir, "frameN_lines_colored_midpts_flow.png"))

    # plot line segments, colored, extended
    background = Image.fromarray(np.ones((height, width, 3), dtype=np.uint8) * 255) if bg_white else Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    img0_lines_colored_extended = plot_lines(background, matched_lines0, extend_lines=True, color=True, number=False)
    imgN_lines_colored_extended = plot_lines(background, matched_linesN, extend_lines=True, color=True, number=False)
    img0_lines_colored_extended.save(os.path.join(output_dir, "frame0_lines_colored_extended.png"))
    imgN_lines_colored_extended.save(os.path.join(output_dir, "frameN_lines_colored_extended.png"))

    # plot line segments, colored, numbered
    img0_lines_colored_numbered = plot_lines(img0, matched_lines0, extend_lines=False, color=True, number=True)
    imgN_lines_colored_numbered = plot_lines(imgN, matched_linesN, extend_lines=False, color=True, number=True)
    img0_lines_colored_numbered.save(os.path.join(output_dir, "frame0_lines_colored_numbered.png"))
    imgN_lines_colored_numbered.save(os.path.join(output_dir, "frameN_lines_colored_numbered.png"))

    # plot line segments, colored, numbered, with flow
    if flow0_midpts is not None:
        img0_lines_colored_numbered_flow = plot_flow(img0_lines_colored_numbered, midpoints0, flow0_midpts)
        imgN_lines_colored_numbered_flow = plot_flow(imgN_lines_colored_numbered, midpointsN, flowN_midpts)
        img0_lines_colored_numbered_flow.save(os.path.join(output_dir, "frame0_lines_colored_numbered_flow.png"))
        imgN_lines_colored_numbered_flow.save(os.path.join(output_dir, "frameN_lines_colored_numbered_flow.png"))

def visualize_flow_field(flow, output_path, scale=1.0):
    """
    Visualize optical flow by drawing then as arrows.
    """
    # dimensions of image
    H, W, _ = flow.shape

    # initialize image
    img = np.ones((H, W, 3), dtype=np.uint8) * 255  # white background

    # visual parameters
    color = (0, 0, 255) # color of OF arrows
    spacing = 20 # between each arrow

    # plot arrows
    for y in range(0, H, spacing):
        for x in range(0, W, spacing):
            dx, dy = flow[y, x]
            end_x = int(x + dx * scale)
            end_y = int(y + dy * scale)

            cv2.arrowedLine(img, (x, y), (end_x, end_y), color, 2, tipLength=0.3)

    # bookkeep
    img_PIL = Image.fromarray(img)
    img_PIL.save(output_path)

    return img_PIL

def visualize_interped_flow_fields(interped_flow, output_dir, scale=1.0):
    """
    Visualize each frame of interpolated optical flow
    """
    # flow out directory
    flow_out_dir = os.path.join(output_dir, "flow_frames")
    if not os.path.exists(flow_out_dir): os.mkdir(flow_out_dir)

    # turn List into np.array
    n = len(interped_flow)
    interped_flow = np.array(interped_flow)

    # intialize output
    output_frames = []
    for i in range(n):
        # get flow for the current frame
        frame_flow = interped_flow[i]

        # plot image
        output_path = os.path.join(flow_out_dir, f"flow_frame_{i}.png")
        img_PIL = visualize_flow_field(frame_flow, output_path, scale=scale)
        output_frames.append(img_PIL)

    return output_frames

"""
Visualize lines on a black image
Args
    interped_lines (List[np.array]) : list of lines, each line is a pair of points [[x1, y1], [x2, y2]]
    output_dir (string) : directory to save the visualization image
    width (int) : width of the output image
    height (int) : height of the output image

"""
def visualize_lines(interped_lines, output_dir, width, height):
    # lines out directory
    lines_out_dir = os.path.join(output_dir, "lines_flow_frames")
    if not os.path.exists(lines_out_dir): os.mkdir(lines_out_dir)

    # line color
    line_color = (0, 0, 255)  # white

    # plot lines
    num_frames = len(interped_lines)
    output_frames = []
    for i in range(num_frames):
        img = np.ones((height, width, 3), dtype=np.uint8) * 255  # black background
        lines = interped_lines[i]

        for line in lines:
            pt1 = tuple(map(int, line[0]))
            pt2 = tuple(map(int, line[1]))
            cv2.line(img, pt1, pt2, color=line_color, thickness=2)

        # save the image
        img_PIL = Image.fromarray(img)
        img_PIL.save(os.path.join(lines_out_dir, f"lines_frame_{i}.png"))
        output_frames.append(img_PIL)

    return output_frames