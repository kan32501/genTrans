from utils.file_io import init_results_dirs, extract_frames_from_mp4
from utils.eval_models import *
from utils.image_operations import *
from utils.line_features import *
from utils.optical_flow import *
from utils.visualization import *
from utils.baselines import *
from diffusers.utils import export_to_video

def generate_transition(args, GlueStick_model, FCVG_model, SEARAFT_model, progress=True, visualize=True):
    # progress
    if progress: print("\n–– GENERATING TRANSITION ––")

    # create a new numbered directory for this iterations in /results, with subfolders
    curr_trial_dir, inputs_dir, conditions_dir, out_frames_dir = init_results_dirs(args.output_dir)

    """
    # produce baseline transition using a simple cross dissolve to compare against:
    # WARNING: this takes a long time
    # make out_frames_dir
    x_dissolve_frames_dir = os.path.join(curr_trial_dir, "xdissolve_out_frames")
    if not os.path.exists(out_frames_dir): os.mkdir(out_frames_dir)

    # get the cross dissolve frames
    start_frame = Image.open(os.path.join(args.video0_frames_dir, os.listdir(args.video0_frames_dir)[args.frame0_index]))
    end_frame = Image.open(os.path.join(args.videoN_frames_dir, os.listdir(args.videoN_frames_dir)[args.frameN_index]))
    x_dissolve_frames = cross_dissolve(start_frame, end_frame, args.frame_count, args.width, args.height)

    # save the frames
    save_out_frames(x_dissolve_frames, 
                    "xdissolve", x_dissolve_frames_dir, curr_trial_dir,
                    bef_and_aft=True,
                    video0_frames_dir=args.video0_frames_dir, frame0_index=args.frame0_index, 
                    videoN_frames_dir=args.videoN_frames_dir, frameN_index=args.frameN_index, 
                    width=args.width, height=args.height)

    """
    # get input frames
    inputs = get_input_frames_by_index(args.video0_frames_dir, args.videoN_frames_dir, 
                                       args.frame0_mask_path, args.frameN_mask_path,
                                       args.width, args.height,
                                       inputs_dir, save=True)
    

    # get preprocessed input images as PIL Images
    frame0_prev, frame0 = inputs["frame0_prev"][1], inputs["frame0"][1]
    frameN, frameN_next = inputs["frameN"][1], inputs["frameN_next"][1]

    # get line matches with gluestick, convert to polar form, get midpoints
    lines0, linesN = infer_gluestick(GlueStick_model, frame0, frameN)
    if progress: 
        print("-> Lines detected")

    # load masks
    frame0_mask = load_mask(args.frame0_mask_path, args.width, args.height, as_numpy=False)
    frameN_mask = load_mask(args.frameN_mask_path, args.width, args.height, as_numpy=False)

    # mask image inverted
    frame0_mask_inv = load_mask(args.frame0_mask_path, args.width, args.height, as_numpy=False, invert=True)
    frameN_mask_inv = load_mask(args.frameN_mask_path, args.width, args.height, as_numpy=False, invert=True)

    # get the line matches across the start and end image in the mask, outside the mask
    matched_lines0, matched_linesN = match_lines(lines0, frame0_mask, linesN, frameN_mask,
                                                 with_flow=False, args=args,
                                                 visualize=visualize, out_dir=curr_trial_dir)
    matched_lines0_inv, matched_linesN_inv = match_lines(lines0, frame0_mask_inv, linesN, frameN_mask_inv,
                                                 with_flow=True, args=args, SEARAFT_model=SEARAFT_model, frame0_prev=frame0_prev, frame0=frame0, frameN=frameN, frameN_next=frameN_next, 
                                                 visualize=visualize, out_dir=curr_trial_dir)

    # combine the matched lines within the masks, combine the matched lines outside the mask
    matched_lines0 = np.concatenate((matched_lines0, matched_lines0_inv), axis=0)
    matched_linesN = np.concatenate((matched_linesN, matched_linesN_inv), axis=0)

    # visualize all line matches
    if visualize:
        lines_dir = curr_trial_dir + "/features_full"
        if not os.path.exists(lines_dir): os.mkdir(lines_dir)
        white = Image.fromarray(np.ones((args.height, args.width, 3), dtype=np.uint8) * 255)
        match_0_img = plot_lines(white, matched_lines0, color=True, number=True)
        match_N_img = plot_lines(white, matched_linesN, color=True, number=True)
        match_0_img.save(os.path.join(lines_dir, "frame0_lines_colored_numbered.png"))
        match_N_img.save(os.path.join(lines_dir, "frameN_lines_colored_numbered.png"))
        match_0_img = plot_lines(white, matched_lines0, color=True, number=False)
        match_N_img = plot_lines(white, matched_linesN, color=True, number=False)
        match_0_img.save(os.path.join(lines_dir, "frame0_lines_colored.png"))
        match_N_img.save(os.path.join(lines_dir, "frameN_lines_colored.png"))
    
    # interpolate the detected lines in line space
    interped_lines = interpolate_lines(matched_lines0, matched_linesN, num_frames=args.frame_count)
    # interped_flow = interpolate_flow(flow0_all, flowN_all, num_frames=args.frame_count)

    # generate c1-c2, the frame-wise conditions
    framewise_cond_imgs = plot_condition_imgs(frame0, interped_lines, lw=2, save=True, out_dir=conditions_dir) 
    # progress update
    if progress: print("-> Generated frame-wise conditions")
    
    # interped_lines_flow = apply_flow_to_lines(interped_lines, interped_flow)
    # visualize_lines(interped_lines_flow, curr_trial_dir, args.width, args.height)

    # run inference on diffusion model 
    if progress: print("-> Running video diffusion pipeline")
    video_frames = FCVG_model(
        frame0, # start image
        frameN, # end image
        framewise_cond_imgs, # control images
        decode_chunk_size=2,
        num_frames=args.frame_count,
        motion_bucket_id=127.0, 
        fps=7,
        control_weight=args.control_weight, 
        width=args.width, 
        height=args.height, 
        min_guidance_scale=3.0, 
        max_guidance_scale=3.0, 
        frames_per_batch=args.batch_frames, 
        num_inference_steps=args.num_inference_steps, 
        overlap=args.overlap).frames
    
    # flatten the output into one list of result images
    inbetween_frames = [img for sublist in video_frames for img in sublist]
    # inbetween_frames_flow = [apply_flow_to_image(inbetween_frames[i], interped_flow[i]) for i in range(len(inbetween_frames))]

    # # output the baseline as a gif
    # baseline_frames_path = "./example/nikolaisavic/f1_surf_DM"
    # baseline_frames = sorted(os.listdir(baseline_frames_path))
    # inbetween_frames = [Image.open(os.path.join(baseline_frames_path, frame)).convert("RGB").resize((args.width, args.height)) for frame in baseline_frames]

    # save generated inbetween frames as PNGs and as a gif in results/xxx/out_frames
    out_gif_path, out_frames = save_out_frames(inbetween_frames, 
                                               "morphwarp", out_frames_dir, curr_trial_dir,
                                               bef_and_aft=True,
                                               video0_frames_dir=args.video0_frames_dir, frame0_index=args.frame0_index,
                                               videoN_frames_dir=args.videoN_frames_dir, frameN_index=args.frameN_index,
                                               width=args.width, height=args.height)
    print("–– OUTPUT TRANSITION GIF SAVED IN " + out_gif_path + " ––")

    # export and save generated inbetween frames as MP4
    out_mp4_path = os.path.join(curr_trial_dir, 'morphwarp.mp4')
    export_to_video(out_frames, out_mp4_path)
    print("–– OUTPUT TRANSITION MP4 SAVED IN " + out_mp4_path + " ––")


def match_lines(lines0, frame0_mask, linesN, frameN_mask,
                with_flow, args, SEARAFT_model=None, frame0_prev=None, frame0=None, frameN=None, frameN_next=None,
                progress=True, 
                visualize=False, out_dir=None):
    """
    match the deteced lines between start and end image
    """
    # filter lines out by the provided masks
    lines0 = filter_lines_in_mask(frame0_mask, lines0)
    linesN = filter_lines_in_mask(frameN_mask, linesN)

    # normalize the lines based on the mask
    norm_lines0 = normalize_lines_by_mask(frame0_mask, lines0)
    norm_linesN = normalize_lines_by_mask(frameN_mask, linesN)

    # extract features of raw lines for visualization
    midpoints0, midpointsN = get_midpoints(lines0), get_midpoints(linesN)

    # extract features of normalized lines
    norm_hough0, norm_houghN = cartesian_to_polar(lines0), cartesian_to_polar(linesN)
    norm_midpoints0, norm_midpointsN = get_midpoints(norm_lines0), get_midpoints(norm_linesN)
   
    # initialize matches
    matched_indices0, matched_indicesN = None, None
    if not with_flow:
        # run a greedy matching algorithm to match lines based on position/orientation/flow metrics
        matched_indices0, matched_indicesN = greedy_match_lines(norm_hough0, norm_houghN, norm_midpoints0, norm_midpointsN, with_flow)
    else:
        # detect optical flow
        flow0_all, flowN_all = infer_SEARAFT(SEARAFT_model, args, frame0_prev, frame0, frameN, frameN_next)

        # normalize the optical flow to [0,1] range
        flow0_all_norm, flowN_all_norm = normalize_flow(flow0_all), normalize_flow(flowN_all)
        if progress: print("-> Optical flow detected outside mask")

        # query flow at midpoints only, and convert to angles
        flow0_midpts = query_flow_at_points(flow0_all_norm, midpoints0)
        flow0_midpts = np.concatenate((flow0_midpts, np.zeros_like(flow0_midpts)), axis=1)
        flow0_midpts_hough = cartesian_to_polar(flow0_midpts)
        flowN_midpts = query_flow_at_points(flowN_all_norm, midpointsN)
        flowN_midpts = np.concatenate((flowN_midpts, np.zeros_like(flowN_midpts)), axis=1)
        flowN_midpts_hough = cartesian_to_polar(flowN_midpts)

        matched_indices0, matched_indicesN = greedy_match_lines(norm_hough0, norm_houghN, norm_midpoints0, norm_midpointsN, with_flow, flow0_midpts_hough, flowN_midpts_hough)

    # reorder the lines and other parameters based on the matched indices
    matched_lines0, matched_linesN = lines0[matched_indices0], linesN[matched_indicesN]

    # visualize the line features
    if visualize:
        # make directories for flow and features
        features_dir = out_dir + "/features_fg" if not with_flow else out_dir + "/features_bg"
        flow_dir = out_dir + "/flow_field"
        # interped_flow_dir = flow_dir + "/interped_flow"
        if not os.path.exists(features_dir): os.mkdir(features_dir)
        if with_flow and not os.path.exists(flow_dir): os.mkdir(flow_dir)
        # if not os.path.exists(interped_flow_dir): os.mkdir(interped_flow_dir)

        # visualize the line features
        matched_midpoints0, matched_midpointsN = get_midpoints(matched_lines0), get_midpoints(matched_linesN)
        matched_flow0 = query_flow_at_points(flow0_all_norm, matched_midpoints0) if with_flow else None
        matched_flowN = query_flow_at_points(flowN_all_norm, matched_midpointsN) if with_flow else None
        visualize_image_features(args.height, args.width,
                                 matched_lines0, matched_linesN, 
                                 matched_midpoints0, matched_midpointsN, 
                                 features_dir,
                                 matched_flow0, matched_flowN,
                                 frame0_mask, frameN_mask)

        # visualize optical flow field
        if with_flow:
            visualize_flow_field(flow0_all, os.path.join(flow_dir, "frame0_flow_raw.png"), scale=100.0)
            visualize_flow_field(flow0_all_norm, os.path.join(flow_dir, "frame0_flow_norm.png"), scale=100.0)
            visualize_flow_field(flowN_all, os.path.join(flow_dir, "frameN_flow_raw.png"), scale=100.0)
            visualize_flow_field(flowN_all_norm, os.path.join(flow_dir, "frameN_flow_norm.png"), scale=100.0)

    return matched_lines0, matched_linesN