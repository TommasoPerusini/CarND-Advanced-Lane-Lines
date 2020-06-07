import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from constants import MEMORY_FRAMES, XM_PER_PIX
from calibration_tools import calibrate_camera, undistort
from thresholding_tools import binarize
from perspective_tools import birdeye
from line_utility import Line, get_fits_by_sliding_windows, draw_back_onto_the_road, get_fits_by_previous_fits

PROCESSED_FRAMES = 0                      # number of processed video frames  
LINE_LT = Line(buffer_len=MEMORY_FRAMES)  # LEFT Lane Line
LINE_RT = Line(buffer_len=MEMORY_FRAMES)  # RIGHT Lane Line


def generate_process_overlay(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    """
    Generate final output frame overlayed with all the blended processing steps
    :param blend_on_road: color image of lane blend onto the road
    :param img_binary: thresholded binary image
    :param img_birdeye: bird's eye view of the thresholded binary image
    :param img_fit: bird's eye view with detected lane-lines highlighted
    :param line_lt: detected left lane-line
    :param line_rt: detected right lane-line
    :param offset_meter: offset from the center of the lane
    :return: final frame showing all results overlayed
    """
    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

    # add thumbnail of bird's eye view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return blend_on_road

def camera_lane_distance(line_lt, line_rt, frame_width):
    """
    Compute offset from center of the inferred lane.
    The offset from the lane center can be computed under the hypothesis that the camera is fixed
    and mounted in the midpoint of the car roof. In this case, we can approximate the car's deviation
    from the lane center as the distance between the center of the image and the midpoint at the bottom
    of the image of the two lane-lines detected.
    :param line_lt: detected left lane-line
    :param line_rt: detected right lane-line
    :param frame_width: width of the undistorted frame
    :return: inferred offset
    """
    if line_lt.detected and line_rt.detected:
        line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
        line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = frame_width / 2
        offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
        offset_meter = XM_PER_PIX * offset_pix
    else:
        offset_meter = -1

    return offset_meter

def ALF_pipeline(frame, keep_state=True):
    """
    Advance Lane Finding pipeline applied to any input frame.
    :param frame: input frame
    :param keep_state: if True, memory of lane Lines is preserved for result smoothing purpose
    :return: results blended over the original frame
    """

    global LINE_LT, LINE_RT, PROCESSED_FRAMES

    # undistort the image using coefficients found in calibration
    img_undistorted = undistort(frame, mtx, dist, verbose=False)

    # binarize the frame s.t. lane lines are highlighted as much as possible
    img_binary = binarize(img_undistorted, verbose=False)

    # compute perspective transform to obtain bird's eye view
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

    # fit 2-degree polynomial curve onto lane lines found
    if PROCESSED_FRAMES > 0 and keep_state and LINE_LT.detected and LINE_RT.detected:
        LINE_LT, LINE_RT, img_fit = get_fits_by_previous_fits(img_birdeye, LINE_LT, LINE_RT, verbose=False)
    else:
        LINE_LT, LINE_RT, img_fit = get_fits_by_sliding_windows(img_birdeye, LINE_LT, LINE_RT, n_windows=9, verbose=False)

    # compute offset in meter from center of the lane
    offset_meter = camera_lane_distance(LINE_LT, LINE_RT, frame_width=frame.shape[1])

    # draw the surface enclosed by lane lines back onto the original frame
    overlay_on_lane = draw_back_onto_the_road(img_undistorted, Minv, LINE_LT, LINE_RT, keep_state)

    # stitch on the top of final output images from different steps of the pipeline
    frame_output = generate_process_overlay(overlay_on_lane, img_binary, img_birdeye, img_fit, LINE_LT, LINE_RT, offset_meter)

    PROCESSED_FRAMES += 1

    return frame_output



if __name__ == "__main__":
    print("Advance Lane Finding (ALF...")

    # Camera Calibration Step
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    TEST_IMG_DIR = "test_images"
    TEST_IMG_DIR = "debug_images"

    MODE = 'video'
    
    # PIPELINE ON VIDEO
    if(MODE == 'video'):
        input_video_filename = 'project_video' 
        input_video = VideoFileClip(input_video_filename + '.mp4')
        # input_video.save_frame('debug_images/frame_test2.jpg', 22)
        processed_video = input_video.fl_image(ALF_pipeline)
        processed_video.write_videofile(input_video_filename + 'FINAL_ALF.mp4', audio=False)
    
    # PIPELINE ON SINGLE FRAMES
    else:
        index = 0
        for test_img in os.listdir(TEST_IMG_DIR):
            index += 1

            print(os.path.join(TEST_IMG_DIR, test_img))
            frame = cv2.imread(os.path.join(TEST_IMG_DIR, test_img))
            
            blend = ALF_pipeline(frame, keep_state=True)

            cv2.imwrite('output_images/{}'.format(test_img), blend)

            # fig = plt.figure()
            # plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
            # debug_filename = 'debug_images/temp' + str(index) + '.png'
            # fig.savefig(debug_filename, dpi=fig.dpi)
            # print(debug_filename)

            # # DECOMMENT FOR GUI SHOW
            # plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
            # plt.show(