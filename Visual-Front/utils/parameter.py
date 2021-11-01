import cv2
import argparse

def read_image(img, img_size):

    if img is None:
      return (None, False)
    interp = cv2.INTER_AREA
    grayim = cv2.resize(img, (img_size[1], img_size[0]), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    grayim = grayim.astype('float32')

    return (grayim, True)


def readParameters():

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='PyTorch SuperPointNet.')

    parser.add_argument('--weights_path', type=str, default='SuperPoint_GhostNet.pth.tar',
        help='Path to pretrained weights file \
            (default: SuperPoint_VGG.pth.tar / SuperPoint_GhostNet.pth.tar / SuperPoint_MobileNet.pth.tar / superpoint_v1).')
    
    parser.add_argument('--H', type=int, default=480,
        help='Input image height (default: 120).')
    
    parser.add_argument('--W', type=int, default=752,
        help='Input image width (default:752 / 640).')
    
    parser.add_argument('--scale', type=int, default=2,
        help='Factor to scale output visualization (default: 2).')
    
    parser.add_argument('--nms_dist', type=int, default=4,
        help='Non Maximum Suppression (NMS) distance (default: 4 / 8 / 12 / 16).')
    
    parser.add_argument('--conf_thresh', type=float, default=0.015,
        help='Detector confidence threshold (default: 0.015).')
    
    parser.add_argument('--nn_thresh', type=float, default=0.7,
        help='Descriptor matching threshold (default: 0.7).')

    parser.add_argument('--max_cnt', type=int, default=150,
        help='Max feature number in feature tracking (default: 150).')
    
    parser.add_argument('--cuda', action='store_false',
        help='Use cuda GPU to speed up network processing speed (default: True)')
    
    parser.add_argument('--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely (default: False).')

    opts = parser.parse_args()

    return opts