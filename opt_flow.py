#!/usr/bin/env python

'''
example to show optical flow

USAGE: opt_flow.py [<video_source>]

Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
from scipy import io


def draw_flow(img, flow, step=4):
    h, w = img.shape[:2]
    win_h = 3
    win_w = 3
    y, x = np.mgrid[win_h:h:step, win_w:w:step].reshape(2, -1).astype(int)

    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.polylines(vis, lines, 0, (0, 0, 255),2)
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 2, (0, 255, 0), -1)
    return vis


def cal_force(img, flow, step=4):
    h, w = flow.shape[:2]
    win_h = 3
    win_w = 3
    y, x = np.mgrid[win_h:h - win_h:step, win_w:w - win_w:step].reshape(2, -1).astype(int)

    # sample particles,and calculate F_int
    particles = np.vstack([x, y])
    force_flow = np.zeros((img.shape[0], img.shape[1]))
    F_ints = np.zeros(particles.shape[1])
    for i in range(particles.shape[1]):
        px, py = particles[:, i]
        win = np.mgrid[(px - 1):(px + 2):1, py - 1:py + 2:1].reshape(2, -1).astype(int)
        sum = [0, 0]
        for j in range(win.shape[1]):
            sum = np.add(flow[win[1, j], win[0, j]], sum)
        particle_ave = sum / 9  # O_ave(x,y)
        particle_act = flow[py, px]  # O(x,y)
        F_int = np.add(particle_act, -1 * particle_ave)
        F_int_val = F_int[0] * F_int[0] + F_int[1] * F_int[1]
        F_ints[i] = F_int_val
        force_flow[py, px] = F_int_val
    particles = np.vstack([x, y, F_ints])
    hsv = np.zeros((img.shape[0], img.shape[1],3),np.uint8)
    hsv[..., 1] = cv2.normalize(force_flow, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 0] = cv2.normalize(force_flow, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = cv2.normalize(force_flow, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    """
    print (np.minimum(force_flow*10, 255))
    hsv[..., 1] = np.minimum(force_flow*100, 255)
    hsv[..., 0] = np.minimum(force_flow*100, 255)
    hsv[..., 2] = np.minimum(force_flow*100, 255)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    """
    return rgb


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = np.minimum(v * 4, 255)
    hsv[..., 1] = np.minimum(v * 4, 255)
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


if __name__ == '__main__':
    import sys

    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    cam = cv2.VideoCapture('1_ab.avi')

    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    show_flow = False
    cur_glitch = prev.copy()
    ii=0
    while True:

        ret, img = cam.read()
        if (ret == True):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prevgray = gray

            cv2.imshow('particle', draw_flow(gray, flow))
            cv2.imshow('video', img)
            cv2.imshow('force', cal_force(gray, flow))
            ii+=1
            print(ii)
            if show_hsv:
                cv2.imshow('flow HSV', draw_hsv(flow))
            if show_glitch:
                cur_glitch = warp_flow(cur_glitch, flow)
                cv2.imshow('glitch', cur_glitch)
            if show_flow:
                flow_t = draw_flow(gray, flow)
                cv2.imshow('flow', flow_t)

            ch = cv2.waitKey(5)
            if ch == 27:
                break
            if ch == ord('1'):
                show_hsv = not show_hsv
                print('HSV flow visualization is', ['off', 'on'][show_hsv])
            if ch == ord('2'):
                show_glitch = not show_glitch
                if show_glitch:
                    cur_glitch = img.copy()
                print('glitch is', ['off', 'on'][show_glitch])
            if ch == ord('3'):
                show_flow = not show_flow
                print('show flow')
    cv2.destroyAllWindows()
