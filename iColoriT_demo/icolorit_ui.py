import sys
import io
import os
import argparse
import numpy as np
import datetime
import glob

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen
from timm.models import create_model
from flask import Flask, render_template, request, url_for, redirect
from flaskwebgui import FlaskUI
# from flaskwebgui import FlaskUI
import webbrowser as wb
from importlib import import_module
from flask_restx import Resource, Api
from skimage import color

import base64
# from threading import Thread
import socket
import torch
#
from io import BytesIO
import cv2
from PIL import Image
import base64
from base64 import b64encode
from gui import gui_draw
from gui import gui_main
import modeling
from einops import rearrange
from gui.lab_gamut import abGrid, lab2rgb_1d, rgb2lab_1d


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_rpb=args.use_rpb,
        avg_hint=args.avg_hint,
        head_mode=args.head_mode,
        mask_cent=args.mask_cent,
    )
    return model

def get_args(filepath):
    parser = argparse.ArgumentParser('Colorization UI', add_help=False)
    # Directories
    parser.add_argument('--model_path', type=str, default='path/to/checkpoints', help='checkpoint path of model')
    parser.add_argument('--target_image', default=filepath, type=str, help='validation dataset path')
    parser.add_argument('--device', default='cpu', help='device to use for testing')

    # Dataset parameters
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')

    # Model parameters
    parser.add_argument('--model', default='icolorit_base_4ch_patch16_224', type=str, help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate (default: 0.1)')
    parser.add_argument('--use_rpb', action='store_true', help='relative positional bias')
    parser.add_argument('--no_use_rpb', action='store_false', dest='use_rpb')
    parser.set_defaults(use_rpb=True)
    parser.add_argument('--avg_hint', action='store_true', help='avg hint')
    parser.add_argument('--no_avg_hint', action='store_false', dest='avg_hint')
    parser.set_defaults(avg_hint=True)
    parser.add_argument('--head_mode', type=str, default='cnn', help='head_mode')
    parser.add_argument('--mask_cent', action='store_true', help='mask_cent')

    args = parser.parse_args()

    return args

class GUIPalette():
    def __init__(self, grid_sz=(6, 3)):
        self.color_width = 25
        self.border = 6
        self.win_width = grid_sz[0] * self.color_width + (grid_sz[0] + 1) * self.border
        self.win_height = grid_sz[1] * self.color_width + (grid_sz[1] + 1) * self.border
        self.setFixedSize(self.win_width, self.win_height)
        self.num_colors = grid_sz[0] * grid_sz[1]
        self.grid_sz = grid_sz
        self.colors = None
        self.color_id = -1
        self.reset()

    def set_colors(self, colors):
        if colors is not None:
            self.colors = (colors[:min(colors.shape[0], self.num_colors), :] * 255).astype(np.uint8)
            self.color_id = -1
            # self.update()

    def paintEvent(self, event):
        # painter = QPainter()
        # painter.begin(self)
        # painter.setRenderHint(QPainter.Antialiasing)
        # painter.fillRect(event.rect(), Qt.white)
        if self.colors is not None:
            for n, c in enumerate(self.colors):
                # ca = QColor(c[0], c[1], c[2], 255)
                # painter.setPen(QPen(Qt.black, 1))
                # painter.setBrush(ca)
                grid_x = n % self.grid_sz[0]
                grid_y = (n - grid_x) // self.grid_sz[0]
                x = grid_x * (self.color_width + self.border) + self.border
                y = grid_y * (self.color_width + self.border) + self.border

                if n == self.color_id:
                    painter.drawEllipse(x, y, self.color_width, self.color_width)
                else:
                    painter.drawRoundedRect(x, y, self.color_width, self.color_width, 2, 2)

        painter.end()

    def sizeHint(self):
        return QSize(self.win_width, self.win_height)

    def reset(self):
        self.colors = None
        self.mouseClicked = False
        self.color_id = -1
        self.update()

    def selected_color(self, pos):
        width = self.color_width + self.border
        dx = pos.x() % width
        dy = pos.y() % width
        if dx >= self.border and dy >= self.border:
            x_id = (pos.x() - dx) // width
            y_id = (pos.y() - dy) // width
            color_id = x_id + y_id * self.grid_sz[0]
            return int(color_id)
        else:
            return -1

    def update_ui(self, color_id):
        self.color_id = int(color_id)
        self.update()
        if color_id >= 0 and self.colors is not None:
            print('choose color (%d) type (%s)' % (color_id, type(color_id)))
            color = self.colors[color_id]
            # self.emit(SIGNAL('update_color'), color)
            self.update_color.emit(color)
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:  # click the point
            color_id = self.selected_color(event.pos())
            self.update_ui(color_id)
            self.mouseClicked = True

    def mouseMoveEvent(self, event):
        if self.mouseClicked:
            color_id = self.selected_color(event.pos())
            self.update_ui(color_id)

    def mouseReleaseEvent(self, event):
        self.mouseClicked = False


class GUIGamut():
    def __init__(self, gamut_size=110):
        self.gamut_size = gamut_size
        self.win_size = gamut_size * 2  # divided by 4
        # self.setFixedSize(self.win_size, self.win_size)
        self.ab_grid = abGrid(gamut_size=gamut_size, D=1)
        self.reset()
        # self.update()

    def set_gamut(self, l_in=50):
        self.l_in = l_in
        self.ab_map, self.mask = self.ab_grid.update_gamut(l_in=l_in)
        # self.update()

    def set_ab(self, color):
        self.color = color
        self.lab = rgb2lab_1d(self.color)
        x, y = self.ab_grid.ab2xy(self.lab[1], self.lab[2])
        self.pos = QPointF(x, y)
        # self.update()

    def is_valid_point(self, pos):
        if pos is None or self.mask is None:
            return False
        else:
            x = pos.x()
            y = pos.y()
            if x >= 0 and y >= 0 and x < self.win_size and y < self.win_size:
                return self.mask[y, x]
            else:
                return False

    def update_ui(self, pos):
        self.pos = pos
        a, b = self.ab_grid.xy2ab(pos.x(), pos.y())
        # get color we need L
        L = self.l_in
        lab = np.array([L, a, b])
        color = lab2rgb_1d(lab, clip=True, dtype='uint8')
        # self.emit(SIGNAL('update_color'), color)
        self.update_color.emit(color)
        # self.update()

    def paintEvent(self):
        # painter = QPainter()
        # painter.begin(self)
        # painter.setRenderHint(QPainter.Antialiasing)
        # painter.fillRect(event.rect(), Qt.white)
        if self.ab_map is not None:
            ab_map = cv2.resize(self.ab_map, (self.win_size, self.win_size))
            pil_image = Image.fromarray(ab_map)

            img_str = cv2.imencode('.png', ab_map)[1].tostring()

            image_io = BytesIO(img_str)
            pil_image.save(image_io, 'PNG')
            dataurl = 'data:image/png;base64,' + b64encode(image_io.getvalue()).decode('ascii')

            return dataurl

        # if self.pos is not None:
            # painter.setPen(QPen(Qt.black, 2, Qt.SolidLine, cap=Qt.RoundCap, join=Qt.RoundJoin))
        #     w = 5
        #     x = self.pos.x()
        #     y = self.pos.y()
        #     painter.drawLine(x - w, y, x + w, y)
        #     painter.drawLine(x, y - w, x, y + w)
        # painter.end()

    def mousePressEvent(self, event):
        pos = event.pos()

        if event.button() == Qt.LeftButton and self.is_valid_point(pos):  # click the point
            self.update_ui(pos)
            self.mouseClicked = True

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self.is_valid_point(pos):
            if self.mouseClicked:
                self.update_ui(pos)

    def mouseReleaseEvent(self, event):
        self.mouseClicked = False

    def sizeHint(self):
        return QSize(self.win_size, self.win_size)

    def reset(self):
        self.ab_map = None
        self.mask = None
        self.color = None
        self.lab = None
        self.pos = None
        self.mouseClicked = False
        # self.update()
class UserEdit(object):
    def __init__(self, mode, win_size, load_size, img_size):
        self.mode = mode
        self.win_size = win_size
        self.img_size = img_size
        self.load_size = load_size
        print('image_size', self.img_size)
        max_width = np.max(self.img_size)
        self.scale = float(max_width) / self.load_size # original image to 224 ration
        self.dw = int((self.win_size - img_size[0]) // 2)
        self.dh = int((self.win_size - img_size[1]) // 2)
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.ui_count = 0
        print(self)

    def scale_point(self, in_x, in_y, w):
        x = int((in_x - self.dw) / float(self.img_w) * self.load_size) + w
        y = int((in_y - self.dh) / float(self.img_h) * self.load_size) + w
        return x, y

    def __str__(self):
        return "add (%s) with win_size %3.3f, load_size %3.3f" % (self.mode, self.win_size, self.load_size)


class PointEdit(UserEdit):
    def __init__(self, win_size, load_size, img_size):
        UserEdit.__init__(self, 'point', win_size, load_size, img_size)

    def add(self, pnt, color, userColor, width, ui_count):
        self.pnt = pnt
        self.color = color
        self.userColor = userColor
        self.width = width
        self.ui_count = ui_count

    def select_old(self, pnt, ui_count):
        self.pnt = pnt
        self.ui_count = ui_count
        return self.userColor, self.width

    def update_color(self, color, userColor):
        self.color = color
        self.userColor = userColor

    def updateInput(self, im, mask, vis_im):
        w = int(self.width / self.scale)
        pnt = self.pnt
        x1, y1 = self.scale_point(pnt.x(), pnt.y(), -w)
        tl = (x1, y1)
        # x2, y2 = self.scale_point(pnt.x(), pnt.y(), w)
        # br = (x2, y2)
        br = (x1+1, y1+1) # hint size fixed to 2
        c = (self.color.red(), self.color.green(), self.color.blue())
        uc = (self.userColor.red(), self.userColor.green(), self.userColor.blue())
        cv2.rectangle(mask, tl, br, 0, -1)
        cv2.rectangle(im, tl, br, c, -1)
        cv2.rectangle(vis_im, tl, br, uc, -1)

    def is_same(self, pnt):
        dx = abs(self.pnt.x() - pnt.x())
        dy = abs(self.pnt.y() - pnt.y())
        return dx <= self.width + 1 and dy <= self.width + 1

    def update_painter(self, painter):
        w = max(3, self.width)
        c = self.color
        r = c.red()
        g = c.green()
        b = c.blue()
        ca = QColor(c.red(), c.green(), c.blue(), 255)
        d_to_black = r * r + g * g + b * b
        d_to_white = (255 - r) * (255 - r) + (255 - g) * (255 - g) + (255 - r) * (255 - r)
        if d_to_black > d_to_white:
            painter.setPen(QPen(Qt.black, 1))
        else:
            painter.setPen(QPen(Qt.white, 1))
        painter.setBrush(ca)
        painter.drawRoundedRect(self.pnt.x() - w, self.pnt.y() - w, 1 + 2 * w, 1 + 2 * w, 2, 2)

class UIControl:
    def __init__(self, win_size=256, load_size=224):
        self.win_size = win_size
        self.load_size = load_size
        self.reset()
        self.userEdit = None
        self.userEdits = []
        self.ui_count = 0

    def setImageSize(self, img_size):
        self.img_size = img_size

    def addStroke(self, prevPnt, nextPnt, color, userColor, width):
        pass

    def erasePoint(self, pnt):
        isErase = False
        for id, ue in enumerate(self.userEdits):
            if ue.is_same(pnt):
                self.userEdits.remove(ue)
                print('remove user edit %d\n' % id)
                isErase = True
                break
        return isErase

    def addPoint(self, pnt, color, userColor, width):
        self.ui_count += 1
        print('process add Point')
        self.userEdit = None
        isNew = True
        for id, ue in enumerate(self.userEdits):
            if ue.is_same(pnt):
                self.userEdit = ue
                isNew = False
                print('select user edit %d\n' % id)
                break

        if self.userEdit is None:
            self.userEdit = PointEdit(self.win_size, self.load_size, self.img_size)
            self.userEdits.append(self.userEdit)
            print('add user edit %d\n' % len(self.userEdits))
            self.userEdit.add(pnt, color, userColor, width, self.ui_count)
            return userColor, width, isNew
        else:
            userColor, width = self.userEdit.select_old(pnt, self.ui_count)
            return userColor, width, isNew

    def movePoint(self, pnt, color, userColor, width):
        self.userEdit.add(pnt, color, userColor, width, self.ui_count)

    def update_color(self, color, userColor):
        self.userEdit.update_color(color, userColor)

    def update_painter(self, painter):
        for ue in self.userEdits:
            if ue is not None:
                ue.update_painter(painter)

    def get_stroke_image(self, im):
        return im

    def used_colors(self):  # get recently used colors
        if len(self.userEdits) == 0:
            return None
        nEdits = len(self.userEdits)
        ui_counts = np.zeros(nEdits)
        ui_colors = np.zeros((nEdits, 3))
        for n, ue in enumerate(self.userEdits):
            ui_counts[n] = ue.ui_count
            c = ue.userColor
            ui_colors[n, :] = [c.red(), c.green(), c.blue()]

        ui_counts = np.array(ui_counts)
        ids = np.argsort(-ui_counts)
        ui_colors = ui_colors[ids, :]
        unique_colors = []
        for ui_color in ui_colors:
            is_exit = False
            for u_color in unique_colors:
                d = np.sum(np.abs(u_color - ui_color))
                if d < 0.1:
                    is_exit = True
                    break

            if not is_exit:
                unique_colors.append(ui_color)

        unique_colors = np.vstack(unique_colors)
        return unique_colors / 255.0

    def get_input(self):
        h = self.load_size
        w = self.load_size
        im = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w, 1), np.uint8)
        vis_im = np.zeros((h, w, 3), np.uint8)

        for ue in self.userEdits:
            ue.updateInput(im, mask, vis_im)

        return im, mask

    def reset(self):
        self.userEdits = []
        self.userEdit = None
        self.ui_count = 0





class GUIDraw():
    def __init__(self, model=None, load_size=224, win_size=512, device='cpu'):
            self.image_file = None
            self.pos = None
            self.model = model
            self.win_size = win_size
            self.load_size = load_size
            self.device = device
            # self.setFixedSize(win_size, win_size)
            self.uiControl = UIControl(win_size=win_size, load_size=load_size)
            # self.move(win_size, win_size)
            # self.movie = True
            # self.init_color()  # initialize color
            self.im_gray3 = None
            # self.eraseMode = False
            # self.ui_mode = 'none'  # stroke or point
            self.image_loaded = False
            self.use_gray = True
            self.total_images = 0
            self.image_id = 0

    def read_image(self, image_file):
        # self.result = None
        self.image_loaded = True
        self.image_file = image_file
        # print(image_file)
        im_bgr = cv2.imread(self.image_file)
        self.im_full = im_bgr.copy()
        # get image for display
        h, w, c = self.im_full.shape
        max_width = max(h, w)
        r = self.win_size / float(max_width)
        self.scale = float(self.win_size) / self.load_size
        print('scale = %f' % self.scale)
        rw = int(round(r * w / 4.0) * 4)
        rh = int(round(r * h / 4.0) * 4)

        self.im_win = cv2.resize(self.im_full, (rw, rh), interpolation=cv2.INTER_CUBIC)

        self.dw = int((self.win_size - rw) // 2)
        self.dh = int((self.win_size - rh) // 2)
        self.win_w = rw
        self.win_h = rh
        self.uiControl.setImageSize((rw, rh))
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        self.im_gray3 = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

        self.gray_win = cv2.resize(self.im_gray3, (rw, rh), interpolation=cv2.INTER_CUBIC)
        im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
        self.im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        lab_win = color.rgb2lab(self.im_win[:, :, ::-1])

        self.im_lab = color.rgb2lab(im_bgr[:, :, ::-1])
        self.im_l = self.im_lab[:, :, 0]
        self.l_win = lab_win[:, :, 0]
        self.im_ab = self.im_lab[:, :, 1:]
        self.im_size = self.im_rgb.shape[0:2]

        self.im_ab0 = np.zeros((2, self.load_size, self.load_size))
        self.im_mask0 = np.zeros((1, self.load_size, self.load_size))
        self.brushWidth = 2 * self.scale

    def init_result(self, image_file):
        self.read_image(image_file.encode('utf-8'))  # read an image
        self.read_image(image_file)  # read an image
        self.reset()

    def get_input(self):
            h = self.load_size
            w = self.load_size
            im = np.zeros((h, w, 3), np.uint8)
            mask = np.zeros((h, w, 1), np.uint8)
            vis_im = np.zeros((h, w, 3), np.uint8)

            for ue in self.userEdits:
                    ue.updateInput(im, mask, vis_im)

            return im, mask
    def compute_result(self, colormodel):
        self.model = colormodel
        im, mask = self.uiControl.get_input()
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2, 0, 1)) # (1, H, W)
        im_lab = color.rgb2lab(im).transpose((2, 0, 1))#(3, H, W)
        self.im_ab0 = im_lab[1:3, :, :]

        # _im_lab is 1) normalized 2) a torch tensor
        _im_lab = self.im_lab.transpose((2,0,1))
        _im_lab = np.concatenate(((_im_lab[[0], :, :]-50) / 100, _im_lab[1:, :, :] / 110), axis=0)
        _im_lab = torch.from_numpy(_im_lab).type(torch.FloatTensor).to(self.device)

        # _img_mask is 1) normalized ab 2) flipped mask
        _img_mask = np.concatenate((self.im_ab0 / 110, (255-self.im_mask0) / 255), axis=0)
        _img_mask = torch.from_numpy(_img_mask).type(torch.FloatTensor).to(self.device)

        # _im_lab is the full color image, _img_mask is the ab_hint+mask
        ab = self.model(_im_lab.unsqueeze(0), _img_mask.unsqueeze(0))
        ab = rearrange(ab, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
                        h=self.load_size//self.model.patch_size, w=self.load_size//self.model.patch_size,
                        p1=self.model.patch_size, p2=self.model.patch_size)[0]
        ab = ab.detach().numpy()

        ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_CUBIC)
        ab_win = ab_win * 110
        pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
        pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
        self.result = pred_rgb
        return self.result
        # self.emit(SIGNAL('update_result'), self.result)
        # self.update()

    def reset(self):
        self.ui_mode = 'none'
        self.pos = None
        self.result = None
        self.user_color = None
        self.color = None
        self.uiControl.reset()
        self.init_color()
        self.compute_result()
        self.update()
    def save_result(self):
        path = os.path.abspath(self.image_file)
        path, ext = os.path.splitext(path)

        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = "_".join([path, suffix])

        print('saving result to <%s>\n' % save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        mask = self.im_mask0.transpose((1, 2, 0)).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(save_path, 'input_mask.png'), mask)
        cv2.imwrite(os.path.join(save_path, 'ours.png'), result_bgr)
    
    # gamut 연관있는 함수
    def mousePressEvent(self, event):
        print('mouse press', event.pos())
        pos = self.valid_point(event.pos())

        if pos is not None:
            if event.button() == Qt.LeftButton:
                self.pos = pos
                self.ui_mode = 'point'
                self.change_color(pos)
                self.update_ui(move_point=False)
                self.compute_result()

            if event.button() == Qt.RightButton:
                # draw the stroke
                self.pos = pos
                self.ui_mode = 'erase'
                self.update_ui(move_point=False)
                self.compute_result()
    def mouseMoveEvent(self, event):
        self.pos = self.valid_point(event.pos())
        if self.pos is not None:
            if self.ui_mode == 'point':
                self.update_ui(move_point=True)
                self.compute_result()

    def paintEvent(self):
        # painter = QPainter()
        # painter.begin(self)
        # painter.fillRect(event.rect(), QColor(49, 54, 49))
        # painter.setRenderHint(QPainter.Antialiasing)
        if self.use_gray or self.result is None:
            im = self.gray_win
        else:
            im = self.result

        if im is not None:
            print(im)
            # qImg = QImage(im.tostring(), im.shape[1], im.shape[0], QImage.Format_RGB888)
            # painter.drawImage(self.dw, self.dh, qImg)

        # self.uiControl.update_painter(painter)
        # painter.end()

    def enable_gray(self):
        self.use_gray = not self.use_gray
        self.update()
    def erase(self):
        self.eraseMode = not self.eraseMode

    def set_color(self, c_rgb):
        # c = QColor(c_rgb[0], c_rgb[1], c_rgb[2])
        self.user_color = c
        snap_qcolor = self.calibrate_color(c, self.pos)
        self.color = snap_qcolor
        # self.emit(SIGNAL('update_color'), str('background-color: %s' % self.color.name()))
        self.update_color.emit(str('background-color: %s' % self.color.name()))
        self.uiControl.update_color(snap_qcolor, self.user_color)
        self.compute_result()

    def calibrate_color(self, c, pos):
        x, y = self.scale_point(pos)

        # snap color based on L color
        color_array = np.array((c.red(), c.green(), c.blue())).astype(
            'uint8')
        mean_L = self.im_l[y, x]
        snap_color = snap_ab(mean_L, color_array)
        # snap_qcolor = QColor(snap_color[0], snap_color[1], snap_color[2])
        return snap_qcolor

    def change_color(self, pos=None):
        if pos is not None:
            x, y = self.scale_point(pos)
            L = self.im_lab[y, x, 0]
            # self.emit(SIGNAL('update_gamut'), L)
            self.update_gammut.emit(L)

            used_colors = self.uiControl.used_colors()
            # self.emit(SIGNAL('used_colors'), used_colors)
            self.used_colors.emit(used_colors)

            snap_color = self.calibrate_color(self.user_color, pos)
            c = np.array((snap_color.red(), snap_color.green(), snap_color.blue()), np.uint8)
            # self.emit(SIGNAL('update_ab'), c)
            self.update_ab.emit(c)

    def init_color(self):
        self.user_color =  np.array([[[128,128,128]]], dtype=np.uint8)  # default color red
        self.color = self.user_color

    def scale_point(self, pnt):
        x = int((pnt.x() - self.dw) / float(self.win_w) * self.load_size)
        y = int((pnt.y() - self.dh) / float(self.win_h) * self.load_size)
        return x, y

if __name__ == '__main__':
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        app = Flask(__name__)

        gamut_img_arr = []
        img_src_arr = []
        uri_arr = []
        pos_arr = []
        @app.route('/')
        def index():
            return render_template('index.html',  gamut_img=gamut_img_arr,filename=img_src_arr, result=uri_arr, id="result", label="result!!")
        @app.route('/upload', methods = ['post'])
        def upload():
            mem_bytes = io.BytesIO()
            cpath = os.getcwd()
            os.chdir(cpath)

            Gamut = GUIGamut()

            G = GUIDraw()

            file = request.files['file']
            filename = file.filename
            img_src = url_for('static', filename = 'image/'+filename)
            image_file = cpath + '/static/image/' + filename
            file.save(os.path.join('./static/image', filename))
            img_src_arr.append(img_src)

            args = get_args(image_file)
            model = get_model(args)
            model.to(args.device)
            checkpoint = torch.load(args.model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            model.eval()

            G.read_image(image_file)
            G.init_color()
            img = G.compute_result(model)
            img = Image.fromarray(img.astype("uint8"))
            img.save(mem_bytes, 'JPEG')
            mem_bytes.seek(0)
            img_base64 = base64.b64encode(mem_bytes.getvalue()).decode('ascii')
            mime = "image/jpeg"
            uri = "data:%s;base64,%s"%(mime, img_base64)
            uri_arr.append(uri)

            Gamut.set_gamut()
            gamut_img = Gamut.paintEvent()
            gamut_img_arr.append(gamut_img)
            return redirect(url_for('index'))


        @app.route('/pos', methods=['post'])
        def pos():
            x_pos = request.form['x_pos']
            y_pos = request.form['y_pos']
            pos_arr.append([x_pos, y_pos])
            print(pos_arr)
            return redirect(url_for('index'))
        app.run()


#

#
#
#
# class ApplicationThread(QtCore.QThread):
#     def __init__(self, application, port=5690):
#         super(ApplicationThread, self).__init__()
#         self.application = application
#         self.port = port
#
#     def __del__(self):
#         self.wait()
#
#     def run(self):
#         self.application.run(port=self.port, threaded=True)
#
#
# class WebPage(QtWebEngineWidgets.QWebEnginePage):
#     def __init__(self, root_url):
#         super(WebPage, self).__init__()
#         self.root_url = root_url
#
#     def home(self):
#         self.load(QtCore.QUrl(self.root_url))
#
#     def acceptNavigationRequest(self, url, kind, is_main_frame):
#         """Open external links in browser and internal links in the webview"""
#         ready_url = url.toEncoded().data().decode()
#         is_clicked = kind == self.NavigationTypeLinkClicked
#         if is_clicked and self.root_url not in ready_url:
#             QDesktopServices.openUrl(url)
#             return False
#         return super(WebPage, self).acceptNavigationRequest(url, kind, is_main_frame)
#
#
# def init_gui(application, port=6006, width=800, height=600,
#              window_title="Colorize", icon="appicon.png", argv=None):
#     if argv is None:
#         argv = sys.argv
#
#     if port == 0:
#         sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         sock.bind(('localhost', 0))
#         port = sock.getsockname()[1]
#         sock.close()
#
#     # Application Level
#     qtapp = QApplication(argv)
#     webapp = ApplicationThread(application, port)
#     webapp.start()
#     qtapp.aboutToQuit.connect(webapp.terminate)
#
#     # Main Window Level
#     window = QMainWindow()
#     window.resize(width, height)
#     window.setWindowTitle(window_title)
#     window.setWindowIcon(QIcon(icon))
#
#     # WebView Level
#     webView = QtWebEngineWidgets.QWebEngineView(window)
#     window.setCentralWidget(webView)
#
#     # WebPage Level
#     page = WebPage('http://localhost:{}'.format(6006))
#     page.home()
#     webView.setPage(page)
#
#     window.show()
#     return qtapp.exec_()
#
#
# if __name__ == '__main__':
#     import warnings
#     warnings.filterwarnings("ignore", category=UserWarning)
#     args = get_args()
#
#     model = get_model(args)
#     model.to(args.device)
#     checkpoint = torch.load(args.model_path, map_location='cpu')
#     model.load_state_dict(checkpoint['model'])
#     model.eval()
#
#     app = QApplication(sys.argv)
#     kwargs = {'host': 'localhost', 'port': 5690, 'threaded': True, 'use_reloader': False, 'debug': True}
#
#     app_ = Flask(__name__)
#     # api = Api(app_)
#     # ui = FlaskUI(app_, width = 800, height = 600)
#     # setting root
#     @app_.route('/')
#     def index():
#         return render_template('index.html')
#     # wb.open_new('http://localhost:8080')
#     flaskThread = ApplicationThread(app_, port=8080)
#     flaskThread.start()
#
#     # flaskThread = Thread(target=app_.run, daemon=True, kwargs=kwargs).start()
#     # app_.run()
#     # ui.run()
#     # app.aboutToQuit.connect(flaskThread.terminate)
#
#     ex = gui_main.IColoriTUI(color_model=model, img_file=args.target_image,
#                              load_size=args.input_size, win_size=256, device=args.device)
#
#     ex.setWindowIcon(QIcon('gui/icon.png'))
#     ex.setWindowTitle('iColoriT')
#
#     # WebView Level
#     webView = QtWebEngineWidgets.QWebEngineView(ex)
#     # ex.setCentralWidget(webView)
#     QtCore.QCoreApplication.processEvents()
#     # WebPage Level
#     page = WebPage('http://localhost:{}'.format(5690))
#     page.home()
#     webView.setPage(page)
#
#     ex.show()
#     app.exec_() # sys.exit() -> none



