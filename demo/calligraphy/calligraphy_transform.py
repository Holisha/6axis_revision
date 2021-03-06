import os
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import save_file, sigmoid, angle2deg

class calligraphy_transform():

    # 將input的6d跟cmd分開
    def read_file(self, path, is_6dcmd = True):
        """
        Read 6dcmd txt files to numpy array.
        """
        if not os.path.isfile(path):
            raise ValueError(f'Error: File not exist! {path}')

        data = list()
        txtFile = open(path)

        if is_6dcmd:
            cmd = list()
            for row in txtFile:
                row = row.split()
                data.append([float(row[2]), float(row[3]), float(row[4]), 
                            float(row[5]), float(row[6]), float(row[7])])
                cmd.append([row[0], row[1], row[8], row[9]])
            
            return np.array(data), np.array(cmd)
        
        else:
            for row in txtFile:
                data.append([float(i) for i in row.split()])
            return np.array(data)

    # 找字的上下左右邊界
    def find_rect(self, data_3d):
        """
        Find the bounding box of data.
        """
        datax = [i[0] for i in data_3d]
        datay = [i[1] for i in data_3d]

        return min(datax), max(datax), min(datay), max(datay)

    # 找z小於thresholdZ的控制點
    def find_draw_points(self, data_3d, thresholdZ, data_cmd=None):
        """
        Find points that z < thresholdZ.
        """
        output_data_3d = list()
        output_data_cmd = list()

        if not (data_cmd is None):
            for i, _ in enumerate(data_3d):
                if data_3d[i][-1] < thresholdZ:
                    output_data_3d.append(data_3d[i])
                    output_data_cmd.append(data_cmd[i])

            return np.array(output_data_3d), np.array(output_data_cmd)
        else:
            for i, _ in enumerate(data_3d):
                if data_3d[i][-1] < thresholdZ:
                    output_data_3d.append(data_3d[i])

            return np.array(output_data_3d)

    # 找有幾筆劃
    def find_stroke(self, data_cmd):
        """
        Return every stroke's start position, including end position.
        """

        stroke = [0]

        # stroke 1
        flag = data_cmd[0][-1]

        for i, data in enumerate(data_cmd):
            if flag != data[-1]:
                stroke.append(i)
                flag = data[-1]
        
        stroke.append(len(data_cmd))

        return stroke

    # 找出z小於thresholdZ的data，在從這些data找出最小的x跟y
    def find_anchor(self, data_3d, thresholdZ):
        """
        Return left, down point.
        """
        data_3d = self.find_draw_points(data_3d, thresholdZ)

        rect = self.find_rect(data_3d)

        # min(datax) , min(datay) , thresholdZ
        return [rect[0], rect[2], thresholdZ]

    # 繪出骨架點
    def visualize_dot_3d(self, data_3d, data_cmd, thresholdZ,  with_thickness = False, show_in_rect=None, plot=True):
        data_3d = self.find_draw_points(data_3d, thresholdZ)

        data = {
            'a': np.array([i[0] for i in data_3d]),
            'b': np.array([i[1] for i in data_3d]),
            'c': np.random.randint(0, 50, len(data_3d)),
            's': np.array([(thresholdZ - i[2])*10+10 for i in data_3d])
            }

        if not with_thickness:
            plt.scatter('a', 'b', c='darkslategray', data=data)
        else:
            plt.scatter('a', 'b', c='darkslategray', s='s', data=data)

        if not show_in_rect is None:
            plt.xlim(show_in_rect[0], show_in_rect[1])
            plt.ylim(show_in_rect[2], show_in_rect[3])

        if plot:
            plt.show()
            

    # 繪出字
    def visualize_line_3d(self, data_3d, data_cmd, thresholdZ, save_path, with_thickness=False, paint_width = 15, show_in_rect=None, plot=True):
        thresholdZ += 2

        if not with_thickness: 
            data_3d, data_cmd = self.find_draw_points(data_3d, thresholdZ, data_cmd=data_cmd)
            stroke = self.find_stroke(data_cmd)
            for i in range(len(stroke)-1):
                line = data_3d[stroke[i]:stroke[i+1]]
                plt.plot([i[0] for i in line], [i[1] for i in line], c='darkslategray')
        else:
            stroke = self.find_stroke(data_cmd)
            for i in range(len(data_3d)-1):
                if (not (i+1 in stroke)) and ((data_3d[i][2] < thresholdZ) and (data_3d[i+1][2] < thresholdZ)):
                    x = [data_3d[i][0], data_3d[i+1][0]]
                    y = [data_3d[i][1], data_3d[i+1][1]]
                    width = (thresholdZ - ((data_3d[i][2] + data_3d[i+1][2])*0.5))*1.8
                    #width = sigmoid((thresholdZ - ((data_3d[i][2] + data_3d[i+1][2])*0.5))*0.2)*paint_width
                    plt.plot(x, y, linewidth=width, c='darkslategray')
        
        if not show_in_rect is None:
            plt.xlim(show_in_rect[0], show_in_rect[1])
            plt.ylim(show_in_rect[2], show_in_rect[3])
        
        if plot:
            plt.savefig(f'{save_path}.png')
            # print(f'{save_path}.png')
            plt.close()
            # plt.show()
            
    
    # 繪出字
    def visualize_line_6d(self, data_6d, data_cmd, thresholdZ, with_thickness=False, paint_width = 15, show_in_rect=None, plot=True):
        data_3d, _ = self.six_to_three(data_6d)
        self.visualize_line_3d(data_3d, data_cmd, thresholdZ, with_thickness, paint_width, show_in_rect, plot)

    def check_length_eq(self, data1, data2):
        if len(data1) != len(data2):
            raise ValueError('Error: len(data1) != len(data2)')

    # 將6d跟cmd整合
    def six_to_cmd(self, data_6d, data_cmd, assign_stroke = True):
        self.check_length_eq(data_6d, data_cmd)
        data_6dcmd = list()
        len_data = len(data_6d)

        if assign_stroke:
            data_cmd = self.assign_cmd_stroke(data_cmd)

        for i in range(len_data):
            data_6dcmd.append([
                data_cmd[i][0],
                data_cmd[i][1],
                data_6d[i][0],
                data_6d[i][1],
                data_6d[i][2],
                data_6d[i][3],
                data_6d[i][4],
                data_6d[i][5],
                data_cmd[i][2],
                data_cmd[i][3]
                ])
        
        return data_6dcmd

    # 將毛筆長設為-185，反向得到6d
    def three_to_six(self, data_3d, data_angle, length=[0,0,-185]):
        self.check_length_eq(data_3d, data_angle)
        data_concate = np.append(data_3d, data_angle, 1)
        out_data_3d, _ = self.six_to_three(data_concate, length)
        out_data_6d = np.append(out_data_3d, data_angle, 1)

        return out_data_6d
    

    def data_6d_cmd_split(self, data_6d, data_cmd, start_stroke, end_stroke, axis=0):
        stroke = self.find_stroke(data_cmd)
        
        # data_6d的start stroke前 & data_6d的end stroke後
        data_6d_1 = np.append(data_6d[:stroke[start_stroke]], data_6d[stroke[end_stroke]:], axis)
        # data_6d的start stroke到data_6d的end stroke
        data_6d_2 = data_6d[stroke[start_stroke]:stroke[end_stroke]]

        data_cmd_1 = np.append(data_cmd[:stroke[start_stroke]], data_cmd[stroke[end_stroke]:], axis)
        data_cmd_2 = data_cmd[stroke[start_stroke]:stroke[end_stroke]]

        return data_6d_1, data_6d_2, data_cmd_1, data_cmd_2
    
    # 每筆最後放上是屬於第幾個stroke
    def assign_cmd_stroke(self, data_cmd):
        stroke = self.find_stroke(data_cmd)
        out_data_cmd = copy.deepcopy(data_cmd)

        for i in range(len(stroke)-1):
            for j in range(stroke[i], stroke[i+1]):
                out_data_cmd[j][-1] = f'stroke{i}'

        return out_data_cmd
    
    def transform_6d_stroke(self, data_6d, data_cmd, stroke_start, stroke_end, thresholdZ, anchor=0, ratio=[1, 1, 1], translate=[0, 0, 0], angle=0):
        """
        Auto set anchor if anchor = 0.
        """
        data_6d_1, data_6d_2, data_cmd_1, data_cmd_2 = self.data_6d_cmd_split(data_6d, data_cmd, stroke_start, stroke_end)
        
        data_3d_2, data_angle_2 = self.six_to_three(data_6d_2)

        if anchor == 0:
            anchor = self.find_anchor(data_3d_2, thresholdZ)

        data_3d_2 = self.transform_3d(data_3d_2, anchor, ratio, translate, angle)
        data_6d_2 = self.three_to_six(data_3d_2, data_angle_2)

        data_6d, data_cmd = self.data_6d_cmd_concate(data_6d_1, data_6d_2, data_cmd_1, data_cmd_2, stroke_start-1)

        return data_6d, data_cmd
    
    def transform_6d(self, data_6d, thresholdZ, anchor=0, ratio=[1, 1, 1], translate=[0, 0, 0], angle=0):
        """
        Auto set anchor if anchor = 0.
        """
        data_3d, data_angle = self.six_to_three(data_6d)

        if anchor == 0:
            anchor = self.find_anchor(data_3d, thresholdZ)

        data_3d = self.transform_3d(data_3d, anchor, ratio, translate, angle)
        data_6d = self.three_to_six(data_3d, data_angle)

        return data_6d

    def transform_to_rect_6d(self, data_6d, to_rect, thresholdZ, ratio_z=0, translate_z=0, center=True, deform=False):
        """
        Auto set ratio_z if ratio_z = 0.
        """
        data_3d, data_angle = self.six_to_three(data_6d)
        data_3d = self.transform_to_rect_3d(data_3d, to_rect, thresholdZ, ratio_z, translate_z, center=True, deform=False)
        data_6d = self.three_to_six(data_3d, data_angle)

        return data_6d

    # 將6d分開的部分都合在一起
    def data_6d_cmd_concate(self, data_6d_1, data_6d_2, data_cmd_1, data_cmd_2, append_to=-1, axis=0):
        stroke = self.find_stroke(data_cmd_1)

        if append_to == -2:
            flag = len(stroke)-1
        else:
            flag = stroke[append_to+1]

        data_6d_u = data_6d_1[:flag]
        data_6d_d = data_6d_1[flag:]
        data_6d = np.append(data_6d_u, data_6d_2, axis)
        data_6d = np.append(data_6d, data_6d_d, axis)

        data_cmd_u = data_cmd_1[:flag]
        data_cmd_d = data_cmd_1[flag:]
        data_cmd = np.append(data_cmd_u, data_cmd_2, axis)
        data_cmd = np.append(data_cmd, data_cmd_d, axis)

        return data_6d, data_cmd

    # 取得投影後的x,y
    def six_to_three(self, data_6d, length=[0,0,185]):
        data_3d = list()
        data_angle = list()

        for row in data_6d:

            x = row[0]
            y = row[1]
            z = row[2]
            a = angle2deg(row[3])
            b = angle2deg(row[4])
            c = angle2deg(row[5])
            
            Ra = np.array([
                [1, 0, 0],
                [0, math.cos(a), -math.sin(a)],
                [0, math.sin(a), math.cos(a)]
            ])

            Rb = np.array([
                [math.cos(b), 0, math.sin(b)],
                [0, 1, 0],
                [-math.sin(b), 0, math.cos(b)]
            ])

            Rc = np.array([
                [math.cos(c), -math.sin(c), 0],
                [math.sin(c), math.cos(c), 0],
                [0, 0, 1]
            ])
            
            R = np.dot(np.dot(Rc, Rb), Ra)
            
            A = np.array([
                [R[0, 0], R[0, 1], R[0, 2], x],
                [R[1, 0], R[1, 1], R[1, 2], y],
                [R[2, 0], R[2, 1], R[2, 2], z],
            ])

            # 毛筆為 x軸 0mm, y軸 0mm, z軸 185 mm
            B = np.array([[length[0]],[length[1]],[length[2]],[1]])

            T = np.dot(A, B)

            data_3d.append([T[0][0], T[1][0], T[2][0]])
            data_angle.append([row[3], row[4], row[5]])

        return np.array(data_3d), np.array(data_angle)

    def check_3d(self, data_3d):
        datay = [i[1] for i in data_3d]
        dataz = [i[2] for i in data_3d]
        if min(dataz) < -10:
            print(f'Warning: z axis is too small in data_3d, z = {min(dataz)}')
        if min(datay) < 250:
            print(f'Warning: y axix is too small in data_3d, y = {min(datay)}')

    def transform_3d(self, data_3d, anchor=[0,0,0], ratio=[1,1,1], translate=[0,0,0], angle=0, thresholdZ=0):
        angle = angle2deg(angle)

        out_data = list()
        
        # shear，將x,y歸0，使每筆資料的x,y相同
        affine1 = np.array([
            [1, 0, 0, -anchor[0]],
            [0, 1, 0, -anchor[1]],
            [0, 0, 1, -anchor[2]],
            [0, 0, 0, 1]
        ])

        # rotation from z
        affine2 = np.array([
            [math.cos(angle), -math.sin(angle), 0, 0],
            [math.sin(angle), math.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # scaling , shear
        affine3 = np.array([
            [ratio[0], 0, 0, anchor[0]+translate[0]],
            [0, ratio[1], 0, anchor[1]+translate[1]],
            [0, 0, ratio[2], anchor[2]+translate[2]],
            [0, 0, 0, 1]
        ])
        
        for row in data_3d:
            if thresholdZ != 0:
                if row[-1] <= thresholdZ:
                    row = np.append(row,1).reshape(4, 1)
                    row = np.dot(affine1, row)
                    row = np.dot(affine2, row)
                    row = np.dot(affine3, row)
                    out_data.append([row[0][0], row[1][0], row[2][0]])
                else:
                    out_data.append([row[0], row[1], row[2]])
            else:
                row = np.append(row,1).reshape(4, 1)
                row = np.dot(affine1, row)
                row = np.dot(affine2, row)
                row = np.dot(affine3, row)
                out_data.append([row[0][0], row[1][0], row[2][0]])

        self.check_3d(out_data)

        return np.array(out_data)

    def transform_to_rect_3d(self, data_3d, to_rect, thresholdZ, ratio_z=0, translate_z=0, center=True, deform=False):
        """
        Auto set ratio_z if ratio_z = 0.
        """

        draw_points = self.find_draw_points(data_3d, thresholdZ)
        from_rect = self.find_rect(draw_points)

        # 一開始長寬(maxx-minx,maxy-miny)
        from_shape = [from_rect[1]-from_rect[0], from_rect[3]-from_rect[2]]
        # 目標的長寬(maxx-minx,maxy-miny)
        to_shape = [to_rect[1]-to_rect[0], to_rect[3]-to_rect[2]]

        # 計算轉換的比例
        width_ratio = to_shape[0]/from_shape[0]
        height_ratio = to_shape[1]/from_shape[1]

        ratio = min(width_ratio, height_ratio)
        # 等比例放大長寬
        ratio = [ratio, ratio, ratio if ratio_z==0 else ratio_z]
        anchor = [from_rect[0], from_rect[2], thresholdZ]
        # 目標最小x到一開始最小x的距離，目標最小y到一開始最小y的距離
        translate = [to_rect[0]-from_rect[0], to_rect[2]-from_rect[2], translate_z]

        if center:
            if width_ratio > height_ratio:
                translate[0] += (to_shape[0] - (from_shape[0]*height_ratio))/2
            else:
                translate[1] += (to_shape[1] - (from_shape[1]*width_ratio))/2

        return self.transform_3d(data_3d, anchor=anchor, ratio=ratio, translate=translate)