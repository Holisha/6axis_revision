from __future__ import print_function
import pandas as pd
import argparse
import re
import os
from PIL import Image, ImageTk
from sys import version_info
from functools import partial


# confirm current version
if version_info.major == 2:
    import Tkinter as tk
elif version_info.major == 3:
    import tkinter as tk

'''
Data setting by different cases
'''
# TODO = 'todo_list.csv'
PATH = os.getcwd()
DIR = 'char00436_stroke'
# RULE = {'alpha': r'^\d\D?\d*', 'beta': r'^\d\D?\d*', 'gamma': r'^\d\D?\d*'}
RULE = {'all': r'^\d\D?\d*'}
DONE = ''

class GUIController:
    def __init__(self, xsize, ysize, xshift=0, yshift=0, csv=None, keep=True):
        self.root = tk.Tk()
        self.root.geometry(f'{xsize}x{ysize}+{xshift}+{yshift}')

        # event binding
        # self.root.bind('<Return>', self.show_input)
        self.root.bind('<Return>', self.get_input)
        # self.root.bind('<Return>', partial(self.store_input, csv))
        self.root.bind('<Escape>', self._quit)
        
        # make widget position dynamically
        self.keep = keep
        self.last_position = [0, 0]     
        
        # control variable
        self.entry_var = {}
        self.input_label = {}

    '''widgets setting'''
    def insert_image(self, img_name, resize=None, location=(0, 0), ipadding=(5, 5)):
        img = Image.open(img_name)
        
        if resize:
            img = img.resize(resize, Image.ANTIALIAS)
        
        # control the position
        self.last_position[0] += location[0]
        self.last_position[1] = location[1]

        tk_image = ImageTk.PhotoImage(img)
        self.image_label = tk.Label(self.root, image=tk_image)
        self.image_label.image = tk_image
        self.image_label.grid(row=self.last_position[0], column=self.last_position[1],
                                ipadx=ipadding[0], ipady=ipadding[1], columnspan=ipadding[1])

        self.last_position[0] += 1
    
    def insert_entry(self, entry_name, col=0):
        text_label = tk.Label(self.root, text=f'{entry_name}:')
        text_label.grid(row=self.last_position[0], column=col, sticky='w')

        text_var = tk.StringVar()
        text_entry = tk.Entry(self.root, textvariable=text_var)
        text_entry.grid(row=self.last_position[0], column=col+1, sticky='news', columnspan=1)

        self.entry_var[entry_name] = text_entry
        self.last_position[0] += 1

        return self.entry_var

    def insert_next(self):
        next_btn = tk.Button(self.root, text='quit', command=self.root.quit)
        next_btn.grid(row=self.last_position[0], column=self.last_position[1], sticky='w')

        self.last_position[1] += 1

    def insert_quit(self):
        quit_btn = tk.Button(self.root, text='quit', command=self.root.destroy)
        quit_btn.grid(row=self.last_position[0], column=self.last_position[1], sticky='w')

        # reset the position
        if self.keep:
            self.last_position = [0, 0]     

    '''widget setting end'''
    
    def _quit(self, event=None):
        self.root.destroy() # terminate tk gui
        os._exit(0)         # terminate current process
    
    def start(self):
        self.root.mainloop()

    def next(self):
        self.root.quit()

    ''' label function '''
    # Modifed if needed
    @staticmethod
    def check_label(label, pattern):
        key = RULE.get('all') if RULE.get('all') else RULE[pattern]
    
        if not re.match(key, label):
            print(f'InputTypeError: current input {label}')
            return False
        
        return True

    def show_input(self, event=None):
        tmp = self.get_input()
        print(tmp)

    def get_input(self, event=None):
        tmp = []
        for key, value in self.entry_var.items():
            if value.get() and self.check_label(value.get(), key):
                # if input value is correct format
                print(f'{key}: {value.get()}')
                tmp.append(value.get())
            
            value.delete(0, tk.END)
        
        # new line
        print('')
        return tmp

    def store_input(self, csv, event=None):
        tmp = self.get_input()
        tmp_series = pd.Series(tmp)

        csv_data = pd.read_csv(csv)
        label_data = csv_data.append(tmp_series, ignore_index=True)
        label_data.to_csv(DONE, index=False)
    ''' label function end '''

def list_dir():
    print(*os.listdir(PATH), sep='\n')
    os._exit(0)

if __name__ == '__main__':
    # argument setting
    parser = argparse.ArgumentParser(description='stroke helper')
    parser.add_argument('--list', action='store_true', default=False,
    help='list all directory under current working directory, and then exit')
    parser.add_argument('--window-x-size', '-wx', type=int, default=800,
    help='control the image window x axis size')
    parser.add_argument('--window-y-size', '-wy', type=int, default=800,
    help='control the image window y axis size')
    parser.add_argument('--image-size', type=tuple, default=(700, 700),
    help='resize image (default: (800, 800))', metavar='( , )')

    args = parser.parse_args()
    
    if args.list:   # --list func
        list_dir()

    image_name = DIR + '.png'
    csv_name = DIR + '.csv'
    csv_data = pd.read_csv(csv_name, header=None)
    
    cnt = 0
    interval = 0
    prev_stroke = csv_data.iloc[0,-1]
    for idx in range(1, csv_data.shape[0]):
        cur_stroke = csv_data.iloc[idx ,-1]

        if prev_stroke is not cur_stroke:
            stroke = csv_data.iloc[interval:idx,]
            prev_stroke = cur_stroke
            cnt += 1
            interval = idx
            break

    print(stroke)
    print(stroke.mean())
    print(stroke.std())
    # print(csv_data.iloc[interval:idx-1, -1])
    # print(cnt)

    """ # call gui
    tK_gui = GUIController(args.window_x_size, args.window_y_size)

    # Tk GUI setting
    tK_gui.insert_image(image_name, args.image_size)
    tK_gui.insert_entry('alpha')
    tK_gui.insert_entry('beta')
    tK_gui.insert_entry('gamma')
    tK_gui.insert_quit()
    tK_gui.start() """
