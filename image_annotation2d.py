# Simple image annotation app
# Advanced zoom for images of various types from small to huge up to several GB
import math
import warnings
import tkinter as tk

from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw

import os
import numpy as np
import skimage.io
from tkinter.filedialog import askopenfilename

class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)

class CanvasImage:
    """ Display and zoom image """
    def __init__(self, placeholder):
        """ Initialize the ImageFrame """
        self.placeholder = placeholder
        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.3  # zoom magnitude
        self.__filter = Image.ANTIALIAS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.__previous_state = 0  # previous state of the keyboard
        self.path = None  # path to the image, should be public for outer classes
        # Create ImageFrame in placeholder widget
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)

        # Annotation setup
        self.__image = None

        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))

        self.container = self.canvas.create_rectangle((0, 0, 500, 500), width=0)
        self.canvas.focus_set()  # set focus on the canvas

    def image_set(self):
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<ButtonPress-3>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B3-Motion>', self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.__wheel)  # zoom for Linux, wheel scroll up

        # Decide if this image huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for the big image
        with warnings.catch_warnings():  # suppress DecompressionBombWarning
            warnings.simplefilter('ignore')
            self.__image = Image.open(self.path)  # open image, but down't load it
        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        self.imzplanes = self.__image.n_frames  # public for outer classes
        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
           self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it have to be 'raw'
                           [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramide scale
        self.__reduction = 2  # reduction degree of image pyramid
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # top pyramid image is around 512 pixels in size
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)

        # Annotation setup
        self.old_x = None
        self.old_y = None
        self.z = 0
        self.linewidth = 5
        self.lines = []
        self.line = None
        self.imt = np.zeros((self.imheight, self.imwidth, self.imzplanes), dtype=np.uint8)
        self.imtz = Image.new("L", (self.imwidth, self.imheight))
        self.imtzd = ImageDraw.Draw(self.imtz)

        self.review = False
        self.path_gt = []
        self.image_gt = []

        self.canvas.bind('<ButtonRelease-1>', self.__reset)
        self.canvas.bind('<B1-Motion>', self.__paint)
        self.canvas.bind('<Up>', self.__key_up)
        self.canvas.bind('<Down>', self.__key_down)

        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 1, round(0.5 + self.imheight / self.__band_width)
        while i < self.imheight:
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imheight - i)  # width of the tile band
            self.__tile[1][3] = band  # set band width
            self.__tile[2] = self.__offset + self.imwidth * i * 3  # tile offset (3 bytes per pixel)
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.seek(self.z)
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]  # set tile
            cropped = self.__image.crop((0, 0, self.imwidth, band))  # crop tile band
            image.paste(cropped.resize((w, int(band * k)+1), self.__filter), (0, int(i * k)))
            i += band
            j += 1
        print('\r' + 30*' ' + '\r', end='')  # hide printed string
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def pack(self, **kw):
        """ Exception: cannot use pack with this widget """
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        """ Exception: cannot use place with this widget """
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if  box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0]  = box_img_int[0]
            box_scroll[2]  = box_img_int[2]
        # Vertical part of the image is in the visible area
        if  box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1]  = box_img_int[1]
            box_scroll[3]  = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            if self.__huge and self.__curr_img < 0:  # show huge image
                h = int((y2 - y1) / self.imscale)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                self.__image.seek(self.z)
                self.__image.size = (self.imwidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # show normal image
                self.placeholder.title('Annotation v0.1: z =' + str(self.z))
                self.__pyramid[max(0, self.__curr_img)].seek(self.z)

                if self.review == False :
                    image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                                        (int(x1 / self.__scale), int(y1 / self.__scale),
                                        int(x2 / self.__scale), int(y2 / self.__scale)))
                else:
                    image_rgb = self.create_rgb(self.__pyramid[max(0, self.__curr_img)], Image.fromarray(self.imt[:,:,self.z]))
                    image = image_rgb.crop(  # crop current img from pyramid
                                        (int(x1 / self.__scale), int(y1 / self.__scale),
                                        int(x2 / self.__scale), int(y2 / self.__scale)))


            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

            for i in range(0, len(self.lines)): # hide lines from previous plane
                l = self.lines[i][0]
                z = self.lines[i][1]
                #self.canvas.delete(l)
                if z != self.z:
                    self.canvas.itemconfigure(l, state='hidden')
                else:
                    if self.review == False:
                        self.canvas.itemconfigure(l, state='normal')
                    else:
                        self.canvas.itemconfigure(l, state='hidden')

    def create_rgb(self, image, image_gt):
        im = np.array(image)
        imt = np.array(image_gt)
        
        imrgb = np.zeros((self.imheight, self.imwidth, 3), dtype=np.uint8)
        imrgb[:,:,1] = im
        imrgb[:,:,2] = im
        im[imt>0] = 255
        imrgb[:,:,0] = im
        
        image_rgb = Image.fromarray(imrgb, 'RGB')
        return image_rgb


    def __move_from(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()  # zoom tile and show it on the canvas

    def __paint(self, event):
        """ Paint annotations on the canvas """
        if self.old_x and self.old_y:
            box_image = self.canvas.coords(self.container)  # get image area

            xe1 = self.canvas.canvasx(self.old_x)
            ye1 = self.canvas.canvasy(self.old_y)
            xe2 = self.canvas.canvasx(event.x)
            ye2 = self.canvas.canvasy(event.y)
            xe1 = max(xe1, box_image[0])
            xe1 = min(xe1, box_image[2])
            ye1 = max(ye1, box_image[1])
            ye1 = min(ye1, box_image[3])
            xe2 = max(xe2, box_image[0])
            xe2 = min(xe2, box_image[2])
            ye2 = max(ye2, box_image[1])
            ye2 = min(ye2, box_image[3])

            xs = self.imwidth / (box_image[2]-box_image[0])
            ys = self.imheight / (box_image[3]-box_image[1])
            x1 = xs * (xe1 - box_image[0])
            x2 = xs * (xe2 - box_image[0])
            y1 = ys * (ye1 - box_image[1])
            y2 = ys * (ye2 - box_image[1])
            x1 = max(x1, 0)
            x1 = min(x1, self.imwidth)
            y1 = max(y1, 0)
            y1 = min(y1, self.imheight)
            x2 = max(x2, 0)
            x2 = min(x2, self.imwidth)
            y2 = max(y2, 0)
            y2 = min(y2, self.imheight)
            # print('-----------------------------------')
            # print(box_canvas)
            # print(box_image)
            # print([self.old_x, self.old_y, event.x, event.y])
            # print([xe1, ye1, xe2, ye2])
            # print([x1, y1, x2, y2])
            # print([xs,ys])

            self.line = self.canvas.create_line(xe1, ye1, xe2, ye2, width=int(xs*self.linewidth), fill='red',
                                            capstyle='round', smooth=1, splinesteps=36)
            self.lines.append([self.line, self.z])
            self.imtzd.line([x1, y1, x2, y2], 255, width=int(xs*self.linewidth))
            self.imt[:, :, self.z] = np.maximum(self.imt[:, :, self.z], np.array(self.imtz))

        self.old_x = event.x
        self.old_y = event.y

    def __clean_lines(self):
        """ Remove annotations from current plane """
        for i in range(0, len(self.lines)):
            l = self.lines[i][0]
            z = self.lines[i][1]
            if z == self.z:
                self.canvas.delete(l)
        self.imtz = Image.new("L", (self.imwidth, self.imheight))
        self.imtzd = ImageDraw.Draw(self.imtz)
        self.imt[:, :, self.z] = np.array(self.imtz)

    def __save_im(self):
        """ Save annotations as [0,255] image """
        skimage.io.imsave(self.path_gt, np.moveaxis(self.imt, 2, 0))

    def __reset(self, event):
        """ Reset last mouse position """
        self.old_x, self.old_y = None, None

    def __key_up(self, event):
        """ Change plane nr - up """
        self.z = self.z + 1
        self.z = min(self.z, self.imzplanes)
        self.imtz = Image.new("L", (self.imwidth, self.imheight))
        self.imtzd = ImageDraw.Draw(self.imtz)
        self.__show_image()

    def __key_down(self, event):
        """ Change plane nr - down """
        self.z = self.z - 1
        self.z = max(self.z, 0)
        self.imtz = Image.new("L", (self.imwidth, self.imheight))
        self.imtzd = ImageDraw.Draw(self.imtz)
        self.__show_image()

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down, smaller
            if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.__delta
            scale        /= self.__delta
        if event.num == 4 or event.delta == 120:  # scroll up, bigger
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.__delta
            scale        *= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def __keystroke(self, event):
        print(event.keycode)
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if self.__image != None:
                if event.keycode in [68, 39, 102]:  # scroll right, keys 'd' or 'Right'
                    self.__scroll_x('scroll',  1, 'unit', event=event)
                elif event.keycode in [65, 37, 100]:  # scroll left, keys 'a' or 'Left'
                    self.__scroll_x('scroll', -1, 'unit', event=event)
                elif event.keycode in [87, 38, 104]:  # scroll up, keys 'w' or 'Up'
                    self.__scroll_y('scroll', -1, 'unit', event=event)
                elif event.keycode in [83, 40, 98]:  # scroll down, keys 's' or 'Down'
                    self.__scroll_y('scroll',  1, 'unit', event=event)
                elif event.keycode == 112: # line width, keys 'Page Up'
                    self.linewidth = self.linewidth + 1
                    self.__show_image()
                elif event.keycode == 117: # line width, keys 'Page Down'
                    self.linewidth = self.linewidth - 1
                    self.linewidth = max(self.linewidth, 1)
                    self.__show_image()
                elif event.keycode == 115: # save file, key 'End'
                    self.__save_im()
                elif event.keycode == 119: # clean, key 'Delete'
                    self.__clean_lines()
                    self.__show_image()
                elif event.keycode == 96: # review, key 'F12'
                    if self.review == False: 
                        self.review = True
                    else:
                        self.review = False
                    self.redraw_figures()
                    self.__show_image()

            if event.keycode == 110: # load image, key 'Home'
                self.load_image()
            if event.keycode == 118: # load GT, key 'Insert'
                self.load_image_gt()
    
    def load_image(self):
        try:
            self.path = askopenfilename(filetypes=[('all files', '.*'), ('image files', '.tif'), ])
            root_ext = os.path.splitext(self.path) 
            self.path_gt = root_ext[0] + '_seg' + root_ext[1]
            self.image_set()
        except:
            print('TODO')
    
    def load_image_gt(self):
        try:
            self.review = True
            self.path_gt = askopenfilename(filetypes=[('all files', '.*'), ('image files', '.tif'), ])
            image_gt = Image.open(self.path_gt)

            self.imt = np.zeros((self.imheight, self.imwidth, self.imzplanes), dtype=np.uint8)
            for z in range(0, self.imzplanes):
                image_gt.seek(z)
                imt = np.array(image_gt)
                self.imt[:,:,z] = imt
            
            self.__clean_lines()
            self.redraw_figures()
            self.__show_image()

            self.lines = []
        except:
            print('TODO')

    def crop(self, bbox):
        """ Crop rectangle from the image and return it """
        if self.__huge:  # image is huge and not totally in RAM
            band = bbox[3] - bbox[1]  # width of the tile band
            self.__tile[1][3] = band  # set the tile height
            self.__tile[2] = self.__offset + self.imwidth * bbox[1] * 3  # set offset of the band
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.seek(self.z)
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]
            return self.__image.crop((bbox[0], 0, bbox[2], band))
        else:  # image is totally in RAM
            return self.__pyramid[0].crop(bbox)

    def destroy(self):
        """ ImageFrame destructor """
        self.__image.close()
        map(lambda i: i.close, self.__pyramid)  # close all pyramid images
        del self.__pyramid[:]  # delete pyramid list
        del self.__pyramid  # delete pyramid variable
        self.canvas.destroy()
        self.__imframe.destroy()

class MainWindow(ttk.Frame):
    """ Main window class """
    def __init__(self, mainframe):
        """ Initialize the main Frame """
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Annotation v0.1: z = 0')
        self.master.geometry('800x600')  # size of the main window
        self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.master.columnconfigure(0, weight=1)
        self.canvas = CanvasImage(self.master)  # create widget
        self.canvas.grid(row=0, column=0)  # show widget

app = MainWindow(tk.Tk())
app.mainloop()