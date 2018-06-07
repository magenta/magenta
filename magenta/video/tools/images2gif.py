# -*- coding: utf-8 -*-
#   Copyright (C) 2012, Almar Klein, Ant1, Marius van Voorden
#
#   This code is subject to the (new) BSD license:
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

""" Module images2gif
Provides functionality for reading and writing animated GIF images.
Use writeGif to write a series of numpy arrays or PIL images as an
animated GIF. Use readGif to read an animated gif as a series of numpy
arrays.
Note that since July 2004, all patents on the LZW compression patent have
expired. Therefore the GIF format may now be used freely.
Acknowledgements
----------------
Many thanks to Ant1 for:
* noting the use of "palette=PIL.Image.ADAPTIVE", which significantly
  improves the results.
* the modifications to save each image with its own palette, or optionally
  the global palette (if its the same).
Many thanks to Marius van Voorden for porting the NeuQuant quantization
algorithm of Anthony Dekker to Python (See the NeuQuant class for its
license).
Many thanks to Alex Robinson for implementing the concept of subrectangles,
which (depening on image content) can give a very significant reduction in
file size.
This code is based on gifmaker (in the scripts folder of the source
distribution of PIL)
Usefull links
-------------
  * http://tronche.com/computer-graphics/gif/
  * http://en.wikipedia.org/wiki/Graphics_Interchange_Format
  * http://www.w3.org/Graphics/GIF/spec-gif89a.txt
"""
# todo: This module should be part of imageio (or at least based on)

import os, time

def encode(x):
  if False:
    return x.encode('utf-8')
  return x

try:
    import PIL
    from PIL import Image
    from PIL.GifImagePlugin import getheader, getdata
except ImportError:
    PIL = None

try:
    import numpy as np
except ImportError:
    np = None

def get_cKDTree():
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    return cKDTree


# getheader gives a 87a header and a color palette (two elements in a list).
# getdata()[0] gives the Image Descriptor up to (including) "LZW min code size".
# getdatas()[1:] is the image data itself in chuncks of 256 bytes (well
# technically the first byte says how many bytes follow, after which that
# amount (max 255) follows).

def checkImages(images):
    """ checkImages(images)
    Check numpy images and correct intensity range etc.
    The same for all movie formats.
    """
    # Init results
    images2 = []

    for im in images:
        if PIL and isinstance(im, PIL.Image.Image):
            # We assume PIL images are allright
            images2.append(im)

        elif np and isinstance(im, np.ndarray):
            # Check and convert dtype
            if im.dtype == np.uint8:
                images2.append(im) # Ok
            elif im.dtype in [np.float32, np.float64]:
                im = im.copy()
                im[im<0] = 0
                im[im>1] = 1
                im *= 255
                images2.append( im.astype(np.uint8) )
            else:
                im = im.astype(np.uint8)
                images2.append(im)
            # Check size
            if im.ndim == 2:
                pass # ok
            elif im.ndim == 3:
                if im.shape[2] not in [3,4]:
                    raise ValueError('This array can not represent an image.')
            else:
                raise ValueError('This array can not represent an image.')
        else:
            raise ValueError('Invalid image type: ' + str(type(im)))

    # Done
    return images2


def intToBin(i):
    """ Integer to two bytes """
    # devide in two parts (bytes)
    i1 = i % 256
    i2 = int( i/256)
    # make string (little endian)
    return chr(i1) + chr(i2)


class GifWriter:
    """ GifWriter()
    Class that contains methods for helping write the animated GIF file.
    """

    def getheaderAnim(self, im):
        """ getheaderAnim(im)
        Get animation header. To replace PILs getheader()[0]
        """
        bb = "GIF89a"
        bb += intToBin(im.size[0])
        bb += intToBin(im.size[1])
        bb += "\x87\x00\x00"
        return bb


    def getImageDescriptor(self, im, xy=None):
        """ getImageDescriptor(im, xy=None)
        Used for the local color table properties per image.
        Otherwise global color table applies to all frames irrespective of
        whether additional colors comes in play that require a redefined
        palette. Still a maximum of 256 color per frame, obviously.
        Written by Ant1 on 2010-08-22
        Modified by Alex Robinson in Janurari 2011 to implement subrectangles.
        """

        # Defaule use full image and place at upper left
        if xy is None:
            xy  = (0,0)

        # Image separator,
        bb = '\x2C'

        # Image position and size
        bb += intToBin( xy[0] ) # Left position
        bb += intToBin( xy[1] ) # Top position
        bb += intToBin( im.size[0] ) # image width
        bb += intToBin( im.size[1] ) # image height

        # packed field: local color table flag1, interlace0, sorted table0,
        # reserved00, lct size111=7=2^(7+1)=256.
        bb += '\x87'

        # LZW minimum size code now comes later, begining of [image data] blocks
        return bb


    def getAppExt(self, loops=float('inf')):
        """ getAppExt(loops=float('inf'))
        Application extention. This part specifies the amount of loops.
        If loops is 0 or inf, it goes on infinitely.
        """

        if loops==0 or loops==float('inf'):
            loops = 2**16-1
            #bb = "" # application extension should not be used
                    # (the extension interprets zero loops
                    # to mean an infinite number of loops)
                    # Mmm, does not seem to work
        if True:
            bb = "\x21\xFF\x0B"  # application extension
            bb += "NETSCAPE2.0"
            bb += "\x03\x01"
            bb += intToBin(loops)
            bb += '\x00'  # end
        return bb


    def getGraphicsControlExt(self, duration=0.1, dispose=2):
        """ getGraphicsControlExt(duration=0.1, dispose=2)
        Graphics Control Extension. A sort of header at the start of
        each image. Specifies duration and transparancy.
        Dispose
        -------
          * 0 - No disposal specified.
          * 1 - Do not dispose. The graphic is to be left in place.
          * 2 - Restore to background color. The area used by the graphic
            must be restored to the background color.
          * 3 -	Restore to previous. The decoder is required to restore the
            area overwritten by the graphic with what was there prior to
            rendering the graphic.
          * 4-7 -To be defined.
        """

        bb = '\x21\xF9\x04'
        bb += chr((dispose & 3) << 2)  # low bit 1 == transparency,
        # 2nd bit 1 == user input , next 3 bits, the low two of which are used,
        # are dispose.
        bb += intToBin( int(duration*100) ) # in 100th of seconds
        bb += '\x00'  # no transparant color
        bb += '\x00'  # end
        return bb


    def handleSubRectangles(self, images, subRectangles):
        """ handleSubRectangles(images)
        Handle the sub-rectangle stuff. If the rectangles are given by the
        user, the values are checked. Otherwise the subrectangles are
        calculated automatically.
        """

        if isinstance(subRectangles, (tuple,list)):
            # xy given directly

            # Check xy
            xy = subRectangles
            if xy is None:
                xy = (0,0)
            if hasattr(xy, '__len__'):
                if len(xy) == len(images):
                    xy = [xxyy for xxyy in xy]
                else:
                    raise ValueError("len(xy) doesn't match amount of images.")
            else:
                xy = [xy for im in images]
            xy[0] = (0,0)

        else:
            # Calculate xy using some basic image processing

            # Check Numpy
            if np is None:
                raise RuntimeError("Need Numpy to use auto-subRectangles.")

            # First make numpy arrays if required
            for i in range(len(images)):
                im = images[i]
                if isinstance(im, Image.Image):
                    tmp = im.convert() # Make without palette
                    a = np.asarray(tmp)
                    if len(a.shape)==0:
                        raise MemoryError("Too little memory to convert PIL image to array")
                    images[i] = a

            # Determine the sub rectangles
            images, xy = self.getSubRectangles(images)

        # Done
        return images, xy


    def getSubRectangles(self, ims):
        """ getSubRectangles(ims)
        Calculate the minimal rectangles that need updating each frame.
        Returns a two-element tuple containing the cropped images and a
        list of x-y positions.
        Calculating the subrectangles takes extra time, obviously. However,
        if the image sizes were reduced, the actual writing of the GIF
        goes faster. In some cases applying this method produces a GIF faster.
        """

        # Check image count
        if len(ims) < 2:
            return ims, [(0,0) for i in ims]

        # We need numpy
        if np is None:
            raise RuntimeError("Need Numpy to calculate sub-rectangles. ")

        # Prepare
        ims2 = [ims[0]]
        xy = [(0,0)]
        t0 = time.time()

        # Iterate over images
        prev = ims[0]
        for im in ims[1:]:

            # Get difference, sum over colors
            diff = np.abs(im-prev)
            if diff.ndim==3:
                diff = diff.sum(2)
            # Get begin and end for both dimensions
            X = np.argwhere(diff.sum(0))
            Y = np.argwhere(diff.sum(1))
            # Get rect coordinates
            if X.size and Y.size:
                x0, x1 = X[0], X[-1]+1
                y0, y1 = Y[0], Y[-1]+1
            else: # No change ... make it minimal
                x0, x1 = 0, 2
                y0, y1 = 0, 2

            # Cut out and store
            im2 = im[y0:y1,x0:x1]
            prev = im
            ims2.append(im2)
            xy.append((x0,y0))

        # Done
        #print('%1.2f seconds to determine subrectangles of  %i images' %
        #    (time.time()-t0, len(ims2)) )
        return ims2, xy


    def convertImagesToPIL(self, images, dither, nq=0):
        """ convertImagesToPIL(images, nq=0)
        Convert images to Paletted PIL images, which can then be
        written to a single animaged GIF.
        """

        # Convert to PIL images
        images2 = []
        for im in images:
            if isinstance(im, Image.Image):
                images2.append(im)
            elif np and isinstance(im, np.ndarray):
                if im.ndim==3 and im.shape[2]==3:
                    im = Image.fromarray(im,'RGB')
                elif im.ndim==3 and im.shape[2]==4:
                    im = Image.fromarray(im[:,:,:3],'RGB')
                elif im.ndim==2:
                    im = Image.fromarray(im,'L')
                images2.append(im)

        # Convert to paletted PIL images
        images, images2 = images2, []
        if nq >= 1:
            # NeuQuant algorithm
            for im in images:
                im = im.convert("RGBA") # NQ assumes RGBA
                nqInstance = NeuQuant(im, int(nq)) # Learn colors from image
                if dither:
                    im = im.convert("RGB").quantize(palette=nqInstance.paletteImage())
                else:
                    im = nqInstance.quantize(im)  # Use to quantize the image itself
                images2.append(im)
        else:
            # Adaptive PIL algorithm
            AD = Image.ADAPTIVE
            for im in images:
                im = im.convert('P', palette=AD, dither=dither)
                images2.append(im)

        # Done
        return images2


    def writeGifToFile(self, fp, images, durations, loops, xys, disposes):
        """ writeGifToFile(fp, images, durations, loops, xys, disposes)
        Given a set of images writes the bytes to the specified stream.
        """

        # Obtain palette for all images and count each occurance
        palettes, occur = [], []
        for im in images:
            #palette = getheader(im)[1]
            palette = getheader(im)[0][-1]
            if not palette:
              #palette = PIL.ImagePalette.ImageColor
                palette = im.palette.tobytes()
            palettes.append(palette)
        for palette in palettes:
            occur.append( palettes.count( palette ) )

        # Select most-used palette as the global one (or first in case no max)
        globalPalette = palettes[ occur.index(max(occur)) ]

        # Init
        frames = 0
        firstFrame = True


        for im, palette in zip(images, palettes):

            if firstFrame:
                # Write header

                # Gather info
                header = self.getheaderAnim(im)
                appext = self.getAppExt(loops)

                # Write
                fp.write(encode(header))
                fp.write(globalPalette)
                fp.write(encode(appext))

                # Next frame is not the first
                firstFrame = False

            if True:
                # Write palette and image data

                # Gather info
                data = getdata(im)
                imdes, data = data[0], data[1:]
                graphext = self.getGraphicsControlExt(durations[frames],
                                                        disposes[frames])
                # Make image descriptor suitable for using 256 local color palette
                lid = self.getImageDescriptor(im, xys[frames])

                # Write local header
                if (palette != globalPalette) or (disposes[frames] != 2):
                    # Use local color palette
                    fp.write(encode(graphext))
                    fp.write(encode(lid)) # write suitable image descriptor
                    fp.write(palette) # write local color table
                    fp.write(encode('\x08')) # LZW minimum size code
                else:
                    # Use global color palette
                    fp.write(encode(graphext))
                    fp.write(imdes) # write suitable image descriptor

                # Write image data
                for d in data:
                    fp.write(d)

            # Prepare for next round
            frames = frames + 1

        fp.write(encode(";"))  # end gif
        return frames




## Exposed functions

def writeGif(filename, images, duration=0.1, repeat=True, dither=False,
                nq=0, subRectangles=True, dispose=None):
    """ writeGif(filename, images, duration=0.1, repeat=True, dither=False,
                    nq=0, subRectangles=True, dispose=None)
    Write an animated gif from the specified images.
    Parameters
    ----------
    filename : string
        The name of the file to write the image to.
    images : list
        Should be a list consisting of PIL images or numpy arrays.
        The latter should be between 0 and 255 for integer types, and
        between 0 and 1 for float types.
    duration : scalar or list of scalars
        The duration for all frames, or (if a list) for each frame.
    repeat : bool or integer
        The amount of loops. If True, loops infinitetely.
    dither : bool
        Whether to apply dithering
    nq : integer
        If nonzero, applies the NeuQuant quantization algorithm to create
        the color palette. This algorithm is superior, but slower than
        the standard PIL algorithm. The value of nq is the quality
        parameter. 1 represents the best quality. 10 is in general a
        good tradeoff between quality and speed. When using this option,
        better results are usually obtained when subRectangles is False.
    subRectangles : False, True, or a list of 2-element tuples
        Whether to use sub-rectangles. If True, the minimal rectangle that
        is required to update each frame is automatically detected. This
        can give significant reductions in file size, particularly if only
        a part of the image changes. One can also give a list of x-y
        coordinates if you want to do the cropping yourself. The default
        is True.
    dispose : int
        How to dispose each frame. 1 means that each frame is to be left
        in place. 2 means the background color should be restored after
        each frame. 3 means the decoder should restore the previous frame.
        If subRectangles==False, the default is 2, otherwise it is 1.
    """

    # Check PIL
    if PIL is None:
        raise RuntimeError("Need PIL to write animated gif files.")

    # Check images
    images = checkImages(images)

    # Instantiate writer object
    gifWriter = GifWriter()

    # Check loops
    if repeat is False:
        loops = 1
    elif repeat is True:
        loops = 0 # zero means infinite
    else:
        loops = int(repeat)

    # Check duration
    if hasattr(duration, '__len__'):
        if len(duration) == len(images):
            duration = [d for d in duration]
        else:
            raise ValueError("len(duration) doesn't match amount of images.")
    else:
        duration = [duration for im in images]

    # Check subrectangles
    if subRectangles:
        images, xy = gifWriter.handleSubRectangles(images, subRectangles)
        defaultDispose = 1 # Leave image in place
    else:
        # Normal mode
        xy = [(0,0) for im in images]
        defaultDispose = 2 # Restore to background color.

    # Check dispose
    if dispose is None:
        dispose = defaultDispose
    if hasattr(dispose, '__len__'):
        if len(dispose) != len(images):
            raise ValueError("len(xy) doesn't match amount of images.")
    else:
        dispose = [dispose for im in images]


    # Make images in a format that we can write easy
    images = gifWriter.convertImagesToPIL(images, dither, nq)

    # Write
    fp = open(filename, 'wb')
    try:
        gifWriter.writeGifToFile(fp, images, duration, loops, xy, dispose)
    finally:
        fp.close()



def readGif(filename, asNumpy=True):
    """ readGif(filename, asNumpy=True)
    Read images from an animated GIF file.  Returns a list of numpy
    arrays, or, if asNumpy is false, a list if PIL images.
    """

    # Check PIL
    if PIL is None:
        raise RuntimeError("Need PIL to read animated gif files.")

    # Check Numpy
    if np is None:
        raise RuntimeError("Need Numpy to read animated gif files.")

    # Check whether it exists
    if not os.path.isfile(filename):
        raise IOError('File not found: '+str(filename))

    # Load file using PIL
    pilIm = PIL.Image.open(filename)
    pilIm.seek(0)

    # Read all images inside
    images = []
    try:
        while True:
            # Get image as numpy array
            tmp = pilIm.convert() # Make without palette
            a = np.asarray(tmp)
            if len(a.shape)==0:
                raise MemoryError("Too little memory to convert PIL image to array")
            # Store, and next
            images.append(a)
            pilIm.seek(pilIm.tell()+1)
    except EOFError:
        pass

    # Convert to normal PIL images if needed
    if not asNumpy:
        images2 = images
        images = []
        for im in images2:
            images.append( PIL.Image.fromarray(im) )

    # Done
    return images


class NeuQuant:
    """ NeuQuant(image, samplefac=10, colors=256)
    samplefac should be an integer number of 1 or higher, 1
    being the highest quality, but the slowest performance.
    With avalue of 10, one tenth of all pixels are used during
    training. This value seems a nice tradeof between speed
    and quality.
    colors is the amount of colors to reduce the image to. This
    should best be a power of two.
    See also:
    http://members.ozemail.com.au/~dekker/NEUQUANT.HTML
    License of the NeuQuant Neural-Net Quantization Algorithm
    ---------------------------------------------------------
    Copyright (c) 1994 Anthony Dekker
    Ported to python by Marius van Voorden in 2010
    NEUQUANT Neural-Net quantization algorithm by Anthony Dekker, 1994.
    See "Kohonen neural networks for optimal colour quantization"
    in "network: Computation in Neural Systems" Vol. 5 (1994) pp 351-367.
    for a discussion of the algorithm.
    See also  http://members.ozemail.com.au/~dekker/NEUQUANT.HTML
    Any party obtaining a copy of these files from the author, directly or
    indirectly, is granted, free of charge, a full and unrestricted irrevocable,
    world-wide, paid up, royalty-free, nonexclusive right and license to deal
    in this software and documentation files (the "Software"), including without
    limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons who receive
    copies from any such party to do so, with the only requirement being
    that this copyright notice remain intact.
    """

    NCYCLES = None # Number of learning cycles
    NETSIZE = None # Number of colours used
    SPECIALS = None # Number of reserved colours used
    BGCOLOR = None # Reserved background colour
    CUTNETSIZE = None
    MAXNETPOS = None

    INITRAD = None # For 256 colours, radius starts at 32
    RADIUSBIASSHIFT = None
    RADIUSBIAS = None
    INITBIASRADIUS = None
    RADIUSDEC = None # Factor of 1/30 each cycle

    ALPHABIASSHIFT = None
    INITALPHA = None # biased by 10 bits

    GAMMA = None
    BETA = None
    BETAGAMMA = None

    network = None # The network itself
    colormap = None # The network itself

    netindex = None # For network lookup - really 256

    bias = None # Bias and freq arrays for learning
    freq = None

    pimage = None

    # Four primes near 500 - assume no image has a length so large
    # that it is divisible by all four primes
    PRIME1 = 499
    PRIME2 = 491
    PRIME3 = 487
    PRIME4 = 503
    MAXPRIME = PRIME4

    pixels = None
    samplefac = None

    a_s = None


    def setconstants(self, samplefac, colors):
        self.NCYCLES = 100 # Number of learning cycles
        self.NETSIZE = colors # Number of colours used
        self.SPECIALS = 3 # Number of reserved colours used
        self.BGCOLOR = self.SPECIALS-1 # Reserved background colour
        self.CUTNETSIZE = self.NETSIZE - self.SPECIALS
        self.MAXNETPOS = self.NETSIZE - 1

        self.INITRAD = self.NETSIZE/8 # For 256 colours, radius starts at 32
        self.RADIUSBIASSHIFT = 6
        self.RADIUSBIAS = 1 << self.RADIUSBIASSHIFT
        self.INITBIASRADIUS = self.INITRAD * self.RADIUSBIAS
        self.RADIUSDEC = 30 # Factor of 1/30 each cycle

        self.ALPHABIASSHIFT = 10 # Alpha starts at 1
        self.INITALPHA = 1 << self.ALPHABIASSHIFT # biased by 10 bits

        self.GAMMA = 1024.0
        self.BETA = 1.0/1024.0
        self.BETAGAMMA = self.BETA * self.GAMMA

        self.network = np.empty((self.NETSIZE, 3), dtype='float64') # The network itself
        self.colormap = np.empty((self.NETSIZE, 4), dtype='int32') # The network itself

        self.netindex = np.empty(256, dtype='int32') # For network lookup - really 256

        self.bias = np.empty(self.NETSIZE, dtype='float64') # Bias and freq arrays for learning
        self.freq = np.empty(self.NETSIZE, dtype='float64')

        self.pixels = None
        self.samplefac = samplefac

        self.a_s = {}

    def __init__(self, image, samplefac=10, colors=256):

        # Check Numpy
        if np is None:
            raise RuntimeError("Need Numpy for the NeuQuant algorithm.")

        # Check image
        if image.size[0] * image.size[1] < NeuQuant.MAXPRIME:
            raise IOError("Image is too small")
        if image.mode != "RGBA":
            raise IOError("Image mode should be RGBA.")

        # Initialize
        self.setconstants(samplefac, colors)
        self.pixels = np.fromstring(image.tostring(), np.uint32)
        self.setUpArrays()

        self.learn()
        self.fix()
        self.inxbuild()

    def writeColourMap(self, rgb, outstream):
        for i in range(self.NETSIZE):
            bb = self.colormap[i,0];
            gg = self.colormap[i,1];
            rr = self.colormap[i,2];
            outstream.write(rr if rgb else bb)
            outstream.write(gg)
            outstream.write(bb if rgb else rr)
        return self.NETSIZE

    def setUpArrays(self):
        self.network[0,0] = 0.0    # Black
        self.network[0,1] = 0.0
        self.network[0,2] = 0.0

        self.network[1,0] = 255.0    # White
        self.network[1,1] = 255.0
        self.network[1,2] = 255.0

        # RESERVED self.BGCOLOR # Background

        for i in range(self.SPECIALS):
            self.freq[i] = 1.0 / self.NETSIZE
            self.bias[i] = 0.0

        for i in range(self.SPECIALS, self.NETSIZE):
            p = self.network[i]
            p[:] = (255.0 * (i-self.SPECIALS)) / self.CUTNETSIZE

            self.freq[i] = 1.0 / self.NETSIZE
            self.bias[i] = 0.0

    # Omitted: setPixels

    def altersingle(self, alpha, i, b, g, r):
        """Move neuron i towards biased (b,g,r) by factor alpha"""
        n = self.network[i] # Alter hit neuron
        n[0] -= (alpha*(n[0] - b))
        n[1] -= (alpha*(n[1] - g))
        n[2] -= (alpha*(n[2] - r))

    def geta(self, alpha, rad):
        try:
            return self.a_s[(alpha, rad)]
        except KeyError:
            length = rad*2-1
            mid = int(length//2)
            q = np.array(list(range(mid-1,-1,-1))+list(range(-1,mid)))
            a = alpha*(rad*rad - q*q)/(rad*rad)
            a[mid] = 0
            self.a_s[(alpha, rad)] = a
            return a

    def alterneigh(self, alpha, rad, i, b, g, r):
        if i-rad >= self.SPECIALS-1:
            lo = i-rad
            start = 0
        else:
            lo = self.SPECIALS-1
            start = (self.SPECIALS-1 - (i-rad))

        if i+rad <= self.NETSIZE:
            hi = i+rad
            end = rad*2-1
        else:
            hi = self.NETSIZE
            end = (self.NETSIZE - (i+rad))

        a = self.geta(alpha, rad)[start:end]

        p = self.network[lo+1:hi]
        p -= np.transpose(np.transpose(p - np.array([b, g, r])) * a)

    #def contest(self, b, g, r):
    #    """ Search for biased BGR values
    #            Finds closest neuron (min dist) and updates self.freq
    #            finds best neuron (min dist-self.bias) and returns position
    #            for frequently chosen neurons, self.freq[i] is high and self.bias[i] is negative
    #            self.bias[i] = self.GAMMA*((1/self.NETSIZE)-self.freq[i])"""
    #
    #    i, j = self.SPECIALS, self.NETSIZE
    #    dists = abs(self.network[i:j] - np.array([b,g,r])).sum(1)
    #    bestpos = i + np.argmin(dists)
    #    biasdists = dists - self.bias[i:j]
    #    bestbiaspos = i + np.argmin(biasdists)
    #    self.freq[i:j] -= self.BETA * self.freq[i:j]
    #    self.bias[i:j] += self.BETAGAMMA * self.freq[i:j]
    #    self.freq[bestpos] += self.BETA
    #    self.bias[bestpos] -= self.BETAGAMMA
    #    return bestbiaspos
    def contest(self, b, g, r):
        """ Search for biased BGR values
                Finds closest neuron (min dist) and updates self.freq
                finds best neuron (min dist-self.bias) and returns position
                for frequently chosen neurons, self.freq[i] is high and self.bias[i] is negative
                self.bias[i] = self.GAMMA*((1/self.NETSIZE)-self.freq[i])"""
        i, j = self.SPECIALS, self.NETSIZE
        dists = abs(self.network[i:j] - np.array([b,g,r])).sum(1)
        bestpos = i + np.argmin(dists)
        biasdists = dists - self.bias[i:j]
        bestbiaspos = i + np.argmin(biasdists)
        self.freq[i:j] *= (1-self.BETA)
        self.bias[i:j] += self.BETAGAMMA * self.freq[i:j]
        self.freq[bestpos] += self.BETA
        self.bias[bestpos] -= self.BETAGAMMA
        return bestbiaspos




    def specialFind(self, b, g, r):
        for i in range(self.SPECIALS):
            n = self.network[i]
            if n[0] == b and n[1] == g and n[2] == r:
                return i
        return -1

    def learn(self):
        biasRadius = self.INITBIASRADIUS
        alphadec = 30 + ((self.samplefac-1)/3)
        lengthcount = self.pixels.size
        samplepixels = lengthcount / self.samplefac
        delta = samplepixels / self.NCYCLES
        alpha = self.INITALPHA

        i = 0;
        rad = biasRadius * 2**self.RADIUSBIASSHIFT
        if rad <= 1:
            rad = 0

        print("Beginning 1D learning: samplepixels = %1.2f  rad = %i" %
                                                    (samplepixels, rad) )
        step = 0
        pos = 0
        if lengthcount%NeuQuant.PRIME1 != 0:
            step = NeuQuant.PRIME1
        elif lengthcount%NeuQuant.PRIME2 != 0:
            step = NeuQuant.PRIME2
        elif lengthcount%NeuQuant.PRIME3 != 0:
            step = NeuQuant.PRIME3
        else:
            step = NeuQuant.PRIME4

        i = 0
        printed_string = ''
        while i < samplepixels:
            if i%100 == 99:
                tmp = '\b'*len(printed_string)
                printed_string = str((i+1)*100/samplepixels)+"%\n"
                print(tmp + printed_string)
            p = self.pixels[pos]
            r = (p >> 16) & 0xff
            g = (p >>  8) & 0xff
            b = (p      ) & 0xff

            if i == 0: # Remember background colour
                self.network[self.BGCOLOR] = [b, g, r]

            j = self.specialFind(b, g, r)
            if j < 0:
                j = self.contest(b, g, r)

            if j >= self.SPECIALS: # Don't learn for specials
                a = (1.0 * alpha) / self.INITALPHA
                self.altersingle(a, j, b, g, r)
                if rad > 0:
                    self.alterneigh(a, rad, j, b, g, r)

            pos = (pos+step)%lengthcount

            i += 1
            if i%delta == 0:
                alpha -= alpha / alphadec
                biasRadius -= biasRadius / self.RADIUSDEC
                rad = biasRadius * 2**self.RADIUSBIASSHIFT
                if rad <= 1:
                    rad = 0

        finalAlpha = (1.0*alpha)/self.INITALPHA
        print("Finished 1D learning: final alpha = %1.2f!" % finalAlpha)

    def fix(self):
        for i in range(self.NETSIZE):
            for j in range(3):
                x = int(0.5 + self.network[i,j])
                x = max(0, x)
                x = min(255, x)
                self.colormap[i,j] = x
            self.colormap[i,3] = i

    def inxbuild(self):
        previouscol = 0
        startpos = 0
        for i in range(self.NETSIZE):
            p = self.colormap[i]
            q = None
            smallpos = i
            smallval = p[1] # Index on g
            # Find smallest in i..self.NETSIZE-1
            for j in range(i+1, self.NETSIZE):
                q = self.colormap[j]
                if q[1] < smallval: # Index on g
                    smallpos = j
                    smallval = q[1] # Index on g

            q = self.colormap[smallpos]
            # Swap p (i) and q (smallpos) entries
            if i != smallpos:
                p[:],q[:] = q, p.copy()

            # smallval entry is now in position i
            if smallval != previouscol:
                self.netindex[previouscol] = (startpos+i) >> 1
                for j in range(previouscol+1, smallval):
                    self.netindex[j] = i
                previouscol = smallval
                startpos = i
        self.netindex[previouscol] = (startpos+self.MAXNETPOS) >> 1
        for j in range(previouscol+1, 256): # Really 256
            self.netindex[j] = self.MAXNETPOS


    def paletteImage(self):
        """ PIL weird interface for making a paletted image: create an image which
            already has the palette, and use that in Image.quantize. This function
            returns this palette image. """
        if self.pimage is None:
            palette = []
            for i in range(self.NETSIZE):
                palette.extend(self.colormap[i][:3])

            palette.extend([0]*(256-self.NETSIZE)*3)

            # a palette image to use for quant
            self.pimage = Image.new("P", (1, 1), 0)
            self.pimage.putpalette(palette)
        return self.pimage


    def quantize(self, image):
        """ Use a kdtree to quickly find the closest palette colors for the pixels """
        if get_cKDTree():
            return self.quantize_with_scipy(image)
        else:
            print('Scipy not available, falling back to slower version.')
            return self.quantize_without_scipy(image)


    def quantize_with_scipy(self, image):
        w,h = image.size
        px = np.asarray(image).copy()
        px2 = px[:,:,:3].reshape((w*h,3))

        cKDTree = get_cKDTree()
        kdtree = cKDTree(self.colormap[:,:3],leafsize=10)
        result = kdtree.query(px2)
        colorindex = result[1]
        print("Distance: %1.2f" % (result[0].sum()/(w*h)) )
        px2[:] = self.colormap[colorindex,:3]

        return Image.fromarray(px).convert("RGB").quantize(palette=self.paletteImage())


    def quantize_without_scipy(self, image):
        """" This function can be used if no scipy is availabe.
        It's 7 times slower though.
        """
        w,h = image.size
        px = np.asarray(image).copy()
        memo = {}
        for j in range(w):
            for i in range(h):
                key = (px[i,j,0],px[i,j,1],px[i,j,2])
                try:
                    val = memo[key]
                except KeyError:
                    val = self.convert(*key)
                    memo[key] = val
                px[i,j,0],px[i,j,1],px[i,j,2] = val
        return Image.fromarray(px).convert("RGB").quantize(palette=self.paletteImage())

    def convert(self, *color):
        i = self.inxsearch(*color)
        return self.colormap[i,:3]

    def inxsearch(self, r, g, b):
        """Search for BGR values 0..255 and return colour index"""
        dists = (self.colormap[:,:3] - np.array([r,g,b]))
        a= np.argmin((dists*dists).sum(1))
        return a



if __name__ == '__main__':
    im = np.zeros((200,200), dtype=np.uint8)
    im[10:30,:] = 100
    im[:,80:120] = 255
    im[-50:-40,:] = 50

    images = [im*1.0, im*0.8, im*0.6, im*0.4, im*0]
    writeGif('lala3.gif',images, duration=0.5, dither=0)
