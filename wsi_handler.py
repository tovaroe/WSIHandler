import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = '1'
import openslide
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion

from wsi_utils import *


class WSIHandler():
    def __init__(self, WSI_path):
        self.WSI_path = WSI_path
        self.WSI = openslide.OpenSlide(WSI_path)
        self.tissue_mask = None
        
    def read_ndpa(self, magnification=0.15625):
              
        # define function to convert ndpa-values to pixel coordinates
        def nm2pix(nm, OffsetFromSlideCenter, mpp, objectivePower, magnification, image_size):
            nm_coord = nm - OffsetFromSlideCenter
            pixel_coord = nm_coord/(mpp*1000)
            pixel_coord = pixel_coord/(objectivePower/magnification)
            pixel_coord = pixel_coord + np.round(image_size/2)
            pixel_coord = int(np.round(pixel_coord))
            return pixel_coord
        
        # get wsi attributes
        mppX = float(self.WSI.properties['openslide.mpp-x'])
        mppY = float(self.WSI.properties['openslide.mpp-y'])
        objectivePower = int(self.WSI.properties['openslide.objective-power'])
        XOffsetFromSlideCentre = float(self.WSI.properties['hamamatsu.XOffsetFromSlideCentre'])
        YOffsetFromSlideCentre = float(self.WSI.properties['hamamatsu.YOffsetFromSlideCentre'])
        width = int(int(self.WSI.properties['openslide.level[0].width'])*magnification/objectivePower)
        height = int(int(self.WSI.properties['openslide.level[0].height'])*magnification/objectivePower)
        nm2pixX = lambda x: nm2pix(x, XOffsetFromSlideCentre, mppX, objectivePower, magnification, width)
        nm2pixY = lambda y: nm2pix(y, YOffsetFromSlideCentre, mppY, objectivePower, magnification, height)
        annotation_image = np.zeros([width, height])
        ndpa_path = self.WSI_path + '.ndpa'
        
        # assertions
        assert (width, height) in self.WSI.level_dimensions, "Width and/or height to not match level_dimensions."
        assert os.path.exists(ndpa_path), f"NDPA file not found. Given path: {ndpa_path}"
        
        # build & walk the xml annotation tree
        xml_tree = ET.parse(ndpa_path)
        annotations = xml_tree.getroot().findall("./ndpviewstate/annotation")
        polymask = Image.new('L', (width, height), 0)
        colormap = {"#ff0000": 1,
                    "#ffff00": 2,
                    "#00ff00": 3,
                    "#0000ff": 4,
                    "#000000": 5,
                    "#ff00ff": 6,
                    "#00ffff": 7,
                    "#ffffff": 8}


        for annotation in annotations:
            if annotation.attrib['displayname'] == 'AnnotateFreehand':
                color = colormap[annotation.attrib['color']]
                
                points = []
                pointlist = annotation.findall("./pointlist/point")
                if not len(pointlist)>1: continue
                for point in pointlist:
                    x = nm2pixX(int(point.findall("./x")[0].text))
                    y = nm2pixY(int(point.findall("./y")[0].text))
                    points.append((x,y))
                
                ImageDraw.Draw(polymask).polygon(points, outline=color, fill=color)

        mask = np.array(polymask)
        return mask
        
        
    def segment_tissue(self, min_area = 100):
        thumbnail = self.get_thumbnail()
        
        cd = colour_deconvolution(thumbnail, 'HE')+1e-9
        mask = binary_erosion(((cd[:,:,0]/2. + cd[:,:,1]/2.)/cd[:,:,2]) < 0.93)
        mask = bw_filter_area(mask, min_area)
        return mask
        
    def obtain_tissue_mask(self, min_area = 100, segment_tissue=True, read_ndpa=True):
        #always uses minimal magnification
        if segment_tissue:
            tissue_mask = self.segment_tissue(min_area)
        if read_ndpa:
            ndpa_mask = self.read_ndpa()
        
        #Todo: think more cases in terms of including different annotations here
        tissue_mask[ndpa_mask > 0] = False
        tissue_mask = bw_filter_area(tissue_mask, min_area)
        self.tissue_mask = tissue_mask
        return self.tissue_mask
        
    def save_tissue_mask(self, path):
        if not self.tissue_mask:
            self.obtain_tissue_mask()
        
        self.tissue_mask
      
    def get_random_tile(self, magnification, width, height):
        if tissue_mask:
            pass
        pass
        
    def get_thumbnail(self, magnification = 0.15625):
        # magnification 0.15625 corresponds to downsample = 256 with 40x objective power
        downsample = int(self.WSI.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])/magnification
        thumbnail_size = self.WSI.level_dimensions[self.WSI.get_best_level_for_downsample(downsample)]
        return self.WSI.get_thumbnail(thumbnail_size)
    
    