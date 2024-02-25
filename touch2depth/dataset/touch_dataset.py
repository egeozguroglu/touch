''' 
TouchDataset: Subclass of the pix2pix BaseDataset Module
'''

from dataset.base_dataset import BaseDataset, get_transform
import cv2
from dataset.image_folder import make_dataset
from PIL import Image
import os

class TouchDataset(BaseDataset):
    """Dataset of touch RGB image and depth map pairs."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Adds new TouchDataset-specific parser options, and rewrites default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. Use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """

        # touch dataset-specific default values:
        parser.set_defaults(# gpu_ids = '0,1,2,3,4', # multi-gpu mode
                            # batch_size = 1 # following the pix2pix paper.
                            serial_batches = False, # if true, takes images in order to make batches, otherwise takes them randomly
                            load_size = 256,
                            preprocess="resize", 
                            no_flip = True,
                            # max_dataset_size=5000,
                            direction = 'AtoB') # touch -> depth
        return parser


    def __init__(self, opt):
        """Initializes the TouchDataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        # retrieve image paths for touch dataset;
        dataset_folder = self.root 
        self.touch_path = os.path.join(dataset_folder, 'touch')
        self.depth_path = os.path.join(dataset_folder, 'depth')

        self.touch_paths =  sorted(make_dataset(self.touch_path, opt.max_dataset_size))
        self.depth_paths = sorted(make_dataset(self.depth_path, opt.max_dataset_size))
        assert len(self.touch_paths) == len(self.depth_paths)
        
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index -- a random integer for data indexing
        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        touch_path = self.touch_paths[index]
        depth_path = self.depth_paths[index]

        touch_img = self.transform(Image.open(touch_path).convert('RGB')) # tensor 
        depth_img = self.transform(Image.open(depth_path).convert('RGB')) # tensor

        # opt.direction: AtoB by default (touch -> depth)
        return {'A': touch_img, 'B': depth_img, 'A_paths': touch_path, 'B_paths': depth_path}

    def __len__(self):
        """Return the total number of touch & depth pairs."""
        return len(self.touch_paths)

def read_rgb(file_path):
    """
    In:
        file_path: Color image png to read.
    Out:
        RGB image as np array [height, width, 3], each value in range [0, 255]. 
        Color channel in the order RGB.
    Purpose:
        Read in a color image.
    """
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

def read_mask(file_path):
    """
    In:
        file_path: Path to binary mask png image.
    Out:
        mask as np array [height, width].
    Purpose:
        Read in a segmentation mask image.
    """
    return cv2.imread(file_path, -1)
