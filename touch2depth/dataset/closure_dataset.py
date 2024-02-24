''' 
ClosureDataset: Subclass of the pix2pix BaseDataset Module
'''

from dataset.base_dataset import BaseDataset, get_transform
import cv2
from dataset.image_folder import make_dataset
from PIL import Image
import os

class ClosureDataset(BaseDataset):
    """Dataset of natural unified whole images & their Gestalt closure variants."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Adds new ClosureDataset-specific parser options, and rewrites default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. Use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--closure_policy', type=str, default = 'patching', help='specify closure generation policy [boundary | patching | superimposition]')

        # closure dataset-specific default values:
        parser.set_defaults(gpu_ids = '0,1,2,3,4,5,6,7', # multi-gpu mode
                            batch_size = 128, 
                            serial_batches = False, # if true, takes images in order to make batches, otherwise takes them randomly
                            preprocess="none", 
                            no_flip = True,
                            direction = 'AtoB') # closure -> whole
        return parser


    def __init__(self, opt):
        """Initializes the ClosureDataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        # retrieve image paths for closure dataset;
        dataset_folder = self.root  # e.g. /local/crv/ege/datasets/closure/closure_patching/train
        self.closure_path = os.path.join(dataset_folder, 'closure')
        self.whole_path = os.path.join(dataset_folder, 'whole')
        # self.mask_path = os.path.join(dataset_folder, 'mask')

        self.whole_paths =  sorted(make_dataset(self.closure_path, opt.max_dataset_size))
        self.closure_paths = sorted(make_dataset(self.closure_path, opt.max_dataset_size))
        assert len(self.closure_paths) == len(self.whole_paths)
        
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index -- a random integer for data indexing
        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        closure_path = os.path.join(self.closure_path, "%s_closure.png" % str(index))
        whole_path = os.path.join(self.whole_path, "%s_whole.png" % str(index))
        # mask_path = os.path.join(self.masks_path, "%s_mask.png" % idx_str)

        closure_img = self.transform(Image.open(closure_path).convert('RGB')) # tensor 
        whole_img = self.transform(Image.open(whole_path).convert('RGB')) # tensor
        # mask = None # We can use segmentation masks for loss

        # opt.direction: AtoB by default (closure -> whole)
        return {'A': closure_img, 'B': whole_img, 'A_paths': closure_path, 'B_paths': whole_path}

    def __len__(self):
        """Return the total number of whole & closure variant pairs."""
        return len(self.closure_paths)
    
def read_segmentation_mask(file_path):
    """
    In:
        file_path: Path to binary mask png image for the object under closure.
    Out:
        segmentation mask as np array [height, width].
    Purpose:
        Read in a segmentation mask image.
    """
    return cv2.imread(file_path, -1)

def read_rgb(file_path):
    """
    In:
        file_path: Color image png to read.
    Out:
        RGB image as np array [height, width, 3], each value in range [0, 255]. Color channel in the order RGB.
    Purpose:
        Read in a color image.
    """
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
