import torch
import numpy as np
import cv2

from ldm.modules.midas.api import load_midas_transform
import torchvision.transforms as tt


class AddMiDaS(object):
    def __init__(self, model_type):
        super().__init__()
        self.transform = load_midas_transform(model_type)

    def pt2np(self, x):
        x = ((x + 1.0) * .5).detach().cpu().numpy()
        return x

    def np2pt(self, x):
        x = torch.from_numpy(x) * 2 - 1.
        return x

    def __call__(self, sample):
        # sample['jpg'] is tensor hwc in [-1, 1] at this point
        x = self.pt2np(sample['jpg'])
        x = self.transform({"image": x})["image"]
        sample['midas_in'] = x
        return sample
    
def prepare_cond_rep(rep):
    return rep[None]

def reconstruct_nns(nn_ids,knn,index, sample_range):
    # remove faulty ids
    nn_ids = nn_ids[nn_ids!=-1]
    if nn_ids.size==0:
        # fallback
        print('fallback as no neighbors found')
        embds = np.zeros((1,knn,768))
    else:
        # sample to avoid duplicates and increase generalization
        nn_ids = np.random.choice(min(len(nn_ids),sample_range),
                                  size=knn,replace=False)

        embds = []
        for idx in nn_ids:
            rec_embds = index.reconstruct(int(idx))
            embds.append(rec_embds)
        # add extra dimension to account for n_pathces which is here always 1
        embds = np.stack(embds)[None]

    return embds


def extract_nns(nns, knn, n_patches=1):
    nns = nns[n_patches]
    return nns['embeddings'][:, :knn]

def load_txt(data):
    return data.decode('utf-8')

def load_int(data):
    return int(data)

def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    batched = {key: [] for key in samples[0]}
    # assert isinstance(samples[0][first_key], (list, tuple)), type(samples[first_key])

    for s in samples:
        [batched[key].append(s[key]) for key in batched]


    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                # import torch

                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
        # result.append(b)
    return result



# def save_exr(path, data):
#     """
#     Args: 
#         path: str 
#         data: np.array (float32) (hight, width, channels)
#     ---------------------------------------------
#     Save data to an EXR file. For optimal results this should be an float32 array.
#     """
#     header = OpenEXR.Header(data.shape[1], data.shape[0])
#     dt = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
#     header['channels'] = dict([(c, dt) for c in 'RGB'])
#     exr = OpenEXR.OutputFile(path, header)
#     R = (data[:,:,0]).astype(np.float32).tobytes()
#     G = (data[:,:,1]).astype(np.float32).tobytes()
#     B = (data[:,:,2]).astype(np.float32).tobytes()
#     exr.writePixels({'R' : R, 'G' : G, 'B' : B})
#     exr.close()

# def load_exr(path):
#     """
#     Args: 
#         path: str 
#     ----------------------------------------------
#     Load an exr file from path.
#     --------------------------------------------
#     returns: np.array (float32) (hight, width, channels)
#     """
#     exr = OpenEXR.InputFile(path)
#     dw = exr.header()['dataWindow']
#     size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

#     # Get correct data type
#     if exr.header()['channels']['R'] == Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)):
#         dt = np.float16
#     else:
#         dt = np.float32

#     # Fill numpy array
#     arr = np.zeros((size[0], size[1], 3), dt)
#     for i, c in enumerate(['R', 'G', 'B']):
#         arr[:,:,i] = np.frombuffer(exr.channel(c), dt).reshape(size)
            
#     return arr.astype(np.float32)

# def load_renders(render_folder):
#     '''
#     Args: 
#         render_folder: str (path to rendered images folder)
#     -------------------------------------------------------
#     loads predicted and true RGB renders of Bunnys. 
#     The names have to include pred, if it is an predicted env. map 
#                               true, if it is an true env. map
#     ---------------------------------------------------------
#     returns:    
#         true (np.array) (num_imgs, hight, width, channels)
#         pred (np.array) (num_imgs, hight, width, channels)
#     '''

#     pred_paths = natsorted(glob(os.path.join(render_folder, '*_Render.png')))

    
#     # find out img resolution and number of images: 
#     true_imgs = None
    
#     # same for predicted images:
#     if pred_paths:
#         img = np.asarray(PIL.Image.open(pred_paths[0]), dtype=np.uintc)
#         pred_imgs = np.zeros( (len(pred_paths), *img.shape))

#         for i, pred_path in enumerate(pred_paths):
#             pred_imgs[i] = np.asarray(PIL.Image.open(pred_path), dtype=np.uintc)
    
#     else: 
#         assert len(pred_paths) != 0, 'No predictions found! Please check if polder path is correct or if predictions have .exr format'

#     return true_imgs, pred_imgs


def sphere_distance(azimuth, zenith, curr_azimuth, curr_zenith):
    '''
    Args:   
        azimuth: float 
        zenith: float 
        curr_azimuth: float 
        curr_zenith: float
    -----------------------------------------------
    Calculates the Haversine Distance on a sphere: https://en.wikipedia.org/wiki/Haversine_formula
    between two points, given by P1 (azimuth, zenith) and P2 (curr_azimuth, curr_zenith)
    ------------------------------------------------
    reutrns: 
        distance: float
    '''
    distance = 2 * np.arcsin( np.sqrt(
                                                np.sin( (zenith - curr_zenith)/2 )**2 + 
                                                np.cos(curr_zenith) * np.cos(zenith) * 
                                                np.sin( (azimuth - curr_azimuth )/2 )**2 
                                            )
                                )

    return distance

def calc_distance_map(param_dic, sun_model, threshold= 5.):
    '''
    Args: 
        param_dic: dict (dictionary of gt parameters)
        sun_model: np.array() hdr_sun_model
        threshold: float
    ------------------------------------------------
    returns an numpy grayscale array of the distance to the sun. 
    In order to keep it easy we just calculate the distance from the center of the sun 
    while setting all vlaues higher than the threshold to distnace 1.

    This means that there might be a discontinuety at the edge of the sun and 
    distance 
    -------------------------------------------------
    returns: 
        np.array (hight, width, channels) distances
    '''

    # calculate angles out of u, v 
    azimuth = (2 * np.pi * param_dic['sunpos_u']) - np.pi
    zenith = (np.pi * param_dic['sunpos_v']) - np.pi/2

    intensity = np.sqrt(np.sum(sun_model**2, axis=2))
    distance_map = np.ones_like(intensity)
    # distance_map[int(distance_map.shape[0]/2):, :] = 0.

    mask = intensity > threshold
    distance_map[mask] = 0.

    width = intensity.shape[1]
    hight = intensity.shape[0]


    # speed this up by using an array computation 
    width = np.arange(intensity.shape[1])
    hight = np.arange(intensity.shape[0])

    curr_width, curr_hight = np.meshgrid(width, hight)

    curr_azimuth = (2 * np.pi * curr_width/intensity.shape[1]) - np.pi
    curr_zenith = (np.pi * curr_hight/intensity.shape[0]) - np.pi/2

    azimuth = np.ones_like(curr_width).astype(float) * azimuth
    zenith = np.ones_like(curr_hight).astype(float) * zenith

    distance_map = sphere_distance(azimuth, zenith,
                                   curr_azimuth, curr_zenith)
    # for curr_width in range(width):
    #     for curr_hight in range(int(hight)):

    #         if distance_map[curr_hight, curr_width] != 0.:
    #             # normalise to u and v 
    #             curr_u = curr_width/width
    #             curr_v = curr_hight/hight

    #             # get angles
    #             curr_azimuth = (2 * np.pi * curr_u) - np.pi
    #             curr_zenith = (np.pi * curr_v) - np.pi/2

    #             distance_map[curr_hight, curr_width] = sphere_distance(azimuth, zenith, 
    #                                                                    curr_azimuth, curr_zenith)
    
    distance_map[mask] = 0. 

    return distance_map

def find_ellipse(sun_model, threshold = 5.):
    '''
    Args: 
        picture: np.array (Sun model)
        threshold: float
    -----------------------------------------------
    Should find the ellipse sorrounding the sun. 

    -----------------------------------------------
    returns: 
        ellipse 
    
    '''
    # Create contour out of HDR image 
    intensity = np.sqrt(np.sum(sun_model**2, axis=2))
    sun_mask = np.zeros_like(intensity)
    sun_mask[intensity > threshold] = 1. 

    # Create open-cv array
    thresh = sun_mask.astype(np.uint8)*255

    # Find contours; If we find two we should try to rotate the image, in order
    # to not split the sun in half at the edge of the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 1: 
        print('rotate until we have only one')
    
    # fit ellipse 
    ellipse = cv2.fitEllipse(contours[0])

    cv2.ellipse(thresh, ellipse, (0, 0, 255), 3)

    # cv2.imshow("Ellipse", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def create_image_grid(images: np.ndarray, grid_size = None):
    """
    Create a grid with the fed images
    Args:
        images (np.array): array of images
        grid_size (tuple(int)): size of grid (grid_width, grid_height)
    Returns:
        grid (np.array): image grid of size grid_size
    """
    # Sanity check
    assert images.ndim == 3 or images.ndim == 4, f'Images has {images.ndim} dimensions (shape: {images.shape})!'
    num, img_h, img_w, c = images.shape
    # If user specifies the grid shape, use it
    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
        # If one of the sides is None, then we must infer it (this was divine inspiration)
        if grid_w is None:
            grid_w = num // grid_h + min(num % grid_h, 1)
        elif grid_h is None:
            grid_h = num // grid_w + min(num % grid_w, 1)

    # Otherwise, we can infer it by the number of images (priority is given to grid_w)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    # Sanity check
    assert grid_w * grid_h >= num, 'Number of rows and columns must be greater than the number of images!'
    # Get the grid
    grid = np.zeros([grid_h * img_h, grid_w * img_h] + list(images.shape[-1:]), dtype=images.dtype)
    # Paste each image in the grid
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y:y + img_h, x:x + img_w, ...] = images[idx]
    return grid


def resize_image_pil(input_image, resolution):
    H, W, C = input_image.shape
    img = tt.ToTensor()(input_image)
    img = tt.CenterCrop(min(H, W))(img)
    
    H = W = float(min(H, W))
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = tt.CenterCrop(max(H,W))(
        tt.Resize(max(W, H))(tt.ToTensor()(input_image)))
    img = np.array(img.permute(1,2,0) * 255).astype(np.uint8)
#     img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img