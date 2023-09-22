import os
from abc import abstractmethod
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset, IterableDataset, DistributedSampler
import torchvision
import pickle
import math
import bisect
from PIL import Image
from glob import glob
from einops import rearrange, repeat
import albumentations
import random
import cv2
import pandas as pd
import torch.distributed as dist
import numpy as np
from natsort import natsorted
import pytorch_lightning as pl
import webdataset as wds
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf



# from ldm.util import instantiate_from_config, load_partial_from_config, default
from ldm.util import instantiate_from_config, default
from ldm.data.util import dict_collation_fn

def load_partial_from_config():
    pass

class NNMemoryDataset(torch.utils.data.Dataset):

    def __init__(self,retriever, k_nn,
                 debug = False,
                 ids = None,
                 max_trials=10):
        super().__init__()
        self.retriever = retriever
        self.is_debug = debug
        assert self.retriever.load_patch_dataset, 'Need a patch dataset to load patches'
        self.k_nn = k_nn
        self.ids = ids
        self.invalids = set()
        self.max_trials = max_trials


    def __getitem__(self, idx,trial_count=0):
        # idx is a id into the datapool
        try:
            query_embeddings = self.retriever.data_pool['embedding'][idx]
        except Exception as e:
            # for debug purposes
            print(f'Catchy: ', e)
            if self.is_debug:
                query_embeddings = torch.rand((512,), dtype=torch.float).numpy()
            else:
                assert self.ids is not None, f'Need to set ids if intending to enable error handling in {self.__class__.__name__}'
                assert len(self.ids) > 1, 'Need more than one id'
                self.invalids.add(idx)
                next_trial_id = np.random.choice(list(set(self.ids).difference(self.invalids)),1)
                trial_count +=1
                print(f'Sampling next id {next_trial_id}')
                if trial_count <= self.max_trials:
                    return self.__getitem__(next_trial_id,trial_count=trial_count)
                else:
                    print(f'Reached max trials, what\'s happening here?')
                    raise e

        # query_embeddings = self.retriever.data_pool['embedding'][idx]
        nns, _ = self.retriever.searcher.search(
            query_embeddings / np.linalg.norm(query_embeddings, axis=0)[np.newaxis]
            , final_num_neighbors=self.k_nn)


        img_ids = self.retriever.data_pool['img_id'][nns]
        patch_coords = self.retriever.data_pool['patch_coords'][nns]

        out = {'mem_idx': idx}
        nn_patches = []
        for img_id, pc in zip(img_ids,patch_coords):
            img = self.retriever.patch_dset[img_id]['image']
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)


            # crop it
            img = img[pc[1]:pc[3],
                      pc[0]:pc[2]]
            nn_patches.append(img)

        out.update({'nn_patches': torch.stack(nn_patches,0)})

        return out

class WrapForFID(torch.utils.data.Dataset):
    def __init__(self, dataset, key="image", num_restrict=None, rearrange=True):
        super().__init__()
        self.key = key
        if num_restrict is not None:
            if isinstance(num_restrict, int):
                print(f"WARNING: restricting input data to {num_restrict} examples (randomly chosen)")
                dataset = torch.utils.data.Subset(dataset, indices=np.random.choice(np.arange(len(dataset)),
                                                                                    size=(num_restrict,),
                                                                                    replace=False))
            elif isinstance(num_restrict, (list, np.ndarray)):
                print(f"WARNING: restricting input data to {len(num_restrict)} pre-defined examples.s")
                dataset = torch.utils.data.Subset(dataset, indices=num_restrict)
            else:
                raise NotImplementedError()

        self.data = dataset
        self.rearrange = rearrange

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # non-hardcore normalization
        x = self.data[i][self.key]
        x = (x + 1.) / 2.
        x = 255 * x
        if type(x) == torch.Tensor:
            x = x.numpy()
        x = x.astype(np.uint8)

        x = torch.tensor(x)  # fidelity 0.3.0 seems to require this
        if self.rearrange:
            x = rearrange(x, 'h w c -> c h w')
        return x


class SubsetSampler(DistributedSampler):
    '''
    Yields a subset of a given dataset. Filter ids sent to the dataset's `__get_item__()`
    based on a range of label ids "label_range" for the label specified by "label_key"
    '''

    def __init__(self, data_source, label_key,
                 label_range: list, shuffle, num_replicas=None, rank=None, seed=0,
                 drop_last=False, activate_dist=True, **kwargs):
        self.dataset = data_source
        self.activate_dist = activate_dist
        assert hasattr(self.dataset, 'get_subset_by_label_range') and callable(
            self.dataset.get_subset_by_label_range), f'data_source of {self.__class__.__name__} has to implement the get_subset_by_label_range()-method'

        if len(label_range) == 2 and int(np.abs(label_range[1] - label_range[0])) != 1:
            # get the range (including the upper value
            label_range = np.arange(start=min(label_range), stop=max(label_range) + 1)

        print(
            f'{self.__class__.__name__} will select subset of {self.dataset.__class__.__name__} based on the following label range')
        print(label_range)

        self.ids = self.dataset.get_subset_by_label_range(label_key=label_key,
                                                          label_range=label_range,
                                                          **kwargs)

        print(f'Length of selected subset is {len(self.ids)}')

        # for distributed training (which is in our case activated all of the time)
        if self.activate_dist:
            if num_replicas is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                num_replicas = dist.get_world_size()
            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")
                rank = dist.get_rank()
            if rank >= num_replicas or rank < 0:
                raise ValueError(
                    "Invalid rank {}, rank should be in the interval"
                    " [0, {}]".format(rank, num_replicas - 1))
        else:
            num_replicas = 1
            rank = 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle

        if self.drop_last and len(self.ids) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.ids) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.ids) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def __iter__(self):

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.ids), generator=g).numpy()
            indices = list(self.ids[indices])
        else:
            indices = list(np.sort(self.ids))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''

    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass


class PatcherDataset(Dataset):

    def __init__(self, dset,
                 patch_size,
                 img_size=None,
                 n_patches=10,
                 sampling_method='random'):
        super().__init__()
        if isinstance(dset, Dataset):
            self.data = dset
        else:
            self.data = instantiate_from_config(dset)

        if isinstance(n_patches, int):
            self.n_patches = n_patches
            self._len = len(self.data)
            self.n_overall_samples = self.n_patches * self._len
        elif isinstance(n_patches, (list, tuple, np.ndarray)):
            assert len(n_patches) == len(self.data)
            print(f'Max number of patches per image is {max(n_patches)}')
            print(f'Min number of patches per image is {min(n_patches)}')
            self.n_patches = n_patches
            self._len = len(self.data)
            self.n_overall_samples = np.sum(self.n_patches)
        else:
            raise TypeError('Invalid type for n_patches, must be (list | tuple | np.ndarray | int) ')

        print(f'N samples in {self.__class__.__name__} is {self.n_overall_samples}')

        # self.n_patches = n_patches
        self.patch_size = patch_size
        self.resizer = albumentations.SmallestMaxSize(max_size=self.patch_size, interpolation=cv2.INTER_CUBIC)
        self.img_size = img_size

        if sampling_method != 'random':
            raise NotImplementedError('Have to implement it')
        else:
            self.random_crop = True

    def _crop_coords(self, size, ):
        height, width = size
        if self.random_crop:
            # random crop
            h_start = random.random()
            w_start = random.random()
            y1 = int((height - self.patch_size) * h_start)
            x1 = int((width - self.patch_size) * w_start)
        else:
            # center crop
            y1 = (height - self.patch_size) // 2
            x1 = (width - self.patch_size) // 2

        return x1, y1

    def __getitem__(self, idx):
        try:
            data = self.data[idx]
        except IndexError as e:
            print(e)
            print('End reached')
            return {}
        image = data['image']
        h, w = image.shape[:2]
        # todo implement aligned sampling
        overlap_y = None
        overlap_x = None

        smaller = min(h, w)
        rescale_factor = None
        if self.img_size is not None:

            image = albumentations.smallest_max_size(image, min(self.img_size),
                                                     interpolation=cv2.INTER_CUBIC)
            if smaller == h:
                h, w = min(self.img_size), max(self.img_size)
            else:
                h, w = max(self.img_size), min(self.img_size)

            image = albumentations.center_crop(image,
                                               crop_height=h,
                                               crop_width=w)
            smaller = min(self.img_size)
            data['image'] = image

        if isinstance(self.n_patches, int):

            if self.patch_size >= smaller:
                n_patches = 1
                image = self.resizer(image=image)['image']
            elif smaller < np.sqrt(self.n_patches) * self.patch_size:
                n_patches = int((smaller // self.patch_size) ** 2)
            else:
                n_patches = self.n_patches
        else:

            n_patches = self.n_patches[idx]

        sampled_patches = []
        coordinates = []
        for n in range(n_patches):
            x_tl, y_tl = self._crop_coords((h, w))
            x_br, y_br = x_tl + self.patch_size, y_tl + self.patch_size
            coordinates.append(np.array([x_tl, y_tl, x_br, y_br]).astype(int))
            patch = image[y_tl:y_br, x_tl:x_br]
            sampled_patches.append(patch)

        patches = np.stack(sampled_patches, axis=0)
        coordinates = np.stack(coordinates, axis=0)
        img_ids = np.asarray([idx] * patches.shape[0]).astype(int)
        data.update({'patch': patches,
                     'patch_coords': coordinates,
                     'img_id': img_ids})
        return data

    def __len__(self):
        return self._len

identity = lambda x: x

class WebDataModuleFromConfig(pl.LightningDataModule):

    def __init__(self, tar_base, batch_size, train=None, validation=None,
                 test=None, num_workers=4, load_ddp = True, n_nodes=1,
                 index_path=None, val_num_workers=None ,**kwargs):
        super().__init__(self)
        print(f'Setting tar base to {tar_base}')
        self.tar_base = tar_base
        self.batch_size = batch_size
        self.num_workers=num_workers
        if val_num_workers:
            self.val_num_workers = val_num_workers
        else:
            self.val_num_workers = self.val_num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.load_ddp = load_ddp
        self.multinode=n_nodes > 1
        self.n_nodes = n_nodes
        if index_path is not None:
            import faiss
            print(f'Read index from {index_path}')
            self.index = faiss.read_index(index_path)
            faiss.downcast_index(self.index.index).make_direct_map()
            print('Finished reading index')
        else:
            self.index=None

    def make_loader(self,dataset_config, train=True, num_workers=None):
        if not num_workers:
            num_workers = self.num_workers

        if 'image_transforms' in dataset_config:
            image_transforms = [instantiate_from_config(tt) for tt in dataset_config.image_transforms]
        else:
            image_transforms = []

        image_transforms.extend([torchvision.transforms.ToTensor(),
                           torchvision.transforms.Lambda(lambda x: rearrange(x * 2. - 1.,'c h w -> h w c'))])
        image_transforms = torchvision.transforms.Compose(image_transforms)


        if 'transforms' in dataset_config:
            transforms_config = OmegaConf.to_container(dataset_config.transforms)
        else:
            transforms_config = dict()


        # hack to add
        if 'nn_indices.npy' in transforms_config:
            assert self.index is not None, 'Need index for reconstruction'
            print('Adding index to transform configs for reconstructing neighbors')
            transforms_config['nn_indices.npy']['params']['index'] = self.index

        transform_dict = {dkey: load_partial_from_config(transforms_config[dkey]) if transforms_config[dkey] != 'identity' else identity for dkey in transforms_config}
        img_key = dataset_config.get('image_key','jpeg')
        transform_dict.update({img_key:image_transforms})


        shuffle = dataset_config.get('shuffle',0)
        #TODO fid strategy when n exmples not known beforehand
        n_examples = dataset_config.get('n_examples',1e6) // self.n_nodes

        shards_to_load = dataset_config.shards
        dset_name = 'unknown'
        if isinstance(shards_to_load,str):
            print(f'Loading tars based on the string {shards_to_load}')
            tars = os.path.join(self.tar_base, shards_to_load)
            start_shard_id, end_shard_id = dataset_config.shards.split('{')[-1].split('}')[0].split('..')
            n_shards = int(end_shard_id) - int(start_shard_id) + 1
            dset_name = dataset_config.shards.split('-')[0]
        elif isinstance(shards_to_load,int):
            print(f'Creating tar list, max shard is {shards_to_load}')
            try:
                tars = [tf for tf in natsorted(glob(os.path.join(self.tar_base,'*.tar'))) if int(tf.split('/')[-1].split('.')[0]) < shards_to_load]
                n_shards = len(tars)
                random.shuffle(tars)

            except ValueError as e:
                print('ValueERROR : ', e)
                print('tarfile names should follow the pattern <zero_padded_number>.tar . Check names of the files')
        else:
            raise ValueError('shards should be either a string containing consecutive shards or an int defining the max shard number')


        print(f'Got {n_shards} shard files in datafolder for {"training" if train else "validation"}')

        if num_workers > 0:
            assert n_shards % num_workers == 0 , f'Number of workers which is {num_workers} does not evenly divide number of shards which is {n_shards}'
        print(f'Loading webdataset based dataloader based on {n_shards} of {dset_name} dataset.')

        #start creating the dataset
        # TODO this is assuming webdataset v1 as we had to change it due to clip retrieval
        nodesplitter = wds.shardlists.split_by_node if self.multinode else wds.shardlists.single_node_only
        epoch_length = n_examples // (self.batch_size)
        wds_version_test = subprocess.run('conda list | grep webdataset',shell=True,
                                          capture_output=True).stdout.decode('utf-8').split(' ')
        isv1 = any(map(lambda x: x.startswith('0.1.'),wds_version_test))
        print(f'Using webdataset-v{"1" if isv1 else "2"} for improved dataloading')
        if isv1:
            dset = wds.WebDataset(tars,
                                  # nodesplitter=nodesplitter # had to comment that after change from v2 -> v1 caused by clip-retrieval https://github.com/rom1504/clip-retrieval
                                  ).shuffle(shuffle)
            with_epoch_args = {'length': n_examples, 'by_node': True}
        else:
            dset = wds.WebDataset(tars,
                                  nodesplitter=nodesplitter
                                  ).shuffle(shuffle)
            with_epoch_args = {'nsamples': n_examples, 'nbatches': epoch_length}
        if 'filters' in dataset_config:
            for stage in tqdm(dataset_config.filters,desc=f'Applying the following filters: {[f for f in dataset_config.filters]}'):
                f = getattr(dset,stage)
                dset = f(dset,*dataset_config.filters[stage].args,**dataset_config.filters[stage].get('kwargs',dict()))

        print(f'Dataset holding {len(dset.pipeline[0].urls)} shards')



        dset = (dset
                .decode('pil')
                # .to_tuple("jpg;png;jpeg pickle cls hls")
                # .map_tuple(image_transforms,load_partial_from_config(nns_transform) if 'target' in nns_transform else identity,identity,identity)
                .map_dict(**transform_dict)
                .repeat()
                .batched(self.batch_size,partial=False,
                         collation_fn=dict_collation_fn)
                .with_length(n_examples)
                .with_epoch(**with_epoch_args)
                )

        loader = wds.WebLoader(dset, batch_size=None,shuffle=False,
                               num_workers=num_workers)



        return loader, n_examples

    def train_dataloader(self):
        assert self.train is not None, 'Need to define "train" argument when intending to obtain train loader'
        loader, dset_size = self.make_loader(self.train, num_workers=self.num_workers)
        # if self.load_ddp:
        #     loader = loader.ddp_equalize(dset_size // self.batch_size)
        return loader

    def val_dataloader(self):
        assert self.train is not None, 'Need to define "validation" argument when intending to obtain train loader'
        loader, _ = self.make_loader(self.validation,train=False, num_workers=self.val_num_workers)
        return loader

    def test_dataloader(self):
        assert self.train is not None, 'Need to define "test" argument when intending to obtain train loader'
        loader, _ =  self.make_loader(self.test,train=False, num_workers=self.val_num_workers)
        return loader

class ShardedQueryDataset(Dataset):

    def __init__(self,
                 shards,
                 rank,
                 dset_config,
                 n_patches_per_side=1,
                 k=None,
                 load_patches=False,
                 debug=False,
                 n_shards=None,
                 check_intersection=False):
        super().__init__()
        self.data = instantiate_from_config(dset_config)
        self.n_patches_per_side = n_patches_per_side
        #
        self.k_nearest = k
        self.is_debug = debug
        self.size = self.data.size
        if n_shards:
            self.n_shards = n_shards
        else:
            self.n_shards = 1

        self.load_patches = load_patches
        assert self.n_patches_per_side <= 8, 'Currently only 16 patches at max supported'
        assert np.log2(self.n_patches_per_side) == np.rint(np.log2(self.n_patches_per_side))
        self.corrupt_ids = []
        self.basepath = '/'.join(shards.split('/')[:-1])

        all_shard_files = natsorted(glob(os.path.join(shards,'nns_*.p')))
        shard_files = all_shard_files[rank*self.n_shards:(rank+1)*self.n_shards]
        print(f'Shard files are {shard_files}')

        # self.shard = os.path.join(shards, f'nns_{rank:02d}.p')
        self.nns = {}
        for s in shard_files:
            print(f'Load nns in {self.__class__.__name__} with base dataset {self.data.__class__.__name__} from shards {s}')
            with open(s, 'rb') as f:
                nn_data = pickle.load(f)

            nns = {list(n.keys())[0]: list(n.values())[0] for n in nn_data}
            if check_intersection:
                print('Checking for intersections between ids...')
                intersection_length = len(set(nns.keys()).intersection(self.nns.keys()))
                if intersection_length > 0:
                    raise ValueError('intersecting ids between different shards. Please check...')
                print(f'Finished intersection check for shard {s} without issues')

            self.nns.update(nns)

        print(f'Loading ids')
        self.ids = list(self.nns.keys())
        print(f'Finished loading of ids which are range [{self.ids[0]},{self.ids[-1]}]')
        self.data = Subset(self.data, self.ids)

        self.offset = int(np.amin(self.ids))
        print(f'Loaded nns in {self.__class__.__name__}; offset is {self.offset}')

        corrupts = os.path.join(self.basepath, 'corrupts.txt')
        if os.path.isfile(corrupts):
            with open(corrupts, 'r') as f:
                self.corrupt_ids = [int(l.rstrip()) for l in f]

        if self.corrupt_ids:
            print(f'Gotcha! Removing the following corrupt ids: {self.corrupt_ids}')
            for idx in self.corrupt_ids:
                try:
                    del self.nns[idx]
                except Exception as e:
                    print('Error: ', e)

            print(f'Finished removal')

    def __getitem__(self, idx):
        if self.corrupt_ids and idx + self.offset in self.corrupt_ids:
            # randomly sample other idx instead
            print('To corrupts')
            idx = int(np.random.choice(np.arange(self.__len__()), 1))
            return self.__getitem__(idx)

        out = self.data[idx]
        has_image = 'image' in out
        if has_image:
            image = out['image']
            # divide image into patches, we are assuming to deal with quadratic images here
            side_length = image.shape[0]
            patch_size = side_length // self.n_patches_per_side
            patches = []

            nn_idx = idx
            if 'nn_idx' in out:
                nn_idx = out['nn_idx']

            nns = self.nns[nn_idx + self.offset][self.n_patches_per_side]

            # depends on what kinda retriever was used for saving the nns
            if self.k_nearest:
                nns = {key: nns[key][:, :self.k_nearest] for key in nns if key != 'nn_patches'}

            # shape of embeddings [n_patches_per_side**2, self.k_nearest, embed_dim]
            embeddings = nns['embeddings']
            out.update({'nn_embeddings': embeddings})

            for row in range(self.n_patches_per_side):
                for col in range(self.n_patches_per_side):
                    current_patch = image[row * patch_size:(row + 1) * patch_size,
                                    col * patch_size:(col + 1) * patch_size]
                    patches.append(current_patch)

            # shape is [n_patches_per_side,side_length,side_length,3]
            out.update({'patches': np.stack(patches)})

        return out

    def __len__(self):
        return len(self.data)


class QueryDataset(Dataset):

    def __init__(self,
                 dset_config,
                 rset_config=None,
                 n_patches_per_side=1,
                 k=None,
                 nns: str = None,
                 load_patches=False,
                 debug=False,**kwargs ):
        """

        :param dset_config: dataset generating the actual data
        :param n_patches_per_side: number of patches per side, overall number of patches will be n_patches_per_side**2
        :param k: different k than the one used for the precomputed n (should be smaller)
        :param nns: path to the file containing the nearest neigbour paths
        :param load_patches: Whether to also load the extracted single patches (to enable other embedders for the patches)
        """
        super().__init__()
        self.data = instantiate_from_config(dset_config)
        self.n_patches_per_side = n_patches_per_side
        #
        self.k_nearest = k
        self.is_debug = debug
        self.size = self.data.size

        self.load_patches = load_patches
        assert self.n_patches_per_side <= 4, 'Currently only 16 patches at max supported'
        assert np.log2(self.n_patches_per_side) == np.rint(np.log2(self.n_patches_per_side))
        self.nns = nns
        self.nn_paths = None
        self.corrupt_ids = []

        if self.nns:
            if os.path.isfile(self.nns):
                self.is_np = self.nns.endswith('_np.p')
                self.basepath = '/'.join(self.nns.split('/')[:-1])
                # TODO hack due to moved paths
                try:
                    with open(self.nns, 'rb') as f:
                        self.nn_paths = pickle.load(f)
                except FileNotFoundError as e:
                    mfile_name = self.nns.split('/')[-1]
                    reldirname = self.basepath.rsplit('/',maxsplit=1)[-1]
                    dpool_size = reldirname.split('p',maxsplit=1)[0]
                    ret_name = reldirname.split('p-',maxsplit=1)[-1].split('_')[0]
                    ret_dset_name = reldirname.split('@')[-1].split('-')[0]
                    new_meta_filepath = os.path.join('/export/compvis-nfs/group/datasets/nitro_queries_',
                                                     f'{dpool_size}p-{ret_name}@'
                                                     f'{ret_dset_name}_128-{self.data.__class__.__name__}', mfile_name)
                    print(f'Catched FileNotFoundError: {e}')
                    print(f'File {self.nns} not found, setting to {new_meta_filepath} and trying again')

                    with open(new_meta_filepath, 'rb') as f:
                        self.nn_paths = pickle.load(f)

                    self.nns = new_meta_filepath
            elif not os.path.isdir(self.nns):
                self.is_np = self.nns.endswith('_np.p')
                self.basepath = '/'.join(self.nns.split('/')[:-1])
                # TODO hack due to moved paths
                try:
                    with open(self.nns, 'rb') as f:
                        self.nn_paths = pickle.load(f)
                except FileNotFoundError as e:
                    mfile_name = self.nns.split('/')[-1]
                    reldirname = self.basepath.rsplit('/', maxsplit=1)[-1]
                    dpool_size = reldirname.split('p', maxsplit=1)[0]
                    ret_name = reldirname.split('p-', maxsplit=1)[-1].split('@')[0]
                    ret_dset_name = reldirname.split('@')[-1].split('-')[0]
                    new_meta_filepath = os.path.join('/export/compvis-nfs/group/datasets/nitro_queries_',
                                                     f'{dpool_size}p-{ret_name}_128@'
                                                     f'{ret_dset_name}-{self.data.__class__.__name__}', mfile_name)
                    print(f'Catched FileNotFoundError: {e}')
                    print(f'File {self.nns} not found, setting to {new_meta_filepath} and trying again')

                    with open(new_meta_filepath, 'rb') as f:
                        self.nn_paths = pickle.load(f)

                    # reset stuff
                    self.nns = new_meta_filepath
                    self.basepath = '/'.join(self.nns.split('/')[:-1])
            else:
                self.basepath = self.nns
                nnps = glob(os.path.join(self.basepath, 'nn_paths_p*.p'))
                assert len(
                    nnps) > 0, f'No nn_paths found in specified directory "{self.basepath}". Is it a single file?'
                self.nn_paths = {}

                print(f'Loading NN paths from {len(nnps)} individual files.')

                for part_ps in nnps:
                    self.is_np = part_ps.endswith('_np.p')
                    with open(part_ps, 'rb') as f:
                        current_data = pickle.load(f)
                    self.nn_paths.update(current_data)

                print(f'Finished loading of {len(self.nn_paths)} NN paths.')

            corrupts = os.path.join(self.basepath, 'corrupts.txt')
            if os.path.isfile(corrupts):
                with open(corrupts, 'r') as f:
                    self.corrupt_ids = [int(l.rstrip()) for l in f]

            patches_file = os.path.join(self.basepath, f'nns-{self.n_patches_per_side ** 2}_patches.p')


            self.patches_loaded = False
            if self.nns and self.load_patches:
                print(f'Searching for patches file under {patches_file}')
                if os.path.isfile(patches_file):
                    self.patches_loaded = True
                    print('NN patches are also loaded.')
                    with open(patches_file, 'rb') as f:
                        self.patch_paths = pickle.load(f)

                    print('Finish loading of NN patches')
                else:
                    assert rset_config is not None, 'Neither are patches precomputed nor is a retrieval dataset specified. ' \
                                                    'Patch loading not possible'
                    self.r_data = instantiate_from_config(rset_config)
                    assert isinstance(self.r_data, PatcherDataset)

                    print('Patches not precomputed, loading them online')

            if self.corrupt_ids:
                print(f'Gotcha! Removing the following corrupt ids: {self.corrupt_ids}')
                for idx in self.corrupt_ids:
                    try:
                        del self.nn_paths[idx]
                    except Exception as e:
                        print('Error: ', e)
                    if self.patches_loaded:
                        del self.patch_paths[idx]

                print(f'Finished removal')

    def load_nns(self, nn_idx):
        relname = self.nn_paths[nn_idx]
        if relname.split('/')[0].endswith('-in_train'):
            relname = relname.replace('-in_train', '')
        fname = os.path.join(self.basepath, relname)
        if self.is_np:
            nns = np.load(fname, allow_pickle=True)
            nns = {key: nns[key] for key in nns.files}['arr_0'].item()[self.n_patches_per_side]
            # print(nns)
            nns = {k: nns[k] for k in nns if k != 'nn_patches'}

        else:

            nns = pd.read_pickle(fname)[self.n_patches_per_side]

        return nns

    def get_subset_by_label_range(self, **kwargs):
        assert hasattr(self.data, 'get_subset_by_label_range') and callable(self.data.get_subset_by_label_range)
        return self.data.get_subset_by_label_range(**kwargs)

    def get_patches(self, idx):
        patch_paths = self.patch_paths[idx]

        nn_patches = []
        for pp in patch_paths:
            patch_filepath = os.path.join(self.basepath, pp)
            img = Image.open(patch_filepath).convert('RGB')
            img = (np.asarray(img) / 127.5 - 1.).astype(np.float32)

            nn_patches.append(img)

        nn_patches = np.stack(nn_patches, axis=0).reshape(self.n_patches_per_side ** 2, -1, *nn_patches[0].shape)

        return nn_patches

    def __getitem__(self, idx):
        if self.corrupt_ids and idx in self.corrupt_ids:
            # randomly sample other idx instead
            print('To corrupts')
            idx = int(np.random.choice(list(self.nn_paths.keys()), 1))
        out = self.data[idx]
        has_image = 'image' in out
        if has_image:
            image = out['image']
            # divide image into patches, we are assuming to deal with quadratic images here
            side_length = image.shape[0]
            patch_size = side_length // self.n_patches_per_side
            patches = []
            if self.nn_paths is not None:
                nn_idx = idx
                if 'nn_idx' in out:
                    nn_idx = out['nn_idx']
                if self.is_debug:
                    try:
                        nns = self.load_nns(nn_idx)
                    except Exception as e:
                        print(f'ERROR: ', e)
                        # construct dummy data
                        nns = {
                            'embeddings': np.random.rand(self.n_patches_per_side ** 2, self.k_nearest, 512),
                            'img_ids': np.random.choice(np.arange(len(self.data)),
                                                        (self.n_patches_per_side ** 2, self.k_nearest)),
                            'patch_coords': repeat(np.asarray([0, 0, 128, 128], dtype=int), 'p -> n k p',
                                                   n=self.n_patches_per_side ** 2, k=self.k_nearest)
                        }
                else:
                    nns = self.load_nns(nn_idx)

                # depends on what kinda retriever was used for saving the nns
                if self.k_nearest:
                    nns = {key: nns[key][:, :self.k_nearest] for key in nns if key != 'nn_patches'}

                if self.patches_loaded:
                    if self.is_debug:
                        try:
                            nn_patches = self.get_patches(nn_idx)
                        except KeyError as e:
                            nn_patches = np.clip(
                                np.random.randn(self.n_patches_per_side ** 2, self.k_nearest, 128, 128, 3), -1, 1)
                    else:
                        nn_patches = self.get_patches(nn_idx)

                    if self.k_nearest:
                        nn_patches = nn_patches[:, :self.k_nearest]
                    # shape of patches [n_patches_per_side**2, self.k_nearest, h_ret_patch, w_ret_patch, 3]
                    out.update({'nn_patches': nn_patches})
                elif self.load_patches and not self.patches_loaded:
                    img_ids = nns['img_ids']
                    nn_pcs = nns['patch_coords']

                    img_ids = rearrange(img_ids, 'p k -> (p k)')
                    nn_pcs = rearrange(nn_pcs, 'p k n -> (p k) n')

                    nn_patches = []

                    for id_, patch_coords in zip(img_ids, nn_pcs):
                        x_tl, y_tl, x_br, y_br = patch_coords
                        nn_patch = self.r_data[id_]['image'][y_tl:y_br, x_tl:x_br]
                        nn_patches.append(nn_patch)

                    nn_patches = rearrange(np.stack(nn_patches), '(p k) h w c -> p k h w c', k=nns['img_ids'].shape[-1])
                    out.update({'nn_patches': nn_patches})

                # shape of embeddings [n_patches_per_side**2, self.k_nearest, embed_dim]
                embeddings = nns['embeddings']
                out.update({'nn_embeddings': embeddings})

                # for item_id, pc in zip(nns['img_id'],nns['patch_coords']):
                #     nn_patch = self.retrieval_data[item_id][pc[1]:pc[3],pc[0]:pc[2]]
                #     nn_patches.append(nn_patches)

            for row in range(self.n_patches_per_side):
                for col in range(self.n_patches_per_side):
                    current_patch = image[row * patch_size:(row + 1) * patch_size,
                                    col * patch_size:(col + 1) * patch_size]
                    patches.append(current_patch)

            # shape is [n_patches_per_side,side_length,side_length,3]
            out.update({'patches': np.stack(patches)})

        return out

    def __len__(self):
        return len(self.data)


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class PatchShuffleWrapper(Dataset):
    def __init__(self, n_patch, base_config):
        self.n_patch = n_patch
        self.base = instantiate_from_config(base_config)

    def get_subset_by_label_range(self, *args, **kwargs):
        return self.base.get_subset_by_label_range(*args, **kwargs)

    def numpy_shuffle(self, x):
        # takes a numpy image
        H, W, C = x.shape
        x = rearrange(x, 'h w c -> c h w')
        patched = rearrange(x, 'c (ph h) (pw w) -> (ph pw) c h w', ph=self.n_patch, pw=self.n_patch,
                            h=H // self.n_patch, w=W // self.n_patch, c=C)

        B = patched.shape[0]
        idx = np.random.choice(np.arange(B), replace=False, size=B)
        patched = patched[idx]
        patched = rearrange(patched, '(ph pw) c h w -> c (ph h) (pw w)', ph=self.n_patch, pw=self.n_patch,
                            h=H // self.n_patch, w=W // self.n_patch, c=C)
        p = rearrange(patched, 'c h w -> h w c')
        return p

    def __getitem__(self, i):
        ex = self.base[i]
        ex["image_shuffled"] = self.numpy_shuffle(ex["image"])
        return ex


class PRNGMixin(object):
    """Adds a prng property which is a numpy RandomState which gets
    reinitialized whenever the pid changes to avoid synchronized sampling
    behavior when used in conjunction with multiprocessing."""

    @property
    def prng(self):
        currentpid = os.getpid()
        if getattr(self, "_initpid", None) != currentpid:
            self._initpid = currentpid
            self._prng = np.random.RandomState()
        return self._prng


class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

class CustomBase(Dataset):
    def __init__(self, image_list_fn, size, *args, data_root=None, **kwargs):
        super().__init__()
        with open(image_list_fn, "r") as f:
            paths = f.read().splitlines()
        if data_root is not None:
            paths = [os.path.join(data_root, p) for p in paths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

if __name__ == '__main__':
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from kornia.geometry import resize
    from einops import rearrange
    import cv2

    dset_config = OmegaConf.load('configs/data/query_datasets/test.yaml')['train']
    dset = instantiate_from_config(dset_config)
    assert dset.n_patches_per_side == 1

    loader = DataLoader(dset, 5, True, num_workers=2)

    nmax = 10
    for c, batch in enumerate(tqdm(loader, total=nmax)):

        if c >= nmax:
            break
        else:
            continue

        query = ((batch['image'].numpy() + 1.) * 127.5).astype(np.uint8)
        nns = batch['nn_patches']
        bs, n_p, k = nns.shape[:3]
        nns = rearrange(resize(rearrange(nns, 'b n k h w c -> (b n k) c h w'),
                               query.shape[1:-1], interpolation='bicubic'), '(b n k) c h w -> (b n) k h w c', b=bs,
                        n=n_p, k=k).numpy()
        nns = ((nns + 1.) * 127.5).astype(np.uint8)

        rows = []
        for q, nn in zip(query, nns):
            row = np.concatenate([q] + list(nn), axis=1)
            rows.append(row)

        grid = cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'nn_ex{c}.png', grid)
