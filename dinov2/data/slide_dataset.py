import os
import h5py
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from scipy.interpolate import griddata  
import matplotlib.pyplot as plt
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

class SlideDataset(Dataset):
    def __init__(self,
                root_path: str,
                exlude_data_path: str = '/homes/gws/jbshen/prov-gigapath/dino/datasets/LUAD-5-gene_TCGA.csv',
                shuffle_tiles: bool = False,
                max_tiles: int = 1000,
                global_crops_scale: float = 0.95,
                local_crops_scale: float = 0.5,
                local_crops_number: int = 8
                    ):

        #self.images getall files in the root_path
        self.images = os.listdir(root_path)
        #read the excluded data
        exclude_data = pd.read_csv(exlude_data_path)
        Ids = exclude_data['slide_id'].values
        #Add .h5 to the excluded data
        Ids = [Id + '.h5' for Id in Ids]

        #remove the excluded data from the images
        self.images = [img for img in self.images if img not in Ids]
        
        self.root_path = root_path
        self.shuffle_tiles = shuffle_tiles
        self.max_tiles = max_tiles
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        print('Number of slides:', len(self.images))

    def shuffle_data(self, images: torch.Tensor, coords: torch.Tensor) -> tuple:
        '''Shuffle the serialized images and coordinates'''
        indices = torch.randperm(len(images))
        images_ = images[indices]
        coords_ = coords[indices]
        return images_, coords_

    def read_assets_from_h5(self, h5_path: str) -> tuple:
        '''Read the assets from the h5 file'''
        assets = {}
        attrs = {}
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                assets[key] = f[key][:]
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
        return assets, attrs
    
    def get_crop(self, images: torch.Tensor, coords: torch.Tensor, percentage: float) -> tuple:
        # Get unique x and y coordinates
        unique_x = coords[:, 0].unique().sort()[0]
        unique_y = coords[:, 1].unique().sort()[0]
        
        # Determine the crop size to maintain the same number of unique x and y coordinates
        crop_size = int(percentage * min(unique_x.size(0), unique_y.size(0)))
            
        # get randomized center crop range for x and y coordinates
        if unique_x.size(0) == crop_size:
            random_pad_x = 0
        else:
            random_pad_x = torch.randint(0, unique_x.size(0) - crop_size, (1,)).item()
        if unique_y.size(0) == crop_size:
            random_pad_y = 0
        else:
            random_pad_y = torch.randint(0, unique_y.size(0) - crop_size, (1,)).item()

        center_x_start = unique_x[random_pad_x]
        center_x_end = unique_x[random_pad_x + crop_size-1]
        center_y_start = unique_y[random_pad_y]
        center_y_end = unique_y[random_pad_y + crop_size-1]

        # Filter the coordinates to get the center crop
        indices = (coords[:, 0] >= center_x_start) & (coords[:, 0] <= center_x_end) & (coords[:, 1] >= center_y_start) & (coords[:, 1] <= center_y_end)
        mask = torch.zeros(len(coords), dtype=torch.bool)
        mask[indices] = True
        cropped_coords = coords[mask]
        cropped_images = images[mask]

        return cropped_images, cropped_coords

    def pad_to_max(self, images_list: list, coords_list: list) -> tuple:
        '''Pad the images and coords to have the same number of patches'''
        max_len = max([len(images) for images in images_list])
        for i in range(len(images_list)):
            if len(images_list[i]) < max_len:
                pad_len = max_len - len(images_list[i])
                pad_images = torch.zeros(pad_len, images_list[i].size(1))
                pad_coords = torch.zeros(pad_len, coords_list[i].size(1))
                images_list[i] = torch.cat([images_list[i], pad_images], dim=0)
                coords_list[i] = torch.cat([coords_list[i], pad_coords], dim=0)
        return images_list, coords_list
    
    def pad_to_min(self, images_list: list, coords_list: list) -> tuple:
        '''Pad the images and coords to have the same number of patches'''
        min_len = min([len(images) for images in images_list])
        #randomly exclude some patches, keep the original order
        for i in range(len(images_list)):
            if len(images_list[i]) > min_len:
                indices = torch.randperm(len(images_list[i]))[:min_len]
                #get boolean mask for the indices
                mask = torch.zeros(len(images_list[i]), dtype=torch.bool)
                mask[indices] = True
                images_list[i] = images_list[i][mask]
                coords_list[i] = coords_list[i][mask]

        return images_list, coords_list
    
    def reduce_images(self, images: torch.Tensor, coords: torch.Tensor, max_tiles: int) -> tuple:
        mask = torch.zeros(len(images), dtype=torch.bool)
        random_indices = torch.randperm(len(images))[:max_tiles]
        mask[random_indices] = True
        reduced_images = images[mask]
        reduced_coords = coords[mask]
        return reduced_images, reduced_coords
    
    #global and local crops
    def multi_crop(self, images: torch.Tensor, coords: torch.Tensor) -> tuple:
        #Crop the images into multiple patches

        global_images_list = []
        global_coords_list = []
        local_images_list = []
        local_coords_list = []
        #plot coords
        
        # plt.clf()
    
        # plt.scatter(coords[:, 0], coords[:, 1])
        # plt.savefig('coords1.png')
        
        # Get the global crop
        for i in range(2):
            global_images, global_coords = self.get_crop(images, coords, self.global_crops_scale)
            if global_images.size(0) > self.max_tiles:
                global_images, global_coords = self.reduce_images(global_images, global_coords, self.max_tiles)
            if self.shuffle_tiles:
                global_images, global_coords = self.shuffle_data(global_images, global_coords)
            global_images_list.append(global_images)
            global_coords_list.append(global_coords)
        
        #pad global crops to have the same number of patches
        #global_images_list, global_coords_list = self.pad_to_max(global_images_list, global_coords_list)
        global_images_list, global_coords_list = self.pad_to_min(global_images_list, global_coords_list)
        
        #get local crops
        max_local_tiles = int(self.max_tiles*self.local_crops_scale*self.local_crops_scale)
        for i in range(self.local_crops_number):
            local_images, local_coords = self.get_crop(images, coords, self.local_crops_scale)
            if local_images.size(0) > max_local_tiles:
                local_images, local_coords = self.reduce_images(local_images, local_coords, max_local_tiles)
            if self.shuffle_tiles:
                local_images, local_coords = self.shuffle_data(local_images, local_coords)
            local_images_list.append(local_images)
            local_coords_list.append(local_coords)
        
        #pad local crops to have the same number of patches
        #local_images_list, local_coords_list = self.pad_to_max(local_images_list, local_coords_list)
        local_images_list, local_coords_list = self.pad_to_min(local_images_list, local_coords_list)

        return global_images_list, global_coords_list, local_images_list, local_coords_list

    def get_images_from_path(self, img_path: str) -> dict:
        '''Get the images from the path'''

        assets, _ = self.read_assets_from_h5(img_path)
        origin_images = torch.from_numpy(assets['features'])
        origin_coords = torch.from_numpy(assets['coords'])
        global_images, global_coords, local_images, local_coords = self.multi_crop(origin_images, origin_coords)

        
        # set the input dict
        data_dict = {"global_crops": global_images,
                        'global_coords': global_coords,
                        "local_crops": local_images,
                        'local_coords': local_coords}

        return data_dict
    
    def get_one_sample(self, idx: int) -> dict:
        '''Get one sample from the dataset'''
        # get the slide id
        slide_id = self.images[idx]
        # get the slide path
        slide_path = os.path.join(self.root_path, slide_id)
        # get the slide images
        data_dict = self.get_images_from_path(slide_path)

        # set the sample dict
        sample = {"global_crops": data_dict["global_crops"],
                    'global_coords': data_dict['global_coords'],
                    "local_crops": data_dict["local_crops"],
                    'local_coords': data_dict['local_coords'],
                    'slide_id': slide_id}
        return sample
    
    def get_sample_with_try(self, idx, n_try=3):
        '''Get the sample with n_try'''
        for _ in range(n_try):
            try:
                sample = self.get_one_sample(idx)
                return sample
            except:
                print('Error in getting the sample, try another index')
                idx = np.random.randint(0, len(self.images))
               
        print('Error in getting the sample, skip the sample')
        return None
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        sample = self.get_sample_with_try(idx)
        return sample

    
if __name__ == '__main__':

    data_path  = '/m-ent1/ent1/xuhw/TCGA-embed/GigaPath-giant-1B/h5_files/'
    #data_path = '/homes/gws/jbshen/remote_files/h5_files/'
    dataset = SlideDataset(root_path = data_path, shuffle_tiles=False, max_tiles=819, global_crops_scale=0.95, local_crops_scale=0.5, local_crops_number=2)
    
    sample = dataset.get_sample_with_try(99)

 
    print(len(sample["global_crops"]))
    print(len(sample['global_coords']))
    print(len(sample["local_crops"]))
    print(len(sample['local_coords']))
    print("global crops 0")
    print(sample['global_crops'][0].size())
    print(sample['global_coords'][0].size())
    print(sample['global_coords'][0])
    print("global crops 1")
    print(sample['global_crops'][1].size())
    print(sample['global_coords'][1].size())
    print(sample['global_coords'][1])
    print("local crops 0")
    print(sample['local_crops'][0].size())
    print(sample['local_coords'][0].size())
    #print the coord one by one
    for i in range(len(sample['local_coords'][0])):
        print(sample['local_coords'][0][i])
    print("local crops 1")
    print(sample['local_crops'][1].size())
    print(sample['local_coords'][1].size())
    #print the coord one by one
    for i in range(len(sample['local_coords'][1])):
        print(sample['local_coords'][1][i])
    
    # check every sample in the dataset, using tdqm to show the progress
    # import tqdm
    # for i in tqdm.tqdm(range(len(dataset))):
    #     dataset.get_sample_with_try(i)
    #     '''try:
    #         dataset.get_sample_with_try(i)
    #     except:
    #         print('Error in getting the sample_{}'.format(i))
    #         continue'''
    

    
        