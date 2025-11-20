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
                exlude_data_path: str = 'prov-gigapath/dino/datasets/LUAD-5-gene_TCGA.csv',
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
        #print('Number of slides:', len(self.images))

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
    
    # def get_crop(self, images: torch.Tensor, coords: torch.Tensor, percentage: float) -> tuple:
    #     # Get unique x and y coordinates
    #     unique_x = coords[:, 0].unique().sort()[0]
    #     unique_y = coords[:, 1].unique().sort()[0]
        
    #     # Determine the crop size to maintain the same number of unique x and y coordinates
    #     crop_size = int(np.ceil(percentage * min(unique_x.size(0), unique_y.size(0))))
    #     #take ceil of the crop_size

    #     # get randomized center crop range for x and y coordinates
    #     random_pad_x = torch.randint(0, unique_x.size(0) - crop_size, (1,)).item()
    #     random_pad_y = torch.randint(0, unique_y.size(0) - crop_size, (1,)).item()

    #     center_x_start = unique_x[random_pad_x]
    #     center_x_end = unique_x[random_pad_x + crop_size]
    #     center_y_start = unique_y[random_pad_y]
    #     center_y_end = unique_y[random_pad_y + crop_size]

    #     # Filter the coordinates to get the center crop
    #     indices = (coords[:, 0] >= center_x_start) & (coords[:, 0] <= center_x_end) & (coords[:, 1] >= center_y_start) & (coords[:, 1] <= center_y_end)
  
    #     cropped_coords = coords[indices]
    #     cropped_images = images[indices]

    #     # Create a grid of coordinates for interpolation
    #     grid_x, grid_y = torch.meshgrid(torch.arange(crop_size), torch.arange(crop_size))
    #     grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).float()  # (crop_size * crop_size, 2)
    #     #times the tile size 256 to get the real coordinates
    #     grid = grid * 256
        
    #     #rearrange the cropped_images with similar order as grid
    #     indices = torch.argsort(cropped_coords[:, 1])
    #     cropped_coords = cropped_coords[indices]
    #     cropped_images = cropped_images[indices]
    #     indices = torch.argsort(cropped_coords[:, 0])
    #     cropped_coords = cropped_coords[indices]
    #     cropped_images = cropped_images[indices]

    #     # Reshape cropped_images for interpolation
    #     cropped_images = cropped_images.view(1, 1, -1, images.size(1))  # Add batch and channel dimensions
    #     interpolated_images = F.interpolate(cropped_images, size=(crop_size*crop_size, cropped_images.size(-1)), mode='nearest')
    #     interpolated_images = interpolated_images.view(crop_size * crop_size, images.size(1))
    #     #remove the batch and channel dimensions
    #     interpolated_images = interpolated_images.squeeze(0).squeeze(1)

    #     #plot interpolated coords
    #     #plt.scatter(grid[:, 0], grid[:, 1])
    #     #plt.savefig('interpolated_coords.png')
        
    #     return interpolated_images, grid
    
    
    # def get_crop(self, images: torch.Tensor, coords: torch.Tensor, percentage: float) -> tuple:
    #     # Get unique x and y coordinates
    #     unique_x = coords[:, 0].unique().sort()[0]
    #     unique_y = coords[:, 1].unique().sort()[0]
        
    #     # Determine the crop size to maintain the same number of unique x and y coordinates
    #     crop_size = int(percentage * min(unique_x.size(0), unique_y.size(0)))
    #     print(crop_size)
    #     # get randomized center crop range for x and y coordinates
    #     #x_pad = (unique_x.size(0) - crop_size) // 2
    #     #y_pad = (unique_y.size(0) - crop_size) // 2
    #     random_pad_x = torch.randint(0, unique_x.size(0) - crop_size, (1,)).item()
    #     random_pad_y = torch.randint(0, unique_y.size(0) - crop_size, (1,)).item()

    #     center_x_start = unique_x[random_pad_x]
    #     center_x_end = unique_x[random_pad_x + crop_size]
    #     center_y_start = unique_y[random_pad_y]
    #     center_y_end = unique_y[random_pad_y + crop_size]

    #     # Filter the coordinates to get the center crop
    #     indices = (coords[:, 0] >= center_x_start) & (coords[:, 0] <= center_x_end) & (coords[:, 1] >= center_y_start) & (coords[:, 1] <= center_y_end)
    #     cropped_coords = coords[indices]
    #     cropped_images = images[indices]
    #     print(cropped_coords.size())
    #     print(cropped_images.size())
    #     #plot cropped coords

    #     #sort the cropped coords by x


    #     # Create a grid for interpolation
    #     grid_x = torch.linspace(center_x_start, center_x_end, crop_size)
    #     grid_y = torch.linspace(center_y_start, center_y_end, crop_size)
    #     grid_xx, grid_yy = torch.meshgrid(grid_x, grid_y)
    #     grid = torch.stack([grid_xx.flatten(), grid_yy.flatten()], dim=1)  # (crop_size*crop_size, 2)

    #     # Interpolate the cropped_images
    #     interpolated_images = griddata(cropped_coords.numpy(), cropped_images.numpy(), grid.numpy(), fill_value=0, method='cubic')     

    #     interpolated_images = torch.tensor(interpolated_images, dtype=images.dtype)  # Convert back to tensor

    #     print(interpolated_images)
    #     print(grid.size())
    #     print(interpolated_images.size())

    #     #plot interpolated coords
    #     plt.scatter(grid[:, 0], grid[:, 1])
    #     plt.savefig('interpolated_coords.png')
    #     return interpolated_images, grid
    
    
    # def get_crop(self, images: torch.Tensor, coords: torch.Tensor, percentage: float) -> tuple:
    #     # Get unique x and y coordinates
    #     unique_x = coords[:, 0].unique().sort()[0]
    #     unique_y = coords[:, 1].unique().sort()[0]
        
    #     # Determine the crop size to maintain the same number of unique x and y coordinates
    #     crop_size = int(percentage * min(unique_x.size(0), unique_y.size(0)))
    #     print(crop_size)
    #     # get randomized center crop range for x and y coordinates
    #     random_pad_x = torch.randint(0, unique_x.size(0) - crop_size, (1,)).item()
    #     random_pad_y = torch.randint(0, unique_y.size(0) - crop_size, (1,)).item()

    #     center_x_start = unique_x[random_pad_x]
    #     center_x_end = unique_x[random_pad_x + crop_size]
    #     center_y_start = unique_y[random_pad_y]
    #     center_y_end = unique_y[random_pad_y + crop_size]

    #     # Filter the coordinates to get the center crop
    #     indices = (coords[:, 0] >= center_x_start) & (coords[:, 0] <= center_x_end) & (coords[:, 1] >= center_y_start) & (coords[:, 1] <= center_y_end)
    #     cropped_coords = coords[indices]
    #     cropped_images = images[indices]
    #     print(cropped_coords.size())
    #     print(cropped_images.size())
    #     #plot cropped coords

    #     #sort the cropped coords by x
        
    #     # Randomly sample to fill missing points from the overall coordinates
    #     num_missing = crop_size * crop_size - cropped_images.size(0)
    #     if num_missing > 0:
    #         overall_sampled_indices = torch.randint(0, images.size(0), (num_missing,))
    #         sampled_images = images[overall_sampled_indices]
    #         sampled_coords = coords[overall_sampled_indices]
            
    #         # Concatenate sampled images and coordinates to the cropped ones
    #         cropped_images = torch.cat([cropped_images, sampled_images], dim=0)
    #         cropped_coords = torch.cat([cropped_coords, sampled_coords], dim=0)
        
    #     # Randomly sample to fill missing points from the cropped coordinates
        
    #     num_missing = crop_size * crop_size - cropped_images.size(0)
    #     if num_missing > 0:
    #         sampled_indices = torch.randint(0, cropped_images.size(0), (num_missing,))
    #         sampled_images = cropped_images[sampled_indices]
    #         sampled_coords = cropped_coords[sampled_indices]
            
    #         # Concatenate sampled images and coordinates to the cropped ones
    #         cropped_images = torch.cat([cropped_images, sampled_images], dim=0)
    #         cropped_coords = torch.cat([cropped_coords, sampled_coords], dim=0)
        
    #     print(cropped_coords.size())
    #     print(cropped_images.size())
    #     #plot cropped coords
    #     plt.scatter(cropped_coords[:, 0], cropped_coords[:, 1])
    #     plt.savefig('interpolated_coords.png')
        
    
    #     return cropped_images, cropped_coords
    
    
    # def get_crop(self, images: torch.Tensor, coords: torch.Tensor, percentage: float) -> tuple:
    #     # Get unique x and y coordinates
    #     unique_x = coords[:, 0].unique().sort()[0]
    #     unique_y = coords[:, 1].unique().sort()[0]
        
    #     # Determine the crop size to maintain the same number of unique x and y coordinates
    #     crop_size = int(percentage * min(unique_x.size(0), unique_y.size(0)))
    #     print(crop_size)
    #     # get randomized center crop range for x and y coordinates
    #     #x_pad = (unique_x.size(0) - crop_size) // 2
    #     #y_pad = (unique_y.size(0) - crop_size) // 2
    #     random_pad_x = torch.randint(0, unique_x.size(0) - crop_size, (1,)).item()
    #     random_pad_y = torch.randint(0, unique_y.size(0) - crop_size, (1,)).item()

    #     center_x_start = unique_x[random_pad_x]
    #     center_x_end = unique_x[random_pad_x + crop_size]
    #     center_y_start = unique_y[random_pad_y]
    #     center_y_end = unique_y[random_pad_y + crop_size]

    #     # Filter the coordinates to get the center crop
    #     indices = (coords[:, 0] >= center_x_start) & (coords[:, 0] <= center_x_end) & (coords[:, 1] >= center_y_start) & (coords[:, 1] <= center_y_end)
    #     cropped_coords = coords[indices]
    #     cropped_images = images[indices]
    #     print(cropped_coords.size())
    #     print(cropped_images.size())
    #     #plot cropped coords

    #     #sort the cropped coords by x


    #     # Create a grid for interpolation
    #     grid_x = torch.linspace(center_x_start, center_x_end, crop_size)
    #     grid_y = torch.linspace(center_y_start, center_y_end, crop_size)
    #     grid_xx, grid_yy = torch.meshgrid(grid_x, grid_y)
    #     grid = torch.stack([grid_xx.flatten(), grid_yy.flatten()], dim=1)  # (crop_size*crop_size, 2)

    #     # Interpolate the cropped_images
    #     interpolated_images = griddata(cropped_coords.numpy(), cropped_images.numpy(), grid.numpy(), fill_value=0, method='cubic')     

    #     interpolated_images = torch.tensor(interpolated_images, dtype=images.dtype)  # Convert back to tensor

    #     print(interpolated_images)
    #     print(grid.size())
    #     print(interpolated_images.size())

    #     #plot interpolated coords
    #     plt.scatter(grid[:, 0], grid[:, 1])
    #     plt.savefig('interpolated_coords.png')
    #     return interpolated_images, grid
    
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

        images_list = []
        coords_list = []
        #plot coords
        '''
        plt.clf()
    
        plt.scatter(coords[:, 0], coords[:, 1])
        plt.savefig('coords1.png')
        '''
        # # Randomly select self.max_tiles patches from the images and coords
        # if images.size(0) > self.max_tiles:
        #     indices = torch.randperm(images.size(0))[:self.max_tiles]
        #     images = images[indices]
        #     coords = coords[indices]
        
        # Get the global crop
        for i in range(2):
            global_images, global_coords = self.get_crop(images, coords, self.global_crops_scale)
            if global_images.size(0) > self.max_tiles:
                global_images, global_coords = self.reduce_images(global_images, global_coords, self.max_tiles)
            if self.shuffle_tiles:
                global_images, global_coords = self.shuffle_data(global_images, global_coords)
            images_list.append(global_images)
            coords_list.append(global_coords)
        
        #get local crops
        max_local_tiles = int(self.max_tiles*self.local_crops_scale*self.local_crops_scale)
        for i in range(self.local_crops_number):
            local_images, local_coords = self.get_crop(images, coords, self.local_crops_scale)
            #compare with self.max_tiles to *self.local_crops_scale*self.local_crops_scale
            if local_images.size(0) > max_local_tiles:
                local_images, local_coords = self.reduce_images(local_images, local_coords, max_local_tiles)
            if self.shuffle_tiles:
                local_images, local_coords = self.shuffle_data(local_images, local_coords)
            images_list.append(local_images)
            coords_list.append(local_coords)

        return images_list, coords_list

    def get_images_from_path(self, img_path: str) -> dict:
        '''Get the images from the path'''

        assets, _ = self.read_assets_from_h5(img_path)
        origin_images = torch.from_numpy(assets['features'])
        origin_coords = torch.from_numpy(assets['coords'])
        images, coords = self.multi_crop(origin_images, origin_coords)

        
        # set the input dict
        data_dict = {'imgs': images,
                'coords': coords}
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
        sample = {'imgs': data_dict['imgs'],
                  'coords': data_dict['coords'],
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
    
    dataset = SlideDataset(root_path = data_path, exlude_data_path = './LUAD-5-gene_TCGA.csv', shuffle_tiles=False, max_tiles=8192, global_crops_scale=1, local_crops_scale=0.5, local_crops_number=1)
 
    sample = dataset.get_sample_with_try(99)

 
    print(len(sample['imgs']))
    print(len(sample['coords']))
    
    # check every sample in the dataset, using tdqm to show the progress
    # import tqdm
    # for i in tqdm.tqdm(range(len(dataset))):
    #     dataset.get_sample_with_try(i)
    #     '''try:
    #         dataset.get_sample_with_try(i)
    #     except:
    #         print('Error in getting the sample_{}'.format(i))
    #         continue'''
    

    
        