from __future__ import annotations

from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import math
from pathlib import Path
from IPython.display import  display, HTML
from base64 import b64encode
import os
from concurrent.futures import ThreadPoolExecutor

class ImageDefaults:
    def __init__(self):
        self.device = "cpu"
        
defaults = ImageDefaults()

class VideoWrapper:
    def __init__(self, video_path, video_size):
        self.video_path = video_path
        self.video_size = video_size
        
    def path(self):
        return self.video_path
    
    def show(self):
        mp4 = open(self.video_path, 'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        
        width, height = self.video_size
        
        return HTML(f"""
        <video width={width} height={height} controls>
              <source src="%s" type="video/mp4">
        </video>
        """ % data_url)

class ImageWrapper:
    def __init__(self, data, image_type):
        self.data = data
        self.image_type = image_type
        
    def resize(self, size=(256, 256)) -> ImageWrapper:
        if self.image_type != "pil":
            raise Exception("resize() only applied for pil images")
            
        if isinstance(size, int):
            size = (size, size)
            
        new_images = [im.resize(size) for im in self.data]
        return ImageWrapper(new_images, "pil")
    
    def normalize(self) -> ImageWrapper:
        if self.image_type != "pt":
            raise Exception("normalize() only applied for pytorch tensors")
        
        normalized = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        return ImageWrapper(normalized, "pt")
    
    def sinrange(self) -> ImageWrapper:
        if self.image_type != "pt":
            raise Exception("sinrange() only applied for pytorch tensors")
        
        return ImageWrapper(self.data * 2 - 1, "pt")
        
    def pil(self) -> Image:
        if self.image_type == "pil":
            return self.data[0] if len(self.data) == 1 else self.data
        
        if self.image_type == "pt":
            make_pil = transforms.ToPILImage()
            pt_images = self.data.cpu()
            pil_images = [make_pil(i) for i in pt_images]
            return pil_images[0] if len(pil_images) == 1 else pil_images
    
    def pt(self) -> torch.Tensor:            
        if self.image_type == "pil":
            pt_images = [transforms.ToTensor()(im) for im in self.data]
            return torch.stack(pt_images).to(defaults.device)
        
        if self.image_type == "pt":
            return self.data
        
    def to(self, device="cpu") -> ImageWrapper:
        if self.image_type != "pt":
            raise Exception("to() only applied for pytorch tensors")
        
        return ImageWrapper(self.data.to(device), "pt")
    
    def cpil(self) -> ImageWrapper:
        images = self.pil()
        if isinstance(images, Image.Image):
            images = [images]
            
        return ImageWrapper(images, "pil")
    
    def cpt(self) -> ImageWrapper:
        return ImageWrapper(self.pt(), "pt")
    
    def show(self, cmap=None, figsize=None, cols=6, max_count=36, scale=2.5, captions=True):        
        if len(self.data) == 1:
            plt.axis("off")
            if self.image_type == "pil":
                plt.imshow(self.data[0], cmap=cmap)
            else:
                plt.imshow(self.data[0].permute(1, 2, 0).cpu(), cmap=cmap)
                
            return
        
        images = self.data.cpu() if self.image_type == "pt" else self.data
        image_count = len(self.data)
        
        if image_count > max_count:
            images = self.data[0:max_count]
            print(f"found {image_count} images to show. But only showing {max_count}")
            
        if image_count < cols:
            cols = image_count
            
        rows = math.ceil(image_count / cols)
        
        if figsize == None:
            figsize = figsize=(cols*scale, rows*scale)
            
        _, ax = plt.subplots(rows, cols, figsize=figsize)
        if (rows == 1):
            for i in range(image_count):
                image = images[i] if self.image_type == "pil" else images[i].permute(1, 2, 0)
                ax[i].imshow(image)
                ax[i].axis("off")
                if captions: ax[i].set_title(f"{i}")
        else:
            for row in range(rows):
                for col in range(cols):
                    i = row * cols + col
                    if i < image_count:
                        image = images[i] if self.image_type == "pil" else images[i].permute(1, 2, 0)
                        ax[row][col].imshow(image)
                        ax[row][col].axis("off")
                        if captions: ax[row][col].set_title(f"{i}")
                    else:
                        ax[row][col].axis("off")
                        
    def to_dir(self, output_dir, prefix="image", max_workers=min(10, os.cpu_count())):
        if self.image_type != "pil":
            raise Exception("to_dir() only applied for pil images")
            
        dir_path = Path(output_dir)
        dir_path.mkdir(exist_ok=True, parents=True)
        
        images = self.data

        def save_image(i):
            try:
                path = Path(output_dir)/f"{prefix}_{i:04}.png"
                images[i].save(path)
            except Exception as e:
                print("image saving error:", e)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return executor.map(save_image, range(len(images)), timeout=60)
            
    def to_video(self, out_path=None, frame_rate=12):
        if self.image_type != "pil":
            raise Exception("to_video() only applied for pil images")
            
        id = int(torch.rand(1)[0].item() * 9999999)
        image_dir = Path(f'/tmp/{id}/images')
        image_dir.mkdir(exist_ok=True, parents=True)

        if out_path == None:
            out_path = f"/tmp/{id}/video.mp4"

        video_path = Path(out_path)
        video_size = self.data[0].size
        images_selector = image_dir/"image_%04d.png"
        
        self.to_dir(image_dir, prefix="image")

        command = f"ffmpeg -v 0 -y -f image2 -framerate {frame_rate} -i {images_selector} -c:v h264_nvenc -preset slow -qp 18 -pix_fmt yuv420p {video_path}"
        os.system(command)
    
        return VideoWrapper(video_path, video_size)

def wrap(input_data) -> ImageWrapper:
    if isinstance(input_data, ImageWrapper):
        return input_data
    
    if isinstance(input_data, torch.Tensor):
        if len(input_data.shape) == 3:
            input_data = input_data.unsqueeze(0)
            
        return ImageWrapper(input_data.detach(), "pt")
    
    if isinstance(input_data, Image.Image):
        return ImageWrapper([input_data], "pil")
    
    if isinstance(input_data, list):
        if isinstance(input_data[0], torch.Tensor):
            images = torch.stack(input_data).squeeze(1).detach()
            return ImageWrapper(images, "pt")
        
        if isinstance(input_data[0], Image.Image):
            return ImageWrapper(input_data, "pil")
        
        if isinstance(input_data[0], ImageWrapper):
            image_list = list(map(lambda w: w.pt(), input_data))
            images = torch.stack(image_list).squeeze(1).detach()
            return ImageWrapper(images, "pt")
    
    raise Exception("not implemented!")                        
                        
def from_dir(dir_path) -> ImageWrapper:
    file_list = [f for f in Path(dir_path).iterdir() if not f.is_dir()]
    image_list = []

    for f in file_list:
        try:
            image_list.append(Image.open(f).convert("RGB"))
        except UnidentifiedImageError:
            None
            
    return ImageWrapper(image_list, "pil")

def from_path(input_data) -> ImageWrapper:
    pil_image = Image.open(input_data).convert("RGB")
    return ImageWrapper([pil_image], "pil")
                        

class DynaPlot:
    def __init__(self, cols=2, figsize=(15, 4)):
        fig, subplots = plt.subplots(1, cols, figsize=(20, 5))
        fig.patch.set_facecolor("white")
        fig.tight_layout()
        out = display(fig, display_id=True)

        self.cols = cols
        self.fig = fig
        self.out = out
        self.subplots = subplots
        
        self.queue = []
        
    def plot(self, subplot_id, *args, **kwargs) -> DynaPlot:
        self.queue.append((
            "plot", subplot_id, args, kwargs
        ))
        return self
    
    def title(self, subplot_id, title)-> DynaPlot:
        self.queue.append((
            "title", subplot_id, title
        ))
        return self
        
    def imshow(self, subplot_id, image)-> DynaPlot:
        self.queue.append((
            "imshow", subplot_id, image
        ))
        return self
        
    def update(self):
        for col in range(self.cols):
            if self.cols == 1:
                self.subplots.clear()
            else:
                self.subplots[col].clear()
        
        for item in self.queue:
            if item[0] == "imshow":
                _, subplot_id, image = item
                if self.cols == 1:
                    self.subplots.imshow(wrap(image).pt().detach().cpu()[0].permute(1, 2, 0))
                    self.subplots.axis("off")
                else:
                    self.subplots[subplot_id].imshow(wrap(image).pt().detach().cpu()[0].permute(1, 2, 0))
                    self.subplots[subplot_id].axis("off")
            
            if item[0] == "plot":
                _, subplot_id, args, kwargs = item
                self.subplots[subplot_id].plot(*args, **kwargs)
                if "label" in kwargs:
                    self.subplots[subplot_id].legend()
                
            if item[0] == "title":
                _, subplot_id, title = item
                self.subplots[subplot_id].title.set_text(title)
                
        self.queue = []
        self.out.update(self.fig)
        
    def close(self):
        plt.close()
    
def dplot(**kwargs) -> DynaPlot:
    return DynaPlot(**kwargs)

# class ImiTools:
#     def __init__(self):
#         self.defaults = defaults
        
#     def wrap(self, path) -> ImageWrapper:
#         return wrap(path)
                        
#     def from_path(self, path) -> ImageWrapper:
#         return from_path(path)
    
#     def from_dir(self, path) -> ImageWrapper:
#         return from_dir(path)
    
#     def dplot(self, **kwargs) -> DynaPlot:
#         return dplot(**kwargs)
    
# I = ImiTools()
# I.defaults.device = device
