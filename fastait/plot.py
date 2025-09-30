import torch
import matplotlib.pyplot as plt
import ipywidgets
from IPython.display import display

def show(images: torch.Tensor, **kwargs):
    """
    Display images interactively with a slider to navigate through them.
    Parameters:
        images (torch.Tensor): A 2D or 3D tensor of images. If 2D, treated as a single image. 
            If 3D, shape should be (N, H, W).
        **kwargs: Additional keyword arguments for plt.imshow().
    """
    if images.ndim == 2:
        images = images.unsqueeze(0)
    elif images.ndim != 3:
        raise ValueError("Input images tensor must be 2D or 3D (N, H, W)")
    
    if torch.is_tensor(images):
        images = images.cpu().numpy()

    plt.ioff()
    
    slider = ipywidgets.IntSlider(orientation='horizontal', value=0, min=0, max=images.shape[0]-1)
    slider.layout.margin = '0px 0% 0px 0%'
    slider.layout.width = '100%'
    
    fig, ax = plt.subplots()
    fig.canvas.header_visible = False
    
    img_display = plt.imshow(images[0], interpolation='none', **kwargs)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis('off')
    
    def update_image(index):
        img_display.set_data(images[index.new])
        img_display.set_clim(vmin=images[index.new].min(), vmax=images[index.new].max())
        fig.canvas.draw_idle()
    
    slider.observe(update_image, names='value')
    footer = slider if images.shape[0] > 1 else None
    
    app = ipywidgets.AppLayout(center=fig.canvas, footer=footer, pane_heights=[0, 10, 1])
    display(app)

def plot_image_grid(images, num_images=18, images_per_row=6, fig_width=20, row_height=3, 
                    start_index=0, step=10, **kwargs):
    """
    Plot a grid of images.

    Parameters:
        images (tensor or array): Collection of images (assumed indexable like images[i]).
        num_images (int): Number of images to display.
        images_per_row (int): Number of images per row in the grid.
        fig_width (float): Figure width in inches.
        row_height (float): Height per row in inches.
        start_index (int): Starting index offset for the images.
        step (int): Step size between selected images.
        **kwargs: Additional keyword arguments for plt.imshow().
    """
    if torch.is_tensor(images):
        images = images.cpu().numpy()
        
    num_images = min(num_images, len(images) - start_index)
    if num_images <= 0:
        raise ValueError("num_images must be positive and within the range of available images.")

    # Compute number of rows
    nrows = (num_images + images_per_row - 1) // images_per_row
    
    # Create figure and axes
    fig, axes = plt.subplots(nrows, images_per_row, figsize=(fig_width, row_height * nrows))
    fig.canvas.header_visible = False     
    
    axes = axes.flatten()  # flatten axes for easier iteration
    for i in range(num_images):
        ax = axes[i]
        index = i*step + start_index
        image = images[index]
        ax.imshow(image, **kwargs)
        ax.set_title(f'Image {index}')
        ax.axis('off')
    
    # Hide unused axes
    for j in range(num_images, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()