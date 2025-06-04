"""Professional plotting styles for TBExciton90."""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np

# Define color schemes
COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#E63946',    # Vibrant red
    'accent': '#F77F00',       # Orange
    'dark': '#264653',         # Dark blue-gray
    'light': '#A8DADC',        # Light blue
    'success': '#06D6A0',      # Teal
    'warning': '#F4A261',      # Sandy brown
    'purple': '#7209B7',       # Purple
    'pink': '#F72585',         # Pink
}

# Scientific color palettes
PALETTES = {
    'default': ['#2E86AB', '#E63946', '#F77F00', '#06D6A0', '#7209B7'],
    'cool': ['#03045E', '#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8'],
    'warm': ['#FFBA08', '#FAA307', '#F48C06', '#E85D04', '#DC2F02'],
    'diverging': ['#0077BB', '#33BBEE', '#BBBBBB', '#EE7733', '#CC3311'],
    'categorical': ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', 
                   '#AA3377', '#BBBBBB', '#000000'],
}


def set_publication_style():
    """Set publication-quality plotting style."""
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        # Fallback for older matplotlib versions
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            # Use default if seaborn styles not available
            plt.style.use('default')
    
    # Update rcParams for publication quality
    params = {
        # Figure
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Fonts
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        
        # Lines
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'lines.markeredgewidth': 1.5,
        
        # Axes
        'axes.linewidth': 1.5,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.prop_cycle': plt.cycler('color', PALETTES['default']),
        
        # Ticks
        'xtick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.size': 6,
        'ytick.major.width': 1.5,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Grid
        'grid.color': '#B0B0B0',
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#CCCCCC',
        'legend.borderpad': 0.5,
        'legend.columnspacing': 1.0,
        'legend.handlelength': 2.0,
    }
    
    plt.rcParams.update(params)


def get_gradient_colormap(color1='#2E86AB', color2='#E63946', n_colors=256):
    """Create a gradient colormap between two colors."""
    from matplotlib.colors import LinearSegmentedColormap
    
    colors = [color1, color2]
    n_bins = n_colors
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return cmap


def add_colorbar(fig, ax, mappable, label='', orientation='vertical', 
                 size='5%', pad=0.1, **kwargs):
    """Add a colorbar to the plot."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    divider = make_axes_locatable(ax)
    if orientation == 'vertical':
        cax = divider.append_axes("right", size=size, pad=pad)
    else:
        cax = divider.append_axes("top", size=size, pad=pad)
    
    cbar = fig.colorbar(mappable, cax=cax, orientation=orientation, **kwargs)
    cbar.set_label(label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Make colorbar outline thinner
    cbar.outline.set_linewidth(1)
    
    return cbar


def add_text_box(ax, text, loc='upper left', **kwargs):
    """Add a styled text box to the plot."""
    default_props = dict(
        boxstyle='round,pad=0.5',
        facecolor='white',
        edgecolor='gray',
        alpha=0.9,
        linewidth=1.5
    )
    default_props.update(kwargs)
    
    # Convert location string to coordinates
    loc_dict = {
        'upper left': (0.05, 0.95),
        'upper right': (0.95, 0.95),
        'lower left': (0.05, 0.05),
        'lower right': (0.95, 0.05),
        'center': (0.5, 0.5),
    }
    
    x, y = loc_dict.get(loc, loc)
    ha = 'left' if 'left' in loc else ('right' if 'right' in loc else 'center')
    va = 'top' if 'upper' in loc else ('bottom' if 'lower' in loc else 'center')
    
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=12, ha=ha, va=va,
            bbox=default_props)


def plot_with_error_band(ax, x, y, yerr, color=None, label=None, alpha=0.3):
    """Plot line with error band."""
    if color is None:
        color = COLORS['primary']
    
    line = ax.plot(x, y, color=color, label=label, linewidth=2)[0]
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha)
    
    return line


def create_figure_grid(nrows=2, ncols=2, figsize=None, **kwargs):
    """Create a figure with subplots using GridSpec for better control."""
    from matplotlib.gridspec import GridSpec
    
    if figsize is None:
        figsize = (8 * ncols, 6 * nrows)
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows, ncols, figure=fig, **kwargs)
    
    axes = []
    for i in range(nrows):
        row_axes = []
        for j in range(ncols):
            ax = fig.add_subplot(gs[i, j])
            row_axes.append(ax)
        axes.append(row_axes)
    
    return fig, axes


def save_figure(fig, filename, dpi=300, transparent=False, **kwargs):
    """Save figure with consistent settings."""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                transparent=transparent, **kwargs)


# Initialize style on import
set_publication_style()