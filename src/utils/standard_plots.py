# -- coding: utf-8 --
# standard_plots.py
"""Module providing standard plotting functions for scatterplots, histograms, polar maps, and binned parameter comparisons."""

# -- built-in libraries --

# -- third-party libraries  --
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#  -- custom modules  --

# %% Standard scatterplots/heatmaps ###################################################################################################################
def scatter_base(func):
    """
    General scatterplot decorator.
    Handles axis creation, scatter call, and basic styling.
    """
    def wrapper(fig, position, x, y,
                fit_kwargs = None,
                legend_kwargs = None,
                cbar_kwargs=None,
                **kwargs):

        if len(x) != len(y):
            raise ValueError("Length of x and y must be the same.")
        
        ax = fig.add_subplot(*position)

        # Core plotting from the wrapped function
        h = func(ax, x, y, **kwargs)

        # Axis limits (basic behavior, can be overridden by later decorators)
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))

        # Optional colorbar if applicable
        if cbar_kwargs is not None:
            if isinstance(h, tuple):       # hist2d -> (counts, xedges, yedges, QuadMesh)
                fig.colorbar(h[3], ax=ax, **cbar_kwargs)
            else:                          # hexbin -> PolyCollection
                fig.colorbar(h, ax=ax, **cbar_kwargs)
        if fit_kwargs is not None:
            add_line_fit(ax, x, y, **fit_kwargs)
        if legend_kwargs is not None:
            ax.legend(**legend_kwargs)

        return ax
    return wrapper


@scatter_base
def scatter(ax, obs, pred, **kwargs):
    """
    Obs vs. pred plot in scatter plot format.
    """
    return ax.scatter(obs, pred, **kwargs)


@scatter_base
def hist2d(ax, obs, pred, **kwargs):
    """
    Obs vs. pred plot in heatmap/hist2d plot format.
    """
    return ax.hist2d(obs, pred, **kwargs)

@scatter_base
def hexmap(ax, obs, pred, **kwargs):
    """
    Obs vs. pred plot in hexagonal heatmap plot format.
    """
    return ax.hexbin(obs, pred, **kwargs)

# %% obs_vs_pred scatterplots/heatmaps ################################################################################################################
def obs_vs_pred_base(func):
    """
    Adds ideal line and performance metrics to an obs-vs-pred plot.
    """
    def wrapper(fig, position, obs, pred,
                data_range = None,
                add_performance_metrics_to_legend=True,
                fit_kwargs = None,
                legend_kwargs = None,
                cbar_kwargs=None,
                ideal_line_kwargs={'color': 'r', 'ls': '--', 'label': 'Ideal Fit', 'lw': 2},
                **kwargs):

        ax = func(fig, position, obs, pred, fit_kwargs, legend_kwargs, cbar_kwargs, **kwargs)

        # Ideal line
        if ideal_line_kwargs is not None:
            ax.plot([obs.min(), obs.max()], [obs.min(), obs.max()], **ideal_line_kwargs)

        # Performance metrics
        if add_performance_metrics_to_legend:
            add_performance_metrics(ax, obs, pred)
        
        # Axis limits
        if data_range is not None:
            ax.set_xlim(data_range)
            ax.set_ylim(data_range)        
        else:
            ax.set_xlim(obs.min(), obs.max())
            ax.set_ylim(obs.min(), obs.max())

        return ax
    return wrapper


@obs_vs_pred_base
@scatter_base
def obs_vs_pred_scatter(ax, obs, pred, **kwargs):
    """
    Obs vs. pred plot in scatter plot format.
    """
    return ax.scatter(obs, pred, **kwargs)

@obs_vs_pred_base
@scatter_base
def obs_vs_pred_hist2d(ax, obs, pred, **kwargs):
    """
    Obs vs. pred plot in heatmap/hist2d plot format.
    """
    return ax.hist2d(obs, pred, **kwargs)

@obs_vs_pred_base
@scatter_base
def obs_vs_pred_hexmap(ax, obs, pred, **kwargs):
    """
    Obs vs. pred plot in hexagonal heatmap plot format.
    """
    return ax.hexbin(obs, pred, **kwargs)


# %% Histograms ################################################################################################################
def histogram_base(plot_func):
    """
    Decorator for creating histograms. Works for standard and line-histograms and supports multiple distributions 
    at once (if val is list).
    """
    def wrapper(fig, position, val,
                data_range = None,
                bins = 40,
                normalize = False,
                legend_kwargs = None,
                **plot_kwargs
                ):
        
        # creating figure
        ax = fig.add_subplot(*position)
        h = plot_func(ax, val, bins, data_range, normalize, **plot_kwargs)
        
        # fixing x extend
        if data_range is not None:
            ax.set_xlim(data_range)
        
        #applying legend
        if legend_kwargs is not None:
            ax.legend(**legend_kwargs)
        
        return ax
    return wrapper


@histogram_base
def hist_outlined(ax, val, bins, data_range, normalize, **kwargs):
    """
    Standard histogram but with only outline drawn.
    """
    if type(val) == np.ndarray:
        val = [val]
    elif type(val) != list:
        raise ValueError('Wrong input type for val')

    x = []
    y = []
    for v in val:
        # Compute histogram
        counts, bin_edges = np.histogram(v, bins=bins, range=data_range)
        if normalize:
            counts = counts.astype(float)/np.sum(counts).astype(float)
        
        # Duplicate edges for stepwise plotting
        x.append(np.repeat(bin_edges, 2)[1:-1])
        y.append(np.repeat(counts, 2))
    
    # Plot as a line
    h = ax.plot(np.stack(x,axis=1), np.stack(y,axis=1), **kwargs)
    ax.set_xlim(data_range)
    ax.set_ylim(0)
    return h


@histogram_base
def hist(ax, val, bins, data_range, normalize, **kwargs):
    """
    Standard histogram.
    """
    if normalize:
        return ax.hist(val, bins=bins, range=data_range, density=True, stacked=True, **kwargs)
    else:
        return ax.hist(val, bins=bins, range=data_range, **kwargs)


# %% Polar Ease2.0 grid plots #####################################################################################################
def polar_base(plot_func):
    """
    Decorator that sorts basic formating for creating map of the arctic (or others with modified proj and crs.)
    """
    def wrapper(fig, position, x, y, val,
                coord_crs="EPSG:6931",
                plot_proj=ccrs.NorthPolarStereo(central_longitude=0),
                land_feature = cfeature.NaturalEarthFeature(category='physical', name='land', scale='50m', facecolor='lightgray'),
                coastline_kwargs={},
                cbar_kwargs=None,
                legend_kwargs=None,
                **plot_kwargs                
                ):
        
        # creating coordinate transformer
        transformer = pyproj.Transformer.from_crs(coord_crs, "EPSG:4326", always_xy=True)

        x = x.squeeze()
        y = y.squeeze()

        # making figure
        ax = fig.add_subplot(*position, projection=plot_proj)
        h = plot_func(ax, x, y, val, transformer, **plot_kwargs)

        # Adding extra layout options
        if coastline_kwargs is not None:
            ax.coastlines(**coastline_kwargs)
        if land_feature is not None:
            ax.add_feature(land_feature)
        if cbar_kwargs is not None:
            fig.colorbar(h, ax=ax, **cbar_kwargs)
        if legend_kwargs is not None:
            ax.legend(**legend_kwargs)
        
        return ax
    return wrapper

@polar_base
def polar_pcolormesh(ax, x, y, val, transformer, **kwargs):
    """
    Mapping function for creating colormesh over the Arctic. x and y can both be 1darrays or 2d meshgrid array.
    """
    # checking and modifying x and y input vals.
    if len(x.shape) == len(y.shape) == 1 and len(val.shape) == 2:
        x, y = np.meshgrid(x,y)
    elif not len(x.shape) == len(y.shape) ==  len(val.shape) == 2:
        raise ValueError('Dimensions of x, y and val does not match.')
    
    lon, lat = transformer.transform(x,y)

    return ax.pcolormesh(lon, lat, val, transform=ccrs.PlateCarree(), **kwargs)

@polar_base
def polar_scatter(ax, x, y, val, transformer, **kwargs):
    """
    Mapping function for making scatterplots over the arctic.
    """
    if not len(x.shape) == len(y.shape) == len(val.shape) == 1:
        raise ValueError('Dimensions of x, y and val does not match.')
    
    lon, lat = transformer.transform(x,y)   
    return ax.scatter(lon, lat, c=val, transform=ccrs.PlateCarree(), **kwargs)


# %% distribution plots ###########################################################################################################
def distribution_plot_base(plot_func):
    """
    Decorator for making standard boxplots or violinplots for one or more nominal features.
    """
    def wrapper(fig, position, vals, labels=None, **kwargs):
        
        # making figure
        ax = fig.add_subplot(*position)
        h = plot_func(ax, vals, **kwargs)

        # fixing ticks
        ax.set_xticks([y + 1 for y in range(len(vals))], labels=labels)

        return ax
    return wrapper


@distribution_plot_base
def boxplot(ax, vals, **kwargs):
    """
    Standard boxplot.
    """
    return ax.boxplot(vals, **kwargs)


@distribution_plot_base
def violinplot(ax, vals, **kwargs):
    """
    Standard violinplots.
    """    
    return ax.violinplot(vals, **kwargs)


# %% Binned param-comparison plots #######################################################################################################
def bin_plot_base(plot_func):
    """
    Decorator for so called "bin plots". These bins a parameter x_param, and quantifies distribution of y_param sorted into bins.
    Standard mean plot as well as boxplot and violinplot versions exists. Per default, a histogram is applied to secondary axis.
    """
    def wrapper(fig, position, x_param, data_range, bins, y_param, 
                discrete_x = False,
                hist_kwargs = {'alpha':0.2},
                fit_kwargs = None,
                **kwargs):
        
        x_param = x_param.squeeze()
        y_param = y_param.squeeze()

        # creating handle
        ax = fig.add_subplot(*position)

        # changing inputs if discrete_x is true. Bins are void if used
        if discrete_x:
            bins = int(data_range[1] - data_range[0])
            data_range = (data_range[0]-0.5,data_range[1]+0.5)
        
        # creation of binning
        bin_edges, y_binned = _bin_y_by_x(x_param, y_param, bins, data_range)
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2

        idx_keep = np.array([i for i in range(len(y_binned)) if len(y_binned[i])!=0])
        x = bin_centers[idx_keep]
        y = [y_binned[i] for i in idx_keep]        
        # plotting
        h = plot_func(ax, x, y, **kwargs)
        
        # applying extra plot options
        if fit_kwargs is not None:
            add_line_fit(ax, x=bin_centers, y=np.array([np.mean(y) for y in y_binned]), weights=np.array([len(y) for y in y_binned]), **fit_kwargs)
        if hist_kwargs is not None:
            ax2 = ax.twinx()
            ax2.hist(x_param, range=data_range, bins=bins, **hist_kwargs)
        
        # set axis extent
        ax.set_xlim(data_range[0],data_range[1])
        
        return ax
    return wrapper

@bin_plot_base
def bin_meanplot(ax, x, y, **kwargs):
    """
    Binning and plotting means as points on figure.
    """
    means = [np.mean(v) for v in y]
    return ax.plot(x, means, **kwargs)

@bin_plot_base
def bin_boxplot(ax, x, y, **kwargs):
    """
    Binning and plotting boxplots for each bin on figure.
    """
    h =  ax.boxplot(y, positions=x, widths=np.diff(x).min()*0.8, manage_ticks=False, **kwargs)
    #ax.autoscale(enable=True, axis="x", tight=True)
    return h

@bin_plot_base
def bin_violinplot(ax, x, y, **kwargs):
    """
    Binning and plotting violinplots for each bin on figure.
    """
    h = ax.violinplot(y, positions=x, widths=np.diff(x).min()*0.8, **kwargs)
    #ax.autoscale(enable=True, axis="x", tight=True)
    return h


def _bin_y_by_x(x, y, bins, data_range):
    """
    Bin x into equal-width intervals and group corresponding y values.

    Parameters
    ----------
    x : array-like
        The variable to bin.
    y : array-like
        The values to group according to x's bins.
    bins : int, optional
        Number of bins. Default is 40.
    data_range : tuple, optional
        The (min, max) range of x to bin. If None, uses (x.min(), x.max()).

    Returns
    -------
    bin_edges : np.ndarray
        The edges of the bins (length = bins + 1).
    y_binned : list of np.ndarray
        A list where each entry contains the y values that fall into the corresponding bin.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if data_range is None:
        data_range = (np.min(x), np.max(x))
    
    # Get bin indices for x
    bin_indices = np.digitize(x, np.linspace(*data_range, bins+1)) - 1
    
    # Prepare list of arrays
    y_binned = []
    for i in range(bins):
        mask = bin_indices == i
        y_binned.append(y[mask])
    
    bin_edges = np.linspace(*data_range, bins+1)
    return bin_edges, y_binned

# %% Feature importance ###########################################################################################################
def plot_feature_importances(fig, position, regressor, input_param, **kwargs):
    """
    Plots the feature importances of a regression model.
    
    Parameters:
    regressor: The trained regression model.
    input_param: List of input feature names.
    figsize: Size of the figure for the plot.
    """

    # Picking feature importance metric based on regressor type
    if type(regressor) == RandomForestRegressor:
        feature_importance = regressor.feature_importances_
    elif type(regressor) == LinearRegression:
        feature_importance = abs(regressor.coef_[0])
    else:
        raise ValueError('Invalid regressor, or have not been implimented.')

    # plotting
    ax = fig.add_subplot(*position)
    ax.bar(input_param, feature_importance, **kwargs)
    ax.set_xlabel('Input Features')
    ax.set_ylabel('Feature Importance')

    return ax


# %% plot features ################################################################################################################
def add_performance_metrics(ax, obs, pred, **kwargs):
        """
        Function for adding RMSE and R^2 of model to figure legend.
        """
        rmse = np.sqrt(np.mean((obs-pred)**2))
        r2 = r2_score(obs, pred)
        ax.plot([],[], ' ', label= f"$RMSE= {rmse:.4f}$")
        ax.plot([],[], ' ', label= f"$R^2= {r2:.4f}$")
        ax.legend(**kwargs)       


def add_line_fit(ax, x, y, 
                 deg = 1,
                 weights = None,
                 label = None,
                 add_eq_to_legend = False,  
                 add_performance_metrics_to_legend = False, 
                 return_pred = False,
                 **kwargs):
    """
    Add polynomial fit to figure, along with various performance metrics, legend details and more.
    """
    
    if len(x) != len(y):
        raise ValueError("Length of predicted values and true values must be the same.")

    # remove nan
    missing = np.isnan(x) | np.isnan(y)
    x = x[~missing]
    y = y[~missing]
    if weights is not None:
        weights = weights[~missing]

    # make fit
    coeffs = np.polyfit(x, y, deg, w=weights)
    y_fit = np.polyval(coeffs, x)   

    # plot
    xp = np.linspace(x.min(),x.max())
    yp = np.polyval(coeffs, xp)
    ax.plot(xp, yp, lw=2, label=label, **kwargs)

    # add elements to legend
    if add_eq_to_legend:
        if deg!=1:
            raise ValueError('add_eq_to_legend currently only works for 1. order polynomials')
        ax.plot([],[], ' ', label= 'y' + r'$\sim$' + f'${coeffs[0]:.2f}$'+ 'x' + f'$ + {coeffs[1]:.2f}$')
        ax.legend()
    if add_performance_metrics_to_legend:
        add_performance_metrics(ax, y, y_fit)    

    # return fit params if requested
    if return_pred:
        return coeffs, y_fit


def create_descrete_cmap(cmap, labels):
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    
    if type(cmap)==list:
        cmap = mcolors.ListedColormap(cmap)

    elif type(cmap)==str:
        base_cmap = plt.cm.get_cmap(cmap, len(labels))
        cmap = mcolors.ListedColormap(base_cmap(np.arange(len(labels))))

    format = mticker.FixedFormatter(labels)
    ticks = list(range(len(labels)))
    boundaries = [tick-0.5 for tick in ticks] + [ticks[-1]+0.5]    

    return cmap,format,ticks,boundaries