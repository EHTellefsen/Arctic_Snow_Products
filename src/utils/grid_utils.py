# -- coding: utf-8 --
# grid_utils.py
"""Module defining Grid and GridDefinition classes for spatial grid representation."""

# -- built-in libraries --
from dataclasses import dataclass
import yaml
from pathlib import Path

# -- third-party libraries  --

#  -- custom modules  --

############################################################½
@dataclass
class GridDefinition():
    """Dataclass for defining grid properties."""
    crs: str
    coords: list    
    extent: list
    grid_cell_size: list
    rows: int
    cols: int
    name: str = None
    description: str = None
    projection: str = None
    
###########################################################
class Grid(GridDefinition):
    """Grid class that inherits directly from GridDefinition"""
    
    @classmethod
    def from_predefined(cls, grid_id: str, filepath: str = None):
        """Create Grid instance from predefined grid configuration in YAML file."""
        if filepath is None:
            # Find the project root (where configs/ folder is located)
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent  # Go up from src/shared/utils/ to project root
            filepath = project_root / "configs" / "grids.yaml"
        
        with open(filepath, 'r') as file:
            config = yaml.safe_load(file)

        if grid_id not in config:
            raise ValueError(f"Grid ID '{grid_id}' not found in configuration file.")

        grid_config = config[grid_id]            
        return cls(**grid_config)
    

    def _get_essential_properties(self):
        """Get essential properties for grid comparison."""
        return {
            'crs': self.crs,
            'coords': tuple(self.coords),
            'extent': tuple(self.extent),
            'grid_cell_size': tuple(self.grid_cell_size),
            'rows': self.rows,
            'cols': self.cols
        }
    

    def is_compatible(self, other):
        """Check if two Grid instances are compatible based on essential properties."""
        if not isinstance(other, Grid):
            return False
        
        return self._get_essential_properties() == other._get_essential_properties()
    

    def __eq__(self, other):
        """Grid equality based on essential properties only"""
        return self.is_compatible(other)
    
    
    def create_grid(self):
        """Create a grid representation (e.g., xarray Dataset) based on grid properties."""
        import numpy as np
        import xarray as xr

        x_min, y_min, x_max, y_max = self.extent
        cell_size_x, cell_size_y = self.grid_cell_size
        
        x_coords = np.arange(x_min + cell_size_x / 2, x_max, cell_size_x)
        y_coords = np.arange(y_max - cell_size_y / 2, y_min, -cell_size_y)

        if len(x_coords) != self.cols or len(y_coords) != self.rows:
            raise ValueError("Calculated grid dimensions do not match specified rows and cols.")

        grid_ds = xr.Dataset(
            coords={
                self.coords[0]: (self.coords[0], x_coords),
                self.coords[1]: (self.coords[1], y_coords)
            }
        )
        return grid_ds

    def __repr__(self):
        """String representation of the Grid instance."""
        return f"Grid(name={self.name}, crs={self.crs}, extent={self.extent}, grid_cell_size={self.grid_cell_size}, rows={self.rows}, cols={self.cols})"
    

    def modify_extent(self, new_extent):
        """Modify the grid extent and update rows and cols accordingly."""
        self.extent = new_extent
        self.rows = int((new_extent[3] - new_extent[1]) / self.grid_cell_size[1])
        self.cols = int((new_extent[2] - new_extent[0]) / self.grid_cell_size[0])


    def modify_grid_cell_size(self, new_cell_size):
        """Modify the grid cell size and update rows and cols accordingly."""
        self.grid_cell_size = new_cell_size
        self.rows = int((self.extent[3] - self.extent[1]) / new_cell_size[1])
        self.cols = int((self.extent[2] - self.extent[0]) / new_cell_size[0])