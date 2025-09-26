from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class GridDefinition():
    crs: str    
    extent: list
    grid_cell_size: list
    rows: int
    cols: int
    name: str = None
    description: str = None
    projection: str = None


class Grid(GridDefinition):
    """Grid class that inherits directly from GridDefinition"""
    
    @classmethod
    def from_predefined(cls, grid_id: str, filepath: str = None):
        if filepath is None:
            # Find the project root (where configs/ folder is located)
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent  # Go up from src/shared/utils/ to project root
            filepath = project_root / "configs" / "grids.yaml"
        
        with open(filepath, 'r') as file:
            config = yaml.safe_load(file)

        if grid_id not in config:
            raise ValueError(f"Grid ID '{grid_id}' not found in configuration file.")

        grid_config = config[grid_id]            
        return cls(**grid_config)
    

    def _get_essential_properties(self):
        return {
            'crs': self.crs,
            'extent': tuple(self.extent),  # Use tuple for hashability
            'grid_cell_size': tuple(self.grid_cell_size),
            'rows': self.rows,
            'cols': self.cols
        }
    

    def is_compatible(self, other):
        if not isinstance(other, Grid):
            return False
        
        return self._get_essential_properties() == other._get_essential_properties()
    

    def __eq__(self, other):
        """Grid equality based on essential properties only"""
        return self.is_compatible(other)
    
    
    def create_grid(self, coords_type='xy'):
        import numpy as np
        import xarray as xr

        if coords_type == 'latlon':
            dims = ("lon", "lat")
        elif coords_type == 'xy':
            dims = ("x", "y")
        else:
            raise ValueError("coords_type must be 'xy' or 'latlon'")

        x_min, y_min, x_max, y_max = self.extent
        cell_size_x, cell_size_y = self.grid_cell_size
        
        x_coords = np.arange(x_min + cell_size_x / 2, x_max, cell_size_x)
        y_coords = np.arange(y_max - cell_size_y / 2, y_min, -cell_size_y)

        if len(x_coords) != self.cols or len(y_coords) != self.rows:
            raise ValueError("Calculated grid dimensions do not match specified rows and cols.")

        grid_ds = xr.Dataset(
            coords={
                "x": (dims[0], x_coords),
                "y": (dims[1], y_coords)
            }
        )

        return grid_ds
