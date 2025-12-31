from pydantic import BaseModel
import toml
import os

class Config(BaseModel):
  dispersion       : float
  effective_area   : float
  baud_rate        : float
  fiber_length     : float
  n_modes          : int
  n_channels       : int
  launch_power     : float
  raman_gain       : float
  channel_spacing  : float
  center_frequency : float
  store            : bool
  pulse_shape      : int
  collision_margin : int
  n_pumps          : int 

class NumericalConfig(BaseModel):
   gvd             : float
   dgd1            : float
   dgd2_g          : float
   dgd2_n          : float
   n_samples_numeric_g : int
   n_samples_numeric_n : int

# Deserialize TOML file into a Pydantic model
def load_toml_to_struct(filepath: str) -> Config:
    """Load simulation configuration from a TOML file into a Config object."""
    data = toml.load(filepath)
    return Config(**data)

def load_nc_toml_to_struct(filepath: str) -> NumericalConfig:
    """Load numerical configuration overrides from TOML into a NumericalConfig object."""
    data = toml.load(filepath)
    return NumericalConfig(**data)

# Serialize a Pydantic model into a TOML file
def save_struct_to_toml(filepath: str, config: Config):
    with open(filepath, "w") as f:
        toml.dump(config.model_dump(), f)
        

def get_next_filename(
  base_name, 
  extension, 
  use_active_naming=True):
    """
    Generate a unique file name by appending an incrementing numeral
    if a file with the same name already exists.

    Args:
        base_name (str): The base name of the file (without extension).
        extension (str): The file extension (with or without a dot).

    Returns:
        str: A unique file name.
    """
    if use_active_naming:
      if not extension.startswith('.'):
          extension = '.' + extension
      
      filename = f"{base_name}{extension}"
      counter = 1

      # Check if the file already exists and increment until it's unique
      while os.path.exists(filename):
          filename = f"{base_name}_{counter}{extension}"
          counter += 1
    else:
      if not extension.startswith('.'):
          extension = '.' + extension
      filename = f"{base_name}{extension}"
    return filename
