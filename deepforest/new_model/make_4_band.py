import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from rasterio.plot import show



# Open the three-band TIFF
with rasterio.open('/home/nadja/DeepForest_new/deepforest/data/2018_BART_4_322000_4882000_image_crop.tif') as src:
    # Read the data and metadata
    three_band_data = src.read()
    profile = src.profile

# Open the single-band TIFF within its own with block
with rasterio.open('/home/nadja/DeepForest_new/deepforest/data/2018_BART_4_322000_4882000_image_crop_CHM.tif') as src:
    # Read the data and metadata
    single_band_data = src.read()
    single_band_profile = src.profile

    # Resample single band data to match the resolution of the three-band data
    resampled_single_band_data = src.read(
        out_shape=(
            src.count,
            int(src.height * (profile['transform'].a / single_band_profile['transform'].a)),
            int(src.width * (profile['transform'].e / single_band_profile['transform'].e))
        ),
        resampling=Resampling.bilinear
    )

# Create the new profile for the four-band TIFF
new_profile = profile.copy()
new_profile.update(count=4)


# Write the four-band TIFF
with rasterio.open('four_band.tif', 'w', **new_profile) as dst:
    # Write the three-band data
    for i in range(3):
        dst.write(three_band_data[i], i + 1)

    # Write the resampled single band data as the fourth band
    dst.write(resampled_single_band_data[0], 4)
# Open the four-band TIFF


with rasterio.open('four_band.tif') as src:
    # Read the first three bands (RGB)
    rgb_data = src.read([1, 2, 3])

    # Plot the RGB image
    plt.figure(figsize=(10, 10))
    show(rgb_data, transform=src.transform)

 

    # Show plot
    plt.show()

