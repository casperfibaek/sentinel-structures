import os
import numpy as np
from glob import glob
from buteo.machine_learning.patch_extraction import extract_patches

folder = "path/to/data/"
outdir = folder + "patches/raw/"
tmpdir = folder + "tmp/"

regions = glob(folder + "vector/regions/region*")
test_areas = [
    folder + "vector/test_areas/aarhus_area.gpkg",
    folder + "vector/test_areas/holsterbro_area.gpkg",
    folder + "vector/test_areas/samsoe_area.gpkg",
]

m10 = [
    folder + "raster/B02.tif",
    folder + "raster/B03.tif",
    folder + "raster/B04.tif",
    folder + "raster/B08.tif",
    folder + "raster/VV_asc.tif",
    folder + "raster/VV_desc.tif",
    folder + "raster/VH_asc_v2.tif",
    folder + "raster/VV_asc_v2.tif",
    folder + "raster/COH_asc.tif",
    folder + "raster/COH_desc.tif",
    folder + "raster/label_area.tif",
    folder + "raster/label_volume.tif",
    folder + "raster/label_people.tif",
]

m20 = [
    folder + "raster/B05.tif",
    folder + "raster/B06.tif",
    folder + "raster/B07.tif",
    folder + "raster/B8A.tif",
    folder + "raster/B11.tif",
    folder + "raster/B12.tif",
]

patch_size = 64

half = patch_size // 2
quart = half // 2
eighth = quart // 2

m10_patches = [(half, half)]
m20_patches = [(quart, quart)]

for idx, region in enumerate(regions):
    print(f"Processing region: {region}")

    extract_patches(
        m10,
        out_dir=tmpdir,
        prefix=f"{idx}_",
        postfix="",
        size=patch_size,
        offsets=m10_patches,
        generate_grid_geom=True,
        generate_zero_offset=True,
        generate_border_patches=True,
        clip_geom=region,
        verify_output=False,
        verification_samples=100,
        verbose=1,
    )

    extract_patches(
        m20,
        out_dir=tmpdir,
        prefix=f"{idx}_",
        postfix="",
        size=half,
        offsets=m20_patches,
        generate_grid_geom=True,
        generate_zero_offset=True,
        generate_border_patches=True,
        clip_geom=region,
        verify_output=False,
        verification_samples=100,
        verbose=1,
    )

for band in m10 + m20:
    name = os.path.splitext(os.path.basename(band))[0] + ".npy"
    tiles = glob(tmpdir + f"*{name}")

    for idx, arr in enumerate(tiles):
        if idx == 0:
            base = np.load(arr)
        else:
            base = np.concatenate([base, np.load(arr)])

    np.save(outdir + name, base)


for idx, region in enumerate(test_areas):
    print(f"Processing region: {region}")

    name = os.path.splitext(os.path.basename(region))[0].split("_")[0]

    extract_patches(
        m10,
        out_dir=tmpdir,
        prefix=f"{name}_",
        postfix="",
        size=patch_size,
        offsets=[
            (quart, quart),
            (half, half),
            (half + quart, half + quart),
        ],
        generate_grid_geom=True,
        generate_zero_offset=True,
        generate_border_patches=True,
        clip_geom=region,
        verify_output=False,
        verification_samples=100,
        verbose=1,
    )

    extract_patches(
        m20,
        out_dir=tmpdir,
        prefix=f"{name}_",
        postfix="",
        size=half,
        offsets=[
            (eighth, eighth),
            (quart, quart),
            (quart + eighth, quart + eighth),
        ],
        generate_grid_geom=True,
        generate_zero_offset=True,
        generate_border_patches=True,
        clip_geom=region,
        verify_output=False,
        verification_samples=100,
        verbose=1,
    )

    for band in m10 + m20:
        bandname = os.path.splitext(os.path.basename(band))[0] + ".npy"
        tiles = glob(tmpdir + f"*{name}_{bandname}")

        for idx, arr in enumerate(tiles):
            if idx == 0:
                base = np.load(arr)
            else:
                base = np.concatenate([base, np.load(arr)])

        np.save(outdir + f"{name}_{bandname}", base)
