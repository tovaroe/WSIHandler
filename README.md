# WSIHandler
Class for WSI functionality
Implements the reading of WSI (ndpi or svs), their respective annotations, can create tissue masks and output random image tiles / ordered image tiles via a generator.

# Dependencies
openslide python 1.2.0 : https://github.com/openslide/openslide-python
scipy 1.6.2
pillow 7.2.0
scikit-image 0.18.1
numpy 1.20.2

# Usage
First, initalize WSIHandler and read a WSI

`wsi = WSIHandler("path/to/wsi.ndpi")`


Read annotations

`wsi.read_annotation()`

Obtain a tissue_mask, i.e., where image tiles will be taken from. By default, all annotations will be EXCLUDED. If you want to only include annotations specify annotation_handling="include" and make sure that ndpa-annotations are in black. You may also choose to ignore annotations by specifying annotation_handling="ignore".

`wsi.obtain_tissue_mask()`

Get any random tile from within the tissue_mask

`wsi.get_random_tile()`

Make a generator to output all tiles from within the tissue_mask

`tile_generator = wsi.tile_generator()`
