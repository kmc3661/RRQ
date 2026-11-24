# Data roots

`calibration128.jsonl` provides the exact 128 caption examples. Resolve
`source_image` relative to your own data directory. Do not copy paths from
another machine. Images are obtained separately from COCO.

`cost50_<model>.json` preserves the exact task document indices, prompts,
processed sizes/grid metadata, and image/input checksums. Obtain AI2D test and
NoCaps validation from their official benchmark sources (the lmms-lab mirrors
are compatible with the dataset revisions used in the experiments). Check
pixel and input-ID hashes instead of assuming a shuffled dataset is identical.
These metadata files do not contain model tensors, pixel data or cache paths.
