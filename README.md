# astronomical_pipeline
completed python based astronomical pipeline

# Background set

conda
---
create env
```bash
conda create -n astro python=3.13 --platform osx-64 # for mac

conda create -n astro python=3.13 --platform linux-64 # for linux
```

conda-forge set
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

install software
```bash
conda install conda-forge::astromatic-<software> # for astromatic softwares, like swarp, scamp, source-extractor, psfex

conda install conda-forge::astrometry # astrometry.ent
```

python pkg
---
astropy, astroquery, photutils, scipy, scikit-image, ray(for multi-processing) etc.