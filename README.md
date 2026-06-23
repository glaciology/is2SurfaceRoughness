# is2SurfaceRoughness
Derive surface roughness from ICESat-2. Then, analyze spatiotemporal patterns!

What is 'surface roughness' you ask? Here, I define surface roughness as the sub-topographic-scale surface morphology that includes crevassing, sastrugi, snow dunes, ice hummocks, supraglacial streams, etc. These features are created by ice dynamics, surface hydrology, and snow deposition and erosion. These scale features are of interest because they reveal how the surface responds to changes in climate forcings (plus ice dynamics, turbulent fluxes, bedform processes, etc.). Quantitatively, surface roughness is simply the RMS or standard deviation of the linearly-detrended 2D surface profile over a certain length scale. 

A better understanding of 'surface roughness' can improve energy balance models, dry snow deposition processes, reveal ice dynamic changes, and altimetry-derived surface roughness can also be used to calibrate, assess, and help model the noise characteristics of sub-surface-reflecting altimeters (radar). 

These scripts are to accompany a manuscript submitted to GRL. 

There are two pipelines: 
1) Derive surface roughness from ATL03 geolocated photons from ICESat-2 (/get_roughness_data)
  - Here, we use cloud-based SlideRule ([https://client.slideruleearth.io/landing](https://docs.slideruleearth.io)) to rapidly analyze all of Greenland (40+ TB of data). Roughness estimates are returned at 200 m postings across all ICESat-2 ground tracks, over the entire satellite mission duration. 
2) Analyze data (/analyze_roughness_data): given the spatio-temporal sampling properties of a near-polar orbiting satellite, geospatial statistical techniques are used to derive continuous maps of surface roughness evolution through time. This manuscript (and scripts within this directory) derive 3 main 'maps': (A) median surface roughness over all of Greenland (B) seasonal amplitude of roughness signal (C) temporal trends of roughness signal.

<img width="916" height="645" alt="Screenshot 2026-06-23 at 06 29 19" src="https://github.com/user-attachments/assets/c3662498-9712-4b2b-909d-ce97dcd98eb0" />

These scripts are built in Python. The derivation of roughness pipeline relies on the prior work of van Tiggelen et al. 2021 (https://doi.org/10.5194/tc-15-2601-2021) for the sliding confidence interval photon selection and photon filtering. We also leverage the incredible SlideRule cloud API (https://docs.slideruleearth.io). AI-assisted development tools were used during implementation,
including code suggestions, debugging support, and documentation, and validated by me, the developer. 
