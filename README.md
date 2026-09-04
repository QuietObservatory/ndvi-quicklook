# NDVI Quicklook

A lightweight Streamlit + Leaflet application for displaying and comparing Sentinel-2 NDVI over a user-drawn area of interest.

## Current MVP

The user can:

1. Navigate anywhere in the world over a satellite imagery basemap.
2. Draw a rectangular area of interest.
3. Calculate Sentinel-2 NDVI for:
   - latest sufficiently clear image over the selected AOI
   - last month
   - last 4 months
   - last year
   - last 5 years
4. Switch the NDVI map between those periods.
5. Compare:
   - cloud-free NDVI using Sentinel-2 SCL
   - unfiltered NDVI
6. Download the selected cloud-free NDVI map as PNG or GeoTIFF.
7. Keep the table of mean NDVI values for all periods.

## Data

### NDVI source
Microsoft Planetary Computer — Sentinel-2 Level-2A.

NDVI is calculated from:
- B04 (red)
- B08 (near infrared)

### Cloud filtering
The cloud-free product uses Sentinel-2 Scene Classification Layer (SCL).

Kept SCL classes:
- 4 vegetation
- 5 bare soil
- 6 water
- 7 unclassified

Excluded:
- no data
- saturated/defective pixels
- dark pixels
- cloud shadows
- medium/high probability clouds
- cirrus
- snow/ice

### Satellite basemap
Esri World Imagery is used only as a lightweight visual context layer. Its imagery date varies geographically and is not assumed to match the Sentinel-2 acquisition date.

## Performance choices

The MVP limits the AOI to 25 km².

To keep processing practical:
- latest = most recent acquisition with at least 70% AOI pixels retained by SCL; if none qualifies, the clearest recent candidate is used
- 1 month = all available acquisitions
- 4 months = all available acquisitions
- 1 year = least-cloudy acquisition per ~10-day block
- 5 years = least-cloudy acquisition per calendar month

Each selected scene is reprojected onto the same geographic grid before temporal averaging. The display grid is approximately 10 m where possible, capped at 600 pixels on its longest side.

## Run locally

Create and activate a clean virtual environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy

### Streamlit Community Cloud
Push the repository to GitHub and deploy `app.py`.

### Hugging Face Spaces
Use the included Dockerfile in a Docker Space.

## Important note

The NDVI overlay is derived from Sentinel-2. The Esri satellite basemap is a contextual reference only, and its acquisition date can differ from the NDVI period.
