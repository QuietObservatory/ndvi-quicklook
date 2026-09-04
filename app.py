import io
import math
from datetime import datetime, timezone

import folium
import numpy as np
import pandas as pd
import planetary_computer
import pystac
import pystac_client
import rasterio
import streamlit as st
from dateutil.relativedelta import relativedelta
from folium.plugins import Draw
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from streamlit_folium import st_folium


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
MAX_AOI_KM2 = 25.0
MAX_MAP_PIXELS = 600

# Sentinel-2 Scene Classification Layer (SCL)
# Clear classes kept in the cloud-free product:
# 4 vegetation, 5 bare soil, 6 water, 7 unclassified.
CLEAR_SCL = {4, 5, 6, 7}

PERIODS = [
    ("Última imagen despejada", "latest"),
    ("Último mes", "1m"),
    ("Últimos 4 meses", "4m"),
    ("Último año", "1y"),
    ("Últimos 5 años", "5y"),
]

ESRI_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_ATTR = (
    "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, "
    "and the GIS User Community"
)

st.set_page_config(
    page_title="NDVI Quicklook",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 2rem;
        max-width: 1800px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.42rem;
    }
    .ndvi-note {
        color: #6b7280;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .ndvi-card {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_catalog():
    return pystac_client.Client.open(
        STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def area_km2(bounds):
    west, south, east, north = bounds
    lat = (south + north) / 2
    width_km = abs(east - west) * 111.32 * math.cos(math.radians(lat))
    height_km = abs(north - south) * 110.57
    return width_km * height_km


def target_grid(bounds):
    west, south, east, north = bounds
    lat = (south + north) / 2

    width_m = max(
        abs(east - west) * 111_320 * math.cos(math.radians(lat)),
        10,
    )
    height_m = max(abs(north - south) * 110_570, 10)

    width = max(32, int(round(width_m / 10)))
    height = max(32, int(round(height_m / 10)))

    scale = min(
        1.0,
        MAX_MAP_PIXELS / max(width, height),
    )
    width = max(32, int(round(width * scale)))
    height = max(32, int(round(height * scale)))

    transform = from_bounds(
        west,
        south,
        east,
        north,
        width,
        height,
    )

    return width, height, transform


def search_items(bounds, start, end):
    catalog = get_catalog()
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bounds,
        datetime=f"{start.isoformat()}/{end.isoformat()}",
    )
    items = list(search.items())
    items.sort(
        key=lambda x: x.datetime
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return items


def dedupe_same_acquisition(items):
    """
    For this small-AOI MVP, keep one Sentinel-2 granule per acquisition time,
    preferring the scene with lower scene-level cloud cover.
    """
    best = {}

    for item in items:
        dt = item.datetime
        if dt is None:
            continue

        key = dt.strftime("%Y-%m-%dT%H:%M")
        cloud = float(item.properties.get("eo:cloud_cover", 100.0))

        if key not in best:
            best[key] = item
        else:
            previous = float(
                best[key].properties.get("eo:cloud_cover", 100.0)
            )
            if cloud < previous:
                best[key] = item

    return sorted(
        best.values(),
        key=lambda item: item.datetime,
        reverse=True,
    )


def choose_items(items, period_key):
    items = dedupe_same_acquisition(items)

    if not items:
        return []

    if period_key == "latest":
        return items[:1]

    if period_key in ("1m", "4m"):
        return items

    grouped = {}

    for item in items:
        dt = item.datetime
        cloud = float(item.properties.get("eo:cloud_cover", 100.0))

        if period_key == "1y":
            ten_day = min(2, (dt.day - 1) // 10)
            key = (dt.year, dt.month, ten_day)
        else:
            key = (dt.year, dt.month)

        if key not in grouped:
            grouped[key] = item
        else:
            old_cloud = float(
                grouped[key].properties.get("eo:cloud_cover", 100.0)
            )
            if cloud < old_cloud:
                grouped[key] = item

    return sorted(
        grouped.values(),
        key=lambda item: item.datetime,
        reverse=True,
    )


def period_dates(now, key):
    if key == "latest":
        return now - relativedelta(days=45), now
    if key == "1m":
        return now - relativedelta(months=1), now
    if key == "4m":
        return now - relativedelta(months=4), now
    if key == "1y":
        return now - relativedelta(years=1), now
    return now - relativedelta(years=5), now


def read_asset_to_grid(
    href,
    bounds,
    width,
    height,
    dst_transform,
    *,
    categorical=False,
):
    if categorical:
        dest = np.zeros((height, width), dtype="uint8")
        resampling = Resampling.nearest
        dst_nodata = 0
    else:
        dest = np.full(
            (height, width),
            np.nan,
            dtype="float32",
        )
        resampling = Resampling.bilinear
        dst_nodata = np.nan

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata if src.nodata is not None else 0,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                dst_nodata=dst_nodata,
                resampling=resampling,
            )

    return dest


@st.cache_data(ttl=3600, show_spinner=False)
def scene_ndvi(item_dict, bounds_tuple, width, height):
    item = planetary_computer.sign(
        pystac.Item.from_dict(item_dict)
    )

    bounds = list(bounds_tuple)
    dst_transform = from_bounds(
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        width,
        height,
    )

    red = read_asset_to_grid(
        item.assets["B04"].href,
        bounds,
        width,
        height,
        dst_transform,
    )

    nir = read_asset_to_grid(
        item.assets["B08"].href,
        bounds,
        width,
        height,
        dst_transform,
    )

    scl = read_asset_to_grid(
        item.assets["SCL"].href,
        bounds,
        width,
        height,
        dst_transform,
        categorical=True,
    )

    valid = (
        np.isfinite(red)
        & np.isfinite(nir)
        & (red > 0)
        & (nir > 0)
    )

    denom = nir + red
    valid &= np.isfinite(denom) & (denom != 0)

    raw = np.full(
        (height, width),
        np.nan,
        dtype="float32",
    )
    raw[valid] = (nir[valid] - red[valid]) / denom[valid]
    raw[(raw < -1) | (raw > 1)] = np.nan

    clear = raw.copy()
    clear_mask = np.isin(scl, list(CLEAR_SCL))
    clear[~clear_mask] = np.nan

    raw_count = int(np.isfinite(raw).sum())
    clear_count = int(np.isfinite(clear).sum())

    return {
        "raw_map": raw,
        "clear_map": clear,
        "raw_mean": (
            float(np.nanmean(raw))
            if raw_count
            else None
        ),
        "clear_mean": (
            float(np.nanmean(clear))
            if clear_count
            else None
        ),
        "raw_count": raw_count,
        "clear_count": clear_count,
        "clear_fraction": (
            float(clear_count / raw_count)
            if raw_count
            else 0.0
        ),
        "date": (
            item.datetime.strftime("%Y-%m-%d")
            if item.datetime
            else ""
        ),
        "scene_cloud": float(
            item.properties.get("eo:cloud_cover", np.nan)
        ),
        "id": item.id,
    }


def safe_nanmean_stack(arrays):
    if not arrays:
        return None

    stack = np.stack(arrays).astype("float32")
    valid_count = np.sum(np.isfinite(stack), axis=0)
    value_sum = np.nansum(stack, axis=0)

    out = np.full(
        stack.shape[1:],
        np.nan,
        dtype="float32",
    )

    valid = valid_count > 0
    out[valid] = value_sum[valid] / valid_count[valid]
    return out



def process_latest_clear_scene(items, bounds, width, height):
    """
    Prefer the most recent scene that is actually usable over the AOI.

    Rule:
    - inspect up to the 12 most recent acquisitions
    - accept the first scene with >= 70% AOI pixels retained by SCL
    - if none reaches 70%, use the scene with the highest AOI clear fraction
    """
    candidates = dedupe_same_acquisition(items)[:12]

    if not candidates:
        return None

    evaluated = []
    progress = st.progress(
        0,
        text="Buscando la imagen reciente más despejada…",
    )

    for i, item in enumerate(candidates):
        try:
            row = scene_ndvi(
                item.to_dict(),
                tuple(bounds),
                width,
                height,
            )
            evaluated.append(row)

            if row["clear_fraction"] >= 0.70:
                progress.empty()
                return {
                    "raw_map": row["raw_map"],
                    "clear_map": row["clear_map"],
                    "raw_mean": row["raw_mean"],
                    "clear_mean": row["clear_mean"],
                    "raw_count": row["raw_count"],
                    "clear_count": row["clear_count"],
                    "clear_fraction": row["clear_fraction"],
                    "scenes": 1,
                    "latest_date": row["date"],
                    "scene_cloud": row["scene_cloud"],
                }
        except Exception:
            pass

        progress.progress(
            (i + 1) / max(len(candidates), 1),
            text=f"Revisando escenas recientes: {i + 1}/{len(candidates)}",
        )

    progress.empty()

    if not evaluated:
        return None

    best = max(
        evaluated,
        key=lambda row: row["clear_fraction"],
    )

    return {
        "raw_map": best["raw_map"],
        "clear_map": best["clear_map"],
        "raw_mean": best["raw_mean"],
        "clear_mean": best["clear_mean"],
        "raw_count": best["raw_count"],
        "clear_count": best["clear_count"],
        "clear_fraction": best["clear_fraction"],
        "scenes": 1,
        "latest_date": best["date"],
        "scene_cloud": best["scene_cloud"],
    }


def process_period(items, bounds, width, height, progress_label):
    raw_maps = []
    clear_maps = []
    metadata = []

    total = max(len(items), 1)
    progress = st.progress(
        0,
        text=progress_label,
    )

    for i, item in enumerate(items):
        try:
            row = scene_ndvi(
                item.to_dict(),
                tuple(bounds),
                width,
                height,
            )
            raw_maps.append(row["raw_map"])
            clear_maps.append(row["clear_map"])
            metadata.append(row)
        except Exception:
            pass

        progress.progress(
            (i + 1) / total,
            text=f"{progress_label} {i + 1}/{len(items)}",
        )

    progress.empty()

    if not metadata:
        return None

    raw_composite = safe_nanmean_stack(raw_maps)
    clear_composite = safe_nanmean_stack(clear_maps)

    raw_valid = int(np.isfinite(raw_composite).sum())
    clear_valid = int(np.isfinite(clear_composite).sum())

    return {
        "raw_map": raw_composite,
        "clear_map": clear_composite,
        "raw_mean": (
            float(np.nanmean(raw_composite))
            if raw_valid
            else None
        ),
        "clear_mean": (
            float(np.nanmean(clear_composite))
            if clear_valid
            else None
        ),
        "raw_count": raw_valid,
        "clear_count": clear_valid,
        "clear_fraction": (
            float(clear_valid / raw_valid)
            if raw_valid
            else 0.0
        ),
        "scenes": len(metadata),
        "latest_date": max(
            row["date"]
            for row in metadata
            if row["date"]
        ),
        "scene_cloud": (
            metadata[0]["scene_cloud"]
            if len(metadata) == 1
            else None
        ),
    }


def colorize_ndvi(ndvi):
    """
    NDVI palette:
    low = red/orange,
    middle = yellow/light green,
    high = dark green.
    NaN = transparent.
    """
    stops = np.array(
        [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
        dtype="float32",
    )

    colors = np.array(
        [
            [194, 39, 32],
            [239, 104, 55],
            [253, 211, 88],
            [184, 220, 111],
            [72, 160, 72],
            [16, 111, 54],
        ],
        dtype="float32",
    )

    rgba = np.zeros(
        ndvi.shape + (4,),
        dtype="uint8",
    )

    valid = np.isfinite(ndvi)
    vals = np.clip(ndvi[valid], stops[0], stops[-1])

    rgb = np.empty(
        (vals.size, 3),
        dtype="float32",
    )

    for channel in range(3):
        rgb[:, channel] = np.interp(
            vals,
            stops,
            colors[:, channel],
        )

    rgba[valid, :3] = np.clip(
        rgb,
        0,
        255,
    ).astype("uint8")
    rgba[valid, 3] = 205

    return rgba


def png_bytes_from_rgba(rgba):
    buffer = io.BytesIO()
    Image.fromarray(
        rgba,
        mode="RGBA",
    ).save(
        buffer,
        format="PNG",
        optimize=True,
    )
    return buffer.getvalue()


def geotiff_bytes(ndvi, bounds):
    west, south, east, north = bounds
    height, width = ndvi.shape

    transform = from_bounds(
        west,
        south,
        east,
        north,
        width,
        height,
    )

    data = np.where(
        np.isfinite(ndvi),
        ndvi,
        -9999.0,
    ).astype("float32")

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=-9999.0,
            compress="deflate",
        ) as dataset:
            dataset.write(data, 1)

        return memfile.read()


def legend_html():
    return """
    <div style="
        position:absolute;
        left:16px;
        bottom:24px;
        z-index:9999;
        background:rgba(255,255,255,.92);
        padding:9px 11px;
        border-radius:8px;
        box-shadow:0 1px 5px rgba(0,0,0,.22);
        font-size:12px;
        min-width:240px;">
        <div style="font-weight:600;margin-bottom:5px;">NDVI libre de nubes</div>
        <div style="
            height:11px;
            border-radius:6px;
            background:linear-gradient(
                90deg,
                rgb(194,39,32) 0%,
                rgb(239,104,55) 20%,
                rgb(253,211,88) 40%,
                rgb(184,220,111) 60%,
                rgb(72,160,72) 80%,
                rgb(16,111,54) 100%
            );">
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:3px;">
            <span>-0.2</span><span>0.0</span><span>0.2</span>
            <span>0.4</span><span>0.6</span><span>0.8</span>
        </div>
    </div>
    """


def fmt(value):
    if value is None:
        return "—"
    if not np.isfinite(value):
        return "—"
    return f"{value:.3f}"


def add_existing_aoi(map_obj, bounds):
    if not bounds:
        return

    west, south, east, north = bounds

    folium.Rectangle(
        bounds=[
            [south, west],
            [north, east],
        ],
        color="#2b7cff",
        weight=2,
        fill=True,
        fill_opacity=0.05,
    ).add_to(map_obj)


def build_map(bounds=None, overlay=None):
    if bounds:
        west, south, east, north = bounds
        center = [
            (south + north) / 2,
            (west + east) / 2,
        ]
        zoom_start = 14
    else:
        center = [41.74, -111.82]
        zoom_start = 10

    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles=ESRI_TILES,
        attr=ESRI_ATTR,
        name="Imagen satelital",
        overlay=False,
        control=True,
        max_zoom=19,
    ).add_to(m)

    if overlay is not None and bounds:
        rgba = colorize_ndvi(overlay)

        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[
                [bounds[1], bounds[0]],
                [bounds[3], bounds[2]],
            ],
            opacity=0.82,
            name="NDVI libre de nubes",
            interactive=False,
            cross_origin=False,
            zindex=5,
        ).add_to(m)

        m.get_root().html.add_child(
            folium.Element(legend_html())
        )

    add_existing_aoi(m, bounds)

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "rectangle": {
                "shapeOptions": {
                    "weight": 2,
                    "color": "#2b7cff",
                }
            },
        },
        edit_options={
            "edit": True,
            "remove": True,
        },
    ).add_to(m)

    folium.LayerControl(
        collapsed=True,
    ).add_to(m)

    if bounds:
        m.fit_bounds(
            [
                [bounds[1], bounds[0]],
                [bounds[3], bounds[2]],
            ],
            padding=(18, 18),
        )

    return m


if "aoi_bounds" not in st.session_state:
    st.session_state.aoi_bounds = None

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None


st.title("🌿 NDVI Quicklook")
st.caption(
    "Visualiza NDVI Sentinel‑2 para cualquier área. "
    "Cambia el periodo, explora el mapa y descarga el resultado."
)

analysis = st.session_state.analysis_results

if analysis:
    available_labels = [
        label
        for label, _ in PERIODS
        if label in analysis["periods"]
    ]

    default_index = 0

    main_col, sidebar_col = st.columns([2.1, 1], gap="medium")

    with sidebar_col:
        st.subheader("Periodo de análisis")

        selected_label = st.radio(
            "Periodo",
            options=available_labels,
            index=default_index,
            label_visibility="collapsed",
        )

        selected_result = analysis["periods"][selected_label]

        st.markdown(
            '<div class="ndvi-card">',
            unsafe_allow_html=True,
        )
        st.markdown("**Área seleccionada**")
        st.metric(
            "Área aproximada",
            f"{area_km2(analysis['bounds']):.2f} km²",
        )
        st.caption(
            f"{analysis['bounds'][1]:.5f}, {analysis['bounds'][0]:.5f} "
            f"→ {analysis['bounds'][3]:.5f}, {analysis['bounds'][2]:.5f}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="ndvi-card">',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Resumen NDVI · {selected_label}**")

        c1, c2 = st.columns(2)
        c1.metric(
            "NDVI libre de nubes",
            fmt(selected_result["clear_mean"]),
        )
        c2.metric(
            "NDVI sin filtrar",
            fmt(selected_result["raw_mean"]),
        )

        st.caption(
            f"Observaciones utilizadas: {selected_result['scenes']}"
        )

        if selected_label == "Última imagen despejada":
            st.caption(
                f"Fecha: {selected_result['latest_date']} · "
                f"Nubes de escena: "
                f"{selected_result['scene_cloud']:.1f}%"
                if selected_result["scene_cloud"] is not None
                else f"Fecha: {selected_result['latest_date']}"
            )

        st.caption(
            f"Píxeles retenidos por SCL: "
            f"{selected_result['clear_fraction'] * 100:.1f}%"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        png_data = png_bytes_from_rgba(
            colorize_ndvi(selected_result["clear_map"])
        )
        tif_data = geotiff_bytes(
            selected_result["clear_map"],
            analysis["bounds"],
        )

        d1, d2 = st.columns(2)

        with d1:
            st.download_button(
                "Descargar PNG",
                data=png_data,
                file_name=(
                    "ndvi_"
                    + selected_label.lower()
                    .replace(" ", "_")
                    .replace("ú", "u")
                    .replace("á", "a")
                    + ".png"
                ),
                mime="image/png",
                use_container_width=True,
            )

        with d2:
            st.download_button(
                "Descargar GeoTIFF",
                data=tif_data,
                file_name=(
                    "ndvi_"
                    + selected_label.lower()
                    .replace(" ", "_")
                    .replace("ú", "u")
                    .replace("á", "a")
                    + ".tif"
                ),
                mime="image/tiff",
                use_container_width=True,
            )

        if st.button(
            "Nueva área",
            use_container_width=True,
        ):
            st.session_state.aoi_bounds = None
            st.session_state.analysis_results = None
            st.rerun()

    with main_col:
        map_obj = build_map(
            bounds=analysis["bounds"],
            overlay=selected_result["clear_map"],
        )

        map_data = st_folium(
            map_obj,
            height=640,
            use_container_width=True,
            returned_objects=[
                "last_active_drawing",
                "all_drawings",
            ],
            key="ndvi_result_map",
        )

else:
    left, right = st.columns([2.1, 1])

    with left:
        map_obj = build_map(
            bounds=st.session_state.aoi_bounds,
            overlay=None,
        )

        map_data = st_folium(
            map_obj,
            height=640,
            use_container_width=True,
            returned_objects=[
                "last_active_drawing",
                "all_drawings",
            ],
            key="ndvi_draw_map",
        )

    drawing = None
    if map_data:
        drawing = map_data.get("last_active_drawing")

    if (
        drawing
        and drawing.get("geometry", {}).get("type") == "Polygon"
    ):
        coords = drawing["geometry"]["coordinates"][0]
        xs = [point[0] for point in coords]
        ys = [point[1] for point in coords]

        st.session_state.aoi_bounds = [
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        ]

    bounds = st.session_state.aoi_bounds

    with right:
        st.subheader("Área seleccionada")

        if bounds:
            km2 = area_km2(bounds)

            st.metric(
                "Área aproximada",
                f"{km2:.2f} km²",
            )

            st.caption(
                f"{bounds[1]:.5f}, {bounds[0]:.5f} "
                f"→ {bounds[3]:.5f}, {bounds[2]:.5f}"
            )

            if km2 > MAX_AOI_KM2:
                st.error(
                    f"Para este MVP, dibuja un área de máximo "
                    f"{MAX_AOI_KM2:.0f} km²."
                )
                run = False
            else:
                run = st.button(
                    "Calcular NDVI",
                    type="primary",
                    use_container_width=True,
                )
        else:
            st.info(
                "Usa la herramienta de rectángulo del mapa "
                "y dibuja tu área de interés."
            )
            run = False

    if run and bounds:
        now = datetime.now(timezone.utc)
        width, height, _ = target_grid(bounds)

        with st.spinner(
            "Buscando observaciones Sentinel‑2…"
        ):
            all_items = search_items(
                bounds,
                now - relativedelta(years=5),
                now,
            )

        if not all_items:
            st.warning(
                "No encontré imágenes Sentinel‑2 para esa zona."
            )
            st.stop()

        full = dedupe_same_acquisition(all_items)
        period_results = {}

        for label, key in PERIODS:
            start, end = period_dates(now, key)

            subset = [
                item
                for item in full
                if item.datetime is not None
                and start <= item.datetime <= end
            ]

            if not subset:
                continue

            if key == "latest":
                result = process_latest_clear_scene(
                    subset,
                    bounds,
                    width,
                    height,
                )
            else:
                subset = choose_items(
                    subset,
                    key,
                )

                result = process_period(
                    subset,
                    bounds,
                    width,
                    height,
                    progress_label=f"{label}:",
                )

            if result:
                period_results[label] = result

        if not period_results:
            st.warning(
                "Encontré escenas, pero no pude producir NDVI "
                "para el área seleccionada."
            )
            st.stop()

        st.session_state.analysis_results = {
            "bounds": bounds,
            "width": width,
            "height": height,
            "periods": period_results,
        }

        st.rerun()


analysis = st.session_state.analysis_results

if analysis:
    st.divider()
    st.subheader("NDVI promedio por periodo")

    rows = []

    for label, _ in PERIODS:
        if label not in analysis["periods"]:
            continue

        result = analysis["periods"][label]

        period_name = label
        if label == "Última imagen despejada":
            period_name = (
                f"{label} ({result['latest_date']})"
            )

        rows.append(
            {
                "Periodo": period_name,
                "NDVI libre de nubes": fmt(
                    result["clear_mean"]
                ),
                "NDVI sin filtrar": fmt(
                    result["raw_mean"]
                ),
                "Observaciones": result["scenes"],
            }
        )

    result_df = pd.DataFrame(rows)

    st.dataframe(
        result_df,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "El mapa NDVI usa la máscara SCL de Sentinel‑2 para excluir "
        "nubes, cirrus, sombras, nieve/hielo y píxeles no válidos. "
        "El mapa satelital de fondo es Esri World Imagery y se usa "
        "solo como contexto visual; su fecha puede no coincidir con "
        "la observación Sentinel‑2 analizada."
    )
