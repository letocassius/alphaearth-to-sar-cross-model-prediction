// Google Earth Engine Code Editor JS
// Export a Sentinel-2 natural-color reference image for the downtown San
// Francisco + Golden Gate AOI used in the SAR -> AlphaEarth reconstruction.

var YEAR = 2024;
var START = ee.Date.fromYMD(YEAR, 1, 1);
var END = START.advance(1, 'year');
var SCALE = 10;

var EXPORT_FOLDER = 'GEE_AlphaEarth_SAR_Project';

var AOI_NAME = 'sf_downtown_golden_gate';
var AOI = ee.Geometry.Rectangle(
  [-122.52, 37.78, -122.35, 37.84],
  null,
  false
);

function maskS2Clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var clear = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(clear).divide(10000);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(AOI)
  .filterDate(START, END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2Clouds);

var sentinel2Image = ee.Image(
  s2.sort('CLOUDY_PIXEL_PERCENTAGE').first()
).clip(AOI);

Map.setOptions('SATELLITE');
Map.centerObject(AOI, 12);
Map.addLayer(
  sentinel2Image,
  {bands: ['B4', 'B3', 'B2'], min: 0.02, max: 0.30},
  'Sentinel-2 natural color'
);
Map.addLayer(AOI, {color: 'yellow'}, 'AOI', true);

print('AOI name', AOI_NAME);
print('Sentinel-2 image', sentinel2Image);

Export.image.toDrive({
  image: sentinel2Image.select(['B4', 'B3', 'B2']),
  description: 'sentinel2_natural_color_' + AOI_NAME + '_' + YEAR,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'sentinel2_natural_color_' + AOI_NAME + '_' + YEAR,
  region: AOI,
  scale: SCALE,
  maxPixels: 1e13
});
