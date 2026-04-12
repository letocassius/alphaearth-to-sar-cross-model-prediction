// Google Earth Engine Code Editor JS
// Export one Sentinel-2 image and a paired AlphaEarth-per-pixel table.
//
// This script is intended for the single-image local experiment documented in
// README.md and reports/single_image_pixel_fraction_report.md.

var YEAR = 2024;
var START = ee.Date('2024-01-01');
var END = ee.Date('2025-01-01');

// The current local experiment uses 5,000 sampled pixels.
var NUM_PIXELS = 5000;

// Default AOI matches the current single-image experiment.
var AOI = ee.Geometry.Rectangle([-122.60, 37.15, -121.70, 37.95], null, false);

var EXPORT_FOLDER = 'GEE_AlphaEarth_SAR_Project';

var EMBEDDING_BANDS = [
  'A00', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07',
  'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15',
  'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23',
  'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31',
  'A32', 'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39',
  'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47',
  'A48', 'A49', 'A50', 'A51', 'A52', 'A53', 'A54', 'A55',
  'A56', 'A57', 'A58', 'A59', 'A60', 'A61', 'A62', 'A63'
];

function maskS2Clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var clear = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(clear).divide(10000);
}

var s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(AOI)
  .filterDate(START, END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2Clouds);

var sentinelImage = ee.Image(
  s2Collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
).clip(AOI);

var alphaEmbedding = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterBounds(AOI)
  .filterDate(START, END)
  .mosaic()
  .select(EMBEDDING_BANDS)
  .clip(AOI);

var sentinelBands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];
var stack = sentinelImage.select(sentinelBands).addBands(alphaEmbedding);

var validMask = sentinelImage.select('B4').mask()
  .and(alphaEmbedding.select('A00').mask());

stack = stack.updateMask(validMask);

var sampled = stack.sample({
  region: AOI,
  scale: 10,
  numPixels: NUM_PIXELS,
  seed: 42,
  geometries: true,
  tileScale: 8
}).map(function(f) {
  var coords = f.geometry().coordinates();
  return f.set({
    longitude: coords.get(0),
    latitude: coords.get(1),
    year: YEAR
  });
});

Map.setOptions('SATELLITE');
Map.centerObject(AOI, 10);
Map.addLayer(
  sentinelImage,
  {bands: ['B4', 'B3', 'B2'], min: 0.02, max: 0.30},
  'Sentinel-2 RGB'
);
Map.addLayer(
  alphaEmbedding.select('A00'),
  {min: -0.5, max: 0.5},
  'AlphaEarth A00',
  false
);
Map.addLayer(AOI, {color: 'yellow'}, 'AOI', false);

print('Sentinel-2 image', sentinelImage);
print('AlphaEarth embedding image', alphaEmbedding);
print('Sample count', sampled.size());
print('Example rows', sampled.limit(5));

Export.table.toDrive({
  collection: sampled,
  description: 'sentinel2_alphaearth_pixel_pairs_' + YEAR + '_n' + NUM_PIXELS,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'sentinel2_alphaearth_pixel_pairs_' + YEAR + '_n' + NUM_PIXELS,
  fileFormat: 'CSV'
});

Export.image.toDrive({
  image: sentinelImage.select(['B4', 'B3', 'B2']),
  description: 'sentinel2_rgb_' + YEAR,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'sentinel2_rgb_' + YEAR,
  region: AOI,
  scale: 10,
  maxPixels: 1e13
});
