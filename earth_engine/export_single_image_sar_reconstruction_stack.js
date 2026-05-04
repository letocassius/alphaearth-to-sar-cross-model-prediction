// Google Earth Engine Code Editor JS
// Export a small Sentinel-1 + AlphaEarth stack for local SAR reconstruction.
//
// Output 1: a colocated 67-band GeoTIFF containing:
//   bands 1-3: S1_VV, S1_VH, S1_VV_div_VH
//   bands 4-67: A00 ... A63 AlphaEarth embeddings
// Output 2: a Sentinel-1-only GeoTIFF for optional inspection.

var YEAR = 2024;
var START = ee.Date(YEAR + '-01-01');
var END = ee.Date((YEAR + 1) + '-01-01');
var SCALE = 10;

var EXPORT_FOLDER = 'GEE_AlphaEarth_SAR_Project';
var AOI_NAME = 'sf_downtown_golden_gate';
var AOI = ee.Geometry.Rectangle([-122.52, 37.78, -122.35, 37.84], null, false);

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

function getSentinel1Composite(regionGeom) {
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(regionGeom)
    .filterDate(START, END)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV', 'VH']))
    .select(['VV', 'VH']);

  print('Sentinel-1 image count:', s1.size());

  var composite = ee.Image(s1.median()).clip(regionGeom);
  var vv = composite.select('VV').rename('S1_VV');
  var vh = composite.select('VH').rename('S1_VH');

  // This is VV - VH in dB space. The Python pipeline expects this band name.
  var vvMinusVh = composite.select('VV')
    .subtract(composite.select('VH'))
    .rename('S1_VV_div_VH');

  return ee.Image.cat([vv, vh, vvMinusVh]);
}

function getAlphaEarthEmbedding(regionGeom) {
  return ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    .filterBounds(regionGeom)
    .filterDate(START, END)
    .mosaic()
    .select(EMBEDDING_BANDS)
    .clip(regionGeom);
}

var sentinel1Image = getSentinel1Composite(AOI);
var alphaEmbedding = getAlphaEarthEmbedding(AOI);
var stack = sentinel1Image.addBands(alphaEmbedding);

var validMask = sentinel1Image.select('S1_VV').mask()
  .and(sentinel1Image.select('S1_VH').mask())
  .and(alphaEmbedding.select('A00').mask());
stack = stack.updateMask(validMask);

Map.setOptions('SATELLITE');
Map.setCenter(-122.435, 37.81, 12);
Map.addLayer(sentinel1Image.select('S1_VV'), {min: -20, max: 0}, 'Sentinel-1 VV');
Map.addLayer(sentinel1Image.select('S1_VH'), {min: -28, max: -5}, 'Sentinel-1 VH', false);
Map.addLayer(alphaEmbedding.select('A00'), {min: -0.5, max: 0.5}, 'AlphaEarth A00', false);
Map.addLayer(AOI, {color: 'yellow'}, 'AOI', true);

print('AOI name', AOI_NAME);
print('Sentinel-1 composite', sentinel1Image);
print('AlphaEarth embedding image', alphaEmbedding);
print('Full stack image', stack);

Export.image.toDrive({
  image: stack,
  description: 'sentinel1_alphaearth_small_stack_' + AOI_NAME + '_' + YEAR,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'sentinel1_alphaearth_small_stack_' + AOI_NAME + '_' + YEAR,
  region: AOI,
  scale: SCALE,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: sentinel1Image,
  description: 'sentinel1_small_vv_vh_' + AOI_NAME + '_' + YEAR,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'sentinel1_small_vv_vh_' + AOI_NAME + '_' + YEAR,
  region: AOI,
  scale: SCALE,
  maxPixels: 1e13
});
