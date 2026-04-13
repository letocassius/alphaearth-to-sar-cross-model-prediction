// Google Earth Engine Code Editor JS
// Export a small Sentinel-1 + AlphaEarth stack for fast full-image reconstruction.
//
// This is the same idea as the larger script, but the AOI is intentionally much
// smaller so you can iterate locally without generating multi-tile 17 GB exports.

var YEAR = 2024;
var START = ee.Date.fromYMD(YEAR, 1, 1);
var END = START.advance(1, 'year');
var SCALE = 10;

var EXPORT_FOLDER = 'GEE_AlphaEarth_SAR_Project';

// Default small AOI focused on downtown San Francisco plus the Golden Gate Bridge.
// This stays much smaller than the earlier full scene but covers both landmarks.
var AOI_NAME = 'sf_downtown_golden_gate';
var AOI = ee.Geometry.Rectangle(
  [-122.52, 37.78, -122.35, 37.84],
  null,
  false
);

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
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .select(['VV', 'VH']);

  var composite = s1.median().clip(regionGeom);
  var vv = composite.select('VV').rename('S1_VV');
  var vh = composite.select('VH').rename('S1_VH');
  var vvMinusVh = vv.subtract(vh).rename('S1_VV_div_VH');

  return vv.addBands(vh).addBands(vvMinusVh);
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
Map.centerObject(AOI, 12);
Map.addLayer(
  sentinel1Image.select('S1_VV'),
  {min: -20, max: 0},
  'Sentinel-1 VV'
);
Map.addLayer(
  sentinel1Image.select('S1_VH'),
  {min: -28, max: -5},
  'Sentinel-1 VH',
  false
);
Map.addLayer(
  alphaEmbedding.select('A00'),
  {min: -0.5, max: 0.5},
  'AlphaEarth A00',
  false
);
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
