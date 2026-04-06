// Reconstructed convenience export based on the committed merged-sample workflow.
// This recreates an AlphaEarth-only point table over the same stratified sample set.

var YEAR = 2024;
var START = ee.Date.fromYMD(YEAR, 1, 1);
var END = START.advance(1, 'year');

var EXPORT_FOLDER = 'GEE_AlphaEarth_SAR_Project';
var SCALE = 10;
var POINTS_PER_CLASS = 80;

var DW_CLASS_NAMES = [
  'water', 'trees', 'grass', 'flooded_vegetation', 'crops',
  'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'
];
var DW_CLASS_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8];
var DW_CLASS_POINTS = [
  POINTS_PER_CLASS, POINTS_PER_CLASS, POINTS_PER_CLASS,
  POINTS_PER_CLASS, POINTS_PER_CLASS, POINTS_PER_CLASS,
  POINTS_PER_CLASS, POINTS_PER_CLASS, POINTS_PER_CLASS
];
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

var regions = [
  {
    name: 'sf_bay_urban',
    geom: ee.Geometry.Rectangle([-122.60, 37.15, -121.70, 37.95], null, false)
  },
  {
    name: 'iowa_ag',
    geom: ee.Geometry.Rectangle([-95.90, 41.60, -94.90, 42.40], null, false)
  },
  {
    name: 'amazon_forest',
    geom: ee.Geometry.Rectangle([-60.50, -3.60, -59.50, -2.60], null, false)
  },
  {
    name: 'california_coast',
    geom: ee.Geometry.Rectangle([-121.90, 34.70, -120.90, 35.50], null, false)
  }
];

function getAlphaEarthImage(regionGeom) {
  return ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    .filterDate(START, END)
    .filterBounds(regionGeom)
    .mosaic()
    .clip(regionGeom);
}

function getSentinel1Image(regionGeom) {
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(regionGeom)
    .filterDate(START, END)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .select(['VV', 'VH']);

  return s1.median().clip(regionGeom).rename(['S1_VV', 'S1_VH']);
}

function getDynamicWorldImage(regionGeom) {
  var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterBounds(regionGeom)
    .filterDate(START, END);
  var label = dw.select('label').reduce(ee.Reducer.mode()).rename('dw_label');
  return label.clip(regionGeom);
}

function makeAlphaOnlySamples(regionObj) {
  var geom = regionObj.geom;
  var alpha = getAlphaEarthImage(geom);
  var s1 = getSentinel1Image(geom);
  var dw = getDynamicWorldImage(geom);

  var commonMask = alpha.select('A00').mask()
    .and(s1.select('S1_VV').mask())
    .and(dw.select('dw_label').mask());

  var samplingStack = alpha.addBands(dw).updateMask(commonMask);
  var samples = samplingStack.stratifiedSample({
    numPoints: POINTS_PER_CLASS,
    classBand: 'dw_label',
    region: geom,
    scale: SCALE,
    seed: 42,
    geometries: true,
    classValues: DW_CLASS_VALUES,
    classPoints: DW_CLASS_POINTS,
    tileScale: 4
  });

  return samples.select(EMBEDDING_BANDS);
}

var alphaAllRegions = ee.FeatureCollection([]);
regions.forEach(function(r) {
  alphaAllRegions = alphaAllRegions.merge(makeAlphaOnlySamples(r));
});

Export.table.toDrive({
  collection: alphaAllRegions,
  description: 'AlphaEarth_Embeddings_' + YEAR,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'AlphaEarth_Embeddings_' + YEAR,
  fileFormat: 'CSV'
});
