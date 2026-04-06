// Reconstructed convenience export based on the committed merged-sample workflow.
// This recreates a Sentinel-1-only point table over the same stratified sample set.

var YEAR = 2024;
var START = ee.Date.fromYMD(YEAR, 1, 1);
var END = START.advance(1, 'year');

var EXPORT_FOLDER = 'GEE_AlphaEarth_SAR_Project';
var SCALE = 10;
var POINTS_PER_CLASS = 80;

var DW_CLASS_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8];
var DW_CLASS_POINTS = [
  POINTS_PER_CLASS, POINTS_PER_CLASS, POINTS_PER_CLASS,
  POINTS_PER_CLASS, POINTS_PER_CLASS, POINTS_PER_CLASS,
  POINTS_PER_CLASS, POINTS_PER_CLASS, POINTS_PER_CLASS
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

  var composite = s1.median().clip(regionGeom);
  var vv = composite.select('VV').rename('S1_VV');
  var vh = composite.select('VH').rename('S1_VH');
  var vvDivVh = vv.subtract(vh).rename('S1_VV_div_VH');

  return vv.addBands(vh).addBands(vvDivVh);
}

function getDynamicWorldLabel(regionGeom) {
  return ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterBounds(regionGeom)
    .filterDate(START, END)
    .select('label')
    .reduce(ee.Reducer.mode())
    .rename('dw_label')
    .clip(regionGeom);
}

function makeSarOnlySamples(regionObj) {
  var geom = regionObj.geom;
  var alpha = getAlphaEarthImage(geom);
  var s1 = getSentinel1Image(geom);
  var dwLabel = getDynamicWorldLabel(geom);

  var commonMask = alpha.select('A00').mask()
    .and(s1.select('S1_VV').mask())
    .and(dwLabel.select('dw_label').mask());

  var samplingStack = s1.addBands(dwLabel).updateMask(commonMask);
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

  return samples.select(['S1_VH', 'S1_VV', 'S1_VV_div_VH']);
}

var sarAllRegions = ee.FeatureCollection([]);
regions.forEach(function(r) {
  sarAllRegions = sarAllRegions.merge(makeSarOnlySamples(r));
});

Export.table.toDrive({
  collection: sarAllRegions,
  description: 'Sentinel1_SAR_' + YEAR,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'Sentinel1_SAR_' + YEAR,
  fileFormat: 'CSV'
});
