var YEAR = 2024;
var START = ee.Date.fromYMD(YEAR, 1, 1);
var END = START.advance(1, 'year');

var EXPORT_FOLDER = 'GEE_AlphaEarth_SAR_Project';

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

function maskS2Clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var clear = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(clear).divide(10000);
}

function getS2ContextImage(regionGeom) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(regionGeom)
    .filterDate(START, END)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskS2Clouds)
    .median()
    .clip(regionGeom);
}

Map.setOptions('SATELLITE');
Map.centerObject(regions[0].geom, 9);

regions.forEach(function(r) {
  var s2Context = getS2ContextImage(r.geom).select(['B4', 'B3', 'B2']);
  Map.addLayer(
    s2Context,
    {bands: ['B4', 'B3', 'B2'], min: 0.02, max: 0.30},
    r.name + '_S2_RGB',
    false
  );

  Export.image.toDrive({
    image: s2Context,
    description: 'sentinel2_context_' + r.name + '_' + YEAR,
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'sentinel2_context_' + r.name + '_' + YEAR,
    region: r.geom,
    scale: 10,
    maxPixels: 1e13
  });
});
