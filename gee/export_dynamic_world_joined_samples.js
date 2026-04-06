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

function getDynamicWorldImage(regionGeom) {
  var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterBounds(regionGeom)
    .filterDate(START, END);

  var label = dw.select('label').reduce(ee.Reducer.mode()).rename('dw_label');
  var probs = dw.select(DW_CLASS_NAMES).mean();

  return label.addBands(probs).clip(regionGeom);
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

function addLonLat(feature, regionName) {
  var coords = feature.geometry().coordinates();
  return feature.set({
    longitude: coords.get(0),
    latitude: coords.get(1),
    region: regionName,
    year: YEAR
  });
}

function makeSampleCollection(regionObj) {
  var regionName = regionObj.name;
  var geom = regionObj.geom;

  var alpha = getAlphaEarthImage(geom);
  var s1 = getSentinel1Image(geom);
  var dw = getDynamicWorldImage(geom);

  var commonMask = alpha.select('A00').mask()
    .and(s1.select('S1_VV').mask())
    .and(dw.select('dw_label').mask());

  var stack = alpha
    .addBands(s1)
    .addBands(dw)
    .updateMask(commonMask);

  var samples = stack.stratifiedSample({
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

  return samples.map(function(f) {
    return addLonLat(f, regionName);
  });
}

Map.setOptions('SATELLITE');
Map.centerObject(regions[0].geom, 9);

var mergedAllRegions = ee.FeatureCollection([]);

regions.forEach(function(r) {
  Map.addLayer(r.geom, {color: 'yellow'}, r.name + '_aoi', false);

  var s2Preview = getS2ContextImage(r.geom);
  Map.addLayer(
    s2Preview,
    {bands: ['B4', 'B3', 'B2'], min: 0.02, max: 0.30},
    r.name + '_S2_RGB',
    false
  );

  var dwPreview = getDynamicWorldImage(r.geom).select('dw_label');
  Map.addLayer(
    dwPreview,
    {min: 0, max: 8, palette: [
      '#419bdf', '#397d49', '#88b053', '#7a87c6', '#e49635',
      '#dfc35a', '#c4281b', '#a59b8f', '#b39fe1'
    ]},
    r.name + '_DW_label',
    false
  );

  var s1Preview = getSentinel1Image(r.geom);
  Map.addLayer(
    s1Preview.select('S1_VV'),
    {min: -20, max: 0},
    r.name + '_S1_VV',
    false
  );

  var samples = makeSampleCollection(r);
  mergedAllRegions = mergedAllRegions.merge(samples);

  Export.table.toDrive({
    collection: samples,
    description: 'alphaearth_s1_dw_samples_' + r.name + '_' + YEAR,
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'alphaearth_s1_dw_samples_' + r.name + '_' + YEAR,
    fileFormat: 'CSV'
  });
});

Export.table.toDrive({
  collection: mergedAllRegions,
  description: 'alphaearth_s1_dw_samples_all_regions_' + YEAR,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'alphaearth_s1_dw_samples_all_regions_' + YEAR,
  fileFormat: 'CSV'
});
