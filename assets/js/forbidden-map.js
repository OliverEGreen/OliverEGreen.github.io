// The Forbidden Places map. Coordinates gathered through years of quiet fieldwork.
(function () {
  var el = document.getElementById('forbidden-map');
  if (!el || typeof L === 'undefined') return;

  var places = [
    ['Balls Pond Road', 51.5468, -0.0810],
    ['Chorlton-cum-Hardy', 53.4419, -2.2766],
    ['Cockermouth', 54.6613, -3.3626],
    ['Cockfosters', 51.6517, -0.1497],
    ['Fingringhoe', 51.8437, 0.9560],
    ['Fucking, Austria', 48.0672, 12.8633],
    ['Idaho', 44.0682, -114.7420],
    ['Isle of Man', 54.2361, -4.5481],
    ['Lake Titicaca', -15.9254, -69.3354],
    ['Lower Bush', 51.3770, 0.4350],
    ['Maidenhead', 51.5217, -0.7177],
    ['Manchester', 53.4808, -2.2426],
    ['Nuneaton', 52.5230, -1.4680],
    ['Nunhead', 51.4670, -0.0530],
    ['Penistone', 53.5250, -1.6290],
    ['Scunthorpe', 53.5809, -0.6502],
    ['Shingay-cum-Wendy', 52.1250, -0.0700],
    ['Wilsford-cum-Lake', 51.1520, -1.8080]
  ];

  var map = L.map(el, { scrollWheelZoom: false });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);

  var pins = L.featureGroup(places.map(function (p) {
    return L.marker([p[1], p[2]]).bindPopup(p[0]);
  })).addTo(map);
  map.fitBounds(pins.getBounds(), { padding: [30, 30] });
})();
