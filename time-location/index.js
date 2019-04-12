const brain = require('brain.js');
const fs = require('fs-extra');
const _ = require('lodash');
const moment = require('moment');


const vorpal = require('vorpal')();

const net = new brain.NeuralNetwork({
  hiddenLayers: [128,64],
});

function round(value, precision) {
  var multiplier = Math.pow(10, precision || 0);
  return Math.round(value * multiplier) / multiplier;
}

vorpal.command('predict', 'Classify').action(async (args, cb) => {
  const mmtMidnight = moment().startOf('day');
  const diffMinutes = time.diff(mmtMidnight, 'minutes');
  [world, place] = net.run(diffMinutes)
  switch(round(place, 1)){
    case 0.1:
      place = 'p'
      break;
    case 0.2:
      place = 'w'
      break;
    case 0.3:
      place = 'i'
      break;
    case 0.4:
      place = 'm'
      break;
    case 0.5:
      place = 's'
      break;
  }
  cb([world * 141, place])
});

vorpal.command('train', '').action(async (args, cb) => {
  // const files = await fs.readdir('train');
  // const bar = new ProgressBar(':bar :percent :eta :elapsed :rate', { total: files.length });

  // files.map()
  let results = _.valuesIn(require('./dht.json'))
  results = _.chain(results)
    .filter(({t,m}) => {
      const world = parseInt(m);
      return (m.length < 5 && !m.includes('d') && !_.isNaN(world))
    })
    .map(({t,m}) => {
      var time = moment(t)
      var mmtMidnight = time.clone().startOf('day');
      var diffMinutes = time.diff(mmtMidnight, 'minutes');

      const world = parseInt(m)
      let place;
      if(m.includes('p')){
        place = 0.1
      }
      if(m.includes('w')){
        place = 0.2
      }
      if(m.includes('i')){
        place = 0.3
      }
      if(m.includes('m')){
        place = 0.4
      }
      if(m.includes('s')){
        place = 0.5
      }
      return {time: diffMinutes / 1440, world: world / 141, place}
    })
    .filter(({place, world}) => {
      return (_.isNumber(place)
        && _.isNumber(world)
        && !_.isNull(place)
        && !_.isNull(world)
        && !_.isUndefined(place)
        && !_.isUndefined(world))
    })
    .map(({time, world, place}) => {
      return {input: time, output: [world, place]}
    })
    .value();
  await fs.writeFile('./thing.json', JSON.stringify(results, null, 2))
  net.train(results);
  cb();
});

vorpal
  .command('save', 'Save the neural net for later reuse')
  .action(async (args, cb) => {
    await fs.writeFile('./save.json', JSON.stringify(net.toJSON(), null, 2));
    cb();
  });

vorpal.command('load', 'Load an exisiting neural net').action((args, cb) => {
  net.fromJSON(require('./save.json'));
  cb();
});

vorpal.show('>');