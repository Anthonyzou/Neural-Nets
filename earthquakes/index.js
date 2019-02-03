const brain = require('brain.js');
const fs = require('fs-extra');
const _ = require('lodash');
const math = require('mathjs');
var ProgressBar = require('progress');


const vorpal = require('vorpal')();

const net = new brain.recurrent.LSTMTimeStep({
  outputSize: 1,
  hiddenLayers: [4, 1],
  learningRate: 0.6
});

vorpal.command('quake', 'Classify').action(async (args, cb) => {
  const files = await fs.readdir('test');
  const bar = new ProgressBar(':bar :percent :eta :elapsed :rate', { total: files.length });

  let result = await Promise.all(
    files.map(async path => {
      const file = await fs.readFile('test/' + path);
      const mean = _(file)
        .split('\n')
        .drop(1)
        .map(parseFloat)
        .filter(_.isNumber)
        .chunk(1000)
        .map(chunk => net.run(chunk))
        .filter(a => a)
        .mean();
      bar.tick()
      return [path, mean];
    })
  );
  result =
    'seg_id,time_to_failure\n' +
    result
      .map(([path, mean]) => {
        return path.replace('.csv', '') + ',' + mean;
      })
      .join('\n');
  await fs.writeFile('submission.csv', result);
  cb()
});

vorpal.command('train', '').action(async (args, cb) => {
  // const files = await fs.readdir('train');
  // const bar = new ProgressBar(':bar :percent :eta :elapsed :rate', { total: files.length });

  // files.map()
  const results = _(await fs.readFile('train/aa', { encoding: 'UTF8' }))
    .split('\n')
    .map(line => {
      let [a, b] = line.split(',');
      // skip any empty lines
      if (line === '') {
        return null;
      }
      a = parseFloat(a);
      b = parseFloat(b);
      return [a, b];
    })
    .filter(a => a)
    .chunk(1000)
    .map(chunk => {
      const [a, b] = _.unzip(chunk);
      a.push(math.mean(b));
      return a;
    })
    .value();
  console.log(results)
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
