const fs = require('fs-extra');
const _ = require('lodash');
const ProgressBar = require('progress');
const tf = require('@tensorflow/tfjs-node');

const vorpal = require('vorpal')();

let model = tf.sequential();
model.add(tf.layers.lstm({
  inputShape: [null, 1],
  units: 1,
  activation: 'sigmoid',
  returnSequences: true
}));
model.add(tf.layers.dense({
  units:1,
}));


//compile
const sgdoptimizer = tf.train.sgd(0.1);
model.compile({
  optimizer: sgdoptimizer,
  loss: tf.losses.meanSquaredError,
});

vorpal.command('train', '').action(async (args, cb) => {
  const results = _(await fs.readFile('train/aa', { encoding: 'UTF8' }))
    .split('\n')
    .map(line => {
      // skip any empty lines
      if (line === '') {
        return null;
      }
      let [a, b] = line.split(',');
      a = parseFloat(a);
      b = parseFloat(b);
      if (a == NaN || b == NaN) {
        return null;
      }
      return [a, b];
    })
    .filter(a => a)
    .chunk(1000)
    .value();
    for(var chunk of results){
      const [a,b] = _.unzip(chunk);
      const aa = tf.tensor(a, [ 1000,1, 1])
      aa.print()
      const bb = tf.tensor(b, [1000, 1])
      await model.fit(aa, bb)
    }


  cb();
});

vorpal.command('predict', 'Classify').action(async (args, cb) => {
  const files = await fs.readdir('test');
  const bar = new ProgressBar(':bar :percent :eta :elapsed :rate', {
    total: files.length,
  });

  let result = await Promise.all(
    files.map(async path => {
      const file = await fs.readFile('test/' + path);
      const mean = _(file)
        .split('\n')
        .drop(1)
        .map(parseFloat)
        .filter(_.isNumber)
        .chunk(1000)
        .map(chunk => model.predict(tf.tensor1d(chunk)))
        .filter(a => a)
        .mean();
      bar.tick();
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
  cb();
});

const save = async () => {
  return fs.writeFile('./save.json', JSON.stringify(net.toJSON(), null, 2));
}
vorpal
  .command('save', 'Save the neural net for later reuse')
  .action(async (args, cb) => {
    await model.save(`file://${__dirname}/model`);
    cb();
  });

vorpal.command('load', 'Load an exisiting neural net').action(async (args, cb) => {
  model = await tf.loadModel(`file:///${__dirname}/model/save.json`);
  cb();
});

vorpal.show('>');
