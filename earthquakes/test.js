const fs = require('fs-extra');
const _ = require('lodash');
const tf = require('@tensorflow/tfjs-node-gpu');

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

fs.readFile('train/aa', { encoding: 'UTF8' })
.then(async results => {
  results
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

  for(var chunk of results) {
    const [a,b] = _.unzip(chunk);
    const aa = tf.tensor(a, [1000, 1, 1])
    aa.print()
    const bb = tf.tensor(b, [1000, 1])
    await model.fit(aa, bb)
  }
})
