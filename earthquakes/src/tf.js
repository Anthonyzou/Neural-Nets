const fs = require('fs-extra');
const _ = require('lodash');
const ProgressBar = require('progress');
const tf = require('@tensorflow/tfjs-node');

class TF {
  constructor(){

    this.model = tf.sequential();
    this.model.add(tf.layers.lstm({
      inputShape: [null, 1],
      units: 1,
      activation: 'sigmoid',
      returnSequences: true
    }));
    this.model.add(tf.layers.dense({
      units:1,
    }));


    //compile
    const sgdoptimizer = tf.train.sgd(0.1);
    this.model.compile({
      optimizer: sgdoptimizer,
      loss: tf.losses.meanSquaredError,
    });

    this.trainDesc = 'Training function';
    this.predictDesc = 'Predict future data'
    this.saveDesc = 'Save the neural net for later reuse';
    this.loadDesc = 'Load an exisiting neural net';
  }

  async train(args){
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
      await this.model.fit(aa, bb)
    }


  }

  async predict(args){
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
          .map(chunk => this.model.predict(tf.tensor1d(chunk)))
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
  }

  async save(args) {
    await this.model.save(`file://${__dirname}/model`);
  }

  async load(args) {
    model = await tf.loadModel(`file:///${__dirname}/model/save.json`);
  }
}

exports.funcs = Object.getOwnPropertyNames(TF.prototype).filter(i => i !== 'constructor')
exports.model = new TF();