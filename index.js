const colors = require('./colors.json');
const { NeuralNetwork } = require('brain.js');
const chalk = require('chalk');
const fs = require('fs');
const _ = require('lodash');

const vorpal = require('vorpal')();

const net = new NeuralNetwork({
  inputLayer: { width: 3 },
  hiddenLayers: [4, 2],
});


vorpal.command('rgb [rgb...]').action((args, cb) => {
  const [r, g, b] = args.rgb;
  console.log(chalk.rgb(r, g, b).bgKeyword('white')('#####'));
  console.log(net.run(args.rgb));
  cb();
});

vorpal.command('train').action((args, cb) => {
  const trainingData = colors.map(({ b, g, r, label }) => ({
    input: [r / 255, g / 255, b / 255],
    output: { [label]: 1 },
  }));

  net.train(_.sampleSize(trainingData, 1000));
  cb();
});

vorpal.command('save').action((args, cb) => {
  fs.writeFile('./save.json', JSON.stringify(net.toJSON()), cb);
});

vorpal.command('load').action((args, cb) => {
  net.fromJSON(require('./save.json'));
  cb();
});

vorpal.show('>');
