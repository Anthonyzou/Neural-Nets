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


vorpal.command('rgb [rgb...]', 'Classify a new color').action((args, cb) => {
  const [r, g, b] = args.rgb;
  console.log(chalk.rgb(r, g, b).bgKeyword('white')('#####'));
  console.log(net.run(args.rgb));
  cb();
});

vorpal.command('train', 'Train a new neural net based of color file').action((args, cb) => {
  const trainingData = colors.map(({ b, g, r, label }) => ({
    input: [r / 255, g / 255, b / 255],
    output: { [label]: 1 },
  }));

  net.train(_.sampleSize(trainingData, 1000));
  cb();
});

vorpal.command('save', 'Save the neural net for later reuse').action((args, cb) => {
  fs.writeFile('./save.json', JSON.stringify(net.toJSON(), null, 2), cb);
});

vorpal.command('load', 'Load an exisiting neural net').action((args, cb) => {
  net.fromJSON(require('./save.json'));
  cb();
});

vorpal.show('>');
