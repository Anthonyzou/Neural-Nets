const colors = require('./colors.json');
const { NeuralNetwork, layer } = require('brain.js');
const chalk = require('chalk');
const _ = require('lodash');

const vorpal = require('vorpal')();

const net = new NeuralNetwork({
  inputLayer: { width: 3 },
  hiddenLayers: [5],
});

vorpal.command('rgb [rgb...]').action((args, cb) => {
  const [r, g, b] = args.rgb;
  console.log(chalk.rgb(r, g, b)('#####'));
  console.log(net.run(args.rgb));
  cb();
});
vorpal.command('train').action((args, cb) => {
  const trainingData = colors.map(({ b, g, r, label }) => {
    return {
      input: [r / 255, g / 255, b / 255],
      output: { [label]: 1 },
    };
  });

  net.train(_.sampleSize(trainingData, 3000));
  cb();
});
vorpal.show('>');
