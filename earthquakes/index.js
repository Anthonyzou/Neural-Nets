

const vorpal = require('vorpal')();
['./src/tf'].map(file => {
  const { funcs, model } = require(file);
  funcs.map(funcName => {
    vorpal.command(funcName, model[`${funcName}Desc`]).action(async (args, cb) => {
      delete require.cache[require.resolve(file)];
      const { model } = require(file);
      await model[funcName](args);
      cb();
    });
  })
})
vorpal.show('>');
