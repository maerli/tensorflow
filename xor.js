
const model = tf.sequential();

model.add(tf.layers.dense({units: 2, inputShape: [2],activation:'sigmoid'}));
model.add(tf.layers.dense({units: 1,activation:'sigmoid'}));

const learning_rate = 0.1;
const optimizer = tf.train.adam(learning_rate);
model.compile({optimizer,loss: 'meanSquaredError'});

const xs_train = tf.tensor2d([
  [0,0],
  [0,1],
  [1,0],
  [1,1]
]);

const ys_train = tf.tensor2d([
  [0],
  [1],
  [1],
  [0]
]);
tf.tidy(() =>{
model.fit(xs_train, ys_train,{epochs:100,shuffle:true}).then(() => {  
  model.predict(tf.tensor2d([[0,0]])).print();
});
});
