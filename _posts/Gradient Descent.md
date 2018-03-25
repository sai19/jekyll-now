---
layout: post
title: Problems with gradient descent
---
In this post we will analyse one important property of gradient descent optimization. Deep neural networks have been successful
on various of machine learning tasks such as classification, object recognition. But the recent studies have found one of the crucial shortcomings of deep learning, i.e. adversial attacks on deep neural networks. 

One of the problems with gradient descent is, it tries to find the easiest possible solution for the given task. To see this, let us do a simple experiment with MNIST dataset. We will add some easy hints to training data but will exclude those hints from test data 
![alt text](https://github.com/sai19/sai19.github.io/blob/master/images/img_0.jpg)

Now let us train the above images using a simple neural network, the keras version of the code is shown below,
```
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=0)

```


