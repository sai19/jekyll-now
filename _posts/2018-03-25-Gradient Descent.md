---
layout: post
title: Problem with the gradient descent
---
In this post we will analyse one important property of gradient descent optimization. Deep neural networks have been successful
on various machine learning tasks such as classification, object recognition. But the recent studies have found one of the crucial shortcomings of deep learning, i.e. adversial attacks on deep neural networks. 

One of the problems with gradient descent is, it tries to find the easiest possible solution for the given task. To see this, let us do a simple experiment with MNIST dataset. We will add some easy hints to training data but will exclude those hints from test data 
![alt text](https://github.com/sai19/sai19.github.io/blob/master/images/img_0.jpg "Logo Title Text 1")

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
Now let us have a look at the training loss profile, As expected the training and validation accuracy both reach 1.0 just after 
1 epoch. But the performance on the test data(remember we did not add any hints) is,
logloss : 0.8712
accuracy: 0.7264
This clearly indicates that, the network has learned to locate the position of the boxes(i.e. hints) in the image really well, as that is an easier task. But the network has failed to learn the other information, which can also be exploited to 
make the predictions. This is what makes the deep neural networks succeptible to adverserial attacks.
But how can we make the network pay attention to other useful information as well?
Well, there are two ways to make this happen:
1. Modify gradient descent
2. Modify the objective function

I currently can not think of a way to modify gradient descent, so let us focus on the second option. Instead of asking the network to classify the images, let us also ask the network to reconstruct the images as well. This should make the network pay attention to other information as well. The code is shown below,
```
input_img = Input(shape=(784,))
encoded = Dense(512, activation='relu')(input_img)
encoded = Dropout(0.2)(encoded)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dropout(0.2)(encoded)
decoded = Dense(784, activation='sigmoid')(encoded)
out = Dense(10,activation="softmax",name="final")(encoded)
model = Model(input_img, [decoded,out])
model.summary()
model.compile(optimizer='rmsprop', 
              loss=['binary_crossentropy', 'categorical_crossentropy'],
              loss_weights=[1.0, 0.1],metrics=["accuracy"])
history = model.fit(x_train, [x_train,y_train],
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_split=0.2)
score = model.evaluate(x_test, [x_test,y_test], verbose=0)

```
We can use the loass weights to control the importance given to each loss. Unlike in previous case, the network fails to
achieve an accuracy of 100%, but the performance on the test data is slightly better(both the networks were trained for 20 epochs)  
Performance on test data(weights = [1.0,0.1]):  
logloss: 0.775  
accuracy: 0.756  
Performance on test data(weights = [1.0,0.01]):  
logloss: 0.7714  
accuracy: 0.7805  
Performance on test data(weights = [1.0,0.5]):  
logloss: 0.7747  
accuracy: 0.8105  

