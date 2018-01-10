# **ImgRec Keras**

## **Run**

#### **Train part**

- If you want to **train from a existing model**, please give the model path.

Example:

> python3 ImgRec_Keras.py train --model /home/binzhang/BinZ/Image_Recog/model/my_model.h5

The model path need to follow the `--model` command

Output:

``` 
Using TensorFlow backend.
Command:  train
Model:  /home/binzhang/BinZ/Image_Recog/model/my_model.h5
Generating data...

--------------------------------------------------
Iteration 1
Train on 45000 samples, validate on 5000 samples
Epoch 1/1
45000/45000 [=============================>] - 4s 78us/step - loss: 0.0376 - acc: 0.9893 - val_loss: 0.0032 - val_acc: 0.9993
```

- Also you can **train without a existing model**.

Example:

> python3 ImgRec_Keras.py train

Please note that the model will be saved every 10 iterations

#### **Evaluate part**

Example:

> python3 ImgRec_Keras.py evaluate --model /home/binzhang/BinZ/Image_Recog/model/my_model.h5

The model path need to follow the `--model` command

Output:

```
Using TensorFlow backend.
Command:  evaluate
Model:  /home/binzhang/BinZ/Image_Recog/model/my_model.h5
Generating data...
Total addition questions: 5000
Vectorization...
Training Data:
(4500, 7, 12)
(4500, 4, 12)
Validation Data:
(500, 7, 12)
(500, 4, 12)
4500/4500 [==============================] - 0s 50us/step
The loss is:  0.0034373
The accuracy is:  99.92222%
```

#### **Test part** - Not finish now

Please note that you need to give the model path to test.

Example:

> python3 ImgRec_Keras.py test --model /home/binzhang/BinZ/Image_Recog/model/my_model.h5

The model path need to follow the `--model` command

Output:

```
Using TensorFlow backend.
Command:  test
Model:  /home/binzhang/BinZ/Image_Recog/model/my_model.h5
There is a loop of addition test, if you want to end, please press CTRL + C


```

Press `CTRL+C` can exit.


#### **Model visualization** - Not finish now

Example:

> python3 ImgRec_Keras.py plot --model /home/binzhang/BinZ/Image_Recog/model/my_model.h5

Output:

`model.png` file in current directory.

![model](/home/binzhang/BinZ/Image_Recog/model.png) 