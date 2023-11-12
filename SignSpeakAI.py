#!/usr/bin/env python
# coding: utf-8

# # Gesture Recognition using ASL

# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report, multilabel_confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
import seaborn as sns

import warnings 
warnings.filterwarnings('ignore')


# ### Loading Dataset

# In[2]:


train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')


# In[3]:


train.head()


# In[4]:


train.isnull()


# In[5]:


train.isna().sum()


# In[6]:


train_df_original = train.copy()

# Split into training, test and validation sets
val_index = int(train.shape[0]*0.2)

train_df = train_df_original.iloc[val_index:]
val_df = train_df_original.iloc[:val_index]


# In[7]:


y = np.array(train_df['label'])
X = np.array(train_df.drop(columns='label'))


# In[8]:


X.shape,y.shape


# In[9]:


import random
r = random.randint(0,(21964-1))
def show_img():
  arr = np.array(X)
  some_value = arr[r]
  some_img = some_value.reshape(28,28)
  plt.imshow(some_img, cmap="gray")
  plt.axis("off")
  plt.show()  

show_img()
print(y[r])


# In[10]:


y_train = pd.get_dummies(y)
y_train.head(5)


# In[11]:


y_val = val_df['label']
X_val = val_df.drop(columns="label",axis=1)
y_val = pd.get_dummies(y_val)


# In[12]:


y_train.shape


# In[13]:


X_val = pd.DataFrame(X_val).values.reshape(X_val.shape[0] ,28, 28, 1)
X_train = pd.DataFrame(X).values.reshape(X.shape[0] ,28, 28, 1)


# In[14]:


X_train.shape,y_train.shape


# In[15]:


generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=False,
    fill_mode="nearest"
)

X_train_flow = generator.flow(X_train, y_train, batch_size=32)

X_val_flow = generator.flow(X_val, y_val, batch_size=32)


# ### Generating Model 

# In[16]:


model = Sequential()

model.add(Conv2D(filters=32,  kernel_size=(3,3), activation="relu", input_shape=(28,28,1)))
model.add(MaxPool2D((2,2),padding='SAME'))
model.add(Dropout(rate=0.2))


model.add(Conv2D(filters=64,  kernel_size=(3,3), activation="relu", input_shape=(28,28,1)))
model.add(MaxPool2D((2,2),padding='SAME'))
model.add(Dropout(rate=0.2))


model.add(Conv2D(filters=521,  kernel_size=(3,3), activation="relu", input_shape=(28,28,1)))
model.add(MaxPool2D((2,2),padding='SAME'))
model.add(Dropout(rate=0.2))



model.add(Flatten())
model.add(Dense(units=521, activation="relu"))
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=24, activation="softmax"))


model.compile(loss="categorical_crossentropy", optimizer='adam',  metrics=["accuracy"])


# In[17]:


model.summary()


# In[18]:


learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001
)
history = model.fit(
    X_train_flow,
    validation_data=X_val_flow,
    # epochs=100,
    epochs=50,
    callbacks=[
               tf.keras.callbacks.EarlyStopping(
                   monitor='val_loss',
                   patience=5,
                   restore_best_weights=True
                   ),
      learning_rate_reduction
    ])


# In[19]:


fig, axes = plt.subplots(2, 1, figsize=(15, 10))
ax = axes.flat

pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot(ax=ax[0])
ax[0].set_title("Accuracy", fontsize = 15)
ax[0].set_ylim(0,1.1)

pd.DataFrame(history.history)[['loss','val_loss']].plot(ax=ax[1])
ax[1].set_title("Loss", fontsize = 15)
plt.show()


# In[20]:


y_test = np.array(test['label'])
X_test = np.array(test.drop(columns='label'))

y_test = pd.get_dummies(y_test)
X_test = pd.DataFrame(X_test).values.reshape(X_test.shape[0] ,28, 28, 1)

y_test = pd.get_dummies(y_test)


# In[21]:


from sklearn.metrics import classification_report
pred = model.predict(X_test)

y_pred = np.argmax(pred,axis=1)
y_test = np.argmax(y_test.values,axis=1)


# ### Accuracy Score

# In[22]:


acc = accuracy_score(y_test,y_pred)

print(f' {acc*100:.2f}% accuracy on the test set')


# ### Classification report and Confusion Matrix

# In[23]:


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
blush_palette = sns.color_palette(["#FFB6C1", "#FF69B4", "#FF1493", "#DB7093", "#C71585", "#FFC0CB", "#FFA07A", "#FF7F50", "#FF6347", "#FF4500", "#DC143C", "#FF0000", "#B22222", "#8B0000", "#800000", "#FFD700", "#FFA500", "#FF8C00", "#FF7F24", "#FF4500", "#FF0000", "#8B0000", "#800000"])

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=blush_palette, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:\n", classification_rep)


# ## The CNN Model gives an accuracy of 94%
