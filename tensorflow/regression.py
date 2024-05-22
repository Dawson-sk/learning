import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

#Get data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
dataset.tail()

#Clean data
dataset.isna().sum()
dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({1:'USA', 2:'Europe', 3:'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()

#Split data into training and test sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Inspect data
#sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')

#Check overall stats
train_dataset.describe().transpose()

#Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

#Normalize
train_dataset.describe().transpose()[['mean', 'std']]

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
    print('First Example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())
    
horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

hp_model = tf.keras.Sequential([
    horsepower_normalizer,
    tf.keras.layers.Dense(units=1)
])

# hp_model.summary()
# print(hp_model.predict(horsepower[:10]))

hp_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = hp_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)

test_results = {}
test_results['hp_model'] = hp_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)
  
x = tf.linspace(0.0, 250, 251)
y = hp_model.predict(x)

def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()
  plt.show()

#Linear regression with multiple inputs. vvv 
linear_model = tf.keras.Sequential([
   normalizer,
   tf.keras.layers.Dense(units=1)
])

linear_model.predict(train_features[:10])

print(linear_model.layers[1].kernel)

linear_model.compile(
   optimizer=tf.keras.optimizers.Adam(leanring_rate=0.1),
   loss='mean_absolute_error'
)

history = linear_model.fit(
   train_features,
   train_labels,
   epochs=100,
   verbose=0,
   validation_split=0.2)

plot_loss(history)

test_results['linear_model'] = linear_model.evaluate(
   test_features, test_labels, verbose=0)
#Linear regression with multiple inputs. ^^^

#Regression with a deep nueral network. vvv
def build_and_compile(norm):
   model = tf.keras.Sequential([
      norm,
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(1)
   ])

   model.compile(loss='mean_absolute_error',
                 optimizer=tf.keras.optimizers.Adam(0.001))
   
   return model

