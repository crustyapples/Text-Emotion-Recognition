import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

df = pd.read_csv('text_emotion.csv')
df.rename(columns={'sentiment' : 'label',
                   'content' : 'text'}, 
                   inplace=True)

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

X_train_, X_test, y_train_, y_test = train_test_split(df.index.values, df.label.values, test_size=0.10, random_state=42, stratify=df.label.values)
X_train, X_val, y_train, y_val = train_test_split(df.loc[X_train_].index.values, df.loc[X_train_].label.values, test_size=0.10, random_state=42, stratify=df.loc[X_train_].label.values)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

df['data_type'] = ['not_set']*df.shape[0]
df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'
df.loc[X_test, 'data_type'] = 'test'

df_train = df.loc[df["data_type"]=="train"]
df_val = df.loc[df["data_type"]=="val"]
df_test = df.loc[df["data_type"]=="test"]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def encode_text(text, max_len=128):
    return tokenizer(text, truncation=True, padding='max_length', max_length=max_len, return_tensors='tf')

df_train_encoded = df_train['text'].apply(encode_text)
df_val_encoded = df_val['text'].apply(encode_text)
df_test_encoded = df_test['text'].apply(encode_text)

def tf_dataset(text_encoded, labels):
    input_ids = [x['input_ids'].numpy()[0] for x in text_encoded]
    attention_mask = [x['attention_mask'].numpy()[0] for x in text_encoded]
    return tf.data.Dataset.from_tensor_slices(((input_ids, attention_mask), labels))

train_ds = tf_dataset(df_train_encoded, df_train.label.values)
val_ds = tf_dataset(df_val_encoded, df_val.label.values)
test_ds = tf_dataset(df_test_encoded, df_test.label.values)

batch_size = 16
train_ds = train_ds.shuffle(len(df_train)).batch(batch_size=batch_size, drop_remainder=False)
val_ds = val_ds.shuffle(len(df_val)).batch(batch_size=batch_size, drop_remainder=False)
test_ds = test_ds.shuffle(len(df_test)).batch(batch_size=batch_size, drop_remainder=False)

def CNN_model():
    input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids') # Assuming a max length of 128
    attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')

    # BERT base uncased encoder
    bert_encoder = TFBertModel.from_pretrained('bert-base-uncased')
    outputs = bert_encoder(input_ids, attention_mask=attention_mask)
    network_layer = outputs[0] 

    # CNN layers
    network_layer = tf.keras.layers.Conv1D(32, (2), activation='relu')(network_layer)
    network_layer = tf.keras.layers.Conv1D(64, (2), activation='relu')(network_layer)
    network_layer = tf.keras.layers.GlobalMaxPool1D()(network_layer) 
    network_layer = tf.keras.layers.Dense(256, activation="relu")(network_layer) 
    network_layer = tf.keras.layers.Dropout(0.5)(network_layer) 
    network_layer = tf.keras.layers.Dense(13, activation="softmax", name='classifier')(network_layer) 

    return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=network_layer)

cnn_model = CNN_model()

from official.nlp import optimization

epochs_count = 10
steps_each_epoch = tf.data.experimental.cardinality(train_ds).numpy()
number_of_training_steps = steps_each_epoch * epochs_count
number_of_warmup_steps = int(0.1*number_of_training_steps)
init_lr = 4e-5

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=number_of_training_steps,
                                          num_warmup_steps=number_of_warmup_steps,
                                          optimizer_type='adamw')

cnn_model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                          metrics=tf.keras.metrics.SparseCategoricalAccuracy('accuracy'))

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print(f'Training BERT CNN model...')
cnn_history = cnn_model.fit(x=train_ds,
                            validation_data=val_ds,
                            epochs=epochs_count,
                            class_weight=class_weight_dict,
                            callbacks=[early_stop])


# Predicting on test dataset
predictions = cnn_model.predict(test_ds)
y_pred = np.argmax(predictions, axis=-1)

print(classification_report(y_test, y_pred))