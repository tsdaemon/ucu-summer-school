from keras.models import Model, Sequential
from keras.layers import *
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle as pc
import pandas as pd

df = pd.read_csv('/input/quora-train.csv')
x = df[['question1', 'question2']]
y = df['is_duplicate']
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_indexes, test_indexes = next(sss.split(x, y))
train = df.iloc[train_indexes]
test = df.iloc[test_indexes]

train_q1_sequences = pc.load(open('/input/train_q1_sequences.pickle', 'wb'))
train_q2_sequences = pc.load(open('/input/train_q2_sequences.pickle', 'wb'))
test_q1_sequences = pc.load(open('/input/test_q1_sequences.pickle', 'wb'))
test_q2_sequences = pc.load(open('/input/test_q2_sequences.pickle', 'wb'))

model = FastTextKeyedVectors.load_word2vec_format('model/fasttext/quora.vec')
all_words = set(model.vocab.keys())
int_vocab = {word:i for i,word in enumerate(all_words)}

embedding_layer = model.get_embedding_layer()
lstm_layer = Bidirectional(LSTM(300,dropout=0.332,recurrent_dropout=0.2))

sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences_2 = embedding_layer(sequence_2_input)
x2 = lstm_layer(embedded_sequences_2)

wmd_input = Input(shape=(1, ))

merged = concatenate([x1, x2, wmd_input])
merged = Dropout(0.4)(merged)
merged = BatchNormalization()(merged)

merged = Dense(130, activation='relu')(merged)
merged = Dropout(0.075)(merged)
merged = BatchNormalization()(merged)

output = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[sequence_1_input, sequence_2_input, wmd_input],outputs=output)
model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['accuracy'])

model_checkpoint_path = '/output/fold-checkpoint.h5'

X_train_q1 = np.vstack([train_q1_sequences, train_q2_sequences])
X_train_q2 = np.vstack([train_q2_sequences, train_q1_sequences])
X_train_wmd = np.concatenate([np.array(train['wmd']), np.array(train['wmd'])])

X_val_q1 = np.vstack([test_q1_sequences, test_q2_sequences])
X_val_q2 = np.vstack([test_q2_sequences, test_q1_sequences])
X_test_wmd = np.concatenate([np.array(test['wmd']), np.array(test['wmd'])])

y_train = np.concatenate([train['is_duplicate'], train['is_duplicate']])
y_val = np.concatenate([test['is_duplicate'], test['is_duplicate']])

# Train.
model.fit([X_train_q1, X_train_q2, X_train_wmd], y_train,
          validation_data=([X_val_q1, X_val_q2, X_test_wmd], y_val),
        batch_size=128,
        epochs=MAX_EPOCHS,
        verbose=1,

        callbacks=[
        # Stop training when the validation loss stops improving.
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=3,
            verbose=1,
            mode='auto',
        ),
        # Save the weights of the best epoch.
        ModelCheckpoint(
            model_checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=2,
        ),
        ],
)