import tensorflow as tf
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import numpy as np
import json

sentences = [
   "I can't believe you betrayed my trust like that!",  # Anger
   "Every creak in the house fills me with terror, imagining the worst.",  # Fear
   "The sight of her smile instantly brightens my day and fills my heart with happiness.",  # Joy
   "Being in your arms makes me feel complete, and I cherish every moment with you.",  # Love
   "The weight of loss sits heavy on my heart, and tears flow freely down my cheeks.",  # Sadness
   "As I opened the gift, a surge of excitement rushed through me. I never expected such a thoughtful gesture!",  # Surprise
   "im updating my blog because i feel shitty",
]

emotion_map =  {
   0: "anger",
   1: "fear",
   2: "joy",
   3: "love",
   4: "sadness",
   5: "surpise"
}


model = tf.keras.models.load_model("model/sentichan.h5")

with open("model/tokenizer.json") as f:
   tokenizer = tokenizer_from_json(json.load(f))

sen_seq = tokenizer.texts_to_sequences(list(v.lower() for v in sentences))
sen_pad = pad_sequences(sen_seq, padding='post', maxlen=40, truncating='post')

pred = model.predict(sen_pad)
for sen, s in zip(sentences, pred):
   print(">> ", sen)
   emotion_idx = np.argmax(s)
   emotion_map[np.argmax(s)]
   
   for idx, v in enumerate(s):
      print("{}: {:.5f}".format(emotion_map[idx], v))
   print()
