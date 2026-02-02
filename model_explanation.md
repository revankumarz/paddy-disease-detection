# Paddy Disease Detection - Model & Training Explanation

Detailed explanation of the notebook from Cell 16 onwards with examples.

---

## Cell 16: Building the Model

### Transfer Learning - The Big Idea

Think of it like this: you want to teach someone to identify paddy diseases from leaf photos.

**Option A (from scratch):** Take a newborn baby, teach them to see, recognize colors, shapes, edges, textures, leaves, spots... then finally teach them disease patterns. Needs millions of examples.

**Option B (transfer learning):** Take an experienced photographer who already knows what edges, textures, shapes, and colors look like. Just teach them "this spot pattern = blast, this yellowing = tungro." Needs much less data.

Cell 16 does **Option B**.

---

### Line by line:

```python
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

**EfficientNetB0** already trained on ImageNet (1.4 million photos of dogs, cars, food, etc.) and learned to see:

```
Layer 1-5:    edges, corners        ─┐
Layer 6-15:   textures, gradients    │  These skills transfer
Layer 16-30:  shapes, parts          │  to ANY image task
Layer 31+:    complex patterns      ─┘
```

**`include_top=False`** removes the last layer that said:

```
Before: "This is 87% dog, 5% cat, 3% car..."  ← useless for us
After:  raw feature map (7, 7, 1280)           ← useful visual features
```

So for a paddy leaf image, EfficientNet outputs a feature map like:

```
Input image: 224 x 224 x 3  (a leaf photo, 150,528 values)
        ↓
EfficientNetB0 processes it
        ↓
Output: 7 x 7 x 1280  (62,720 values encoding what it "sees")
```

Those 1280 channels might represent things like:

```
Channel 23:  "amount of brown color detected"
Channel 156: "circular spot patterns"
Channel 891: "edge sharpness of lesions"
Channel 45:  "yellow discoloration intensity"
... and 1276 other learned features
```

---

```python
base_model.trainable = False
```

**Freezing** = locking all EfficientNet weights. Example:

```
Layer "edge_detector": weight = 0.73   → LOCKED, won't change
Layer "texture_finder": weight = -0.41 → LOCKED, won't change
Layer "shape_recognizer": weight = 1.2 → LOCKED, won't change
```

Why? These weights took millions of images to learn. Our 8K images would mess them up.

---

```python
layers.GlobalAveragePooling2D(),
```

The EfficientNet output is a 7x7 grid for each of the 1280 features. This layer averages each grid into one number:

```
Feature channel 23 ("brown color"):

Grid:                          Average:
[0.1, 0.3, 0.8, 0.9, 0.7, 0.2, 0.1]
[0.2, 0.5, 0.9, 0.9, 0.8, 0.3, 0.1]
[0.1, 0.4, 0.7, 0.8, 0.6, 0.2, 0.0]    →  0.45
[0.0, 0.2, 0.5, 0.6, 0.4, 0.1, 0.0]
[0.0, 0.1, 0.3, 0.4, 0.3, 0.1, 0.0]
[0.0, 0.0, 0.1, 0.2, 0.1, 0.0, 0.0]
[0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]

Do this for all 1280 channels:
(7, 7, 1280) → (1280,)

Result: [0.45, 0.12, 0.88, 0.03, ..., 0.67]  ← 1280 numbers
```

Each number now summarizes "how much of this feature is in the whole image."

---

```python
layers.BatchNormalization(),
```

The 1280 values might have wildly different scales:

```
Before: [0.45, 87.3, -0.002, 15.6, ...]  ← messy, hard to learn from

After:  [0.12, 0.95, -0.87, 0.34, ...]   ← normalized, mean≈0, std≈1
```

Like converting currencies to one standard before comparing prices.

---

```python
layers.Dropout(0.3),
```

During each training step, randomly kills 30% of neurons:

```
Step 1: [0.12, DEAD, -0.87, 0.34, DEAD, 0.56, 0.78, DEAD, ...]
Step 2: [DEAD, 0.95, -0.87, DEAD, 0.23, 0.56, DEAD, 0.91, ...]
Step 3: [0.12, 0.95, DEAD, 0.34, 0.23, DEAD, 0.78, 0.91, ...]
```

This is like training a football team where random players sit out each practice. Every player must learn to perform, no one can depend on one star player. Prevents the model from relying on any single feature.

---

```python
layers.Dense(256, activation='relu'),
```

**Dense = every input connects to every output.**

```
1280 inputs x 256 outputs = 327,680 connections (weights)

Input (1280 values):   [0.12, 0.95, -0.87, 0.34, ...]
                         ↓↘↓↘↓↘↓↘
                       multiply by weights, add up
                         ↓↘↓↘↓↘↓↘
Output (256 values):   [2.3, -1.1, 0.7, ...]
```

**ReLU activation** then applies: `if negative → make it 0`:

```
Before ReLU: [ 2.3, -1.1,  0.7, -0.4,  1.8, -2.3]
After ReLU:  [ 2.3,  0.0,  0.7,  0.0,  1.8,  0.0]
```

This layer combines the 1280 EfficientNet features into 256 new features that are more specific to paddy diseases.

---

```python
layers.BatchNormalization(),
layers.Dropout(0.3),
```

Same as before -- normalize, then randomly drop 30%. Applied again to the 256 neurons for the same reasons.

---

```python
layers.Dense(num_classes, activation='softmax')
```

256 inputs → 10 outputs (one per disease):

```
Neuron 0 → bacterial_leaf_blight
Neuron 1 → bacterial_leaf_streak
Neuron 2 → bacterial_panicle_blight
Neuron 3 → blast
Neuron 4 → brown_spot
Neuron 5 → dead_heart
Neuron 6 → downy_mildew
Neuron 7 → hispa
Neuron 8 → normal
Neuron 9 → tungro
```

**`activation='softmax'`** converts raw numbers into **probabilities** that sum to 1.0:

```
256 inputs: [2.3, 0.0, 0.7, 0.0, 1.8, ...]
                    ↓
            multiply by weights, add up
                    ↓
Raw scores: [1.2, 0.3, 0.1, 4.5, 0.8, 0.5, 0.2, 0.7, 0.4, 0.3]
                    ↓
              softmax (convert to probabilities)
                    ↓
Probabilities: [0.04, 0.02, 0.01, 0.78, 0.03, 0.02, 0.01, 0.03, 0.02, 0.04]
                                    ↑
                              blast = 78%     ← MODEL'S ANSWER
```

Softmax formula example for the "blast" score of 4.5:

```
e^4.5 / (e^1.2 + e^0.3 + e^0.1 + e^4.5 + e^0.8 + ...) = 90.0 / 115.4 = 0.78
```

Higher raw score → higher probability. All 10 probabilities add to 1.0.

---

### Compile

```python
optimizer=keras.optimizers.Adam(learning_rate=0.001)
```

**Adam** decides how to update each weight. Example:

```
Current weight: 0.500
Gradient says:  "increase this weight to reduce error"
Adam calculates: step = 0.001 x (smart adjustment)
New weight:     0.503
```

It's "smart" because it adapts per weight -- weights that need big changes get bigger steps, weights near their optimal value get tiny steps.

---

```python
loss='categorical_crossentropy'
```

Measures how wrong the prediction is. Example:

```
True label:     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  ← blast
                         ↕
Prediction A:   [.02,.01,.01,.85,.03,.02,.01,.02,.02,.01]  ← confident & correct
Loss A = -log(0.85) = 0.16   ← LOW loss, good!

Prediction B:   [.10,.10,.10,.10,.10,.10,.10,.10,.10,.10]  ← confused
Loss B = -log(0.10) = 2.30   ← HIGH loss, bad!

Prediction C:   [.50,.01,.01,.02,.01,.01,.01,.01,.01,.41]  ← wrong answer
Loss C = -log(0.02) = 3.91   ← VERY HIGH loss, very bad!
```

Training works to make this loss number as small as possible.

---

## Cell 17: Custom CNN Alternative (Not Used)

This defines a simpler model from scratch. **Not executed** (it's commented out at the bottom). It's there for comparison.

```python
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3))
```

A convolutional layer slides 32 small **3x3 filters** across the image. Each filter detects a specific pattern (horizontal edge, vertical edge, diagonal, etc.). The output is 32 **feature maps** -- one per filter, each showing where that pattern appears in the image.

```python
layers.MaxPooling2D((2, 2))
```

Shrinks each feature map by half. Takes every 2x2 block and keeps only the maximum value. This reduces computation and makes the model tolerant to small shifts in position.

The four blocks progressively learn more abstract features:

```
Block 1 (32 filters)  → edges, corners
Block 2 (64 filters)  → textures, simple shapes
Block 3 (128 filters) → parts of objects (spots, lesions)
Block 4 (256 filters) → complex structures (disease patterns)
```

This would be much weaker than EfficientNet because it learns from only 8K images instead of 1.4 million.

---

## Cell 19: Training Callbacks

Think of callbacks as **automatic assistants** watching the training.

### EarlyStopping -- the "stop wasting time" assistant

```
Epoch 1:  val_loss = 2.10
Epoch 2:  val_loss = 1.80  ← improved
Epoch 3:  val_loss = 1.50  ← improved
Epoch 4:  val_loss = 1.45  ← improved
Epoch 5:  val_loss = 1.46  ← worse (patience count: 1/5)
Epoch 6:  val_loss = 1.48  ← worse (patience count: 2/5)
Epoch 7:  val_loss = 1.44  ← improved! Reset patience
Epoch 8:  val_loss = 1.50  ← worse (1/5)
Epoch 9:  val_loss = 1.52  ← worse (2/5)
Epoch 10: val_loss = 1.55  ← worse (3/5)
Epoch 11: val_loss = 1.58  ← worse (4/5)
Epoch 12: val_loss = 1.60  ← worse (5/5) → STOP! Restore epoch 7 weights.
```

### ModelCheckpoint -- the "save your progress" assistant

```
Epoch 1: val_accuracy = 55% → saved! (best so far)
Epoch 2: val_accuracy = 68% → saved! (new best)
Epoch 3: val_accuracy = 72% → saved! (new best)
Epoch 4: val_accuracy = 70% → skipped (not best)
Epoch 5: val_accuracy = 75% → saved! (new best)
```

### ReduceLROnPlateau -- the "take smaller steps" assistant

```
Epoch 1-5:  val_loss improving, LR = 0.001
Epoch 6:    val_loss stuck (1/3)
Epoch 7:    val_loss stuck (2/3)
Epoch 8:    val_loss stuck (3/3) → LR = 0.0005 (halved!)
Epoch 9-11: val_loss improving again with smaller steps
Epoch 12:   val_loss stuck (1/3)
Epoch 13:   val_loss stuck (2/3)
Epoch 14:   val_loss stuck (3/3) → LR = 0.00025 (halved again!)
```

---

## Cell 20: Phase 1 Training (Frozen Base)

```python
history = model.fit(train_generator, epochs=30, ...)
```

What happens in ONE epoch (all 8,330 images):

```
Batch 1:   images[0-31]      → predict → loss=2.3 → update weights
Batch 2:   images[32-63]     → predict → loss=2.1 → update weights
Batch 3:   images[64-95]     → predict → loss=1.9 → update weights
...
Batch 260: images[8288-8329] → predict → loss=0.8 → update weights

Then validate (no weight updates):
Val batch 1:  images[0-31]      → predict → record accuracy
Val batch 2:  images[32-63]     → predict → record accuracy
...
Val batch 65: images[2048-2077] → predict → record accuracy

Result: "Epoch 1 - loss: 1.45 - accuracy: 0.52 - val_loss: 1.30 - val_accuracy: 0.58"
```

Only **333,578 weights** are being updated (the custom layers). EfficientNet's 4 million weights are frozen.

---

## Cell 21: Phase 2 Fine-Tuning

```python
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
```

Now we unlock the top 20 EfficientNet layers:

```
Layers 1-215:   FROZEN   (basic edges, textures - universally useful)
Layers 216-236: UNLOCKED (high-level features - adapt to paddy diseases)
Custom layers:  UNLOCKED (already training)
```

Example of what fine-tuning changes:

```
Before fine-tuning:
  Channel 156 detects: "generic circular patterns" (from ImageNet)

After fine-tuning:
  Channel 156 detects: "brown circular lesions on green leaves" (paddy-specific)
```

```python
optimizer=keras.optimizers.Adam(learning_rate=0.0001)  # 10x smaller
```

Why 10x smaller? Example:

```
EfficientNet weight currently = 0.7342 (learned from 1.4M images)

With LR 0.001:  new weight = 0.7342 + 0.015 = 0.7492  ← too aggressive
With LR 0.0001: new weight = 0.7342 + 0.0015 = 0.7357 ← gentle nudge
```

---

## Cell 22: Save Final Model

```python
model.save(os.path.join(MODEL_DIR, 'final_model.keras'))
```

Saves the complete model after fine-tuning. You now have two saved models:

- `best_model.keras` -- the one with highest validation accuracy during training
- `final_model.keras` -- the one after all training is done (might be slightly worse if last epochs overfit)

---

## Cell 24: Training Visualization

Plots two charts from the stored training history.

### What good training looks like

```
Accuracy:                    Loss:
1.0|          ___──────     3.0|──╮
   |       __/                 |   ╲
   |     _/    ← both rise     |    ╲__    ← both drop
   |   _/       together       |       ╲___
   |  /                        |           ╲───────
0.0|/______________________  0.0|______________________
   0    10    20    30            0    10    20    30
     train ── val ──                train ── val ──
```

### What overfitting looks like

```
Accuracy:                    Loss:
1.0|     train──────────    3.0|──╮
   |       /                   |   ╲  train keeps dropping
   |      /   val plateaus     |    ╲_______________
   |     / ─ ─ ─ ─ ─ ─ ─      |       ╱─ ─ ─ ─ ─ ─
   |   _/                     |      ╱ val goes UP
0.0|/______________________  0.0|____╱_________________
```

---

## Cell 26: Evaluation

```python
val_loss, val_accuracy = model.evaluate(val_generator)
```

Runs all 2,077 validation images through the model one final time and reports overall accuracy and loss.

---

## Cell 27: Classification Report

```python
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
```

- `model.predict` returns probabilities for all 10 classes for each image
- `np.argmax` picks the class with the highest probability as the prediction

```python
classification_report(y_true, y_pred, target_names=class_names)
```

Prints per-class metrics:

```
                          precision  recall  f1-score  support

bacterial_leaf_blight        0.82     0.75     0.78       96
blast                        0.93     0.95     0.94      348
brown_spot                   0.85     0.80     0.82      193
...
```

Example for **blast**:

- **Precision 0.93** -- model said "blast" 374 times. 348 were actually blast. 26 were wrong → 348/374 = 0.93
- **Recall 0.95** -- there were 367 real blast images. Model found 348 of them. Missed 19 → 348/367 = 0.95
- **F1 0.94** -- harmonic mean = 2 x (0.93 x 0.95) / (0.93 + 0.95) = 0.94
- **Support** -- how many actual images of that class exist in validation

---

## Cell 28: Confusion Matrix

A 10x10 grid where:

- **Rows** = actual disease
- **Columns** = predicted disease
- **Diagonal** = correct predictions (bright/high numbers = good)
- **Off-diagonal** = mistakes (tells you exactly which diseases get confused with which)

Example (simplified to 4 classes):

```
                    PREDICTED
                blast  brown  normal  tungro
ACTUAL blast    [ 340,    5,     2,     1  ]  ← 340 correct
       brown    [   8,  165,    15,     5  ]  ← 165 correct, confused 15 with normal
       normal   [   3,   12,   330,     8  ]  ← 330 correct
       tungro   [   2,    4,     6,   196  ]  ← 196 correct
```

Reading: row "brown", column "normal" = 15 means **15 brown spot images were wrongly predicted as normal**. This tells you exactly where the model struggles.

---

## Cell 30: Single Image Prediction

Full example of predicting one image:

```
Input: photo of a rice leaf with brown spots

Step 1 - Resize:    480x640 → 224x224
Step 2 - Normalize: pixel [0,255] → [0.0, 1.0]
         example: pixel value 178 → 178/255 = 0.698
Step 3 - Add batch dim: (224,224,3) → (1,224,224,3)

Step 4 - Model predicts:
  EfficientNet → features → Dense layers → softmax

  Output: [0.02, 0.01, 0.01, 0.05, 0.82, 0.02, 0.01, 0.03, 0.02, 0.01]
                                           ↑
  Index 4 = brown_spot, confidence = 82%

Result: {"class": "brown_spot", "confidence": 0.82}
```

---

## Cell 31: Visual Prediction Test

Takes 8 random validation images, predicts each, and displays them with:

- **Green title** = model predicted correctly
- **Red title** = model predicted wrong
- Shows predicted class, true class, and confidence percentage

---

## Cell 33-34: Kaggle Submission

```python
test_generator = test_datagen.flow_from_directory(
    os.path.dirname(TEST_DIR),
    classes=['test_images'],
    class_mode=None,
    shuffle=False
)
```

- `class_mode=None` -- no labels (these are unlabeled test images)
- `shuffle=False` -- keep order so filenames match predictions

```python
test_predictions = model.predict(test_generator)
predicted_classes = np.argmax(test_predictions, axis=1)
predicted_labels = [class_names[idx] for idx in predicted_classes]
```

Runs all 3,469 test images through the model, gets the predicted class for each.

```python
submission = pd.DataFrame({'image_id': filenames, 'label': predicted_labels})
submission.to_csv('../submission.csv', index=False)
```

Saves as CSV in the format Kaggle expects:

```
image_id,label
200001.jpg,blast
200002.jpg,normal
200003.jpg,hispa
200004.jpg,brown_spot
200005.jpg,blast
...
(3,469 rows)
```

Upload this file to Kaggle and they compare your predictions vs hidden true labels to give you a score.

---

## Summary: The Full Flow

```
Raw leaf image (480x640x3)
        ↓  resize
(224x224x3)
        ↓  normalize (/255)
[0.0 - 1.0] values
        ↓  EfficientNetB0 (frozen/fine-tuned)
(7x7x1280) feature maps
        ↓  GlobalAveragePooling
(1280,) feature vector
        ↓  BatchNorm + Dropout
(1280,) normalized
        ↓  Dense(256, relu)
(256,) disease features
        ↓  BatchNorm + Dropout
(256,) normalized
        ↓  Dense(10, softmax)
(10,) probabilities
        ↓  argmax
Predicted class: "blast" (78% confidence)
```
