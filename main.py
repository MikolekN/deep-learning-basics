import numpy as np
import tensorflow as tf

from class_to_number import class_names
from constants import INPUT_SHAPE, NUM_CLASSES
from load_dataset import load_dataset
from model import create_model, train_model, save_model, predict_image, _initialize, _uninitialize

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs available: {physical_devices}")
else:
    print("No GPU detected. TensorFlow will run on CPU.")

if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_ds, val_ds = load_dataset()


wandb_config = _initialize()
model = create_model(wandb_config, INPUT_SHAPE, NUM_CLASSES)
history = train_model(model, train_ds, val_ds)
save_model(model)
_uninitialize()

prediction = predict_image(model, "data/testing/our/A-3/A-3.jpg")
print(f"Prediction: {prediction}")
predicted_class = np.argmax(prediction, axis=1)
print(f"Predicted class: {predicted_class[0]}")
predicted_class_label = class_names[predicted_class[0]]
print(f"Predicted class label: {predicted_class_label}")



# wandb.init(project="polish-road-signs-classification", name=create_run_name(config=hyperparams), config=hyperparams)
# wandb_config = wandb.config
#
# model = build_model(wandb_config, INPUT_SHAPE, NUM_CLASSES)
#
# # na ten moment wsparcie tylko dla optimizer Adam
# if wandb.config['optimizer'] == 'adam':
#     optimizer = Adam(learning_rate=wandb.config['learning_rate'])
# else:
#     raise ValueError(f"Unknown optimizer: {wandb.config['optimizer']}")
#
#
# model.compile(optimizer=optimizer,
#                     loss='sparse_categorical_crossentropy',
#                     metrics=['accuracy'])
#
# # Print model summary
# model.summary()

# history = model.fit( # w przykładach jest rozbicie na x i y, wywala error ConnectionAbortedError: [WinError 10053] Nawiązane połączenie zostało przerwane przez oprogramowanie zainstalowane w komputerze-hoście
#     train_ds,
#     steps_per_epoch=int(np.ceil(len(list(train_ds)) / 10)),
#     epochs=wandb.config['epochs'],
#     batch_size=wandb.config['batch_size'],
#     validation_data=val_ds,
#     validation_steps=int(np.ceil(len(list(val_ds)) / 10)),
#     verbose=1,
#     callbacks=[  #  rozwazyc dodanie EarlyStopping
#         WandbMetricsLogger(log_freq=1),
#         WandbModelCheckpoint(filepath=os.path.join("checkpoints", create_checkpoint_name(), "checkpoint_{epoch:02d}.keras"),
#                              save_freq="epoch")
#     ]
# )
#
# model.save(f'best_models/{create_model_name()}.keras')

# wandb.finish(exit_code=0)

# final = Sequential([
#     Conv2D(60, (5, 5), input_shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNEL_NUMBER), activation='relu'),
#     Conv2D(60, (5, 5), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#
#     Conv2D(30, (3, 3), activation='relu'),
#     Conv2D(30, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#
#     Flatten(),
#     Dense(500, activation='relu'),
#     Dropout(0.5),
#     Dense(len(class_names), activation='softmax')
# ])
# final.compile(Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# final.summary()
#
# history = final.fit(
#     train_ds,
#     steps_per_epoch=int(np.ceil(len(list(train_ds)) / 10)),
#     epochs=EPOCHS,
#     validation_data=val_ds,
#     validation_steps=int(np.ceil(len(list(val_ds)) / 10)),
#     verbose=1
# )

# plot_metric(history=history, metric_name='accuracy')
# plot_metric(history=history, metric_name='loss')