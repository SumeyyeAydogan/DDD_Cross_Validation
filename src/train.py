import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, AUC
from src.losses import create_simple_masked_loss

def train_model(model, train_ds, val_ds, epochs=10, callbacks=None, initial_epoch=0):
    """
    Train the model with custom callbacks and sample_weight support.
    
    Args:
        model: Keras model to train
        train_ds: Training dataset (should return (x, y, sample_weight) tuples)
        val_ds: Validation dataset  
        epochs: Number of training epochs
        callbacks: List of Keras callbacks
        initial_epoch: Starting epoch number
    """

    # Use standard loss with sample_weight support
    loss_fn = create_simple_masked_loss()

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=loss_fn, #loss_fn, 'binary_crossentropy'
        metrics=['accuracy'],  # Keep basic accuracy unweighted
        weighted_metrics=[Precision(name='precision'), Recall(name='recall'), AUC(name='auc')],  # These will use sample_weight
        #metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
    )
    
    # Prepare callbacks
    if callbacks is None:
        callbacks = []
    """ batch = next(iter(train_ds))
    print([t.shape for t in batch])  #(32, 224,224,3), (32,1), (32,)

    # Try with a single batch:
    model.train_on_batch(*batch) """
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1
    )
    
    '''
    train_small = train_ds.take(4)  # 4*16=64
    val_small = val_ds.take(4)
    history = model.fit(train_small, epochs=4, validation_data=val_small, class_weight=None)
    '''
    return history
