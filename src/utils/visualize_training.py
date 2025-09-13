from matplotlib import pyplot as plt

def plot_history(history,save_path):
    plt.figure(figsize=(12, 5))

    # ---- Loss ----
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # ---- Dice ----
    if 'multiclss_soft_iou' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['multiclss_soft_iou'], label='Train IOU')
        plt.plot(history.history['val_multiclss_soft_iou'], label='Val IOU')
        plt.title("IOU Curve")
        plt.xlabel("Epoch")
        plt.ylabel("IOU")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path + "/training_curves.png", dpi=150)
    plt.show()
