import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.cm as cm
import cv2
from sklearn.model_selection import train_test_split
import zipfile
import kaggle
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# تعيين نفس البذرة العشوائية لقابلية التكرار
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# تحميل البيانات من Kaggle
def download_dataset():
    try:
        # Check if dataset already exists
        if os.path.exists('./chest_xray'):
            print("Dataset found in ./chest_xray directory.")
            return True

        print("\nDataset not found. Please follow these steps:")
        print("1. Download the dataset from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("2. Extract the downloaded zip file")
        print("3. Place the 'chest_xray' folder in this directory:", os.getcwd())
        print("\nExpected folder structure:")
        print("chest_xray/")
        print("├── train/")
        print("│   ├── NORMAL/")
        print("│   └── PNEUMONIA/")
        print("├── test/")
        print("│   ├── NORMAL/")
        print("│   └── PNEUMONIA/")
        print("└── val/")
        print("    ├── NORMAL/")
        print("    └── PNEUMONIA/")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

# تحقق من وجود البيانات أو تحميلها
if not os.path.exists('./chest_xray'):
    if not download_dataset():
        print("\nPlease place the dataset in the correct location and run the program again.")
        exit(1)

# تحديد مسارات المجلدات
data_dir = './chest_xray'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
val_dir = os.path.join(data_dir, 'val')

# تحديد المعلمات
IMG_SIZE = 224  # حجم الصورة لـ EfficientNetB3
BATCH_SIZE = 32
EPOCHS = 20

# استكشاف البيانات وتصويرها
def explore_data():
    # حساب عدد الصور في كل فئة
    normal_train = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
    pneumonia_train = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
    normal_test = len(os.listdir(os.path.join(test_dir, 'NORMAL')))
    pneumonia_test = len(os.listdir(os.path.join(test_dir, 'PNEUMONIA')))
    normal_val = len(os.listdir(os.path.join(val_dir, 'NORMAL')))
    pneumonia_val = len(os.listdir(os.path.join(val_dir, 'PNEUMONIA')))
    
    print(f"مجموعة التدريب: عادي = {normal_train}, التهاب رئوي = {pneumonia_train}")
    print(f"مجموعة الاختبار: عادي = {normal_test}, التهاب رئوي = {pneumonia_test}")
    print(f"مجموعة التحقق: عادي = {normal_val}, التهاب رئوي = {pneumonia_val}")
    
    # رسم توزيع الفئات
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # مجموعة التدريب
    data_train = {'عادي': normal_train, 'التهاب رئوي': pneumonia_train}
    axes[0].bar(data_train.keys(), data_train.values(), color=['skyblue', 'salmon'])
    axes[0].set_title('توزيع الفئات - مجموعة التدريب')
    axes[0].set_ylabel('عدد الصور')
    
    # مجموعة الاختبار
    data_test = {'عادي': normal_test, 'التهاب رئوي': pneumonia_test}
    axes[1].bar(data_test.keys(), data_test.values(), color=['skyblue', 'salmon'])
    axes[1].set_title('توزيع الفئات - مجموعة الاختبار')
    
    # مجموعة التحقق
    data_val = {'عادي': normal_val, 'التهاب رئوي': pneumonia_val}
    axes[2].bar(data_val.keys(), data_val.values(), color=['skyblue', 'salmon'])
    axes[2].set_title('توزيع الفئات - مجموعة التحقق')
    
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.show()
    
    # عرض بعض الصور من كل فئة
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # عرض صور عادية
    normal_images = os.listdir(os.path.join(train_dir, 'NORMAL'))
    for i in range(4):
        img_path = os.path.join(train_dir, 'NORMAL', normal_images[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(img)
        axes[0, i].set_title('عادي')
        axes[0, i].axis('off')
    
    # عرض صور التهاب رئوي
    pneumonia_images = os.listdir(os.path.join(train_dir, 'PNEUMONIA'))
    for i in range(4):
        img_path = os.path.join(train_dir, 'PNEUMONIA', pneumonia_images[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(img)
        axes[1, i].set_title('التهاب رئوي')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()

# تحضير مولدات البيانات مع زيادة البيانات
def prepare_data_generators():
    # مولد بيانات التدريب مع زيادة البيانات
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # مولدات بيانات الاختبار والتحقق - فقط إعادة قياس
    test_val_datagen = ImageDataGenerator(rescale=1./255)
    
    # إنشاء مولدات تدفق البيانات
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED
    )
    
    validation_generator = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=SEED
    )
    
    test_generator = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        seed=SEED
    )
    
    return train_generator, validation_generator, test_generator

# عرض صور من مولد البيانات مع زيادة البيانات
def visualize_augmentation():
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # اختيار صورة عشوائية
    img_path = os.path.join(train_dir, 'PNEUMONIA', os.listdir(os.path.join(train_dir, 'PNEUMONIA'))[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape((1,) + img.shape)
    
    # عرض الصور المزادة
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # عرض الصورة الأصلية
    axes[0].imshow(img[0])
    axes[0].set_title('الصورة الأصلية')
    axes[0].axis('off')
    
    # عرض الصور المزادة
    i = 1
    for batch in datagen.flow(img, batch_size=1):
        axes[i].imshow(batch[0])
        axes[i].set_title(f'صورة مزادة {i}')
        axes[i].axis('off')
        i += 1
        if i >= 6:
            break
    
    plt.tight_layout()
    plt.savefig('augmented_images.png')
    plt.show()

# إنشاء نموذج EfficientNetB3
def create_model():
    # تحميل النموذج المدرب مسبقًا بدون الطبقة العليا
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # تجميد طبقات النموذج الأساسي
    for layer in base_model.layers:
        layer.trainable = False
    
    # إضافة طبقاتنا المخصصة
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    # إنشاء النموذج النهائي
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # تجميع النموذج
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

# إجراء تدريب النموذج
def train_model(model, train_generator, validation_generator):
    # تحديد استراتيجيات التدريب
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # تدريب النموذج
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return history

# فتح طبقات النموذج الأساسي للتدريب الدقيق
def unfreeze_and_fine_tune(model, base_model, train_generator, validation_generator):
    # فتح الطبقات الأخيرة من النموذج الأساسي
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # إعادة تجميع النموذج مع معدل تعلم أقل
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # تحديد استراتيجيات التدريب
    checkpoint = ModelCheckpoint(
        'best_model_fine_tuned.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    # تدريب النموذج
    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return fine_tune_history

# تقييم النموذج
def evaluate_model(model, test_generator):
    # التنبؤ بالتصنيفات
    test_generator.reset()
    predictions = model.predict(test_generator, steps=len(test_generator))
    predicted_classes = (predictions > 0.5).astype(int)
    true_classes = test_generator.classes
    
    # حساب مصفوفة الالتباس
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # عرض مصفوفة الالتباس
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['عادي', 'التهاب رئوي'],
                yticklabels=['عادي', 'التهاب رئوي'])
    plt.xlabel('التصنيف المتوقع')
    plt.ylabel('التصنيف الحقيقي')
    plt.title('مصفوفة الالتباس')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # تقرير التصنيف
    print("تقرير التصنيف:")
    print(classification_report(true_classes, predicted_classes, 
                               target_names=['عادي', 'التهاب رئوي']))
    
    # منحنى ROC
    fpr, tpr, _ = roc_curve(true_classes, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'منحنى ROC (مساحة = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('معدل الإيجابيات الخاطئة')
    plt.ylabel('معدل الإيجابيات الصحيحة')
    plt.title('منحنى ROC')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()
    
    # حساب الدقة
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    print(f"دقة النموذج: {accuracy:.4f}")
    
    return accuracy, cm, roc_auc

# رسم منحنيات التدريب
def plot_training_curves(history, fine_tune_history=None):
    # تحضير البيانات
    if fine_tune_history:
        # دمج تاريخ التدريب
        acc = history.history['accuracy'] + fine_tune_history.history['accuracy']
        val_acc = history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']
        loss = history.history['loss'] + fine_tune_history.history['loss']
        val_loss = history.history['val_loss'] + fine_tune_history.history['val_loss']
        epochs_range = range(1, len(acc) + 1)
    else:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(1, len(acc) + 1)
    
    # رسم منحنى الدقة
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='دقة التدريب')
    plt.plot(epochs_range, val_acc, label='دقة التحقق')
    plt.legend(loc='lower right')
    plt.title('دقة التدريب والتحقق')
    
    # رسم منحنى الخسارة
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='خسارة التدريب')
    plt.plot(epochs_range, val_loss, label='خسارة التحقق')
    plt.legend(loc='upper right')
    plt.title('خسارة التدريب والتحقق')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

# اختبار النموذج على صور جديدة
def test_on_new_images(model):
    # اختيار بعض الصور من مجموعة الاختبار
    normal_test_images = os.listdir(os.path.join(test_dir, 'NORMAL'))
    pneumonia_test_images = os.listdir(os.path.join(test_dir, 'PNEUMONIA'))
    
    # اختيار صور عشوائية
    selected_normal = random.sample(normal_test_images, min(5, len(normal_test_images)))
    selected_pneumonia = random.sample(pneumonia_test_images, min(5, len(pneumonia_test_images)))
    
    # تحضير الصور للاختبار
    test_images = []
    true_labels = []
    
    # إضافة صور عادية
    for img_name in selected_normal:
        img_path = os.path.join(test_dir, 'NORMAL', img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        test_images.append(img)
        true_labels.append(0)  # عادي
    
    # إضافة صور التهاب رئوي
    for img_name in selected_pneumonia:
        img_path = os.path.join(test_dir, 'PNEUMONIA', img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        test_images.append(img)
        true_labels.append(1)  # التهاب رئوي
    
    if not test_images:
        print("Error: No valid images found for testing")
        return
    
    # تحويل إلى مصفوفة NumPy
    test_images = np.array(test_images) / 255.0
    
    # التنبؤ بالتصنيفات
    predictions = model.predict(test_images, batch_size=4)  # Process in smaller batches
    predicted_classes = (predictions > 0.5).astype(int)
    
    # عرض النتائج
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, (img, true_label, pred_label, pred_prob) in enumerate(zip(test_images, true_labels, predicted_classes, predictions)):
        if i >= len(axes):  # Prevent index out of range
            break
        axes[i].imshow(img)
        
        # تحديد لون العنوان بناءً على صحة التنبؤ
        title_color = 'green' if true_label == pred_label[0] else 'red'
        
        # تعيين العنوان
        label_text = 'عادي' if true_label == 0 else 'التهاب رئوي'
        pred_text = 'عادي' if pred_label[0] == 0 else 'التهاب رئوي'
        
        axes[i].set_title(f'حقيقي: {label_text}\nمتوقع: {pred_text}\nاحتمالية: {pred_prob[0]:.4f}', 
                         color=title_color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions.png')
    plt.show()

# تصور طبقات التنشيط
def visualize_activations(model, img_path):
    # تحميل الصورة
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img / 255.0, axis=0)
    
    # إنشاء نموذج لاستخراج ميزات الطبقة الوسطى
    layer_outputs = [layer.output for layer in model.layers[0].layers if isinstance(layer.output, tf.Tensor) and len(layer.output.shape) == 4]
    activation_model = Model(inputs=model.layers[0].input, outputs=layer_outputs[:5])  # الحصول على أول 5 طبقات فقط
    
    # الحصول على التنشيطات
    activations = activation_model.predict(img_array)
    
    # عرض الصورة الأصلية
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title('الصورة الأصلية')
    plt.axis('off')
    plt.savefig('original_image_for_activation.png')
    plt.show()
    
    # عرض تنشيطات الطبقات
    for i, activation in enumerate(activations):
        plt.figure(figsize=(15, 8))
        
        # عرض الخرائط الـ 16 الأولى فقط
        display_num = min(16, activation.shape[3])
        
        for j in range(display_num):
            plt.subplot(4, 4, j+1)
            plt.imshow(activation[0, :, :, j], cmap='viridis')
            plt.title(f'خريطة {j+1}')
            plt.axis('off')
        
        plt.suptitle(f'تنشيطات الطبقة {i+1}')
        plt.tight_layout()
        plt.savefig(f'layer_{i+1}_activations.png')
        plt.show()

# الوظيفة الرئيسية
def main():
    # استكشاف البيانات
    print("استكشاف البيانات وتصويرها...")
    explore_data()
    
    # عرض صور مزادة
    print("تصوير زيادة البيانات...")
    visualize_augmentation()
    
    # تحضير مولدات البيانات
    print("تحضير مولدات البيانات...")
    train_generator, validation_generator, test_generator = prepare_data_generators()
    
    # إنشاء النموذج
    print("إنشاء نموذج EfficientNetB3...")
    model, base_model = create_model()
    
    # عرض ملخص النموذج
    model.summary()
    
    # تدريب النموذج (المرحلة الأولى)
    print("بدء تدريب النموذج (المرحلة الأولى)...")
    history = train_model(model, train_generator, validation_generator)
    
    # التدريب الدقيق (المرحلة الثانية)
    print("بدء التدريب الدقيق (المرحلة الثانية)...")
    fine_tune_history = unfreeze_and_fine_tune(model, base_model, train_generator, validation_generator)
    
    # تقييم النموذج
    print("تقييم النموذج...")
    accuracy, confusion_mat, roc_auc = evaluate_model(model, test_generator)
    
    # رسم منحنيات التدريب
    print("رسم منحنيات التدريب...")
    plot_training_curves(history, fine_tune_history)
    
    # اختبار النموذج على صور جديدة
    print("اختبار النموذج على صور جديدة...")
    test_on_new_images(model)
    
    # تصور طبقات التنشيط
    print("تصور طبقات التنشيط...")
    # اختيار صورة عشوائية من مجموعة الاختبار
    sample_img_path = os.path.join(test_dir, 'PNEUMONIA', os.listdir(os.path.join(test_dir, 'PNEUMONIA'))[0])
    visualize_activations(model, sample_img_path)
    
    # طباعة النتائج النهائية
    print("\n===== النتائج النهائية =====")
    print(f"دقة النموذج على مجموعة الاختبار: {accuracy:.4f}")
    print(f"مساحة تحت منحنى ROC: {roc_auc:.4f}")
    
    if accuracy >= 0.98:
        print("تم تحقيق هدف الدقة بنجاح! (≥ 98%)")
    else:
        print(f"لم يتم تحقيق هدف الدقة. الدقة الحالية: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
        