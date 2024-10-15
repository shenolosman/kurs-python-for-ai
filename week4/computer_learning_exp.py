import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt


# Bilgisayarla görme (Computer Vision), bilgisayarların görüntü ve videoları anlamasını sağlayan bir yapay zeka alanıdır. Bu alanda yapılan çalışmalar, görsel verilerden anlamlı bilgilerin çıkarılmasını, nesne tanıma, görüntü sınıflandırma, yüz tanıma, hareket takibi, ve segmentasyon gibi görevleri içerir. Computer Vision, makine öğrenimi ve derin öğrenme teknikleriyle birlikte kullanılarak büyük ilerlemeler kaydetmiştir.

# Computer Vision'in Temel Kavramları
# Görüntü İşleme: Bir görüntüyü matematiksel olarak işlemek ve dönüştürmek için kullanılan tekniklerdir. Görüntüdeki pikseller üzerinde doğrudan işlemler yapılır.

# Gri tonlama: Renkli bir görüntüyü siyah-beyaz hale getirme.
# Filtreleme: Gürültüyü azaltmak veya belirli özellikleri çıkarmak için kullanılır.
# Nesne Tespiti (Object Detection): Görüntü veya videolarda belirli nesneleri tanıma ve bu nesnelerin konumlarını bulma.

# Görüntü Sınıflandırma: Bir görüntünün hangi kategoriye ait olduğunu belirler. Örneğin, bir görüntünün kedi, köpek ya da araba içerip içermediğini belirlemek.

# Segmentasyon: Görüntüyü pikseller bazında sınıflandırarak farklı nesneleri veya bölgeleri ayırır.

# Semantic Segmentation: Her bir pikseli bir sınıfa atama (örneğin, insan, araç, arka plan).
# Instance Segmentation: Aynı sınıfa ait farklı nesneleri ayırma (örneğin, farklı insanları ayırma).
# Optik Akış (Optical Flow): Bir video içerisinde hareket eden nesnelerin yön ve hızını analiz eder.

# Derin Öğrenme ile Computer Vision
# Derin öğrenme kullanılarak görme görevlerinde büyük başarılar elde edilmiştir. Özellikle Convolutional Neural Networks (CNN), görüntülerdeki özellikleri öğrenme konusunda çok başarılıdır. CNN, yerel özellikleri öğrenerek görüntülerdeki desenleri tanıma yeteneğine sahiptir.

# Python ile Basit Bir Computer Vision Uygulaması: CNN ile Görüntü Sınıflandırma
# Aşağıdaki örnek, popüler bir veri seti olan CIFAR-10'da yer alan renkli görüntüler üzerinde bir CNN modeli eğiterek, görüntü sınıflandırma işlemi yapar. CIFAR-10 veri seti, 10 farklı sınıfa ait küçük görüntüler içerir (örneğin, uçak, araba, kuş, kedi, köpek vb.).


# CIFAR-10 veri setini yükleme
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Verilerin normalize edilmesi (0-255 aralığındaki değerleri 0-1 aralığına getirme)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Sınıf isimleri
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Eğitim verilerinden birkaç örnek görüntü gösterme
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(class_names[y_train[i][0]])
plt.show()

# CNN modelini oluşturma
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        # Fully connected katmanlar
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Modeli derleme
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Modeli eğitme
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Test verileri üzerinde doğruluğu hesaplama
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Doğruluğu: {test_acc * 100:.2f}%")

# Test setinden bir görüntü üzerinde tahmin yapma
predictions = model.predict(X_test)
predicted_label = class_names[predictions[0].argmax()]

# Görüntüyü gösterme ve tahmin edilen etiketi yazdırma
plt.imshow(X_test[0])
plt.title(f"Gerçek: {class_names[y_test[0][0]]}, Tahmin: {predicted_label}")
plt.show()


# Kodun Açıklaması:
# CIFAR-10 Veri Seti: Bu veri seti, 10 sınıfa ait 60,000 adet 32x32 boyutlarında renkli görüntüden oluşur. 50,000'i eğitim, 10,000'i test verisidir.
# Convolutional Neural Network (CNN): CNN, görüntülerdeki lokal özellikleri öğrenmek için kullanılan bir sinir ağı türüdür. Bu modelde üç adet convolution ve pooling katmanı bulunmaktadır.
# Girdi Şekli: CNN'de görüntülerin girdi boyutu 32x32 piksel ve 3 kanal (RGB) içerir.
# Softmax Çıktı Katmanı: Son katmanda 10 sınıfa ait olasılıkları döndüren bir softmax katmanı bulunur.
# Eğitim ve Değerlendirme: Model 10 epoch boyunca eğitilir ve ardından test verileri üzerinde doğruluk hesaplanır.
# Computer Vision'da Kullanılan Diğer Teknikler:
# Nesne Tespiti (Object Detection): Görüntüdeki belirli nesneleri ve bu nesnelerin bulunduğu alanları tespit etmek. Yaygın algoritmalar arasında YOLO (You Only Look Once), Faster R-CNN, ve SSD (Single Shot Detector) bulunur.

# Semantic Segmentation: Görüntüdeki her bir pikseli sınıflandırma. Örneğin, bir görüntüde insanların, arabaların, binaların nerede olduğunu pikseller düzeyinde belirleme. U-Net ve SegNet gibi modeller bu görev için kullanılır.

# Face Recognition (Yüz Tanıma): Yüzleri tanımak ve belirli kişileri tespit etmek için kullanılan bir yöntem. Haar Cascades, Fisherfaces, ve daha ileri düzeyde CNN'ler bu işlemde kullanılır.

# Optical Character Recognition (OCR): Görüntülerdeki yazılı metinleri tanıyıp çıkarma. Tesseract gibi araçlar, metin tespiti ve tanıma için yaygın olarak kullanılır.

# Transfer Learning: Büyük bir modelin önceden öğrenilmiş özelliklerini kullanarak yeni bir problem için hızlıca model geliştirme. Örneğin, ResNet, Inception, VGG gibi modeller, önceden büyük veri setlerinde eğitilmiş ve yeni veri setlerinde yeniden eğitilebilir


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Pre-trained ResNet50 modeli ve ağırlıkları (ImageNet veri seti ile eğitilmiş)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Yeni sınıflandırıcı katmanları ekleme
x = Flatten()(base_model.output)
x = Dense(1024, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)

# Yeni modelin oluşturulması
model = Model(inputs=base_model.input, outputs=predictions)

# Önceden eğitilmiş katmanları dondurma (bunlar eğitilmeyecek)
for layer in base_model.layers:
    layer.trainable = False

# Modelin derlenmesi
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ImageDataGenerator ile veri artırma (data augmentation)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# Veri seti klasöründen veri yükleme ve model eğitimi
train_generator = train_datagen.flow_from_directory(
    "path_to_train_data",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

# Modeli eğitme
model.fit(train_generator, epochs=10)

# Bu örnekte ResNet-50'yi kullanarak, görüntü sınıflandırması yapmak için transfer öğrenme yöntemi uygulanmıştır. Model, önceki ImageNet ağırlıklarıyla başlar ve yeni bir problem için ince ayar yapılır.

# Computer Vision Uygulama Alanları:
# Otonom Araçlar: Çevredeki nesneleri tanıyarak güvenli sürüş sağlama.
# Tıbbi Görüntüleme: MRI, röntgen gibi tıbbi görüntüleri analiz ederek hastalık tespiti.
# Yüz Tanıma: Güvenlik ve kimlik doğrulama sistemleri.
# Tarım: Bitki sağlığı izleme ve ürün kalitesi kontrolü.
# Perakende: Müşteri davranışlarını izleme ve stok takibi.
# Bilgisayarla görme, görüntü ve video verilerinin analizini sağlayarak birçok sektörde önemli yeniliklere yol açmaktadır.