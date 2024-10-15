import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Deep learning (derin öğrenme), makine öğreniminin bir alt dalı olup, insan beyninin çalışma biçiminden ilham alarak yapay sinir ağlarını (neural networks) kullanır. Derin öğrenme, özellikle büyük veri kümeleri üzerinde öğrenme yapabilen ve çok katmanlı sinir ağlarıyla çalışarak karmaşık veri temsillerini modelleyebilen bir tekniktir. Görüntü işleme, doğal dil işleme, ses tanıma gibi alanlarda çığır açan sonuçlar elde edilmesini sağlamıştır.

# Yapay Sinir Ağı (Artificial Neural Network) ve Derin Sinir Ağı (Deep Neural Network)
# Yapay Sinir Ağı (ANN): Birkaç katmandan oluşur. Her katmanda nöronlar (nodes) vardır ve her nöron, bir önceki katmandan gelen verileri alır, belirli ağırlıklarla çarpar, bir aktivasyon fonksiyonundan geçirir ve bir sonraki katmana iletir.

# Derin Sinir Ağı (DNN): Daha fazla gizli katman (hidden layers) içeren bir yapay sinir ağıdır. Derin sinir ağları, daha karmaşık desenleri ve ilişkileri öğrenebilir.

# Derin Öğrenmenin Temel Yapısı:
# Girdi Katmanı (Input Layer): Veri bu katmandan ağa girer.
# Gizli Katmanlar (Hidden Layers): Ağın derinliğini artıran katmanlardır. Verinin işlenmesi burada gerçekleşir.
# Çıktı Katmanı (Output Layer): Modelin son tahminleri veya çıktı değerleri burada üretilir.
# Aktivasyon Fonksiyonu (Activation Function): Nöronların karar verebilmesi için kullanılan fonksiyonlardır (örneğin, ReLU, Sigmoid, Tanh).
# Loss Function (Kayıp Fonksiyonu): Modelin çıktıları ile gerçek değerler arasındaki farkı ölçer.
# Optimizer (Optimizasyon Algoritması): Modelin parametrelerini güncelleyerek hatayı (loss) en aza indirmeye çalışır (örneğin, Adam, SGD).
# Python’da Keras ile Basit Bir Derin Öğrenme Örneği
# Aşağıdaki örnekte, bir sinir ağı kullanarak MNIST veri seti üzerinde basit bir el yazısı rakam tanıma modeli oluşturulacaktır. Keras kütüphanesi, derin öğrenme modellerini kolayca oluşturmak için yaygın olarak kullanılır.


# MNIST veri setini yükleme
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Verilerin normalize edilmesi (0-255 aralığındaki değerleri 0-1 aralığına getirir)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Verileri düzleştirme (28x28 boyutundaki görüntüler 784 boyutlu bir vektöre çevriliyor)
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

# Modelin oluşturulması
model = models.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=(784,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),  # 10 sınıf (0-9 arası rakamlar)
    ]
)

# Modeli derleme (optimizasyon, kayıp fonksiyonu ve değerlendirme metriği seçilir)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Modeli eğitme
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Modelin test seti üzerinde değerlendirilmesi
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Doğruluğu:", test_acc)

# Test veri setinden örnek bir görüntü gösterme
plt.imshow(X_test[0].reshape(28, 28), cmap="gray")
plt.title(f"Gerçek etiket: {y_test[0]}")
plt.show()

# Tahmin yapma
predictions = model.predict(X_test)
predicted_label = np.argmax(predictions[0])
print(f"Tahmin edilen etiket: {predicted_label}")



# Kodun Açıklaması:
# Veri Seti: MNIST veri seti, el yazısı rakamları içerir. Bu veri setinde her rakam 28x28 piksel boyutunda gri tonlamalı görüntülerden oluşur.
# Model: Üç katmandan oluşan basit bir sinir ağı kurulmuştur:
# İlk gizli katmanda 128 nöron vardır ve aktivasyon fonksiyonu olarak ReLU kullanılır.
# İkinci gizli katmanda 64 nöron vardır ve yine ReLU kullanılır.
# Son katman, 10 sınıf (0-9 arası rakamlar) için softmax fonksiyonunu kullanır.
# Loss Function: sparse_categorical_crossentropy kullanılmıştır, çünkü sınıflandırma problemi vardır ve etiketler tam sayıdır.
# Optimizer: Adam optimizasyon algoritması kullanılmıştır.
# Eğitim ve Değerlendirme: Model eğitim verileri ile eğitilir ve test verileri üzerinde doğruluğu değerlendirilir.
# Derin Öğrenme'nin Gücü:
# Özellik Öğrenimi: Derin öğrenme modelleri, verideki desenleri otomatik olarak öğrenebilir ve ek bir özellik mühendisliğine gerek kalmaz.
# Yüksek Performans: Büyük veri kümeleri ve yüksek hesaplama gücüyle derin öğrenme, birçok karmaşık problemde geleneksel makine öğrenme yöntemlerinden daha iyi performans gösterir.
# Katman Derinliği: Katman sayısı ve nöron sayısı arttıkça modelin karmaşık ilişkileri öğrenme kapasitesi artar, ancak bu aynı zamanda aşırı öğrenme (overfitting) riskini de beraberinde getirir.
# Bu örnek, derin öğrenmenin temel ilkelerini anlamanızı sağlar. Derin sinir ağları, görüntü işleme, dil modelleme ve diğer birçok alanda etkili çözümler sunar.