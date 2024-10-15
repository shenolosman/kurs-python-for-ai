import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Supervised learning, makine öğrenmesinde en yaygın kullanılan yöntemlerden biridir. Bu öğrenme türünde model, etiketli veriler (yani, girdilere karşılık gelen doğru çıktılar) ile eğitilir. Amacı, girdi verileri ile çıktı verileri arasındaki ilişkileri öğrenerek, yeni verilerle karşılaştığında doğru tahminler yapabilmektir.

# Supervised learning iki ana kategoriye ayrılır:

# Sınıflandırma (Classification): Verileri sınıflara ayırmayı amaçlar. Örneğin, bir e-posta’nın spam olup olmadığını belirlemek bir sınıflandırma problemidir.

# Regresyon (Regression): Sürekli bir değer tahmin etmeye çalışır. Örneğin, bir evin fiyatını tahmin etmek bir regresyon problemidir.

# Adımlar:
# Veri Toplama: Giriş ve çıkış verilerinin toplandığı bir veri seti oluşturulur.
# Veri Ön İşleme: Veriler temizlenir ve modelin eğitimi için uygun hale getirilir.
# Model Seçimi: Hangi algoritmanın kullanılacağına karar verilir (örneğin, lojistik regresyon, karar ağaçları, SVM).
# Model Eğitimi: Veri seti kullanılarak model eğitilir.
# Model Değerlendirmesi: Model test seti üzerinde değerlendirilir.
# Tahmin Yapma: Yeni verilerle model tahmin yapar.
# Python'da Bir Örnek: Lojistik Regresyon ile Sınıflandırma
# Aşağıda basit bir sınıflandırma örneği yer almakta. Bu örnekte bir veri seti ile model eğitip, yeni verilerle sınıflandırma yapıyoruz:


# example data set
data = {
    "Feature1": [2, 3, 5, 7, 8, 10, 12, 15, 18, 20],
    "Feature2": [1, 4, 5, 8, 10, 13, 14, 16, 17, 19],
    "Label": [0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
}

df = pd.DataFrame(data)

# entry (X) ve output (y) variables
X = df[["Feature1", "Feature2"]]
y = df["Label"]

# splitting education and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model creating and training
model = LogisticRegression()
model.fit(X_train, y_train)

# guessing
y_pred = model.predict(X_test)

# model accuracy calculating
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy: {:.2f}%".format(accuracy * 100))
