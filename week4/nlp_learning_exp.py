# Doğal dil işleme (NLP - Natural Language Processing), bilgisayarların insan dillerini anlama, yorumlama ve oluşturma becerisini geliştiren bir yapay zeka dalıdır. NLP, dilbilim ile bilgisayar biliminin kesişiminde yer alır ve dil yapısını anlamak için makine öğrenimi ve derin öğrenme tekniklerinden yararlanır.

# NLP'nin Temel Bileşenleri:
# Tokenizasyon: Metni kelime veya cümle gibi daha küçük birimlere ayırma işlemi.
# Lemmatizasyon/Stemming: Kelimeleri kök forma indirme. Lemmatizasyon dilbilgisel kurallara göre yapılır, stemming ise daha basit ve kısaltma esaslıdır.
# Parça-of-Speech (POS) Tagging: Kelimelerin dilbilgisel kategorilerini (isim, fiil vb.) etiketleme.
# Entity Recognition (Varlık Tanıma): Metinde geçen önemli varlıkları (örneğin, isimler, tarihler, yerler) bulma.
# Duygu Analizi: Bir metnin duygu durumunu (olumlu, olumsuz, nötr) analiz etme.
# Metin Sınıflandırma: Metinleri belirli kategorilere ayırma (örneğin, spam tespiti, haber türü sınıflandırması).
# Dil Modelleme: Bir metni anlayarak sonraki kelimeleri tahmin etme veya yeni metinler oluşturma.
# NLP'de Yaygın Kullanılan Teknikler:
# TF-IDF (Term Frequency-Inverse Document Frequency): Kelimelerin bir belgede ne kadar önemli olduğunu belirlemek için kullanılan istatistiksel bir yöntemdir.
# Word Embeddings: Kelimeleri vektörler ile temsil etme. Bu temsil yöntemi, kelimeler arasındaki anlam ilişkilerini vektörler aracılığıyla öğrenir (örneğin, Word2Vec, GloVe).
# Transformer Modelleri: Dil modellerini eğitmek için kullanılan gelişmiş derin öğrenme modelleridir. Özellikle BERT, GPT gibi modeller son yıllarda büyük başarı sağlamıştır.
# Python ile Basit Bir NLP Örneği
# Aşağıda, NLTK (Natural Language Toolkit) ve scikit-learn kullanarak metin verilerini işleyen basit bir Python örneği bulunmaktadır. Bu örnekte, metin sınıflandırması yapılacaktır.

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# NLTK ile stopword (gereksiz kelime) listesi ve tokenizer kullanımı
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Veri seti: Basit bir olumlu ve olumsuz cümleler listesi
texts = [
    "I love this product, it is amazing!",
    "This is the worst experience I've ever had.",
    "I feel so happy using this service.",
    "This product is terrible and disappointing.",
    "Best purchase I've made, totally worth it!",
    "Not satisfied at all, I want a refund.",
    "The quality is excellent and I highly recommend it.",
    "Awful service, never coming back again."
]

# Etiketler: 1 = olumlu, 0 = olumsuz
labels = [1, 0, 1, 0, 1, 0, 1, 0]

# TF-IDF ve Naive Bayes sınıflandırıcıyı içeren bir pipeline oluşturma
model = make_pipeline(TfidfVectorizer(stop_words=stop_words), MultinomialNB())

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)

# Modeli eğitme
model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Model doğruluğunu hesaplama
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy * 100:.2f}%")

# Test verilerinin tahmin edilen etiketleri
for i, text in enumerate(X_test):
    print(f"Metin: {text}\nGerçek Etiket: {y_test[i]} - Tahmin Edilen Etiket: {y_pred[i]}")


# Kodun Açıklaması:
# Veri Seti: Basit bir olumlu ve olumsuz cümlelerden oluşan küçük bir veri seti kullanıyoruz.
# TF-IDF Vektörleştirme: TfidfVectorizer ile metinleri sayısal vektörlere dönüştürüyoruz. Bu yöntem, her kelimenin bir belgede ne kadar sık geçtiğini hesaplayarak metni temsil eder.
# Naive Bayes: MultinomialNB sınıflandırıcısı, vektörleştirilen metinleri kullanarak sınıflandırma yapar.
# Model Eğitimi ve Testi: Veriyi eğitim ve test kümelerine böldükten sonra model eğitilir. Ardından test verisi üzerinde tahminler yapılır.
# Doğruluk: accuracy_score ile modelin doğruluğu hesaplanır.



#----------------*********************************--------------------------------
# NLP'nin İleri Teknikleri:
# BERT (Bidirectional Encoder Representations from Transformers): Özellikle dil modellemede çığır açan bir modeldir. Hem soldan sağa hem de sağdan sola tüm cümleyi analiz ederek dilin bağlamını öğrenir.
# GPT (Generative Pretrained Transformer): Büyük miktarda metin verisi üzerinde önceden eğitilmiş ve doğal dil oluşturma konusunda başarılı olan bir modeldir.
# Sequence-to-Sequence Modelleri: Makine çevirisi, metin özeti oluşturma gibi sıralı veri problemlerini çözmek için kullanılan modellerdir.


# from transformers import BertTokenizer, TFBertForSequenceClassification
# from transformers import InputExample, InputFeatures
# import tensorflow as tf

# # BERT Tokenizer ve modeli
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = TFBertForSequenceClassification.from_pretrained(
#     "bert-base-uncased", num_labels=2
# )

# # Örnek cümleler
# texts = ["I love this product, it's amazing!", "This is the worst experience ever."]

# # Tokenize etme
# inputs = tokenizer(texts, return_tensors="tf", padding=True, truncation=True)

# # Model ile tahmin yapma
# outputs = model(inputs)
# logits = outputs.logits
# predictions = tf.argmax(logits, axis=-1)

# # Sonuçlar
# for text, prediction in zip(texts, predictions):
#     print(f"Metin: {text} -> Tahmin Edilen Etiket: {prediction.numpy()}")


# Bu kod, BERT gibi gelişmiş bir NLP modelini kullanarak metin sınıflandırma yapar. BERT, dilin hem geçmiş hem de gelecek bağlamını anlamada oldukça başarılıdır ve modern NLP uygulamalarında sıkça kullanılır.

# NLP’nin Uygulama Alanları:
# Metin Sınıflandırma: Spam tespiti, haber türü belirleme gibi metinleri kategorilere ayırma işlemleri.
# Duygu Analizi: Müşteri yorumları, sosyal medya analizleri ile kullanıcıların olumlu, olumsuz veya nötr görüşlerini belirleme.
# Makine Çevirisi: Bir dilden başka bir dile otomatik çeviri yapma.
# Chatbot ve Sesli Asistanlar: İnsanlarla doğal dilde etkileşim kurabilen sistemler.
# Metin Özeti: Uzun metinleri otomatik olarak özetleme.
# Bu örnekler, doğal dil işleme alanında kullanılan temel teknikleri ve Python kütüphanelerini tanımanızı sağlar. NLP, geniş bir uygulama yelpazesi ile modern yapay zeka projelerinde kritik bir rol oynar.