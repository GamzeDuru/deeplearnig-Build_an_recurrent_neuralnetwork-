import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#imdb puanlarına göre yorum yapılan filmlerin analizi
#veri setimizi dahil ediyoruz
#en sık kullanılan 10k kelimeyi indiriyoruz
(X_train,y_train), (X_test,y_test)=tf.keras.datasets.imdb.load_data(num_words=10000)

#Derin öğrenme eğitimi için üç veri kümesine ihtiyacımız olduğunu unutmayın: 80-10-10 bölünme oranına sahip eğitim, doğrulama ve test veri kümeleri.
#bu sayılardan örneğin 45 iyi 98 kotuyu temsil ediyor
print(X_train[0])

#verilerin dağılımına bakıyoruz
print(f"X_train: {len(X_train)}")
print(f"X_test: {len(X_test)}")

#veri setlerini birleştiriyoruz
X=np.concatenate((X_train, X_test), axis=0)
y=np.concatenate((y_train, y_test), axis=0)

#padding
"""
Film incelemeleri farklı uzunluklarda olduğu için CNN'lerde yaptığımız gibi incelemelerin başına bir miktar sıfır ekleyebiliriz,
 böylece tüm örnekler aynı uzunlukta olur. Buradaki max_len parametresi tüm incelemeleri 1024 kelimeye yeniden boyutlandıracaktır.
 Bu uzunluğun altındaki tüm örneklere fazladan sıfır verilecek ve bu uzunluğun üzerindeki tüm incelemeler kırpılacaktır.
 Burada max_len parametresi için herhangi bir sayı belirtmemiş olsaydık, tüm diziler en uzun dizi uzunluğuna kadar doldurulacaktı. 
 Bizim durumumuzda bu, eğitim süresini önemli ölçüde artıracaktır.
"""
X=tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=1024)

#veri setimizi bölüyoruz

X_train=X[:40000]
y_train=y[:40000]

X_val=X[40000:45000]
y_val=y[40000:45000]


X_test=X[45000:50000]
y_test=y[45000:50000]


#bölünmenin istediğimiz şekilde olup olmadığını kontrol ediyoruz
print(f"X_train= {len(X_train)}")
print(f"X_val= {len(X_val)}")
print(f"X_test= {len(X_test)}")

#model oluşturma aşaması
model=tf.keras.Sequential()

"""
 ilk katman “gömme katmanı” olarak adlandırılmaktadır
 Kelime gömme, kelimeleri veya metni sayısal bir şekilde temsil etme yöntemidir.
 10000 kelimeyi 256 boyutlu bir düzleme çevireceğiz
 input_dim parametresini kullanarak giriş katmanının giriş olarak 10000 kelime içerdiğini belirtiyoruz
 Gömmeden sonra dropout'u ekliyoruz.
 Daha sonra bir LSTM kapısı, diğer bir deyişle LSTM katmanı ekleyeceğiz.
 LSTM katmanına dropout ekliyoruz. Ve düşmeli yoğun bir katman
 . Son olarak çıktı katmanını ekliyoruz
 Sigmoid aktivasyon fonksiyonu ile ağ 0 ile 1 arasında bir değer üretecektir. Yani değer 0,5'in altındaysa sınıf 0 yani olumsuz yorumların sınıfı, üzerindeyse sınıf olumlu yorumlar için 1 olacaktır.
"""

model.add(tf.keras.layers.Embedding(input_dim = 10000, output_dim =256))
model.add(tf.keras.layers.Dropout(0.7))

model.add(tf.keras.layers.LSTM((256)))
model.add(tf.keras.layers.Dropout(0.7))

model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.7))

model.add(tf.keras.layers.Dense(1, activation="sigmoid"))


#modeli optimize etme
"""
 Bu sefer “İkili Çapraz Entropi” kayıp fonksiyonunu kullanacağız çünkü bu bir ikili sınıflandırma problemidir. 
 İkiden fazla sınıfımız olsaydı “Seyrek Kategorik Çapraz Entropi”yi kullanmak zorunda kalacağımızı unutmayın
"""
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


#modeli eğitiyoruz 5 dönem boyunca
results = model.fit(X_train, y_train, epochs=5, validation_data=(X_val,y_val))

#kaybı grafiğe dökme
plt.plot(results.history["loss"],label="Train")
plt.plot(results.history["val_loss"],label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.show()

#accuracy 
plt.plot(results.history["accuracy"],label="Train")
plt.plot(results.history["val_accuracy"],label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.show()


#
model.evaluate(X_test,y_test)

"""
 Şimdi rastgele tek bir örnek seçelim ve modelin doğru tahmin edip etmediğini görelim. 
 Tahmin yöntemi veri grupları üzerinde çalışır ancak yalnızca bir örnek üzerinde tahmin yaptığımız için 
 örneği yeniden şekillendirmemiz gerekir. İncelemeyi 1'e 1024 olarak yeniden şekillendireceğiz. 
 Her incelemeyi 1024 uzunluğa sahip olacak şekilde doldurduğumuzu unutmayın. 
 Daha sonra bunları karşılaştırmak için tahmin sonucunu ve örneğin veri kümesinde bulunan etiketi
 yazdırırız. Modelimizin çıktısı 0.2. Unutmayın, çıktı 0,5'ten küçükse 0'a ait diyoruz.
 Ve sonuç sıfıra çok yakın olduğundan modelin bu incelemeyi “negatif” olarak öngördüğünü söyleyebiliriz.
 Etiket de sıfır olduğundan modelin doğru bir tahmin yaptığı sonucuna varabiliriz.
"""
prediction_result=model.predict(X_test[789].reshape(1,1024))
print(f"label: {y_test[789]} | Prediction={prediction_result}")



